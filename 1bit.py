import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = None
LOAD_DTYPE = "float16"
QUANTIZE_LM_HEAD = False
RUN_SMOKE_TEST = True
PREFER_CUDA = True
SMOKE_TEST_PROMPT = "hello \n"
SMOKE_TEST_MAX_NEW_TOKENS = 12


def local_model_ready(path: Path) -> bool:
    if not path.exists():
        return False
    config_ok = (path / "config.json").exists()
    weight_ok = any(
        (path / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    return config_ok and weight_ok


def resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    bashrc = Path.home() / ".bashrc"
    if not bashrc.exists():
        return None

    for line in bashrc.read_text().splitlines():
        line = line.strip()
        if line.startswith("export HF_TOKEN="):
            value = line.split("=", 1)[1].strip().strip("\"'")
            return value or None
    return None


def enable_fast_downloads() -> None:
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        return
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        return
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("Enabled hf_transfer for faster downloads.")


def ternary_quantize(weight: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, float]:
    scale = weight.detach().abs().mean().item()
    if scale <= eps:
        return torch.zeros_like(weight), 0.0

    normalized = weight / scale
    ternary = torch.round(normalized).clamp_(-1, 1)
    quantized = ternary * scale
    return quantized.to(weight.dtype), scale


def should_quantize(module_name: str, module: nn.Module, quantize_lm_head: bool) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if module_name.endswith("lm_head") and not quantize_lm_head:
        return False
    return True


@torch.no_grad()
def convert_model_to_ternary(model: nn.Module, quantize_lm_head: bool) -> dict[str, float]:
    quantized_layers = 0
    skipped_layers = 0
    total_params = 0
    ternary_nonzero = 0

    for module_name, module in model.named_modules():
        if not should_quantize(module_name, module, quantize_lm_head):
            continue

        quantized_weight, scale = ternary_quantize(module.weight.data)
        module.weight.data.copy_(quantized_weight)

        quantized_layers += 1
        total_params += quantized_weight.numel()
        ternary_nonzero += quantized_weight.ne(0).sum().item()

        if scale == 0.0:
            skipped_layers += 1

    density = ternary_nonzero / total_params if total_params else 0.0
    return {
        "quantized_layers": quantized_layers,
        "zero_scale_layers": skipped_layers,
        "total_params": total_params,
        "nonzero_params": ternary_nonzero,
        "density": density,
    }


def repo_cache_dir(repo_dir: Path, model_id: str) -> Path:
    return repo_dir / "models" / model_id.split("/")[-1]


def resolve_load_source(repo_dir: Path, model_id: str) -> tuple[str, Path, bool]:
    local_model_dir = repo_cache_dir(repo_dir, model_id)
    use_local = local_model_ready(local_model_dir)
    return (str(local_model_dir) if use_local else model_id), local_model_dir, use_local


def get_runtime_device() -> torch.device:
    if PREFER_CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str, dtype: torch.dtype) -> tuple[nn.Module, AutoTokenizer, Path]:
    repo_dir = Path(__file__).resolve().parent
    load_source, local_model_dir, use_local = resolve_load_source(repo_dir, model_id)
    hf_token = resolve_hf_token()
    enable_fast_downloads()

    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    if use_local:
        print(f"Loading model from local dir: {local_model_dir}")
    else:
        print(f"Local model cache not found. Downloading from: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(load_source, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_source,
        dtype=dtype,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.to(get_runtime_device())

    if not use_local:
        local_model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_model_dir)
        model.save_pretrained(local_model_dir)

    return model, tokenizer, local_model_dir


def save_converted_model(model: nn.Module, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


@torch.no_grad()
def smoke_test(output_dir: Path, prompt: str, max_new_tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = get_runtime_device()
    model_dtype = getattr(torch, LOAD_DTYPE) if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    dtype = getattr(torch, LOAD_DTYPE)

    model, tokenizer, local_model_dir = load_model_and_tokenizer(DEFAULT_MODEL_ID, dtype)

    output_dir = (
        Path(OUTPUT_DIR)
        if OUTPUT_DIR
        else local_model_dir.parent / f"{local_model_dir.name}-ternary-1bit-codex"
    )

    stats = convert_model_to_ternary(model, quantize_lm_head=QUANTIZE_LM_HEAD)
    print(
        "Converted linear layers to ternary weights: "
        f"{stats['quantized_layers']} layers, "
        f"{stats['nonzero_params']}/{stats['total_params']} non-zero "
        f"({stats['density']:.2%} density)."
    )
    if stats["zero_scale_layers"]:
        print(f"Encountered {stats['zero_scale_layers']} all-zero layers.")

    save_converted_model(model, tokenizer, output_dir)
    print(f"Saved ternary model to: {output_dir}")

    if RUN_SMOKE_TEST:
        text = smoke_test(output_dir, SMOKE_TEST_PROMPT, SMOKE_TEST_MAX_NEW_TOKENS)
        print("Smoke test output:")
        print(text)


if __name__ == "__main__":
    main()
