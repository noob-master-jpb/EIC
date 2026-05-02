from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load
from transformers import AutoModelForCausalLM, AutoTokenizer

from bit import (
    MODEL_ID,
    LearnableBitLinear,
    apply_ternary_surgery_to_block,
    get_transformer_blocks,
)

REPO_DIR = Path(__file__).resolve().parent
MODEL_DIR = REPO_DIR / "models" / f"{MODEL_ID.split('/')[-1]}-ternary-advanced"

PREFER_CUDA = True
MAX_NEW_TOKENS = 64
DO_SAMPLE = False

SMOKE_PROMPTS = [
    "Hello! Briefly explain what quantization does in neural networks.",
]


def runtime_device() -> str:
    if PREFER_CUDA and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_ternary_model(model_dir: Path, dtype: torch.dtype) -> torch.nn.Module:
    """
    Two-phase load for a ternary-quantized model saved by bit.py:

    Phase 1 — Rebuild architecture:
        Load the base Gemma model (standard weights from config), then apply
        ternary surgery to every transformer block so that MLP Linear layers
        become LearnableBitLinear (with alpha_pos, alpha_neg, delta_param).

    Phase 2 — Restore weights:
        Load the saved safetensors state dict and inject it with strict=False
        so the custom ternary parameters land in the right places.
        Any truly missing or unexpected keys are printed for diagnostics.
    """
    print("  Phase 1: loading base architecture ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    print("  Phase 1: applying ternary surgery to all blocks ...")
    blocks = get_transformer_blocks(model)
    total_ternary, total_attn = 0, 0
    for block in blocks:
        t, a = apply_ternary_surgery_to_block(block)
        total_ternary += t
        total_attn    += a
    print(f"  Surgery complete: {total_ternary} MLP→ternary, {total_attn} attn dequantized")

    print("  Phase 2: loading saved weights ...")
    # Collect all safetensors shards in the model directory
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    state_dict = {}
    for shard in shards:
        state_dict.update(safetensors_load(str(shard), device="cpu"))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if unexpected:
        print(f"  [WARNING] {len(unexpected)} still-unexpected keys (should be 0 after surgery):")
        for k in unexpected[:10]:
            print(f"    {k}")
    if missing:
        print(f"  [INFO] {len(missing)} missing keys (base weights not in checkpoint):")
        for k in missing[:5]:
            print(f"    {k}")

    print("  Weights loaded successfully.")
    return model


def main() -> None:
    if not MODEL_DIR.exists():
        print(f"Error: model path does not exist: {MODEL_DIR}")
        return

    device = runtime_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from: {MODEL_DIR}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_ternary_model(MODEL_DIR, dtype)
    model.to(device)
    model.eval()

    for index, prompt in enumerate(SMOKE_PROMPTS, start=1):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n=== Smoke Prompt {index} ===")
        print(prompt)
        print("--- Response ---")
        print(text)


if __name__ == "__main__":
    main()
