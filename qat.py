import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


MODEL_ID = "google/gemma-4-E2B-it"
REPO_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = REPO_DIR / "models" / MODEL_ID.split("/")[-1]
OUTPUT_DIR = REPO_DIR / "models" / f"{MODEL_ID.split('/')[-1]}-ternary-qat"
DATASET_PATH = REPO_DIR / "Datasets" / "chat_reasoning_qat_mix.parquet"

LOAD_DTYPE = torch.bfloat16
MAX_LENGTH = 1024
TRAIN_SPLIT = "train"
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 5
NUM_TRAIN_EPOCHS = 0.05
MAX_STEPS = -1
LOGGING_STEPS = 1
SAVE_STEPS = 500
SEED = 3407
DISTILL_TEMPERATURE = 2.0
DISTILL_ALPHA = 0.75
QUANTIZE_LM_HEAD = False
RUN_SMOKE_TEST = False
USE_CUDA_FOR_STUDENT = True
KEEP_TEACHER_ON_CPU = False
USE_GRADIENT_CHECKPOINTING = True
OPTIMIZER = "adamw_torch"
SAVE_ONLY_MODEL = True
DATALOADER_PIN_MEMORY = True
DATALOADER_NUM_WORKERS = 10
REPORT_TO = "none"
LOGGING_STRATEGY = "steps"
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "no"
USE_BF16_IF_AVAILABLE = True
USE_FP16_IF_AVAILABLE = True
SMOKE_TEST_PROMPT = "Write one sentence about GPU kernel optimization."
SMOKE_TEST_MAX_NEW_TOKENS = 64


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def env_dtype(name: str, default: torch.dtype) -> torch.dtype:
    value = os.environ.get(name)
    if value is None:
        return default
    return getattr(torch, value)


def configure_runtime() -> None:
    torch.set_float32_matmul_precision("high")


def get_train_device() -> str:
    if env_bool("USE_CUDA_FOR_STUDENT", USE_CUDA_FOR_STUDENT) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_load_dtype(device: str) -> torch.dtype:
    dtype = env_dtype("LOAD_DTYPE", LOAD_DTYPE)
    if device == "cuda" and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


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


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight.abs().mean().clamp_min(1e-8)
        ternary = torch.round(self.weight / scale).clamp_(-1, 1)
        quantized_weight = ternary * scale
        ste_weight = self.weight + (quantized_weight - self.weight).detach()
        return F.linear(x, ste_weight, self.bias)


def should_quantize(module_name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    protected = ["lm_head", "q_proj", "k_proj", "v_proj", "o_proj"]
    if module_name.endswith("lm_head"):
        return env_bool("QUANTIZE_LM_HEAD", QUANTIZE_LM_HEAD)
    if module_name.endswith(tuple(protected)):
        return False
    return True


def replace_linear_with_bitlinear(module: nn.Module, prefix: str = "") -> int:
    replacements = 0
    for name, child in list(module.named_children()):
        module_name = f"{prefix}.{name}" if prefix else name
        if should_quantize(module_name, child):
            bit_layer = BitLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
            )
            bit_layer.to(device=child.weight.device, dtype=child.weight.dtype)
            bit_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                bit_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, bit_layer)
            replacements += 1
        else:
            replacements += replace_linear_with_bitlinear(child, module_name)
    return replacements


def format_prompt(example: dict, tokenizer: AutoTokenizer) -> dict:
    messages = [
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": example["solution"]},
    ]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = (
            "### INSTRUCTION ###\n"
            f"{example['problem']}\n\n"
            "### RESPONSE ###\n"
            f"{example['solution']}"
        )
    return {"text": text}


def tokenize_batch(batch: dict, tokenizer: AutoTokenizer) -> dict:
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=env_int("MAX_LENGTH", MAX_LENGTH),
        padding=False,
    )
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized


def collate_causal_batch(features: list[dict], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    max_len = max(len(item["input_ids"]) for item in features)

    input_ids = []
    attention_mask = []
    labels = []
    for item in features:
        length = len(item["input_ids"])
        pad_len = max_len - length
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class DistillationTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_model: nn.Module | None,
        distill_alpha: float,
        temperature: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_alpha = distill_alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
        )
        ce_loss = student_outputs.loss

        if self.teacher_model is None:
            raise ValueError("Pure distillation requires a teacher model.")
        with torch.no_grad():
            teacher_device = next(self.teacher_model.parameters()).device
            teacher_inputs = {
                "input_ids": inputs["input_ids"].to(teacher_device),
                "attention_mask": inputs["attention_mask"].to(teacher_device),
            }
            teacher_outputs = self.teacher_model(
                **teacher_inputs,
            )

        mask = inputs["attention_mask"].bool()
        student_logits = student_outputs.logits[mask]
        teacher_logits = teacher_outputs.logits[mask.to(teacher_outputs.logits.device)].to(student_logits.device)

        if student_logits.numel() == 0 or teacher_logits.numel() == 0:
            distill_loss = student_outputs.logits.new_tensor(0.0)
        else:
            temp = self.temperature
            student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
            distill_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
            ) * (temp ** 2)
        loss = (self.distill_alpha * distill_loss) + ((1.0 - self.distill_alpha) * ce_loss)

        if return_outputs:
            return loss, student_outputs
        return loss


def load_model_and_tokenizer(model_id: str, dtype: torch.dtype) -> tuple[nn.Module, AutoTokenizer]:
    hf_token = resolve_hf_token()
    enable_fast_downloads()

    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    load_source = str(MODEL_CACHE_DIR) if local_model_ready(MODEL_CACHE_DIR) else model_id
    print(f"Loading model from: {load_source}")

    tokenizer = AutoTokenizer.from_pretrained(load_source, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_source,
        dtype=dtype,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    return model, tokenizer


@torch.no_grad()
def materialize_ternary_weights(model: nn.Module) -> dict[str, float]:
    quantized_layers = 0
    total_params = 0
    nonzero_params = 0

    for module in model.modules():
        if not isinstance(module, BitLinear):
            continue

        scale = module.weight.detach().abs().mean().clamp_min(1e-8)
        ternary = torch.round(module.weight / scale).clamp_(-1, 1)
        quantized_weight = (ternary * scale).to(module.weight.dtype)
        module.weight.data.copy_(quantized_weight)

        quantized_layers += 1
        total_params += quantized_weight.numel()
        nonzero_params += quantized_weight.ne(0).sum().item()

    density = nonzero_params / total_params if total_params else 0.0
    return {
        "quantized_layers": quantized_layers,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "density": density,
    }


def prepare_datasets(tokenizer: AutoTokenizer):
    train_dataset = load_dataset(
        "parquet",
        data_files=str(DATASET_PATH),
        split=os.environ.get("TRAIN_SPLIT", TRAIN_SPLIT),
    )
    train_dataset = train_dataset.map(
        lambda example: format_prompt(example, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    train_dataset = train_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    return train_dataset


def save_qat_model(model: nn.Module, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


@torch.no_grad()
def smoke_test(output_dir: Path, tokenizer: AutoTokenizer) -> str:
    device = get_train_device()
    model_dtype = resolve_load_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    prompt = os.environ.get("SMOKE_TEST_PROMPT", SMOKE_TEST_PROMPT)
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=env_int("SMOKE_TEST_MAX_NEW_TOKENS", SMOKE_TEST_MAX_NEW_TOKENS),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)


def main() -> None:
    configure_runtime()
    torch.manual_seed(SEED)
    device = get_train_device()
    load_dtype = resolve_load_dtype(device)
    output_dir = Path(os.environ.get("OUTPUT_DIR", str(OUTPUT_DIR)))
    distill_alpha = env_float("DISTILL_ALPHA", DISTILL_ALPHA)
    if distill_alpha <= 0.0:
        raise ValueError("Set DISTILL_ALPHA > 0.0 for teacher-supervised distillation training.")
    print(f"Runtime device: {device}; load dtype: {load_dtype}")

    teacher_model = None
    tokenizer = None
    if distill_alpha > 0.0:
        teacher_model, tokenizer = load_model_and_tokenizer(MODEL_ID, load_dtype)
        if env_bool("KEEP_TEACHER_ON_CPU", KEEP_TEACHER_ON_CPU):
            teacher_model.to("cpu")
        elif device == "cuda":
            teacher_model.to("cuda")
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)

    student_model, student_tokenizer = load_model_and_tokenizer(MODEL_ID, load_dtype)
    tokenizer = tokenizer or student_tokenizer
    replaced = replace_linear_with_bitlinear(student_model)
    print(f"Replaced {replaced} linear layers with BitLinear.")
    student_model.config.use_cache = False

    train_dataset = prepare_datasets(tokenizer)

    use_bf16 = (
        env_bool("USE_CUDA_FOR_STUDENT", USE_CUDA_FOR_STUDENT)
        and USE_BF16_IF_AVAILABLE
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = (
        env_bool("USE_CUDA_FOR_STUDENT", USE_CUDA_FOR_STUDENT)
        and USE_FP16_IF_AVAILABLE
        and torch.cuda.is_available()
        and not use_bf16
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=env_int(
            "PER_DEVICE_TRAIN_BATCH_SIZE",
            PER_DEVICE_TRAIN_BATCH_SIZE,
        ),
        gradient_accumulation_steps=env_int(
            "GRADIENT_ACCUMULATION_STEPS",
            GRADIENT_ACCUMULATION_STEPS,
        ),
        learning_rate=env_float("LEARNING_RATE", LEARNING_RATE),
        weight_decay=WEIGHT_DECAY,
        warmup_steps=env_int("WARMUP_STEPS", WARMUP_STEPS),
        num_train_epochs=env_float("NUM_TRAIN_EPOCHS", NUM_TRAIN_EPOCHS),
        max_steps=env_int("MAX_STEPS", MAX_STEPS),
        logging_steps=env_int("LOGGING_STEPS", LOGGING_STEPS),
        save_steps=env_int("SAVE_STEPS", SAVE_STEPS),
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        logging_strategy=LOGGING_STRATEGY,
        report_to=REPORT_TO,
        remove_unused_columns=False,
        dataloader_pin_memory=(
            DATALOADER_PIN_MEMORY
            and env_bool("USE_CUDA_FOR_STUDENT", USE_CUDA_FOR_STUDENT)
            and torch.cuda.is_available()
        ),
        dataloader_num_workers=env_int("DATALOADER_NUM_WORKERS", DATALOADER_NUM_WORKERS),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=env_bool("USE_GRADIENT_CHECKPOINTING", USE_GRADIENT_CHECKPOINTING),
        optim=os.environ.get("OPTIMIZER", OPTIMIZER),
        save_only_model=SAVE_ONLY_MODEL,
        seed=SEED,
        use_cpu=device != "cuda",
    )

    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        distill_alpha=distill_alpha,
        temperature=DISTILL_TEMPERATURE,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda features: collate_causal_batch(features, tokenizer),
    )

    print("Starting distillation + ternary QAT...")
    train_result = trainer.train()
    print(f"Training complete. Final training loss: {train_result.training_loss:.6f}")

    stats = materialize_ternary_weights(student_model)
    print(
        "Materialized QAT BitLinear weights to ternary values: "
        f"{stats['quantized_layers']} layers, "
        f"{stats['nonzero_params']}/{stats['total_params']} non-zero "
        f"({stats['density']:.2%} density)."
    )

    save_qat_model(student_model, tokenizer, output_dir)
    print(f"Saved QAT model to: {output_dir}")

    if env_bool("RUN_SMOKE_TEST", RUN_SMOKE_TEST):
        text = smoke_test(output_dir, tokenizer)
        print("Smoke test output:")
        print(text)


if __name__ == "__main__":
    main()
