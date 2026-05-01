import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


MODEL_ID = "Qwen/Qwen3-0.6B"
REPO_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = REPO_DIR / "models" / MODEL_ID.split("/")[-1]
OUTPUT_DIR = REPO_DIR / "models" / f"{MODEL_ID.split('/')[-1]}-ternary-qat"
DATASET_PATH = REPO_DIR / "Datasets" / "nvidia_compute_eval.parquet"

LOAD_DTYPE = torch.float16
MAX_LENGTH = 256
TRAIN_SPLIT = "train[:128]"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0
NUM_TRAIN_EPOCHS = 1.0
MAX_STEPS = 1
LOGGING_STEPS = 1
SAVE_STEPS = 1
SEED = 3407
DISTILL_TEMPERATURE = 2.0
DISTILL_ALPHA = 0.5
QUANTIZE_LM_HEAD = False
RUN_SMOKE_TEST = True
USE_CUDA_FOR_STUDENT = True
KEEP_TEACHER_ON_CPU = True
USE_GRADIENT_CHECKPOINTING = True
OPTIMIZER = "adafactor"
SAVE_ONLY_MODEL = True
DATALOADER_PIN_MEMORY = True
REPORT_TO = "none"
LOGGING_STRATEGY = "steps"
SAVE_STRATEGY = "steps"
EVAL_STRATEGY = "no"
USE_BF16_IF_AVAILABLE = True
USE_FP16_IF_AVAILABLE = True
SMOKE_TEST_PROMPT = "Write one sentence about CUDA kernel optimization."
SMOKE_TEST_MAX_NEW_TOKENS = 16


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
        quantized = torch.round(self.weight / scale).clamp_(-1, 1)
        ste_weight = self.weight + (quantized - self.weight).detach()
        return F.linear(x, ste_weight * scale, self.bias)


def should_quantize(module_name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if module_name.endswith("lm_head") and not QUANTIZE_LM_HEAD:
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
            bit_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                bit_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, bit_layer)
            replacements += 1
        else:
            replacements += replace_linear_with_bitlinear(child, module_name)
    return replacements


def format_prompt(example: dict) -> dict:
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
        max_length=MAX_LENGTH,
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
    def __init__(self, *args, teacher_model: nn.Module, distill_alpha: float, temperature: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_alpha = distill_alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        student_loss = student_outputs.loss

        with torch.no_grad():
            teacher_device = next(self.teacher_model.parameters()).device
            teacher_inputs = {
                "input_ids": inputs["input_ids"].to(teacher_device),
                "attention_mask": inputs["attention_mask"].to(teacher_device),
            }
            teacher_outputs = self.teacher_model(
                **teacher_inputs,
            )

        mask = labels.ne(-100)
        student_logits = student_outputs.logits[mask]
        teacher_mask = mask.to(teacher_outputs.logits.device)
        teacher_targets = teacher_outputs.logits.argmax(dim=-1)[teacher_mask].to(student_logits.device)

        if student_logits.numel() == 0:
            distill_loss = student_loss.new_tensor(0.0)
        else:
            temp = self.temperature
            distill_loss = F.cross_entropy(student_logits / temp, teacher_targets) * (temp ** 2)

        loss = (1.0 - self.distill_alpha) * student_loss + self.distill_alpha * distill_loss
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


def prepare_datasets(tokenizer: AutoTokenizer):
    train_dataset = load_dataset("parquet", data_files=str(DATASET_PATH), split=TRAIN_SPLIT)
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
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
    device = "cuda" if USE_CUDA_FOR_STUDENT and torch.cuda.is_available() else "cpu"
    model_dtype = LOAD_DTYPE if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    inputs = tokenizer(SMOKE_TEST_PROMPT, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=SMOKE_TEST_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    torch.manual_seed(SEED)

    teacher_model, tokenizer = load_model_and_tokenizer(MODEL_ID, LOAD_DTYPE)
    if KEEP_TEACHER_ON_CPU:
        teacher_model.to("cpu")
    elif torch.cuda.is_available():
        teacher_model.to("cuda")
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    student_model, _ = load_model_and_tokenizer(MODEL_ID, LOAD_DTYPE)
    replaced = replace_linear_with_bitlinear(student_model)
    print(f"Replaced {replaced} linear layers with BitLinear.")
    student_model.config.use_cache = False

    train_dataset = prepare_datasets(tokenizer)

    use_bf16 = (
        USE_CUDA_FOR_STUDENT
        and USE_BF16_IF_AVAILABLE
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = (
        USE_CUDA_FOR_STUDENT
        and USE_FP16_IF_AVAILABLE
        and torch.cuda.is_available()
        and not use_bf16
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        logging_strategy=LOGGING_STRATEGY,
        report_to=REPORT_TO,
        remove_unused_columns=False,
        dataloader_pin_memory=(
            DATALOADER_PIN_MEMORY and USE_CUDA_FOR_STUDENT and torch.cuda.is_available()
        ),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        optim=OPTIMIZER,
        save_only_model=SAVE_ONLY_MODEL,
        seed=SEED,
        use_cpu=not (USE_CUDA_FOR_STUDENT and torch.cuda.is_available()),
    )

    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        distill_alpha=DISTILL_ALPHA,
        temperature=DISTILL_TEMPERATURE,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda features: collate_causal_batch(features, tokenizer),
    )

    print("Starting distillation + ternary QAT...")
    train_result = trainer.train()
    print(f"Training complete. Final training loss: {train_result.training_loss:.6f}")

    save_qat_model(student_model, tokenizer, OUTPUT_DIR)
    print(f"Saved QAT model to: {OUTPUT_DIR}")

    if RUN_SMOKE_TEST:
        text = smoke_test(OUTPUT_DIR, tokenizer)
        print("Smoke test output:")
        print(text)


if __name__ == "__main__":
    main()
