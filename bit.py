import os
import inspect
import math
import json
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from contextlib import nullcontext
from torch.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID        = "google/gemma-4-E2B-it"
REPO_DIR        = Path(__file__).resolve().parent

DATASET_SOURCES = [
    "Datasets/smoltalk_chat.parquet",
    "Datasets/open_thoughts_reasoning.parquet",
    "Datasets/cass_part1.parquet",
    "Datasets/cass_part2.parquet",
    "Datasets/numina_math_reasoning.parquet",
    "Datasets/openhermes.parquet",
    
]

OUTPUT_DIR     = REPO_DIR / "models"      / f"{MODEL_ID.split('/')[-1]}-ternary-advanced"
CHECKPOINT_DIR = REPO_DIR / "checkpoints" / f"{MODEL_ID.split('/')[-1]}-ternary"

# -- Data ----------------------------------
MAX_LENGTH               = 1000
TRAIN_SPLIT              = "train"
CALIBRATION_BATCH_SIZE   = 8
CALIBRATION_SAMPLES      = 256
DATASET_VAL_SPLIT_FACTOR = 0.01

# -- Training ------------------------------
STEPS_PER_BLOCK    = 32
EPOCHS_PER_BLOCK   = 4
LEARNING_RATE      = 2e-4
ATTN_LEARNING_RATE = 2e-3   # lower LR for full-precision attention weights
GRAD_CLIP          = 1.0

# -- Validation ----------------------------
VAL      = False
VAL_STEP = 1

# -- Checkpointing -------------------------
CHECKPOINT_EVERY_N_BLOCKS = 0
RESUME_FROM_BLOCK         = 0

# -- Teacher placement ---------------------
TEACHER_ON_CPU      = False
TEACHER_CPU_THREADS = 16

# -- Distillation --------------------------
DISTILL_TEMPERATURE = 2.0
DISTILL_ALPHA       = 0 # FIX: was 0, JSD branch was completely dead
ATTN_LOSS_WEIGHT    = 0.1

# -- Attention handling --------------------
# ATTN_PRESERVE=True      q/k/v/o_proj are never replaced with LearnableBitLinear
# UNFREEZE_ATTENTION=True those layers ARE trained at full precision
ATTN_PRESERVE       = True
UNFREEZE_ATTENTION  = True

# -- Student loading -----------------------
STUDENT_4BIT = False   # bf16 full-precision warm-start

# ==========================================
# HELPERS
# ==========================================
def get_train_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def model_accepts_mm_token_type_ids(model: nn.Module) -> bool:
    try:
        return "mm_token_type_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def ensure_mm_token_type_ids(inputs: dict, model: nn.Module) -> dict:
    if "mm_token_type_ids" in inputs:
        return inputs
    if not model_accepts_mm_token_type_ids(model):
        return inputs
    inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    return inputs


def estimate_vram(model_id: str, teacher_on_cpu: bool, student_4bit: bool) -> None:
    param_map = {"31b": 31e9, "27b": 27e9, "13b": 13e9, "7b": 7e9, "3b": 3e9, "2b": 2e9}
    params     = next((v for k, v in param_map.items() if k in model_id.lower()), 7e9)
    student_gb = params * (0.5 if student_4bit else 2.0) / 1e9
    teacher_gb = 0.0 if teacher_on_cpu else params * 2.0 / 1e9
    total_gb   = student_gb + teacher_gb

    print("\n" + "-" * 60)
    print("  VRAM estimate:")
    print(f"    Student            : ~{student_gb:.1f} GB ({'4-bit NF4' if student_4bit else 'bf16'})")
    if teacher_on_cpu:
        print(f"    Teacher            : CPU  (saves ~{params * 2 / 1e9:.1f} GB GPU VRAM)")
    else:
        print(f"    Teacher (bf16)     : ~{teacher_gb:.1f} GB")
    print(f"    Total GPU          : ~{total_gb:.1f} GB")
    if torch.cuda.is_available():
        avail = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    GPU capacity       : ~{avail:.1f} GB")
        if total_gb > avail * 0.90:
            print(f"  [WARNING] Estimated usage ({total_gb:.1f} GB) > 90% of GPU RAM.")
    print("-" * 60 + "\n")


def move_model_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Robustly move every parameter and buffer to `device`.

    Standard .to(device) silently leaves some tensors on CPU when the model
    was loaded with low_cpu_mem_usage=True or with bitsandbytes quantization.
    This function walks every named parameter and buffer and forces the move,
    which is the fix for 0% GPU utilization during training.
    """
    try:
        model = model.to(device)
    except Exception as e:
        print(f"  [WARNING] model.to({device}) raised: {e} — moving per-parameter.")

    moved, skipped = 0, 0
    for _name, param in model.named_parameters():
        try:
            if param.device != device:
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
                moved += 1
        except Exception:
            skipped += 1

    for _name, buf in model.named_buffers():
        try:
            if buf.device != device:
                buf.data = buf.data.to(device)
        except Exception:
            pass

    total = sum(1 for _ in model.named_parameters())
    print(f"  move_model_to_device({device}): {total} params total, "
          f"{moved} moved, {skipped} skipped.")
    try:
        print(f"  First parameter device: {next(model.parameters()).device}")
    except StopIteration:
        pass
    return model


# ==========================================
# 1. LEARNABLE TERNARY LAYER (TTQ / STE)
# ==========================================
class LearnableBitLinear(nn.Linear):
    """
    Ternary linear layer — Trained Ternary Quantization (TTQ).
    Weights -> {-alpha_neg*scale, 0, +alpha_pos*scale} via learnable threshold.
    STE keeps full-precision gradients flowing. MLP layers only.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.delta_param = nn.Parameter(torch.tensor(0.05))
        self.alpha_pos   = nn.Parameter(torch.tensor(1.0))
        self.alpha_neg   = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w     = self.weight
        scale = w.abs().mean().clamp_min(1e-8).detach()
        w_n   = w / scale
        mask_p  = (w_n >  self.delta_param).to(w.dtype)
        mask_n  = (w_n < -self.delta_param).to(w.dtype)
        quant_w = (mask_p * self.alpha_pos + mask_n * -self.alpha_neg) * scale
        ste_w   = w + (quant_w - w).detach()
        return F.linear(x, ste_w, self.bias)


# ==========================================
# 2. BLOCK SURGERY  (visited-set safe)
# ==========================================
ATTN_PROJ_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj"}


def apply_ternary_surgery_to_block(
    block: nn.Module,
    _visited: set | None = None,
) -> tuple[int, int]:
    """
    Single-pass surgery:
      - MLP / non-attention Linear or Linear4bit  ->  LearnableBitLinear
      - Attention projection Linear4bit           ->  plain nn.Linear (bf16)
        (already-plain nn.Linear attention projections are left untouched)

    Returns (ternary_replacements, attn_dequantized).
    Visited-set prevents double-processing shared sub-modules.
    """
    if _visited is None:
        _visited = set()

    ternary_n  = 0
    attn_deq_n = 0

    for name, child in list(block.named_children()):
        cid = id(child)
        if cid in _visited:
            continue
        _visited.add(cid)

        is_attn = any(p in name for p in ATTN_PROJ_NAMES)

        if isinstance(child, (nn.Linear, bnb.nn.Linear4bit)):
            if is_attn:
                # Dequantize 4-bit attention projections to plain bf16 nn.Linear
                # so requires_grad_(True) works. Plain nn.Linear: leave alone.
                if isinstance(child, bnb.nn.Linear4bit):
                    fp = nn.Linear(
                        child.in_features, child.out_features,
                        bias=child.bias is not None,
                    )
                    fp.to(device=child.weight.device, dtype=torch.bfloat16)
                    dq = bnb.functional.dequantize_4bit(
                        child.weight.data, child.weight.quant_state
                    ).to(torch.bfloat16)
                    fp.weight.data.copy_(dq)
                    if child.bias is not None:
                        fp.bias.data.copy_(child.bias.data)
                    setattr(block, name, fp)
                    attn_deq_n += 1
            else:
                # MLP / other projections -> ternary
                bit = LearnableBitLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                )
                bit.to(device=child.weight.device, dtype=torch.bfloat16)
                if isinstance(child, bnb.nn.Linear4bit):
                    dq = bnb.functional.dequantize_4bit(
                        child.weight.data, child.weight.quant_state
                    ).to(torch.bfloat16)
                    bit.weight.data.copy_(dq)
                else:
                    bit.weight.data.copy_(child.weight.data.to(torch.bfloat16))
                if child.bias is not None:
                    bit.bias.data.copy_(child.bias.data)
                setattr(block, name, bit)
                ternary_n += 1
        else:
            tc, ac = apply_ternary_surgery_to_block(child, _visited)
            ternary_n  += tc
            attn_deq_n += ac

    return ternary_n, attn_deq_n


# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================
def compute_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    s = F.softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    m = (0.5 * (s + t)).clamp_min(1e-8)
    kl_s = (s * (s.clamp_min(1e-8).log() - m.log())).sum(-1).mean()
    kl_t = (t * (t.clamp_min(1e-8).log() - m.log())).sum(-1).mean()
    return 0.5 * (kl_s + kl_t) * (temperature ** 2)


def compute_kl_loss(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    """KL divergence averaged over every independent distribution in the batch.

    Attention tensors arrive as [B, H, Sq, Sk].  ``batchmean`` divides only by
    the first dimension, giving sum / B instead of sum / (B*H*Sq).  We flatten
    all non-distribution axes so that batchmean correctly normalises by the
    number of individual distributions (one per query-position per head).
    """
    eps = 1e-8
    # Flatten to [N, D] where D is the distribution dimension (last dim).
    s = student_probs.reshape(-1, student_probs.shape[-1])
    t = teacher_probs.reshape(-1, teacher_probs.shape[-1])
    return F.kl_div(s.clamp_min(eps).log(), t.clamp_min(eps), reduction="batchmean")


# ==========================================
# 4. TEACHER FORWARD  (device-agnostic)
# ==========================================
def teacher_forward(
    teacher: nn.Module,
    inputs: dict,
    use_attn: bool,
    teacher_device: torch.device,
) -> object:
    t_inputs = {k: v.to(teacher_device) for k, v in inputs.items() if k != "labels"}
    with torch.no_grad():
        ctx = (
            autocast("cuda", dtype=torch.bfloat16)
            if str(teacher_device) != "cpu" and torch.cuda.is_available()
            else nullcontext()
        )
        with ctx:
            return teacher(**t_inputs, output_attentions=use_attn)


# ==========================================
# 5. DATASET PREP
# ==========================================
def _first_present(example: dict, keys: list) -> str | None:
    for k in keys:
        if k in example and example[k] is not None:
            v = example[k]
            s = v if isinstance(v, str) else str(v)
            if s.strip():
                return s.strip()
    return None


def normalize_schema(example: dict) -> dict:
    for msg_key in ("messages", "conversations", "conversation"):
        raw = example.get(msg_key)
        if not isinstance(raw, list):
            continue
        problem = solution = None
        for msg in raw:
            if not isinstance(msg, dict):
                continue
            role    = str(msg.get("role", msg.get("from", ""))).lower()
            content = msg.get("content", msg.get("value", ""))
            if not content:
                continue
            if role in ("user", "human") and problem is None:
                problem = str(content).strip()
            elif role in ("assistant", "gpt", "model", "bot") and solution is None:
                solution = str(content).strip()
        if problem and solution:
            return {"problem": problem, "solution": solution}
    p_keys = ["problem", "query", "instruction", "prompt", "question", "Problem"]
    s_keys = ["solution", "answer", "response", "Response", "output", "completion"]
    return {
        "problem":  _first_present(example, p_keys),
        "solution": _first_present(example, s_keys),
    }


def format_prompt(example: dict, tokenizer: AutoTokenizer) -> dict:
    messages = [
        {"role": "user",      "content": example["problem"]},
        {"role": "assistant", "content": example["solution"]},
    ]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        text = (
            f"### INSTRUCTION ###\n{example['problem']}\n\n"
            f"### RESPONSE ###\n{example['solution']}"
        )
    return {"text": text}


def tokenize_batch(batch: dict, tokenizer: AutoTokenizer) -> dict:
    tok = tokenizer(
        batch["text"],
        truncation=True,
        max_length=env_int("MAX_LENGTH", MAX_LENGTH),
        padding=False,
    )
    tok["labels"] = [ids[:] for ids in tok["input_ids"]]
    return tok


def collate_causal_batch(features: list, tokenizer: AutoTokenizer) -> dict:
    pad_id  = tokenizer.pad_token_id
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"]         + [pad_id] * pad)
        attention_mask.append(f["attention_mask"] + [0]      * pad)
        labels.append(f["labels"]               + [-100]  * pad)
    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
    }


def build_loader(dataset, tokenizer, batch_size, seed, max_samples, shuffle):
    if dataset is None:
        return None
    ds = dataset
    if max_samples > 0 and ds.num_rows > max_samples:
        idx = torch.randperm(ds.num_rows, generator=torch.Generator().manual_seed(seed))
        ds  = ds.select(idx[:max_samples].tolist())
    if shuffle:
        ds = ds.shuffle(seed=seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=env_int("CALIBRATION_NUM_WORKERS", max(1, (os.cpu_count() or 1) // 2)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda f: collate_causal_batch(f, tokenizer),
    )


def get_calibration_datasets(sources: list, tokenizer: AutoTokenizer):
    train_split      = os.environ.get("TRAIN_SPLIT", TRAIN_SPLIT)
    shuffle_each     = env_bool("CALIBRATION_SHUFFLE_EACH_FILE", True)
    val_split_factor = env_float("DATASET_VAL_SPLIT_FACTOR", DATASET_VAL_SPLIT_FACTOR)
    seed             = env_int("CALIBRATION_SEED", 42)
    train_parts, val_parts = [], []

    for i, source in enumerate(sources):
        try:
            path    = Path(source)
            fmt     = "parquet" if path.suffix == ".parquet" else "json"
            ds_dict = (
                load_dataset(fmt, data_files=str(path))
                if path.exists()
                else load_dataset(source)
            )
            available = list(ds_dict.keys())
            cur_split = train_split if train_split in available else available[0]
            ds_train  = ds_dict[cur_split]
            ds_val    = ds_dict.get("validation", ds_dict.get("test"))

            if not ds_val and ds_train.num_rows > 10 and val_split_factor > 0:
                split    = ds_train.train_test_split(test_size=val_split_factor, seed=seed)
                ds_train = split["train"]
                ds_val   = split["test"]

            def process(ds, seed_offset=0):
                if shuffle_each and ds.num_rows > 1:
                    ds = ds.shuffle(seed=seed + i + seed_offset)
                ds = ds.map(normalize_schema, remove_columns=ds.column_names)
                ds = ds.filter(lambda ex: ex["problem"] and ex["solution"])
                if ds.num_rows == 0:
                    return None
                ds = ds.map(lambda ex: format_prompt(ex, tokenizer), remove_columns=ds.column_names)
                ds = ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True, remove_columns=["text"])
                return ds

            pt = process(ds_train)
            if pt: train_parts.append(pt)
            if ds_val:
                pv = process(ds_val, seed_offset=1000)
                if pv: val_parts.append(pv)
        except Exception as e:
            print(f"  [WARNING] Error loading '{source}': {e}")

    if not train_parts:
        raise ValueError("No valid training samples found for calibration.")

    train_ds = concatenate_datasets(train_parts)
    val_ds   = concatenate_datasets(val_parts) if val_parts else None
    print(
        f"  Calibration data: {train_ds.num_rows:,} train rows"
        + (f", {val_ds.num_rows:,} val rows" if val_ds else ", no val split")
    )
    return train_ds, val_ds


# ==========================================
# 6. MODEL UTILITIES
# ==========================================
def get_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    base = model.model if hasattr(model, "model") else model
    if hasattr(base, "language_model"):
        base = base.language_model
    for path in (
        ("layers",),
        ("decoder", "layers"),
        ("transformer", "h"),
        ("transformer", "blocks"),
        ("blocks",),
    ):
        cur = base
        for attr in path:
            if not hasattr(cur, attr): cur = None; break
            cur = getattr(cur, attr)
        if isinstance(cur, (nn.ModuleList, list, tuple)):
            return cur
    raise AttributeError("Cannot locate transformer blocks in model.")


# ==========================================
# 7. CHECKPOINTING
# ==========================================
def save_checkpoint(student: nn.Module, tokenizer: AutoTokenizer, block_idx: int, ckpt_dir: Path):
    block_dir = ckpt_dir / f"block_{block_idx:04d}"
    block_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(block_dir, safe_serialization=True)
    tokenizer.save_pretrained(block_dir)
    meta = {"completed_block": block_idx, "timestamp": time.time()}
    (block_dir / "ckpt_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Checkpoint saved -> {block_dir}")


def latest_checkpoint(ckpt_dir: Path) -> int:
    if not ckpt_dir.exists():
        return -1
    indices = []
    for meta_file in ckpt_dir.glob("block_*/ckpt_meta.json"):
        try:
            meta = json.loads(meta_file.read_text())
            indices.append(int(meta["completed_block"]))
        except Exception:
            pass
    return max(indices) if indices else -1


# ==========================================
# 8. BLOCK-WISE QAT ENGINE
# ==========================================
def execute_block_wise_qat(
    student:          nn.Module,
    teacher:          nn.Module | None,
    train_dataset,
    val_dataset,
    tokenizer:        AutoTokenizer,
    batch_size:       int,
    steps_per_block:  int,
    epochs_per_block: int,
    distill_alpha:    float,
    temperature:      float,
    attn_weight:      float,
    compute_dtype:    torch.dtype,
    teacher_device:   torch.device,
    student_device:   torch.device,
    ckpt_dir:         Path,
    ckpt_every:       int,
    resume_from:      int,
):
    blocks        = get_transformer_blocks(student)
    seed          = env_int("CALIBRATION_SEED", 42)
    max_samp      = env_int("CALIBRATION_SAMPLES", CALIBRATION_SAMPLES)
    grad_clip     = env_float("GRAD_CLIP", GRAD_CLIP)
    use_attn      = attn_weight > 0 and teacher is not None
    # compute_attn: request output_attentions whenever attn_weight > 0 so the
    # alignment metric is always visible in the log, even when UNFREEZE_ATTENTION
    # is False.  attn_in_loss: only add attn_loss to the backpropped loss when
    # attention params are actually trainable (frozen Q/K block the gradient
    # anyway, so including it would be a wasted backward pass).
    compute_attn  = use_attn          # alias — kept for readability below
    # attn_in_loss is resolved per-block (depends on unfreeze_attn), set below.
    verbose       = env_bool("VERBOSE", True)
    v_every       = env_int("VERBOSE_EVERY", 1)
    run_val       = env_bool("VAL", VAL)
    val_step      = env_int("VAL_STEP", VAL_STEP)
    unfreeze_attn = env_bool("UNFREEZE_ATTENTION", UNFREEZE_ATTENTION)
    attn_lr       = env_float("ATTN_LEARNING_RATE", ATTN_LEARNING_RATE)
    mlp_lr        = env_float("LEARNING_RATE", LEARNING_RATE)

    val_loader = (
        build_loader(val_dataset, tokenizer, batch_size, seed + 10_000, 0, False)
        if run_val else None
    )

    # Build autocast context once; nullcontext on CPU avoids overhead.
    fwd_ctx = autocast("cuda", dtype=compute_dtype) if torch.cuda.is_available() else nullcontext()

    width = 110
    print("\n" + "=" * width)
    print(
        f"{'LAYER':<8} | {'STEPS':<6} | {'START':<10} | {'TRAIN':<10} | "
        f"{'VAL':<10} | {'V-BATCHES':<10} | {'ATTN':<10} | {'TIME(s)':<8}"
    )
    print("-" * width)

    for i, block in enumerate(blocks):
        if i < resume_from:
            print(f"Layer {i + 1:02d}   | SKIPPED (resume_from={resume_from})")
            continue

        t0 = time.time()
        if verbose:
            print(f"\n--- Block {i + 1}/{len(blocks)}: sampling up to {max_samp} rows ...")

        cal_loader = build_loader(
            train_dataset, tokenizer, batch_size,
            seed=seed + i, max_samples=max_samp, shuffle=True,
        )

        # ── Surgery ──────────────────────────────────────────────────
        ternary_n, attn_deq_n = apply_ternary_surgery_to_block(block)
        if verbose:
            print(f"Block {i + 1}: {ternary_n} MLP layers -> ternary | "
                  f"{attn_deq_n} attn projections dequantized to fp")

        # ── Selective unfreezing with per-group LRs ───────────────────
        student.requires_grad_(False)

        mlp_params  = []
        attn_params = []
        seen_ids    = set()

        def _add(param: nn.Parameter, bucket: list) -> None:
            pid = id(param)
            if pid in seen_ids or not param.is_floating_point():
                return
            seen_ids.add(pid)
            param.requires_grad_(True)
            bucket.append(param)

        for mod in block.modules():
            if isinstance(mod, LearnableBitLinear):
                for p in mod.parameters():
                    _add(p, mlp_params)

        if unfreeze_attn:
            for name, param in block.named_parameters():
                if any(
                    name.endswith(f"{proj}.weight") or name.endswith(f"{proj}.bias")
                    for proj in ATTN_PROJ_NAMES
                ):
                    _add(param, attn_params)
                elif "attn" in name.lower() or "attention" in name.lower():
                    _add(param, attn_params)

        if verbose:
            print(f"Block {i + 1}: {len(mlp_params)} MLP params (lr={mlp_lr:.0e}) | "
                  f"{len(attn_params)} attn params (lr={attn_lr:.0e})")

        param_groups = [{"params": mlp_params, "lr": mlp_lr}]
        if attn_params:
            param_groups.append({"params": attn_params, "lr": attn_lr})
        optimizer = torch.optim.AdamW(param_groups)

        total_steps   = 0
        start_loss    = 0.0
        running_train = 0.0
        running_attn  = 0.0

        for _epoch in range(epochs_per_block):
            student.train()
            epoch_steps = 0
            for batch in cal_loader:
                # limit steps per epoch to `steps_per_block` (if >0)
                if 0 < steps_per_block <= epoch_steps:
                    break

                optimizer.zero_grad()
                inputs = {k: v.to(student_device) for k, v in batch.items()}
                inputs = ensure_mm_token_type_ids(inputs, student)

                # Teacher forward (CPU or GPU — never holds GPU mem when on CPU)
                t_out = None
                if teacher is not None:
                    t_out = teacher_forward(teacher, inputs, compute_attn, teacher_device)

                # Student forward on GPU with autocast
                with fwd_ctx:
                    s_out = student(**inputs, output_attentions=compute_attn)

                ce_loss = (
                    s_out.loss
                    if "labels" in inputs and s_out.loss is not None
                    else s_out.logits.new_tensor(0.0)
                )

                # Attention distillation — always computed for monitoring;
                # only enters the loss when attention params are being trained
                # (frozen Q/K would yield zero gradient to MLP params anyway).
                attn_loss = s_out.logits.new_tensor(0.0)
                if compute_attn and t_out is not None and t_out.attentions is not None:
                    t_attn = t_out.attentions[i].to(student_device)
                    s_attn = s_out.attentions[i]
                    attn_loss = compute_kl_loss(s_attn, t_attn)

                # JSD distillation
                jsd_loss = s_out.logits.new_tensor(0.0)
                if t_out is not None and distill_alpha > 0:
                    mask = inputs["attention_mask"].bool()
                    if mask.any():
                        s_logits = s_out.logits[mask]
                        t_logits = t_out.logits.to(student_device)[mask]
                        jsd_loss = compute_jsd_loss(s_logits, t_logits, temperature)

                # attn_loss contributes to backprop only when attention weights
                # are unfrozen; otherwise log it but don't waste a backward pass.
                attn_in_loss = attn_weight if unfreeze_attn else 0.0
                loss = (
                    distill_alpha    * jsd_loss
                    + (1.0 - distill_alpha) * ce_loss
                    + attn_in_loss   * attn_loss
                )

                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(mlp_params + attn_params, grad_clip)
                optimizer.step()

                if total_steps == 0:
                    start_loss = loss.item()
                running_train += loss.item()
                running_attn  += attn_loss.item() if isinstance(attn_loss, torch.Tensor) else 0.0
                total_steps   += 1
                epoch_steps   += 1

                if verbose and total_steps % v_every == 0:
                    print(
                        f"  B{i + 1} step {total_steps}: "
                        f"loss={loss.item():.4f}  "
                        f"ce={ce_loss.item():.4f}  "
                        f"jsd={jsd_loss.item() if isinstance(jsd_loss, torch.Tensor) else 0:.4f}  "
                        f"attn={attn_loss.item() if isinstance(attn_loss, torch.Tensor) else 0:.4f}"
                    )

        # Validation (first batch only for speed)
        val_loss, val_batches = 0.0, 0
        if val_loader and run_val:
            student.eval()
            with torch.no_grad():
                for bid, batch in enumerate(val_loader, 1):
                    inputs = {k: v.to(student_device) for k, v in batch.items()}
                    inputs = ensure_mm_token_type_ids(inputs, student)
                    with fwd_ctx:
                        s_out = student(**inputs, output_attentions=compute_attn)
                    ce = s_out.loss if s_out.loss is not None else s_out.logits.new_tensor(0.0)

                    t_out = None
                    if teacher is not None:
                        t_out = teacher_forward(teacher, inputs, use_attn, teacher_device)

                    mask   = inputs["attention_mask"].bool()
                    d_loss = s_out.logits.new_tensor(0.0)
                    va     = s_out.logits.new_tensor(0.0)
                    if t_out is not None:
                        if mask.any():
                            sl = s_out.logits[mask]
                            tl = t_out.logits.to(student_device)[mask]
                            d_loss = compute_jsd_loss(sl, tl, temperature)
                        if compute_attn and t_out.attentions:
                            va = compute_kl_loss(
                                s_out.attentions[i],
                                t_out.attentions[i].to(student_device),
                            )
                    attn_in_loss_val = attn_weight if unfreeze_attn else 0.0
                    batch_loss  = distill_alpha * d_loss + (1 - distill_alpha) * ce + attn_in_loss_val * va
                    val_loss   += batch_loss.item()
                    val_batches += 1
                    if verbose and bid % val_step == 0:
                        print(f"  B{i + 1} val batch {bid}: loss={batch_loss.item():.4f}")
                    break  # one batch only

            val_loss = val_loss / val_batches if val_batches else 0.0

        avg_train = running_train / total_steps if total_steps else 0.0
        avg_attn  = running_attn  / total_steps if total_steps else 0.0
        elapsed   = time.time() - t0

        print(
            f"Layer {i + 1:02d}   | {total_steps:<6} | {start_loss:<10.4f} | "
            f"{avg_train:<10.4f} | {val_loss:<10.4f} | {val_batches:<10} | "
            f"{avg_attn:<10.4f} | {elapsed:<8.1f}"
        )

        for p in block.parameters():
            p.requires_grad_(False)

        if ckpt_every > 0 and (i + 1) % ckpt_every == 0:
            save_checkpoint(student, tokenizer, i, ckpt_dir)

    print("=" * width)
    print(f"All {len(blocks)} blocks processed.\n")
    return student


# ==========================================
# 9. MAIN
# ==========================================
def main():
    student_device = torch.device(get_train_device())
    compute_dtype  = get_gpu_dtype()

    teacher_on_cpu = env_bool("TEACHER_ON_CPU",            TEACHER_ON_CPU)
    cpu_threads    = env_int( "TEACHER_CPU_THREADS",        TEACHER_CPU_THREADS)
    distill_alpha  = env_float("DISTILL_ALPHA",             DISTILL_ALPHA)
    temperature    = env_float("DISTILL_TEMPERATURE",       DISTILL_TEMPERATURE)
    attn_weight    = env_float("ATTN_LOSS_WEIGHT",          ATTN_LOSS_WEIGHT)
    student_4bit   = env_bool( "STUDENT_4BIT",              STUDENT_4BIT)
    ckpt_every     = env_int(  "CHECKPOINT_EVERY_N_BLOCKS", CHECKPOINT_EVERY_N_BLOCKS)
    resume_from    = env_int(  "RESUME_FROM_BLOCK",         RESUME_FROM_BLOCK)

    teacher_device = torch.device("cpu") if teacher_on_cpu else student_device
    attn_impl      = "eager" if attn_weight > 0 else None

    if teacher_on_cpu:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
        print(f"CPU teacher mode: {cpu_threads} intra-op / "
              f"{max(1, cpu_threads // 2)} inter-op threads")

    print(f"\nDevice:            {student_device} | Dtype: {compute_dtype}")
    print(f"Teacher:           {'CPU (%d threads)' % cpu_threads if teacher_on_cpu else student_device}")
    print(f"Student 4-bit:     {student_4bit}")
    print(f"ATTN_PRESERVE:     True (always)")
    print(f"UNFREEZE_ATTENTION:{env_bool('UNFREEZE_ATTENTION', UNFREEZE_ATTENTION)}  "
          f"attn_lr={env_float('ATTN_LEARNING_RATE', ATTN_LEARNING_RATE):.0e}")
    print(f"MLP lr:            {env_float('LEARNING_RATE', LEARNING_RATE):.0e}")
    print(f"Distill alpha:     {distill_alpha}  | Temperature: {temperature}")
    print(f"Attn loss weight:  {attn_weight}")
    print(f"Grad clip:         {env_float('GRAD_CLIP', GRAD_CLIP)}")
    print(f"Checkpoint every:  {ckpt_every} blocks -> {CHECKPOINT_DIR}")
    if resume_from > 0:
        print(f"Resume from block: {resume_from}")

    estimate_vram(MODEL_ID, teacher_on_cpu, student_4bit)

    if resume_from == 0:
        found = latest_checkpoint(CHECKPOINT_DIR)
        if found >= 0:
            resume_from = found + 1
            print(f"  Auto-resume: checkpoint at block {found}, resuming from {resume_from}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Teacher ──────────────────────────────────────────────────────
    teacher = None
    if distill_alpha > 0 or attn_weight > 0:
        print(f"\nLoading Teacher ({compute_dtype}) -> {teacher_device} ...")
        t_kwargs = {"torch_dtype": compute_dtype, "low_cpu_mem_usage": True}
        if attn_impl:
            t_kwargs["attn_implementation"] = attn_impl
        teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, **t_kwargs)
        teacher = move_model_to_device(teacher, teacher_device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        print("\nSkipping teacher (DISTILL_ALPHA=0 and ATTN_LOSS_WEIGHT=0).")

    # ── Student ───────────────────────────────────────────────────────
    if student_4bit:
        print("Loading Student (4-bit NF4) ...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        # device_map is required for bitsandbytes to place the model on GPU;
        # without it the model stays on CPU even when CUDA is available.
        s_kwargs = {
            "quantization_config": bnb_cfg,
            "low_cpu_mem_usage": True,
            "device_map": {"" : student_device},
        }
    else:
        print(f"Loading Student ({compute_dtype} / full-precision warm-start) ...")
        # Explicitly target student_device so the model never lands on CPU
        # by accident (e.g. when the teacher was loaded onto CPU first).
        s_kwargs = {
            "torch_dtype": compute_dtype,
            "low_cpu_mem_usage": True,
            "device_map": {"" : student_device},
        }

    if attn_impl:
        s_kwargs["attn_implementation"] = attn_impl

    student = AutoModelForCausalLM.from_pretrained(MODEL_ID, **s_kwargs)

    # CRITICAL: explicitly walk every parameter and move to GPU.
    # low_cpu_mem_usage=True can leave tensors on CPU after from_pretrained,
    # causing 0% GPU utilization during the entire training run.
    student = move_model_to_device(student, student_device)

    student.train()
    student.config.use_cache = False

    # ── Datasets ─────────────────────────────────────────────────────
    print("\nPreparing calibration datasets ...")
    train_ds, val_ds = get_calibration_datasets(DATASET_SOURCES, tokenizer)

    block_samp = env_int("CALIBRATION_SAMPLES", CALIBRATION_SAMPLES)
    if block_samp <= 0 and STEPS_PER_BLOCK > 0:
        block_samp = CALIBRATION_BATCH_SIZE * STEPS_PER_BLOCK
    eff_samp      = block_samp if block_samp > 0 else train_ds.num_rows
    train_batches = math.ceil(eff_samp / CALIBRATION_BATCH_SIZE)
    print(f"  ~{train_batches} train batches per block | steps_per_block={STEPS_PER_BLOCK}")

    # ── QAT ──────────────────────────────────────────────────────────
    print("\nStarting Block-Wise QAT ...")
    student = execute_block_wise_qat(
        student, teacher, train_ds, val_ds, tokenizer,
        CALIBRATION_BATCH_SIZE, STEPS_PER_BLOCK, EPOCHS_PER_BLOCK,
        distill_alpha, temperature, attn_weight, compute_dtype,
        teacher_device, student_device,
        CHECKPOINT_DIR, ckpt_every, resume_from,
    )

    # ── Finalise ternary weights ──────────────────────────────────────
    print("Finalizing ternary weights ...")
    total_p, nonzero_p, mod_count = 0, 0, 0
    with torch.no_grad():
        for _n, mod in student.named_modules():
            if not isinstance(mod, LearnableBitLinear):
                continue
            scale   = mod.weight.detach().abs().mean().clamp_min(1e-8)
            w_n     = mod.weight.detach() / scale
            mask_p  = (w_n >  mod.delta_param).to(torch.bfloat16)
            mask_n  = (w_n < -mod.delta_param).to(torch.bfloat16)
            final_w = (mask_p * mod.alpha_pos + mask_n * -mod.alpha_neg) * scale
            mod.weight.data.copy_(final_w)
            total_p   += mod.weight.numel()
            nonzero_p += final_w.ne(0).sum().item()
            mod_count += 1

    density  = nonzero_p / total_p if total_p > 0 else 0.0
    n_blocks = len(get_transformer_blocks(student))

    print(f"\n{'METRIC':<32} | VALUE")
    print("-" * 50)
    print(f"{'Transformer blocks':<32} | {n_blocks}")
    print(f"{'Ternary modules':<32} | {mod_count}")
    print(f"{'Total ternary params':<32} | {total_p:,}")
    print(f"{'Non-zero density':<32} | {density:.4%}")
    print(f"{'Effective bit-width':<32} | ~1.58-bit TTQ")
    print(f"{'Attention layers':<32} | full bf16, trained")
    print("-" * 50 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved -> {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()