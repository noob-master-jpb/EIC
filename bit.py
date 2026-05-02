import os
from pathlib import Path

# Optimize VRAM allocation for AMD ROCm (MI300X)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb

# ==========================================
# CONFIGURATION & DATASETS
# ==========================================
MODEL_ID = "google/gemma-4-E2B-it" 
REPO_DIR = Path(__file__).resolve().parent

# Datasets to pool for calibration
DATASET_SOURCES = [
    "Datasets/smoltalk_chat.parquet",
    "Datasets/chat_reasoning_qat_mix.parquet"
]

OUTPUT_DIR = REPO_DIR / "models" / f"{MODEL_ID.split('/')[-1]}-ternary-advanced"

MAX_LENGTH = 1024
TRAIN_SPLIT = "train"
CALIBRATION_BATCH_SIZE = 8
CALIBRATION_SAMPLES_PER_FILE = 512
DATASET_VAL_SPLIT_FACTOR = 0.05
STEPS_PER_BLOCK = 100
EPOCHS_PER_BLOCK = 1
LEARNING_RATE = 1e-4

# Advanced Recovery Hyperparameters
DISTILL_TEMPERATURE = 2.0
DISTILL_ALPHA = 0.8
ATTN_LOSS_WEIGHT = 0.5

def get_train_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else default

def env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val is not None else default

def env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None: return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

# ==========================================
# 1. ADVANCED TERNARY LAYER (TTQ)
# ==========================================
class LearnableBitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        # Learnable threshold and scales for dynamic sparsity
        self.delta_param = nn.Parameter(torch.tensor(0.05))
        self.alpha_pos = nn.Parameter(torch.tensor(1.0))
        self.alpha_neg = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        scale = w.abs().mean().clamp_min(1e-8).detach()
        w_norm = w / scale
        
        # Apply learnable threshold
        mask_p = (w_norm > self.delta_param).to(w.dtype)
        mask_n = (w_norm < -self.delta_param).to(w.dtype)
        
        quant_w = (mask_p * self.alpha_pos) + (mask_n * -self.alpha_neg)
        quant_w = quant_w * scale
        
        # Straight-Through Estimator trick for gradient flow
        ste_weight = w + (quant_w - w).detach()
        return F.linear(x, ste_weight, self.bias)

# ==========================================
# 2. SELECTIVE BLOCK SURGERY (4-BIT AWARE)
# ==========================================
def apply_ternary_surgery_to_block(block: nn.Module) -> int:
    replacements = 0
    # CRITICAL: Protect the Attention mechanism
    protected = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    for name, child in list(block.named_children()):
        if isinstance(child, (nn.Linear, bnb.nn.Linear4bit)) and not any(p in name for p in protected):
            
            in_features = child.in_features
            out_features = child.out_features
            
            bit_layer = LearnableBitLinear(in_features, out_features, bias=child.bias is not None)
            bit_layer.to(device=child.weight.device, dtype=torch.bfloat16)

            # Dequantize 4-bit logic for initial Ternary setup
            if isinstance(child, bnb.nn.Linear4bit):
                dequantized_weight = bnb.functional.dequantize_4bit(
                    child.weight.data, 
                    child.weight.quant_state
                ).to(torch.bfloat16)
                bit_layer.weight.data.copy_(dequantized_weight)
            else:
                bit_layer.weight.data.copy_(child.weight.data.to(torch.bfloat16))

            if child.bias is not None:
                bit_layer.bias.data.copy_(child.bias.data)
            
            setattr(block, name, bit_layer)
            replacements += 1
        else:
            replacements += apply_ternary_surgery_to_block(child)
            
    return replacements

# ==========================================
# 3. JSD LOSS FUNCTION
# ==========================================
def compute_jsd_loss(student_logits, teacher_logits, temperature):
    s_probs = F.softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    m_probs = 0.5 * (s_probs + t_probs).clamp_min(1e-8)
    
    s_jsd = (s_probs * (torch.log(s_probs.clamp_min(1e-8)) - torch.log(m_probs))).sum(dim=-1).mean()
    t_jsd = (t_probs * (torch.log(t_probs.clamp_min(1e-8)) - torch.log(m_probs))).sum(dim=-1).mean()
    
    return 0.5 * (s_jsd + t_jsd) * (temperature ** 2)

# ==========================================
# 4. DATASET PREP 
# ==========================================
def format_prompt(example: dict, tokenizer: AutoTokenizer) -> dict:
    messages = [{"role": "user", "content": example["problem"]}, {"role": "assistant", "content": example["solution"]}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        text = f"### INSTRUCTION ###\n{example['problem']}\n\n### RESPONSE ###\n{example['solution']}"
    return {"text": text}

def tokenize_batch(batch: dict, tokenizer: AutoTokenizer) -> dict:
    tokenized = tokenizer(batch["text"], truncation=True, max_length=env_int("MAX_LENGTH", MAX_LENGTH), padding=False)
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized

def collate_causal_batch(features: list[dict], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    max_len = max(len(item["input_ids"]) for item in features)
    input_ids, attention_mask, labels = [], [], []
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

def _first_present(example: dict, keys: list[str]) -> str | None:
    for key in keys:
        if key in example and example[key] is not None:
            val = example[key]
            if isinstance(val, str) and val.strip(): return val.strip()
            elif str(val).strip(): return str(val).strip()
    return None

def normalize_schema(example: dict) -> dict:
    p_keys = ["problem", "query", "instruction", "prompt", "question", "Problem"]
    s_keys = ["solution", "answer", "response", "Response", "output", "completion"]
    return {"problem": _first_present(example, p_keys), "solution": _first_present(example, s_keys)}

def get_calibration_loader(sources: list[str], tokenizer: AutoTokenizer, batch_size: int):
    train_split = os.environ.get("TRAIN_SPLIT", TRAIN_SPLIT)
    shuffle_each = env_bool("CALIBRATION_SHUFFLE_EACH_FILE", True)
    samples_per_source = env_int("CALIBRATION_SAMPLES_PER_FILE", CALIBRATION_SAMPLES_PER_FILE)
    val_split_factor = env_float("DATASET_VAL_SPLIT_FACTOR", DATASET_VAL_SPLIT_FACTOR)
    seed = env_int("CALIBRATION_SEED", 42)

    train_parts, val_parts = [], []
    
    for i, source in enumerate(sources):
        try:
            path = Path(source)
            ds_dict = load_dataset("parquet" if path.suffix == ".parquet" else "json", data_files=str(path)) if path.exists() else load_dataset(source)

            available_splits = list(ds_dict.keys())
            current_train_split = train_split if train_split in available_splits else available_splits[0]
            ds_train = ds_dict[current_train_split]
            ds_val = ds_dict.get("validation", ds_dict.get("test"))
            
            if not ds_val and ds_train.num_rows > 10 and val_split_factor > 0:
                split_ds = ds_train.train_test_split(test_size=val_split_factor, seed=seed)
                ds_train, ds_val = split_ds["train"], split_ds["test"]

            def process_ds(ds, is_train=True):
                if shuffle_each and ds.num_rows > 1: ds = ds.shuffle(seed=seed + i)
                limit = samples_per_source if is_train else max(1, samples_per_source // 10)
                if limit > 0 and ds.num_rows > limit: ds = ds.select(range(limit))
                ds = ds.map(normalize_schema, remove_columns=ds.column_names)
                ds = ds.filter(lambda ex: ex["problem"] is not None and ex["solution"] is not None)
                if ds.num_rows == 0: return None
                ds = ds.map(lambda ex: format_prompt(ex, tokenizer), remove_columns=ds.column_names)
                ds = ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True, remove_columns=["text"])
                return ds

            ptrain = process_ds(ds_train, True)
            if ptrain: train_parts.append(ptrain)
            if ds_val:
                pval = process_ds(ds_val, False)
                if pval: val_parts.append(pval)

        except Exception as e:
            print(f"      Error loading '{source}': {e}")

    train_loader = DataLoader(concatenate_datasets(train_parts), batch_size=batch_size, shuffle=True, collate_fn=lambda f: collate_causal_batch(f, tokenizer))
    val_loader = DataLoader(concatenate_datasets(val_parts), batch_size=batch_size, shuffle=False, collate_fn=lambda f: collate_causal_batch(f, tokenizer)) if val_parts else None
    
    return train_loader, val_loader

# ==========================================
# 5. TRANSFORMER BLOCK RESOLUTION
# ==========================================
def get_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    base_model = model.model if hasattr(model, "model") else model
    if hasattr(base_model, "language_model"): base_model = base_model.language_model
    candidates = (("layers",), ("decoder", "layers"), ("transformer", "h"), ("transformer", "blocks"), ("blocks",))
    for path in candidates:
        current = base_model
        for attr in path:
            if not hasattr(current, attr):
                current = None
                break
            current = getattr(current, attr)
        if isinstance(current, (nn.ModuleList, list, tuple)): return current
    raise AttributeError("Unable to locate transformer blocks.")

# ==========================================
# 6. ADVANCED BLOCK-WISE QAT ENGINE
# ==========================================
def execute_block_wise_qat(
    student, teacher, calibration_loader, validation_loader,
    steps_per_block, epochs_per_block, distill_alpha, temperature, attn_weight
):
    device = next(student.parameters()).device
    blocks = get_transformer_blocks(student)
    
    print(f"\n{'='*90}")
    print(f"{'LAYER':<10} | {'STEPS':<8} | {'START LOSS':<12} | {'TRAIN LOSS':<12} | {'VAL LOSS':<12} | {'ATTN LOSS':<12}")
    print(f"{'-'*90}")

    for i, block in enumerate(blocks):
        _ = apply_ternary_surgery_to_block(block)
        
        # --- FIXED SELECTIVE UNFREEZING ---
        student.requires_grad_(False)
        
        trainable_params = []
        for module in block.modules():
            if isinstance(module, LearnableBitLinear):
                for param in module.parameters():
                    param.requires_grad_(True)
                    trainable_params.append(param)
        
        optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
        # -----------------------------------

        total_steps = 0
        start_loss, running_train_loss, running_attn_loss = 0.0, 0.0, 0.0

        for epoch in range(epochs_per_block):
            student.train()
            for batch in calibration_loader:
                if 0 < steps_per_block <= total_steps: break
                    
                optimizer.zero_grad()
                inputs = {k: v.to(device) for k, v in batch.items()}
                
                with torch.no_grad():
                    teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                    teacher_outputs = teacher(**teacher_inputs, output_attentions=True)
                    
                student_outputs = student(**inputs, output_attentions=True)
                
                ce_loss = student_outputs.loss if "labels" in inputs and student_outputs.loss is not None else student_outputs.logits.new_tensor(0.0)

                # Attention Distillation
                t_attn = teacher_outputs.attentions[i]
                s_attn = student_outputs.attentions[i]
                attn_loss = F.mse_loss(s_attn, t_attn)

                # JSD Distillation
                mask = inputs["attention_mask"].bool()
                if mask.any().item():
                    s_logits = student_outputs.logits[mask]
                    t_logits = teacher_outputs.logits[mask.to(teacher_outputs.logits.device)].to(s_logits.device)
                    jsd_loss = compute_jsd_loss(s_logits, t_logits, temperature)
                else:
                    jsd_loss = student_outputs.logits.new_tensor(0.0)

                loss = (distill_alpha * jsd_loss) + ((1.0 - distill_alpha) * ce_loss) + (attn_weight * attn_loss)
                loss.backward()
                optimizer.step()
                
                if total_steps == 0: start_loss = loss.item()
                running_train_loss += loss.item()
                running_attn_loss += attn_loss.item()
                total_steps += 1

        val_loss = 0.0
        if validation_loader:
            student.eval()
            total_val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for batch in validation_loader:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    outputs = student(**inputs, output_attentions=True)
                    ce = outputs.loss if outputs.loss is not None else outputs.logits.new_tensor(0.0)
                    
                    t_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                    t_outputs = teacher(**t_inputs, output_attentions=True)
                    mask = inputs["attention_mask"].bool()
                    
                    d_loss = outputs.logits.new_tensor(0.0)
                    if mask.any().item():
                        s_logits, t_logits = outputs.logits[mask], t_outputs.logits[mask.to(t_outputs.logits.device)].to(outputs.logits.device)
                        d_loss = compute_jsd_loss(s_logits, t_logits, temperature)
                    
                    batch_attn_loss = F.mse_loss(outputs.attentions[i], t_outputs.attentions[i])
                    total_val_loss += ((distill_alpha * d_loss) + ((1.0 - distill_alpha) * ce) + (attn_weight * batch_attn_loss)).item()
                    val_batches += 1
            val_loss = total_val_loss / val_batches if val_batches > 0 else 0.0
                
        avg_train = running_train_loss / total_steps if total_steps > 0 else 0
        avg_attn = running_attn_loss / total_steps if total_steps > 0 else 0
        print(f"Layer {i+1:02d}   | {total_steps:<8} | {start_loss:<12.4f} | {avg_train:<12.4f} | {val_loss:<12.4f} | {avg_attn:<12.4f}")

        # Ensure everything in the block is locked down before moving to the next
        for param in block.parameters():
            param.requires_grad_(False)
        
    print(f"{'='*90}\n")
    return student

# ==========================================
# 7. MAIN PIPELINE (WITH 4-BIT WARM START)
# ==========================================
def main():
    device = get_train_device()
    dtype = torch.bfloat16
    print(f"Runtime Initialized. Device: {device} | Dtype: {dtype}")

    distill_alpha = env_float("DISTILL_ALPHA", DISTILL_ALPHA)
    temperature = env_float("DISTILL_TEMPERATURE", DISTILL_TEMPERATURE)
    attn_weight = env_float("ATTN_LOSS_WEIGHT", ATTN_LOSS_WEIGHT)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading Teacher Model (16-bit BF16)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=dtype, 
        low_cpu_mem_usage=True
    ).to(device)
    teacher.eval()
    for param in teacher.parameters(): param.requires_grad_(False)
        
    print("Loading Student Model (4-bit NF4 Warm-Start)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        low_cpu_mem_usage=True
    ).to(device)
    student.train()
    student.config.use_cache = False
    
    print("\nPreparing diverse calibration samples from requested sources...")
    calibration_loader, validation_loader = get_calibration_loader(DATASET_SOURCES, tokenizer, CALIBRATION_BATCH_SIZE)
    
    print(f"\nStarting Advanced Block-Wise QAT ({len(calibration_loader)} train batches per layer)...")
    
    student = execute_block_wise_qat(
        student, teacher, calibration_loader, validation_loader,
        STEPS_PER_BLOCK, EPOCHS_PER_BLOCK, distill_alpha, temperature, attn_weight
    )
    
    total_p, nonzero_p = 0, 0
    with torch.no_grad():
        for n, module in student.named_modules():
            if isinstance(module, LearnableBitLinear):
                scale = module.weight.detach().abs().mean().clamp_min(1e-8)
                w_norm = module.weight.detach() / scale
                mask_p = (w_norm > module.delta_param).to(dtype)
                mask_n = (w_norm < -module.delta_param).to(dtype)
                
                final_weight = ((mask_p * module.alpha_pos) + (mask_n * -module.alpha_neg)) * scale
                
                total_p += module.weight.numel()
                nonzero_p += final_weight.ne(0).sum().item()
                module.weight.data.copy_(final_weight)

    print(f"\n{'METRIC':<30} | {'VALUE':<10}")
    print(f"{'-'*45}")
    print(f"{'Total Transformer Layers':<30} | {len(get_transformer_blocks(student)):<10}")
    print(f"{'Total Ternary Params':<30} | {total_p:<10}")
    print(f"{'Non-Zero Density':<30} | {nonzero_p/total_p if total_p > 0 else 0:<10.4%}")
    print(f"{'Quantization State':<30} | {'1.58-bit TTQ' if total_p > 0 else 'N/A'}")
    print(f"{'-'*45}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Optimization Complete.")

if __name__ == "__main__":
    main()