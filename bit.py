import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ==========================================
# 1. DEFINE THE TERNARY LAYER (1.58-Bit)
# ==========================================
class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        # Calculate scale (mean of absolute weights)
        scale = self.weight.abs().mean()
        
        # Quantize weights to -1, 0, 1
        quantized = torch.round(self.weight / (scale + 1e-8)).clamp(-1, 1)
        
        # Straight-Through Estimator (STE) for gradient flow
        ste_weights = quantized.detach() - self.weight.detach() + self.weight
        
        # Linear transformation
        out = F.linear(x, ste_weights) * scale
        if self.bias is not None:
            out += self.bias
        return out

# ==========================================
# 2. DEFINE THE SURGERY FUNCTION
# ==========================================
def apply_ternary_surgery(target_model):
    print("Executing ternary layer surgery...")
    for name, module in target_model.named_children():
        # Replace standard linear layers, but protect the final output head
        if isinstance(module, nn.Linear) and name != "lm_head":
            bit_layer = BitLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None
            )
            # Copy original 16-bit weights into the new ternary layer
            bit_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                bit_layer.bias.data.copy_(module.bias.data)
            
            # Inject the layer back into the model
            setattr(target_model, name, bit_layer)
        else:
            # Recursively check deeper blocks (attention, MLP, etc.)
            apply_ternary_surgery(module)
    return target_model

# ==========================================
# 3. DATA PREPARATION
# ==========================================
def format_prompt(example):
    text = "### INSTRUCTION ###\n"
    text += example["problem"] + "\n\n"
    text += "### RESPONSE ###\n"
    text += example["solution"] + "<eos>"
    return {"text": text}

print("Loading and formatting dataset...")
dataset = load_dataset("parquet", data_files="Datasets/nvidia_compute_eval.parquet", split="train")
formatted_dataset = dataset.map(format_prompt)

# ==========================================
# 4. LOAD BASE MODEL & TOKENIZER
# ==========================================
model_id = "Qwen/Qwen3-0.6B" # Replace with exact HF repo ID if testing a specific 0.5B/1.5B variant
repo_dir = Path(__file__).resolve().parent
local_model_dir = repo_dir / "models" / model_id.split("/")[-1]
local_model_dir.mkdir(parents=True, exist_ok=True)

def local_model_ready(path):
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

def resolve_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    bashrc = Path.home() / ".bashrc"
    if not bashrc.exists():
        return None
    for line in bashrc.read_text().splitlines():
        line = line.strip()
        if line.startswith("export HF_TOKEN="):
            value = line.split("=", 1)[1].strip()
            if value.startswith("\"") and value.endswith("\""):
                value = value[1:-1]
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            return value or None
    return None

def enable_fast_downloads():
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        return
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        return
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("Enabled hf_transfer for faster downloads.")

use_local = local_model_ready(local_model_dir)
load_source = str(local_model_dir) if use_local else model_id
hf_token = resolve_hf_token()
enable_fast_downloads()
if hf_token:
    os.environ.setdefault("HF_TOKEN", hf_token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

if use_local:
    print(f"Loading base model from local dir: {local_model_dir}")
else:
    print(f"Local model not found. Downloading from: {model_id}")
    if not hf_token:
        print("HF_TOKEN not set. Downloads may be slower and rate-limited.")

tokenizer = AutoTokenizer.from_pretrained(load_source, token=hf_token)
# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if not use_local:
    tokenizer.save_pretrained(local_model_dir)

model = AutoModelForCausalLM.from_pretrained(
    load_source,
    dtype=torch.bfloat16,
    device_map="auto", # Automatically utilizes your MI300X VRAM
    token=hf_token,
)
if not use_local:
    model.save_pretrained(local_model_dir)

# ==========================================
# 5. APPLY SURGERY
# ==========================================
model = apply_ternary_surgery(model)
print("Surgery complete. Model is now in 1.58-bit architecture.")

# ==========================================
# 6. TRAINING CONFIGURATION (QAT)
# ==========================================
print("Initializing QAT Trainer...")
training_args = SFTConfig(
    output_dir="./qwen3-ternary-rocm",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,          # Keep low to prevent shock from ternary constraints
    bf16=True,                   # Gradients flow in BF16, forward pass is ternary
    num_train_epochs=1,          # Start with 1 epoch for testing
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",            # Turn off wandb/tensorboard for clean testing
    dataloader_pin_memory=True,  # GPU acceleration
    dataset_text_field="text",
    max_length=2048,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
)

# ==========================================
# 7. EXECUTE TRAINING
# ==========================================
print("Starting Quantization-Aware Training...")
trainer.train()

# Save the final recovered model
print("Saving final ternary model...")
trainer.save_model("./qwen3-ternary-rocm-final")
tokenizer.save_pretrained("./qwen3-ternary-rocm-final")
print("Done!")