"""
model_download.py

Downloads google/gemma-4-E2B-it (processor + weights) into
./models/gemma-4-E2B-it, which is exactly where test.py expects it.

Set the HF_TOKEN environment variable if the model is gated.
"""

import os
from pathlib import Path

from transformers import AutoProcessor, AutoModelForCausalLM

HF_MODEL_ID = "google/gemma-4-E2B-it"
HF_TOKEN = os.environ.get("HF_TOKEN")

REPO_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = REPO_DIR / "models" / "gemma-4-E2B-it"

LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
save_path = str(LOCAL_MODEL_DIR)

print(f"[model_download] Source : {HF_MODEL_ID}")
print(f"[model_download] Target : {save_path}\n")

# Processor / tokenizer
print("[model_download] Downloading processor / tokenizer …")
processor = AutoProcessor.from_pretrained(HF_MODEL_ID, token=HF_TOKEN)
processor.save_pretrained(save_path)
print("[model_download] Processor saved.\n")

# Model weights
print("[model_download] Downloading model weights …")
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
)
model.save_pretrained(save_path)

print(f"\n✓ Done! Model ready at: {save_path}")
