from unsloth import FastLanguageModel
import torch
import time
import gc
import sys
import json
import os
import atexit

class Logger(object):
    def __init__(self, filename="benchmark_output.txt"):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

_logger = Logger("benchmark_output.txt")
sys.stdout = _logger
sys.stderr = _logger  # also capture warnings and tracebacks
atexit.register(_logger.close)

# --- CONFIG ---
CONFIG = {
    "model_path":    "/root/EIC/gemma-4-31B-it-finetuned",
    "prompts_file":  "/root/EIC/prompts.json",
    "max_seq_length": 8192,
    "max_new_tokens": 2048,
    "temperature":    0.7,
    "top_p":          0.9,
    "output_file":   "benchmark_output.txt",
}

print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = CONFIG["model_path"],
    max_seq_length  = CONFIG["max_seq_length"],
    dtype           = None,
    load_in_4bit    = False,  # Full precision for the 192GB MI300X
)
FastLanguageModel.for_inference(model)

# Set padding once globally — left-pad is required for decoder-only batched inference
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 1. LOAD PROMPTS FROM JSON ---
_prompts_path = CONFIG["prompts_file"]
if not os.path.exists(_prompts_path):
    raise FileNotFoundError(f"prompts.json not found at: {_prompts_path}")

with open(_prompts_path, "r") as f:
    _prompt_data = json.load(f)

smoke_instruction = _prompt_data["smoke_instruction"]
prompt_entries    = _prompt_data["prompts"]          # list of {label, use_smoke_instruction, code}
task_labels       = [p["label"] for p in prompt_entries]

print(f"Loaded {len(prompt_entries)} prompts from {_prompts_path}")
for i, entry in enumerate(prompt_entries):
    print(f"  [{i+1}] {entry['label']}")

# --- 2. PROMPT BUILDER ---
def prepare_batch(thinking_enabled=False):
    if thinking_enabled:
        base_instruction = "Think step-by-step. Analyze the CUDA memory models, library dependencies, and hardware constraints. Document your reasoning, then output the final ROCm/HIP C++ code with native compatibility."
    else:
        base_instruction = "Directly convert this CUDA code to natively compatible ROCm/HIP. Change all headers and library calls appropriately. Do not provide explanations, output only the C++ code."

    prompts = []
    for entry in prompt_entries:
        extra = f"\n\n{smoke_instruction}" if entry["use_smoke_instruction"] else ""
        content = f"{base_instruction}{extra}\n\n```cpp\n{entry['code']}\n```"
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    return tokenizer(text=prompts, return_tensors="pt", padding=True).to("cuda")

# --- 3. BENCHMARK ENGINE ---
def run_benchmark(model_name, inputs, description):
    print(f"\n{'='*80}")
    print(f" BENCHMARK: {model_name} | {description}")
    print(f"{'='*80}")
    
    # Reset GPU memory stats for accurate measurement
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_mem = torch.cuda.memory_allocated()

    # Synchronize before timing so we measure true GPU time, not Python scheduling lag
    torch.cuda.synchronize()
    start_time = time.time()

    # Generate batch — wrapped to survive OOM or other runtime failures
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens = CONFIG["max_new_tokens"],
            do_sample      = True,
            temperature    = CONFIG["temperature"],
            top_p          = CONFIG["top_p"],
            use_cache      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return

    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate Performance Metrics
    peak_mem = torch.cuda.max_memory_allocated()

    # Detailed VRAM Breakdown
    base_model_gb       = start_mem / (1024**3)
    inference_overhead_gb = (peak_mem - start_mem) / (1024**3)
    total_peak_gb       = peak_mem / (1024**3)

    total_time = end_time - start_time
    batch_size = outputs.shape[0]

    # Per-sequence actual input lengths (accounts for left-padding correctly)
    actual_input_lens = inputs['attention_mask'].sum(dim=1)          # shape: [batch]
    actual_generated  = outputs.shape[1] - actual_input_lens         # shape: [batch]
    total_generated_tokens = actual_generated.sum().item()

    tps = total_generated_tokens / total_time

    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Tokens Gen:         {int(total_generated_tokens)} tokens (Batch Size: {batch_size})")
    print(f"Speed:              {tps:.2f} tokens/second")
    print(f"Base Model VRAM:    {base_model_gb:.2f} GB (The static weights in 16-bit)")
    print(f"Inference Overhead: {inference_overhead_gb:.2f} GB (The dynamic 'Spike' for KV Cache)")
    print(f"Total Peak VRAM:    {total_peak_gb:.2f} GB (What you see in rocm-smi)\n")

    # Decode full outputs using the globally loaded task_labels (auto-scales to any JSON)
    for idx, label in enumerate(task_labels):
        seq_input_len = actual_input_lens[idx].item()
        print(f"--- Output [{idx+1}/{batch_size}]: {label} ---")
        print(tokenizer.decode(outputs[idx][seq_input_len:], skip_special_tokens=True).strip())
        print()

# --- 4. EXECUTION FLOW ---

if __name__ == "__main__":
    batch_size = len(prompt_entries)
    print(f"Preparing Batch Tensors for BASE MODEL runs (Batch Size = {batch_size})...")
    inputs_no_think = prepare_batch(thinking_enabled=False)
    inputs_think    = prepare_batch(thinking_enabled=True)

    # 1. Base Model Benchmarks
    with model.disable_adapter():
        run_benchmark("BASE MODEL", inputs_no_think, f"Mode: Thinking DISABLED (Batch={batch_size})")
        run_benchmark("BASE MODEL", inputs_think,    f"Mode: Thinking ENABLED  (Batch={batch_size})")

    # Free base-model batch tensors before LoRA runs to avoid holding both on GPU
    del inputs_no_think, inputs_think
    torch.cuda.empty_cache()

    # 2. LoRA Model Benchmarks
    print(f"Preparing Batch Tensors for LORA MODEL runs (Batch Size = {batch_size})...")
    inputs_no_think = prepare_batch(thinking_enabled=False)
    inputs_think    = prepare_batch(thinking_enabled=True)

    run_benchmark("LORA MODEL", inputs_no_think, f"Mode: Thinking DISABLED (Batch={batch_size})")
    run_benchmark("LORA MODEL", inputs_think,    f"Mode: Thinking ENABLED  (Batch={batch_size})")

    del inputs_no_think, inputs_think
    torch.cuda.empty_cache()

    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE.")
    print("="*80)
