from unsloth import FastLanguageModel
import torch
import time
import gc
import sys
import json
import os
import atexit


# ─── DUAL-STREAM LOGGER ──────────────────────────────────────────────────────
# Terminal gets everything (progress bars etc).
# The log file gets ONLY clean benchmark output — no loading bars.
class BenchmarkLogger:
    """
    Writes clean benchmark results to the log file.
    Terminal output is unchanged (loading bars, tqdm, etc. all show normally).
    We intercept only the lines we explicitly write via log_write().
    """
    def __init__(self, filename):
        self._file = open(filename, "w", encoding="utf-8")

    def log_write(self, message):
        """Write a line directly to the log file only."""
        self._file.write(message)
        self._file.flush()

    def close(self):
        self._file.close()

_log = BenchmarkLogger(CONFIG["output_file"] if "CONFIG" in dir() else "benchmark_output.txt")

# We register close first, then define CONFIG below — atexit order is fine.
atexit.register(_log.close)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CONFIG = {
    "model_path":    "/root/EIC/gemma-4-31B-it-finetuned",
    "prompts_file":  "/root/EIC/prompts.json",
    "max_seq_length": 8192,
    "max_new_tokens": 2048,
    "temperature":    0.7,
    "top_p":          0.9,
    "output_file":   "benchmark_output.txt",
}

# Re-open logger with the correct filename now that CONFIG is defined
_log.close()
_log = BenchmarkLogger(CONFIG["output_file"])

# ─── MODEL LOAD ──────────────────────────────────────────────────────────────
# Loading bars appear on terminal only — they are NOT captured to the log.
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = CONFIG["model_path"],
    max_seq_length  = CONFIG["max_seq_length"],
    dtype           = None,
    load_in_4bit    = False,
)
FastLanguageModel.for_inference(model)

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── 1. LOAD PROMPTS ─────────────────────────────────────────────────────────
_prompts_path = CONFIG["prompts_file"]
if not os.path.exists(_prompts_path):
    raise FileNotFoundError(f"prompts.json not found at: {_prompts_path}")

with open(_prompts_path, "r") as f:
    _prompt_data = json.load(f)

smoke_instruction = _prompt_data["smoke_instruction"]
prompt_entries    = _prompt_data["prompts"]
task_labels       = [p["label"] for p in prompt_entries]

print(f"\nLoaded {len(prompt_entries)} prompts from {_prompts_path}")
for i, entry in enumerate(prompt_entries):
    print(f"  [{i+1:02d}] {entry['label']}")

# ─── 2. PROMPT BUILDER ───────────────────────────────────────────────────────
def prepare_batch(thinking_enabled=False):
    if thinking_enabled:
        base_instruction = (
            "Think step-by-step. Analyze the CUDA memory models, library dependencies, "
            "and hardware constraints. Document your reasoning, then output the final "
            "ROCm/HIP C++ code with native compatibility."
        )
    else:
        base_instruction = (
            "Directly convert this CUDA code to natively compatible ROCm/HIP. "
            "Change all headers and library calls appropriately. "
            "Do not provide explanations, output only the C++ code."
        )

    prompts = []
    for entry in prompt_entries:
        extra = f"\n\n{smoke_instruction}" if entry["use_smoke_instruction"] else ""
        content = f"{base_instruction}{extra}\n\n```cpp\n{entry['code']}\n```"
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    return tokenizer(text=prompts, return_tensors="pt", padding=True).to("cuda")

# ─── 3. TOKEN-LEVEL PROGRESS PROCESSOR ──────────────────────────────────────
from transformers import LogitsProcessor

class LiveProgressBar(LogitsProcessor):
    """
    Called by HuggingFace generate() after EVERY token step for the full batch.
    This gives us true real-time progress instead of a fake end-jump.
    """
    def __init__(self, batch_size: int, max_new_tokens: int, model_name: str):
        from tqdm import tqdm
        self._total = batch_size * max_new_tokens
        self._bar = tqdm(
            total=self._total,
            desc=f"  Generating ({model_name})",
            unit="tok",
            ncols=95,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} tok  [{elapsed}<{remaining}  {rate_fmt}]",
            file=sys.__stdout__,
            leave=True,
            dynamic_ncols=False,
        )
        self._step = batch_size   # each call = 1 new token per sequence in batch

    def __call__(self, input_ids, scores):
        self._bar.update(self._step)
        return scores

    def close(self):
        # Fill any remainder (early stopping) then close cleanly
        remaining = self._total - self._bar.n
        if remaining > 0:
            self._bar.update(remaining)
        self._bar.close()

# ─── 4. BENCHMARK ENGINE ─────────────────────────────────────────────────────
def run_benchmark(model_name, inputs, description):
    header = f"\n{'='*80}\n BENCHMARK: {model_name} | {description}\n{'='*80}"
    print(header)
    _log.log_write(header + "\n")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_mem  = torch.cuda.memory_allocated()
    batch_size = inputs["input_ids"].shape[0]

    # Real-time progress bar — advances one tick per token, per sequence
    prog = LiveProgressBar(batch_size, CONFIG["max_new_tokens"], model_name)

    # ── Generation ───────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    start_time = time.time()

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens    = CONFIG["max_new_tokens"],
            do_sample         = True,
            temperature       = CONFIG["temperature"],
            top_p             = CONFIG["top_p"],
            use_cache         = True,
            pad_token_id      = tokenizer.eos_token_id,
            logits_processor  = [prog],   # ← hook for real-time ticks
        )
    except Exception as e:
        prog.close()
        err = f"[ERROR] Generation failed: {e}"
        print(err)
        _log.log_write(err + "\n")
        return
    finally:
        prog.close()

    torch.cuda.synchronize()
    end_time = time.time()

    # ── Metrics ──────────────────────────────────────────────────────────────
    peak_mem = torch.cuda.max_memory_allocated()

    base_model_gb       = start_mem / (1024**3)
    inference_overhead_gb = (peak_mem - start_mem) / (1024**3)
    total_peak_gb       = peak_mem / (1024**3)

    total_time = end_time - start_time

    actual_input_lens      = inputs['attention_mask'].sum(dim=1)
    actual_generated       = outputs.shape[1] - actual_input_lens
    total_generated_tokens = actual_generated.sum().item()
    tps = total_generated_tokens / total_time

    metrics = (
        f"Total Time:         {total_time:.2f} seconds\n"
        f"Tokens Gen:         {int(total_generated_tokens)} tokens (Batch Size: {batch_size})\n"
        f"Speed:              {tps:.2f} tokens/second\n"
        f"Base Model VRAM:    {base_model_gb:.2f} GB (static weights in 16-bit)\n"
        f"Inference Overhead: {inference_overhead_gb:.2f} GB (KV Cache spike)\n"
        f"Total Peak VRAM:    {total_peak_gb:.2f} GB (visible in rocm-smi)\n"
    )
    print(metrics)
    _log.log_write(metrics)

    # ── Decode & save outputs (log-only, clean — no prompt echo) ─────────────
    for idx, label in enumerate(task_labels):
        seq_input_len = actual_input_lens[idx].item()
        raw = tokenizer.decode(outputs[idx][seq_input_len:], skip_special_tokens=True).strip()

        # Strip prompt echo: model sometimes re-prints the user message before answering.
        # The echo always starts with "user\n" or the base_instruction prefix — trim it.
        for marker in ["```cpp", "model\nthought", "user\n"]:
            if raw.startswith(marker):
                break
            # Find the LAST occurrence of ```cpp and keep from there
        code_start = raw.rfind("```cpp")
        if code_start != -1:
            raw = raw[code_start:]

        entry_out = f"\n{'─'*78}\nOutput [{idx+1:02d}/{batch_size}]: {label}\n{'─'*78}\n{raw}\n"
        _log.log_write(entry_out)

    # Summary line to terminal
    summary = f"  ✓ Done | {tps:.1f} TPS | {total_peak_gb:.1f} GB VRAM | Saved to {CONFIG['output_file']}"
    print(summary)

# ─── 4. EXECUTION FLOW ───────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_size = len(prompt_entries)

    _log.log_write(f"EIC Benchmark — Gemma 4 31B LoRA on AMD MI300X\n")
    _log.log_write(f"Batch Size: {batch_size} | Max New Tokens: {CONFIG['max_new_tokens']} | Seq Len: {CONFIG['max_seq_length']}\n")
    _log.log_write("="*80 + "\n")

    print(f"\nPreparing Batch Tensors (Batch Size = {batch_size})...")
    inputs_no_think = prepare_batch(thinking_enabled=False)
    inputs_think    = prepare_batch(thinking_enabled=True)

    # 1. Base Model
    _log.log_write("\n### RUN 1 & 2: BASE MODEL (No LoRA Adapter)\n")
    with model.disable_adapter():
        run_benchmark("BASE MODEL", inputs_no_think, f"Thinking DISABLED (Batch={batch_size})")
        run_benchmark("BASE MODEL", inputs_think,    f"Thinking ENABLED  (Batch={batch_size})")

    del inputs_no_think, inputs_think
    torch.cuda.empty_cache()

    # 2. LoRA Model
    _log.log_write("\n### RUN 3 & 4: LORA MODEL (Fine-tuned Adapter Active)\n")
    print(f"\nPreparing Batch Tensors for LORA MODEL runs (Batch Size = {batch_size})...")
    inputs_no_think = prepare_batch(thinking_enabled=False)
    inputs_think    = prepare_batch(thinking_enabled=True)

    run_benchmark("LORA MODEL", inputs_no_think, f"Thinking DISABLED (Batch={batch_size})")
    run_benchmark("LORA MODEL", inputs_think,    f"Thinking ENABLED  (Batch={batch_size})")

    del inputs_no_think, inputs_think
    torch.cuda.empty_cache()

    done_msg = "\n" + "="*80 + "\n BENCHMARK COMPLETE. Results saved to: " + CONFIG["output_file"] + "\n" + "="*80
    print(done_msg)
    _log.log_write(done_msg + "\n")
