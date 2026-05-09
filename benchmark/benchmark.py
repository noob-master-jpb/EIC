from unsloth import FastLanguageModel
import torch, time, gc, sys, json, os, atexit
from transformers import LogitsProcessor
from tqdm import tqdm
from collections import defaultdict

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CONFIG = {
    "model_path":          "/root/EIC/gemma-4-31B-it-finetuned",
    "prompts_file":        "/root/EIC/prompts.json",
    "max_seq_length":      8192,
    "stress_max_new_tokens": 2048,   # default for stress tests
    "unit_max_new_tokens":   512,    # unit snippets are short
    "unit_batch_size":       10,     # unit tests run in groups of 10
    "temperature":           0.7,
    "top_p":                 0.9,
    "output_file":          "benchmark_output.txt",
}

# ─── LOGGER ──────────────────────────────────────────────────────────────────
class BenchmarkLogger:
    def __init__(self, fn):
        self._f = open(fn, "w", encoding="utf-8")
    def log_write(self, msg):
        self._f.write(msg); self._f.flush()
    def close(self):
        self._f.close()

_log = BenchmarkLogger(CONFIG["output_file"])
atexit.register(_log.close)

# ─── MODEL LOAD ──────────────────────────────────────────────────────────────
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

# ─── LOAD & SPLIT PROMPTS ────────────────────────────────────────────────────
with open(CONFIG["prompts_file"], "r") as f:
    _data = json.load(f)

smoke_instruction = _data.get("smoke_instruction", "")
all_prompts       = _data["prompts"]

stress_prompts = [p for p in all_prompts if p.get("type", "stress") == "stress"]
unit_prompts   = [p for p in all_prompts if p.get("type") == "unit"]

# Group stress prompts by their individual token budget
stress_groups = defaultdict(list)
for p in stress_prompts:
    tok = p.get("max_new_tokens", CONFIG["stress_max_new_tokens"])
    stress_groups[tok].append(p)

print(f"\nLoaded {len(all_prompts)} prompts — {len(stress_prompts)} stress | {len(unit_prompts)} unit tests")
print(f"\nStress prompts:")
for i, p in enumerate(stress_prompts):
    tok_note = f" [max_tokens={p['max_new_tokens']}]" if "max_new_tokens" in p else ""
    print(f"  [{i+1:02d}] {p['label']}{tok_note}")
print(f"\nUnit test categories: {set(p['category'] for p in unit_prompts)}")

# ─── LIVE PROGRESS BAR ───────────────────────────────────────────────────────
class LiveProgressBar(LogitsProcessor):
    def __init__(self, batch_size, max_new_tokens, desc):
        self._total = batch_size * max_new_tokens
        self._bar = tqdm(
            total=self._total, desc=f"  {desc}", unit="tok", ncols=95,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} tok  [{elapsed}<{remaining}  {rate_fmt}]",
            file=sys.__stdout__, leave=True,
        )
        self._step = batch_size
    def __call__(self, input_ids, scores):
        self._bar.update(self._step); return scores
    def close(self):
        rem = self._total - self._bar.n
        if rem > 0: self._bar.update(rem)
        self._bar.close()

# ─── BATCH BUILDER ───────────────────────────────────────────────────────────
def build_inputs(entries, thinking=False, is_unit=False):
    if is_unit:
        instruction = (
            "Convert this CUDA snippet to native ROCm/HIP. "
            "Replace ALL CUDA headers, types, and API calls with their HIP equivalents. "
            "Output only the converted C++ code, no explanations."
        )
    elif thinking:
        instruction = (
            "Think step-by-step. Analyze the CUDA memory models, library dependencies, "
            "and hardware constraints. Document your reasoning, then output the final "
            "ROCm/HIP C++ code with native compatibility."
        )
    else:
        instruction = (
            "Directly convert this CUDA code to natively compatible ROCm/HIP. "
            "Change all headers and library calls appropriately. "
            "Do not provide explanations, output only the C++ code."
        )

    prompts = []
    for entry in entries:
        extra = f"\n\n{smoke_instruction}" if entry.get("use_smoke_instruction") else ""
        content = f"{instruction}{extra}\n\n```cpp\n{entry['code']}\n```"
        msgs = [{"role": "user", "content": content}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return tokenizer(text=prompts, return_tensors="pt", padding=True).to("cuda")

# ─── PASS/FAIL SCORER ────────────────────────────────────────────────────────
_CUDA_ONLY = [
    "cudaMalloc","cudaFree","cudaMemcpy","cudaMemset","cudaHostAlloc",
    "cudaStreamCreate","cudaEventCreate","cudaDeviceSynchronize",
    "cudaStreamSynchronize","cudaStreamWaitEvent","cudaEventElapsedTime",
    "cudaEventRecord","cudaEventSynchronize",
    "__shfl_sync","__shfl_up_sync","__shfl_down_sync",
    "__shfl_xor_sync","__ballot_sync","__any_sync","__all_sync",
]

def score_output(raw, entry):
    expected  = entry.get("expected_hip", "")
    forbidden = entry.get("forbidden", [])
    has_exp   = (expected.lower() in raw.lower()) if expected else True
    leaked    = [s for s in (forbidden + _CUDA_ONLY) if s in raw]
    leaked    = list(dict.fromkeys(leaked))
    if not has_exp:
        return "FAIL", f"Missing: {expected}"
    if leaked:
        return "WARN", f"CUDA leaked: {', '.join(leaked[:3])}"
    return "PASS", "OK"

def decode_output(outputs, inputs, idx):
    input_len = inputs["attention_mask"].sum(dim=1)[idx].item()
    raw = tokenizer.decode(outputs[idx][input_len:], skip_special_tokens=True).strip()
    code_start = raw.rfind("```cpp")
    if code_start != -1:
        raw = raw[code_start:]
    return raw

# ─── STRESS BENCHMARK ────────────────────────────────────────────────────────
def run_stress_benchmark(model_name, entries, max_new_tokens, thinking=False):
    mode = "Thinking ENABLED" if thinking else "Thinking DISABLED"
    header = f"\n{'='*80}\n STRESS: {model_name} | {mode} | Batch={len(entries)} | MaxTok={max_new_tokens}\n{'='*80}"
    print(header); _log.log_write(header + "\n")

    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    start_mem  = torch.cuda.memory_allocated()
    inputs     = build_inputs(entries, thinking=thinking, is_unit=False)
    batch_size = inputs["input_ids"].shape[0]
    prog       = LiveProgressBar(batch_size, max_new_tokens, f"Stress {model_name}")

    torch.cuda.synchronize(); t0 = time.time()
    try:
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"], use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[prog],
        )
    except Exception as e:
        prog.close(); err = f"[ERROR] {e}"; print(err); _log.log_write(err+"\n"); return
    finally:
        prog.close()

    torch.cuda.synchronize(); elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated()
    actual_lens = inputs["attention_mask"].sum(dim=1)
    total_gen   = (outputs.shape[1] - actual_lens).sum().item()
    tps         = total_gen / elapsed

    metrics = (
        f"Total Time:         {elapsed:.2f}s\n"
        f"Tokens Generated:   {int(total_gen)} (Batch={batch_size})\n"
        f"Speed:              {tps:.2f} tok/s\n"
        f"Base VRAM:          {start_mem/(1024**3):.2f} GB\n"
        f"KV Overhead:        {(peak_mem-start_mem)/(1024**3):.2f} GB\n"
        f"Peak VRAM:          {peak_mem/(1024**3):.2f} GB\n"
    )
    print(metrics); _log.log_write(metrics)

    for idx, entry in enumerate(entries):
        raw = decode_output(outputs, inputs, idx)
        out = f"\n{'─'*78}\nOutput [{idx+1:02d}/{batch_size}]: {entry['label']}\n{'─'*78}\n{raw}\n"
        _log.log_write(out)

    print(f"  ✓ {tps:.1f} TPS | {peak_mem/(1024**3):.1f} GB VRAM")
    del inputs, outputs

# ─── UNIT BENCHMARK ──────────────────────────────────────────────────────────
def run_unit_benchmark(model_name):
    header = f"\n{'='*80}\n UNIT TESTS: {model_name} | {len(unit_prompts)} snippets\n{'='*80}"
    print(header); _log.log_write(header + "\n")

    UBATCH   = CONFIG["unit_batch_size"]
    MAX_TOK  = CONFIG["unit_max_new_tokens"]
    batches  = [unit_prompts[i:i+UBATCH] for i in range(0, len(unit_prompts), UBATCH)]
    all_results = []

    for bn, batch in enumerate(batches, 1):
        gc.collect(); torch.cuda.empty_cache()
        inputs = build_inputs(batch, is_unit=True)
        prog   = LiveProgressBar(len(batch), MAX_TOK, f"Unit {model_name} batch {bn}/{len(batches)}")

        torch.cuda.synchronize(); t0 = time.time()
        try:
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_TOK,
                do_sample=True, temperature=0.2,
                top_p=CONFIG["top_p"], use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=[prog],
            )
        except Exception as e:
            prog.close(); print(f"[ERROR] Unit batch {bn}: {e}"); continue
        finally:
            prog.close()

        torch.cuda.synchronize(); elapsed = time.time() - t0
        total_gen = (outputs.shape[1] - inputs["attention_mask"].sum(dim=1)).sum().item()
        tps = total_gen / elapsed

        for i, entry in enumerate(batch):
            raw    = decode_output(outputs, inputs, i)
            verdict, reason = score_output(raw, entry)
            all_results.append({"label": entry["label"], "category": entry["category"],
                                 "verdict": verdict, "reason": reason, "output": raw})
            icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[verdict]
            _log.log_write(f"\n{'─'*78}\n{icon} [{verdict}] {entry['label']}\n   {reason}\n{'─'*78}\n{raw}\n")

        print(f"  Batch {bn}/{len(batches)} | {tps:.1f} TPS")
        del inputs, outputs

    # ── Score table ───────────────────────────────────────────────────────────
    total  = len(all_results)
    passed = sum(1 for r in all_results if r["verdict"] == "PASS")
    warned = sum(1 for r in all_results if r["verdict"] == "WARN")
    failed = sum(1 for r in all_results if r["verdict"] == "FAIL")
    score  = (passed + 0.5 * warned) / total * 100 if total else 0

    score_block = (
        f"\n{'─'*78}\n"
        f" UNIT TEST SCORE — {model_name}\n"
        f"{'─'*78}\n"
        f"  Total   : {total}\n"
        f"  ✅ PASS  : {passed:>3}  ({passed/total*100:.1f}%)\n"
        f"  ⚠️  WARN  : {warned:>3}  ({warned/total*100:.1f}%)\n"
        f"  ❌ FAIL  : {failed:>3}  ({failed/total*100:.1f}%)\n"
        f"  Score   : {score:.1f} / 100\n"
    )

    # Per-category breakdown
    cats = defaultdict(lambda: {"PASS":0,"WARN":0,"FAIL":0})
    for r in all_results:
        cats[r["category"]][r["verdict"]] += 1
    score_block += f"\n  {'Category':<14} {'PASS':>5} {'WARN':>5} {'FAIL':>5} {'Score':>7}\n"
    score_block += "  " + "─"*36 + "\n"
    for cat, c in cats.items():
        t = sum(c.values())
        s = (c["PASS"] + 0.5*c["WARN"]) / t * 100
        score_block += f"  {cat:<14} {c['PASS']:>5} {c['WARN']:>5} {c['FAIL']:>5} {s:>6.1f}%\n"

    fails = [r for r in all_results if r["verdict"] == "FAIL"]
    if fails:
        score_block += "\n  ❌ Failed:\n" + "\n".join(f"     - {r['label']}" for r in fails) + "\n"

    print(score_block); _log.log_write(score_block)
    return score

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _log.log_write(
        f"EIC Benchmark — Gemma 4 31B LoRA | AMD MI300X\n"
        f"Stress: {len(stress_prompts)} kernels | Unit: {len(unit_prompts)} snippets\n"
        + "="*80 + "\n"
    )

    unit_scores = {}

    for model_name, ctx in [("BASE MODEL", model.disable_adapter()), ("LORA MODEL", None)]:
        maybe_ctx = ctx if ctx else torch.inference_mode()

        if ctx:
            cm = ctx
        else:
            cm = __import__("contextlib").nullcontext()

        with cm:
            sec = f"\n{'#'*80}\n# {model_name}\n{'#'*80}"
            print(sec); _log.log_write(sec + "\n")

            # ── Stress phase ──────────────────────────────────────────────────
            _log.log_write("\n## STRESS TESTS\n")
            for max_tok, entries in sorted(stress_groups.items()):
                inputs_no  = build_inputs(entries, thinking=False)
                inputs_yes = build_inputs(entries, thinking=True)
                run_stress_benchmark(model_name, entries, max_tok, thinking=False)
                del inputs_no
                run_stress_benchmark(model_name, entries, max_tok, thinking=True)
                del inputs_yes
                torch.cuda.empty_cache()

            # ── Unit phase ────────────────────────────────────────────────────
            _log.log_write("\n## UNIT TESTS\n")
            unit_scores[model_name] = run_unit_benchmark(model_name)

        torch.cuda.empty_cache()

    # ── Final comparison ──────────────────────────────────────────────────────
    final = f"\n{'='*80}\n FINAL UNIT TEST SCORES\n{'='*80}\n"
    for name, s in unit_scores.items():
        bar = "█" * int(s / 5)
        final += f"  {name:<15} {bar:<20} {s:.1f}/100\n"
    final += "="*80
    print(final); _log.log_write(final + "\n")

    done = f"\n✓ Complete. Results saved to: {CONFIG['output_file']}"
    print(done); _log.log_write(done + "\n")
