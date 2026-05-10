import os
import sys
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN        = 512
MAX_EVAL_TOKENS = 4096   # tokens to score for perplexity; None = full test set
MAX_NEW_TOKENS  = 200

# ── Dataset Config (Mirrored from billm.py) ───────────────────────────────────

DATASETS = [
    "/root/EIC/Datasets/openhermes.parquet",
    "/root/EIC/Datasets/oss-ins-75k.parquet",
    "/root/EIC/Datasets/nvidia_compute_eval.parquet",
    "/root/EIC/Datasets/cass_diverse_selected.parquet",
    "/root/EIC/Datasets/cass_part1.parquet",
    "/root/EIC/Datasets/codefeedback_part1.parquet",
    "/root/EIC/Datasets/cuda_to_rocm_distill_glm5.jsonl",
    "/root/EIC/Datasets/open_thoughts_reasoning.parquet",
    "/root/EIC/Datasets/smoltalk_chat.parquet",
    "/root/EIC/Datasets/numina_math_reasoning.parquet",
    # "wikitext",
]

COLUMN_MAPPING = {
    "/root/EIC/Datasets/openhermes.parquet":            ["problem", "solution"],
    "/root/EIC/Datasets/oss-ins-75k.parquet":           ["problem", "solution"],
    "/root/EIC/Datasets/nvidia_compute_eval.parquet":   ["problem", "solution"],
    "/root/EIC/Datasets/cass_diverse_selected.parquet": ["problem", "answer"],
    "/root/EIC/Datasets/cass_part1.parquet":            ["problem", "answer"],
    "/root/EIC/Datasets/codefeedback_part1.parquet":    ["query", "answer"],
    "/root/EIC/Datasets/cuda_to_rocm_distill_glm5.jsonl": ["problem", "response"],
    "/root/EIC/Datasets/open_thoughts_reasoning.parquet": ["problem", "solution"],
    "/root/EIC/Datasets/smoltalk_chat.parquet":           ["problem", "solution"],
    "/root/EIC/Datasets/numina_math_reasoning.parquet":   ["problem", "solution"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(model_path):
    print(f"[Eval] Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        model.eval()
    except Exception as e:
        print(f"[Eval] Error loading model: {e}")
        sys.exit(1)
    print(f"[Eval] Model loaded on device: {model.device}")
    return tokenizer, model


@torch.no_grad()
def measure_perplexity(model, tokenizer):
    """
    Measure perplexity on the dataset mixture defined in DATASETS.
    Uses a sliding-window approach.
    """
    all_text = []

    for ds_path in DATASETS:
        print(f"[Eval] Loading {os.path.basename(ds_path) if '/' in ds_path else ds_path}...")
        try:
            if os.path.exists(ds_path):
                if ds_path.endswith(".parquet"):
                    ds = load_dataset("parquet", data_files=ds_path, split="train")
                elif ds_path.endswith(".jsonl"):
                    ds = load_dataset("json", data_files=ds_path, split="train")
                else:
                    continue
            elif ds_path == "wikitext":
                # Use test split for wikitext if available
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            else:
                ds = load_dataset(ds_path, split="test" if "test" in ds_path else "train")

            cols = COLUMN_MAPPING.get(ds_path)
            # Take a small slice for eval to keep it fast
            subset = ds.select(range(min(200, len(ds))))

            if cols:
                for i in range(len(subset)):
                    row = [str(subset[c][i]) for c in cols if c in subset.column_names and subset[c][i]]
                    if row: all_text.append("\n\n".join(row))
            else:
                possible = ["text", "content", "instruction", "output"]
                text_field = next((f for f in possible if f in ds.column_names), None)
                if text_field:
                    all_text.append("\n\n".join(subset[text_field]))

        except Exception as e:
            print(f"[Warn] Failed to load {ds_path}: {e}")

    if not all_text:
        print("[Error] No evaluation data found!")
        return float('inf')

    enc = tokenizer("\n\n".join(all_text), return_tensors="pt").input_ids
    
    if MAX_EVAL_TOKENS and enc.size(1) > MAX_EVAL_TOKENS:
        print(f"[Perplexity] Capping to {MAX_EVAL_TOKENS} tokens (full set: {enc.size(1)})")
        enc = enc[:, :MAX_EVAL_TOKENS]

    device  = next(model.parameters()).device
    stride  = SEQ_LEN
    max_len = min(getattr(model.config, "max_position_embeddings", 8192), 2048)
    nlls    = []

    for i in range(0, enc.size(1), stride):
        begin      = max(i + stride - max_len, 0)
        end        = i + stride
        input_ids  = enc[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        loss = model(input_ids, labels=target_ids).loss
        nlls.append(loss.cpu() * stride)

        del input_ids, target_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ppl = math.exp(sum(nlls) / (len(nlls) * stride))
    print(f"[Perplexity] PPL = {ppl:.3f}  (lower is better)\n")
    return ppl


def run_generation(model, tokenizer):
    """
    Greedy generation on a fixed prompt set.
    Greedy (do_sample=False) is preferred for quantized model eval:
      - deterministic and reproducible
      - no temperature scaling needed (ternary weights already reduce variance)
    """
    prompts = {
        "Basic Greeting"  : "Hello! How are you doing today?",
        "Arithmetic"      : "If I have 5 apples, eat 2, and then buy 3 more, how many apples do I have?",
        "Science"         : "Explain quantum entanglement using a simple analogy for a 10-year-old.",
        "Coding"          : "Write a Python function that returns the nth Fibonacci number recursively.",
        "Instruction"     : "List three tips for improving sleep quality.",
    }

    device = next(model.parameters()).device
    print("=" * 60)
    print(f"Generation  |  device={device}  |  greedy (do_sample=False)")
    print("=" * 60 + "\n")

    for category, prompt in prompts.items():
        print(f"[{category}]\nPrompt: {prompt}")

        messages = [{"role": "user", "content": prompt}]
        try:
            text   = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(device)
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,            # greedy — deterministic
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs.input_ids.shape[1]
        response     = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        print(f"\nResponse:\n{response.strip()}\n")
        print("-" * 60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEFAULT_MODEL_PATH = "/root/EIC/gemma-4-31B-it-merged-billm"
    # DEFAULT_MODEL_PATH = "./gemma-4-E4B-billm"
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        print(f"Error: path '{model_path}' does not exist.")
        print("Usage: python billm_eval.py [path_to_model]")
        sys.exit(1)

    tokenizer, model = load_model(model_path)
    measure_perplexity(model, tokenizer)
    run_generation(model, tokenizer)
