"""
data_counter.py
---------------
Counts real token counts for all datasets in ./Datasets/ using the
GLM-5 tokenizer (zai-org/GLM-5 on HuggingFace).

GLM-5 is hosted under the `zai-org` organisation (released Feb 2026).
AutoTokenizer with trust_remote_code=True is required for GLM models.

Column mapping handled automatically:
  - 'problem' or 'query'  -> input (prompt)
  - 'answer' or 'solution' -> output (response)
"""

import os
import sys
import pandas as pd
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOKENIZER_ID = "zai-org/GLM-5"

DATASETS = [
    {"file": "./Datasets/cass_part1.parquet", "input_col": "problem", "output_col": "answer"},
    {"file": "./Datasets/cass_part2.parquet", "input_col": "problem", "output_col": "answer"},
{"file": "./Datasets/cass_diverse_selected.parquet", "input_col": "problem", "output_col": "answer"}]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def batch_tokenize(tokenizer, texts, batch_size=512):
    """Return a list of token counts for each text using batch encoding."""
    counts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        counts.extend(len(ids) for ids in encoded["input_ids"])
    return counts


def analyze_dataset(tokenizer, dataset_cfg):
    file_path   = dataset_cfg["file"]
    input_col   = dataset_cfg["input_col"]
    output_col  = dataset_cfg["output_col"]
    name        = os.path.basename(file_path)

    print(f"\n  Loading  {name} ...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  [ERROR] Could not load {file_path}: {e}")
        return None

    if input_col not in df.columns or output_col not in df.columns:
        print(f"  [ERROR] Expected columns '{input_col}' and '{output_col}' not found in {name}.")
        print(f"         Available columns: {list(df.columns)}")
        return None

    input_texts  = df[input_col].fillna("").astype(str).tolist()
    output_texts = df[output_col].fillna("").astype(str).tolist()

    print(f"  Tokenising {len(df):,} rows (input) ...")
    input_token_counts  = batch_tokenize(tokenizer, input_texts)

    print(f"  Tokenising {len(df):,} rows (output) ...")
    output_token_counts = batch_tokenize(tokenizer, output_texts)

    total_input_tokens  = sum(input_token_counts)
    total_output_tokens = sum(output_token_counts)
    total_tokens        = total_input_tokens + total_output_tokens
    max_input           = max(input_token_counts)
    max_output          = max(output_token_counts)
    avg_input           = total_input_tokens / len(df)
    avg_output          = total_output_tokens / len(df)

    return {
        "name"               : name,
        "rows"               : len(df),
        "input_col"          : input_col,
        "output_col"         : output_col,
        "total_input_tokens" : total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens"       : total_tokens,
        "max_input_tokens"   : max_input,
        "max_output_tokens"  : max_output,
        "avg_input_tokens"   : avg_input,
        "avg_output_tokens"  : avg_output,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"  GLM-5 Token Counter")
    print(f"  Tokenizer : {TOKENIZER_ID}")
    print("=" * 70)

    print(f"\nLoading tokenizer '{TOKENIZER_ID}' (trust_remote_code=True) ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_ID,
            trust_remote_code=True,
        )
        print(f"  Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"[ERROR] Could not load tokenizer: {e}")
        sys.exit(1)

    results = []
    for cfg in DATASETS:
        res = analyze_dataset(tokenizer, cfg)
        if res is not None:
            results.append(res)

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("                    DATASET TOKEN COUNT SUMMARY (GLM-5)")
    print("=" * 70)

    header = (
        f"{'Dataset':<38} {'Rows':>7}  {'Input Tok':>12}  "
        f"{'Output Tok':>12}  {'Total Tok':>12}  {'MaxIn':>7}  {'MaxOut':>7}"
    )
    print(header)
    print("-" * 105)

    grand_rows   = 0
    grand_input  = 0
    grand_output = 0
    grand_total  = 0

    for r in results:
        print(
            f"{r['name']:<38} {r['rows']:>7,}  "
            f"{r['total_input_tokens']:>12,}  "
            f"{r['total_output_tokens']:>12,}  "
            f"{r['total_tokens']:>12,}  "
            f"{r['max_input_tokens']:>7,}  "
            f"{r['max_output_tokens']:>7,}"
        )
        grand_rows   += r["rows"]
        grand_input  += r["total_input_tokens"]
        grand_output += r["total_output_tokens"]
        grand_total  += r["total_tokens"]

    print("-" * 105)
    print(
        f"{'GRAND TOTAL':<38} {grand_rows:>7,}  "
        f"{grand_input:>12,}  "
        f"{grand_output:>12,}  "
        f"{grand_total:>12,}"
    )
    print("=" * 70)

    print("\n--- Per-Dataset Averages ---")
    for r in results:
        print(
            f"  {r['name']:<38}  "
            f"avg_input={r['avg_input_tokens']:>7.1f} tok/row  "
            f"avg_output={r['avg_output_tokens']:>8.1f} tok/row"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
