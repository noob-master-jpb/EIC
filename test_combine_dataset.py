"""
Full test of combine_dataset() across all 9 parquet datasets in d:/EIC/Datasets/.

Column schemas discovered:
  Group A  (ahmedheakl_cass_source_grpo_part1/2)     -> ['problem', 'answer']
  Group B  (codefeedback_filtered_part1/2/3/4)        -> ['query',   'answer']
  Group C  (nvidia_compute_eval_new,
            openhermes-coding-tasks_new,
            oss-ins-75k)                              -> ['problem', 'solution']

Output mapping chosen: {"user": "problem", "agent": "solution"}
Output path          : d:/EIC/Datasets/combined_output.parquet
"""

import os
import pandas as pd
from data import combine_dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.join("d:\\", "EIC", "Datasets")

def p(name):
    return os.path.join(BASE, name)

PATHS = [
    # Group A  [problem, answer]
    p("ahmedheakl_cass_source_grpo_part1.parquet"),
    p("ahmedheakl_cass_source_grpo_part2.parquet"),
    # Group B  [query, answer]
    p("codefeedback_filtered_part1.parquet"),
    p("codefeedback_filtered_part2.parquet"),
    p("codefeedback_filtered_part3.parquet"),
    p("codefeedback_filtered_part4.parquet"),
    # Group C  [problem, solution]
    p("nvidia_compute_eval_new.parquet"),
    p("openhermes-coding-tasks_new.parquet"),
    p("oss-ins-75k.parquet"),
]

# ── column_mapping: path -> [user_col, agent_col] ─────────────────────────────
COLUMN_MAPPING = {
    # Group A
    p("ahmedheakl_cass_source_grpo_part1.parquet"): ["problem", "answer"],
    p("ahmedheakl_cass_source_grpo_part2.parquet"): ["problem", "answer"],
    # Group B
    p("codefeedback_filtered_part1.parquet"): ["query", "answer"],
    p("codefeedback_filtered_part2.parquet"): ["query", "answer"],
    p("codefeedback_filtered_part3.parquet"): ["query", "answer"],
    p("codefeedback_filtered_part4.parquet"): ["query", "answer"],
    # Group C
    p("nvidia_compute_eval_new.parquet"): ["problem", "solution"],
    p("openhermes-coding-tasks_new.parquet"): ["problem", "solution"],
    p("oss-ins-75k.parquet"): ["problem", "solution"],
}

# ── output_mapping: unified column names ──────────────────────────────────────
#   dict form  -> {"user": "<col>", "agent": "<col>"}
OUTPUT_MAPPING = {"user": "problem", "agent": "solution"}

# ── output_path ───────────────────────────────────────────────────────────────
OUTPUT_PATH = p("combined_output.parquet")


def main():
    print("=" * 70)
    print("combine_dataset() -- Full Test")
    print("=" * 70)

    print(f"\nDatasets to combine   : {len(PATHS)}")
    print(f"Output columns        : {OUTPUT_MAPPING}")
    print(f"Output path           : {OUTPUT_PATH}\n")

    # ── per-dataset preview ───────────────────────────────────────────────────
    print("-" * 70)
    print("Input dataset column analysis:")
    print("-" * 70)
    for path in PATHS:
        src = COLUMN_MAPPING[path]
        df_peek = pd.read_parquet(path)
        fname = os.path.basename(path)
        print(f"  {fname:<52}  {list(df_peek.columns)}  rows={len(df_peek):>6}"
              f"  mapped=[{src[0]!r} -> 'problem', {src[1]!r} -> 'solution']")

    # ── run combine_dataset ───────────────────────────────────────────────────
    print("\nRunning combine_dataset() ...")
    combined: pd.DataFrame = combine_dataset(
        paths=PATHS,
        column_mapping=COLUMN_MAPPING,
        output_mapping=OUTPUT_MAPPING,
        output_path=OUTPUT_PATH,
    )

    # ── results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Combined shape   : {combined.shape}")
    print(f"  Columns          : {list(combined.columns)}")
    print(f"  Total rows       : {len(combined):,}")
    print(f"  Null 'problem'   : {combined['problem'].isna().sum():,}")
    print(f"  Null 'solution'  : {combined['solution'].isna().sum():,}")
    print(f"  Saved to         : {OUTPUT_PATH}")

    # ── expected row counts per group ─────────────────────────────────────────
    expected_rows = {
        "ahmedheakl (part1+2)"         : 31727 + 32309,
        "codefeedback_filtered (p1-4)" : 51926 + 51838 + 51885 + 526,
        "nvidia_compute_eval"          : 566,
        "openhermes-coding-tasks"      : 5561,
        "oss-ins-75k"                  : 74791,
    }
    total_expected = sum(expected_rows.values())
    print(f"\n  Expected row counts by group:")
    for name, cnt in expected_rows.items():
        print(f"    {name:<40} {cnt:>7,}")
    print(f"    {'TOTAL':<40} {total_expected:>7,}")
    assert len(combined) == total_expected, (
        f"Row count mismatch! Got {len(combined):,}, expected {total_expected:,}"
    )
    print(f"\n  OK Row count matches expected total: {total_expected:,}")

    # ── sample rows ───────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Sample rows (first 3):")
    print("-" * 70)
    for i, row in combined.head(3).iterrows():
        print(f"\n  Row {i}:")
        print(f"    problem  : {row['problem'][:120]!r}")
        print(f"    solution : {row['solution'][:120]!r}")

    print("\n" + "=" * 70)
    print("PASSED combine_dataset() test PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
