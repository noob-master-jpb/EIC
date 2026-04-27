import pandas as pd
import os

files = [
    r'd:\EIC\Datasets\ahmedheakl_cass_source_grpo_part1.parquet',
    r'd:\EIC\Datasets\ahmedheakl_cass_source_grpo_part2.parquet',
    r'd:\EIC\Datasets\codefeedback_filtered_part1.parquet',
    r'd:\EIC\Datasets\codefeedback_filtered_part2.parquet',
    r'd:\EIC\Datasets\codefeedback_filtered_part3.parquet',
    r'd:\EIC\Datasets\codefeedback_filtered_part4.parquet',
    r'd:\EIC\Datasets\nvidia_compute_eval_new.parquet',
    r'd:\EIC\Datasets\openhermes-coding-tasks_new.parquet',
    r'd:\EIC\Datasets\oss-ins-75k.parquet',
]

for f in files:
    df = pd.read_parquet(f)
    print(f'=== {os.path.basename(f)} ===')
    print(f'  Shape: {df.shape}')
    print(f'  Columns: {list(df.columns)}')
    for col in df.columns:
        sample = df[col].iloc[0]
        print(f'  [{col}] dtype={df[col].dtype}, sample_type={type(sample).__name__}, sample_preview={repr(str(sample)[:150])}')
    print()
