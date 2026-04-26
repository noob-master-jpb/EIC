import os
from datasets import load_dataset

# 1. Download the dataset
print("Downloading m-a-p/CodeFeedback-Filtered-Instruction...")
ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction")

# 2. Setup path and save as parquet
output_dir = "Datasets"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "codefeedback_filtered.parquet")

print(f"Saving to {output_file}...")
ds['train'].to_parquet(output_file)

print("Done.")
