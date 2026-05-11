from huggingface_hub import snapshot_download
import os

local_dir = r"D:\EIC\models\gemma-4-E2B-it"
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading unsloth/gemma-4-E2B-it to {local_dir}...")
snapshot_download(repo_id="unsloth/gemma-4-E2B-it", local_dir=local_dir)
print("Download complete!")
