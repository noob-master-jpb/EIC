import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

# Maximize hardware thread usage for background math operations
os.environ["OMP_NUM_THREADS"]      = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"]      = "16"

DATASET_DIR = "./Datasets"
CASS_FILES  = [
    os.path.join(DATASET_DIR, "cass_part1.parquet"),
    os.path.join(DATASET_DIR, "cass_part2.parquet"),
]
OUTPUT_FILE = os.path.join(DATASET_DIR, "cass_diverse_selected.parquet")
N_SAMPLES = 12_000
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def load_cass_data(files: list[str]) -> pd.DataFrame:
    dfs = []
    for f in files:
        print(f"[load] {f} ...")
        dfs.append(pd.read_parquet(f, columns=["problem", "answer"]))
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[load] Total rows: {len(combined):,}")
    return combined

def main():
    # 1. Hardware Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[hardware] Utilizing device: {device.upper()}")
    
    # 2. Load Data
    df = load_cass_data(CASS_FILES)
    problems = df["problem"].tolist()
    
    # 3. Hardware Accelerated Embeddings
    print(f"\n[embed] Loading model {EMBED_MODEL} into video memory...")
    model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5", 
    model_kwargs={"torch_dtype": torch.float16}, 
    device="cuda"
)
    
    print(f"[embed] Encoding {len(problems):,} problems...")
    # Push batch size to 512 to fully utilize available video memory
    embeddings = model.encode(
    problems, 
    batch_size=600, 
    show_progress_bar=True,
)
    
    # 4. Multi-Threaded Clustering
    print(f"\n[kmeans] Clustering into {N_SAMPLES:,} groups...")
    km = MiniBatchKMeans(
        n_clusters=N_SAMPLES,
        batch_size=2048, 
        n_init="auto",
        random_state=42
    )
    km.fit(embeddings)
    
    print("[kmeans] Finding closest sample to each centroid...")
    closest_indices = pairwise_distances_argmin(
        km.cluster_centers_,
        embeddings,
        metric="cosine"
    )
    
    unique_indices = np.unique(closest_indices)
    print(f"[kmeans] Unique representatives: {len(unique_indices):,}")
    
    # 5. Save Results
    diverse_df = df.iloc[unique_indices].reset_index(drop=True)
    diverse_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n[done] Saved diverse dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()