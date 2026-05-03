import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import time
import os

# Suppress Hugging Face / Transformers expected key warnings
import transformers
transformers.logging.set_verbosity_error()

def get_diversity_score(model, file_paths, batch_size=600):
    print(f"[load] Loading datasets: {[os.path.basename(f) for f in file_paths]}")
    dfs = []
    for fp in file_paths:
        df = pd.read_parquet(fp, columns=["problem"])
        dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    texts = df_combined["problem"].tolist()
    N = len(texts)
    
    print(f"[embed] Encoding {N:,} texts...")
    t0 = time.time()
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_tensor=True
    )
    print(f"[embed] Encoding finished in {time.time()-t0:.2f}s")
    
    print("[math] Calculating highly-optimized diversity metrics on GPU...")
    embeddings = embeddings.float()
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    variance = torch.var(embeddings, dim=0).sum().item()
    
    sum_embeddings = torch.sum(embeddings, dim=0) 
    sum_sq_norm = torch.sum(sum_embeddings ** 2).item()
    
    total_pairwise_sim = sum_sq_norm - N
    num_pairs = N * (N - 1) if N > 1 else 1
    
    avg_pairwise_sim = total_pairwise_sim / num_pairs
    avg_pairwise_dist = 1.0 - avg_pairwise_sim
    
    return {
        "N": N,
        "variance": variance,
        "avg_pairwise_sim": avg_pairwise_sim,
        "avg_pairwise_dist": avg_pairwise_dist
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[hardware] Utilizing device: {device.upper()}")
    
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"[embed] Loading embedding model: {model_name}")
    # Load model once for memory efficiency and speed
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    print("\n--- Evaluating Original Dataset (Part 1 + Part 2) ---")
    orig_files = [
        r"d:\EIC\Datasets\cass_part1.parquet",
        r"d:\EIC\Datasets\cass_part2.parquet"
    ]
    orig_metrics = get_diversity_score(model, orig_files)
    
    print("\n--- Evaluating Selected Diverse Dataset ---")
    diverse_file = [r"d:\EIC\Datasets\cass_diverse_selected.parquet"]
    diverse_metrics = get_diversity_score(model, diverse_file)
    
    print("\n" + "="*80)
    print("                             DIVERSITY COMPARISON")
    print("="*80)
    print(f"{'Metric':<25} | {'Original Dataset':<18} | {'Selected Dataset':<16} | {'Ratio (Sel/Orig)'}")
    print("-" * 80)
    print(f"{'Dataset Size':<25} | {orig_metrics['N']:<18,} | {diverse_metrics['N']:<16,} | N/A")
    print(f"{'Total Variance':<25} | {orig_metrics['variance']:<18.5f} | {diverse_metrics['variance']:<16.5f} | {diverse_metrics['variance'] / orig_metrics['variance']:.4f}x")
    print(f"{'Avg Pairwise Cosine Sim':<25} | {orig_metrics['avg_pairwise_sim']:<18.5f} | {diverse_metrics['avg_pairwise_sim']:<16.5f} | {diverse_metrics['avg_pairwise_sim'] / orig_metrics['avg_pairwise_sim']:.4f}x")
    print(f"{'Avg Pairwise Cosine Dist':<25} | {orig_metrics['avg_pairwise_dist']:<18.5f} | {diverse_metrics['avg_pairwise_dist']:<16.5f} | {diverse_metrics['avg_pairwise_dist'] / orig_metrics['avg_pairwise_dist']:.4f}x")
    print("="*80)

if __name__ == "__main__":
    main()
