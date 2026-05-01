import pandas as pd
import os

def create_dataset():
    input_file = r"d:\EIC\Datasets\nvidia_compute_eval_glm5.jsonl"
    output_file = r"d:\EIC\Datasets\cuda_to_rocm_distill.parquet"

    print(f"Loading data from {input_file}...")
    df = pd.read_json(input_file, lines=True)

    print("Transforming dataset...")
    # Prepend the requested prompt to the original 'response' column
    prompt_prefix = "Convert this CUDA kernal to ROCm/HIP kernal\n\n"
    
    # Create the new dataset dataframe
    new_df = pd.DataFrame()
    new_df['Problem'] = prompt_prefix + df['response'].astype(str)
    new_df['Response'] = ""  # Leave the response blank for the AI to fill during distillation

    print(f"Saving new dataset to {output_file}...")
    new_df.to_parquet(output_file, index=False)

    print("\n============================================================")
    print("                DATASET CREATION SUCCESSFUL")
    print("============================================================")
    print(f"Output File : {output_file}")
    print(f"Total Rows  : {len(new_df):,}")
    print("------------------------------------------------------------")
    print("Sample Output (Row 0):")
    print("------------------------------------------------------------")
    print("[PROBLEM COLUMN]")
    
    sample_prob = new_df.iloc[0]['Problem']
    # Print just the first 300 chars to not flood terminal
    print(f"{sample_prob[:300]}...\n")
    
    print(f"[RESPONSE COLUMN]")
    print(f"'{new_df.iloc[0]['Response']}' (Blank)\n")
    print("============================================================")

if __name__ == "__main__":
    create_dataset()
