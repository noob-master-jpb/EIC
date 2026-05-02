import pandas as pd
import os

def convert_dataset():
    input_file = r"d:\EIC\Datasets\cuda_to_rocm_distill_glm5.jsonl"
    output_file = r"d:\EIC\Datasets\cuda_to_rocm_final.parquet"

    print(f"Loading data from {input_file}...")
    df = pd.read_json(input_file, lines=True)

    # Strip away 'index' and 'usage' columns
    # Follow the format of cass_diverse_selected.parquet (problem, answer)
    print("Formatting columns to match (problem, answer)...")
    final_df = df[['problem', 'response']].copy()
    final_df.rename(columns={'response': 'answer'}, inplace=True)

    print(f"Saving final dataset to {output_file}...")
    final_df.to_parquet(output_file, index=False)

    print("\n============================================================")
    print("                DATASET CONVERSION SUCCESSFUL")
    print("============================================================")
    print(f"Output File : {output_file}")
    print(f"Columns     : {final_df.columns.tolist()}")
    print(f"Total Rows  : {len(final_df):,}")
    print("------------------------------------------------------------")
    print("Sample Output (Row 0):")
    print("------------------------------------------------------------")
    print(f"[PROBLEM]\n{final_df.iloc[0]['problem'][:200]}...\n")
    print(f"[ANSWER]\n{final_df.iloc[0]['answer'][:200]}...")
    print("============================================================")

if __name__ == "__main__":
    convert_dataset()
