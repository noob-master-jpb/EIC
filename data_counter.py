import pandas as pd
import sys

def analyze_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if 'problem' not in df.columns or 'solution' not in df.columns:
        print("Error: Dataset must contain 'problem' and 'solution' columns.")
        sys.exit(1)

    total_problem_chars = 0
    total_solution_chars = 0
    
    total_problem_words = 0
    total_solution_words = 0
    
    unique_chars = set()
    
    max_problem_len = 0
    max_solution_len = 0
    
    print("\n--- Per-Row Character Counts ---")
    print("(Showing first 15 rows to prevent terminal flooding. Change 'MAX_ROWS_TO_PRINT' in the script to see more)")
    MAX_ROWS_TO_PRINT = 15
    
    for index, row in df.iterrows():
        problem_text = str(row['problem']) if pd.notna(row['problem']) else ""
        solution_text = str(row['solution']) if pd.notna(row['solution']) else ""
        
        prob_len = len(problem_text)
        sol_len = len(solution_text)
        
        total_problem_chars += prob_len
        total_solution_chars += sol_len
        
        # Word counts for tokenizer context
        total_problem_words += len(problem_text.split())
        total_solution_words += len(solution_text.split())
        
        # Max lengths for deciding maximum sequence length
        max_problem_len = max(max_problem_len, prob_len)
        max_solution_len = max(max_solution_len, sol_len)
        
        # Unique characters to know the raw character vocabulary size
        unique_chars.update(problem_text)
        unique_chars.update(solution_text)
        
        if index < MAX_ROWS_TO_PRINT:
            print(f"Row = {index} | No. of characters under problem: {prob_len} | No. of characters under solution: {sol_len}")
        elif index == MAX_ROWS_TO_PRINT:
            print("...\n(Remaining rows omitted for terminal readability)\n")

    total_chars = total_problem_chars + total_solution_chars
    total_words = total_problem_words + total_solution_words
    
    print("\n============================================================")
    print("                     DATASET SUMMARY")
    print("============================================================")
    print(f"Total Rows: {len(df):,}")
    print(f"Total characters under 'problem' : {total_problem_chars:,}")
    print(f"Total characters under 'solution': {total_solution_chars:,}")
    print(f"Total characters in dataset      : {total_chars:,}")
    
    print("\n============================================================")
    print("          TOKENIZER PLANNING METRICS (EXTRA INFO)")
    print("============================================================")
    print(f"Total unique characters (Char Vocab Size): {len(unique_chars):,}")
    print(f"Total approximate words (Whitespace split): {total_words:,}")
    print(f"Max characters in a single 'problem' : {max_problem_len:,}")
    print(f"Max characters in a single 'solution': {max_solution_len:,}")
    print(f"Average characters per 'problem'     : {total_problem_chars / len(df):,.2f}")
    print(f"Average characters per 'solution'    : {total_solution_chars / len(df):,.2f}")
    
    # Standard rule of thumb for English text tokenizers (like Llama/Gemma)
    # is roughly ~4 characters per token or ~0.75 words per token.
    approx_tokens = total_chars / 4.0
    print(f"\nApproximate total tokens (assuming ~4 chars/token): {approx_tokens:,.0f}")

if __name__ == "__main__":
    file_path = r"d:\EIC\Datasets\combined_output.parquet"
    analyze_dataset(file_path)
