import pandas as pd
import sys
import os

def analyze_multiple_datasets(file_paths):
    total_problem_chars = 0
    total_solution_chars = 0
    total_rows = 0
    total_problem_words = 0
    total_solution_words = 0
    unique_chars = set()
    max_problem_len = 0
    max_solution_len = 0
    
    # Track overall index for per-row reporting
    overall_row_count = 0
    MAX_ROWS_TO_PRINT = 15
    
    print("\n--- Per-Row Character Counts ---")
    print(f"(Showing first {MAX_ROWS_TO_PRINT} rows across all files to prevent terminal flooding)")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        # Try to identify columns (handling common naming conventions)
        prob_col = next((c for c in ['problem', 'query', 'instruction'] if c in df.columns), None)
        sol_col = next((c for c in ['solution', 'answer', 'output'] if c in df.columns), None)

        if not prob_col or not sol_col:
            print(f"Error in {file_path}: Could not find problem/solution columns. Available: {df.columns.tolist()}")
            continue

        for _, row in df.iterrows():
            problem_text = str(row[prob_col]) if pd.notna(row[prob_col]) else ""
            solution_text = str(row[sol_col]) if pd.notna(row[sol_col]) else ""
            
            prob_len = len(problem_text)
            sol_len = len(solution_text)
            
            total_problem_chars += prob_len
            total_solution_chars += sol_len
            
            total_problem_words += len(problem_text.split())
            total_solution_words += len(solution_text.split())
            
            max_problem_len = max(max_problem_len, prob_len)
            max_solution_len = max(max_solution_len, sol_len)
            
            unique_chars.update(problem_text)
            unique_chars.update(solution_text)
            
            if overall_row_count < MAX_ROWS_TO_PRINT:
                print(f"Row = {overall_row_count} | No. of characters under problem: {prob_len} | No. of characters under solution: {sol_len}")
            elif overall_row_count == MAX_ROWS_TO_PRINT:
                print("...\n(Remaining rows omitted for terminal readability)\n")
                
            overall_row_count += 1
        
        total_rows += len(df)

    if total_rows == 0:
        print("No data processed.")
        return

    total_chars = total_problem_chars + total_solution_chars
    total_words = total_problem_words + total_solution_words
    
    print("\n============================================================")
    print("                COMBINED DATASET SUMMARY")
    print("============================================================")
    print(f"Files Processed  : {len(file_paths)}")
    print(f"Total Rows       : {total_rows:,}")
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
    print(f"Average characters per 'problem'     : {total_problem_chars / total_rows:,.2f}")
    print(f"Average characters per 'solution'    : {total_solution_chars / total_rows:,.2f}")
    
    approx_tokens = total_chars / 4.0
    print(f"\nApproximate total tokens (assuming ~4 chars/token): {approx_tokens:,.0f}")

if __name__ == "__main__":
    files = [
        r"d:\EIC\Datasets\cass_part1.parquet",
        r"d:\EIC\Datasets\cass_part2.parquet",
        r"d:\EIC\Datasets\nvidia_compute_eval.parquet"
    ]
    analyze_multiple_datasets(files)
