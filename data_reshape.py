import pyarrow.parquet as pq
import pyarrow as pa
import json
import sys
import os

def main():
    # 1. Parse arguments for robust file handling
    input_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join("Datasets", "nvidia_compute_eval.parquet")
    output_parquet = sys.argv[2] if len(sys.argv) > 2 else os.path.join("Datasets", "organized_nvidia_compute_eval.parquet")
    output_txt = sys.argv[3] if len(sys.argv) > 3 else os.path.join("Datasets", "organized_nvidia_compute_eval.txt")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Loading dataset from: {input_file}")
    # Load the parquet dataset using PyArrow for massive speedup
    table = pq.read_table(input_file)
    records = table.to_pylist()
    
    print(f"Processing {len(records)} records...")
    
    output_records = []
    
    with open(output_txt, "w", encoding="utf-8") as f_out:
        for line_idx, row in enumerate(records, start=1):
            
            # --- 1. Extract raw data ---
            workspace = ""
            context_files = row.get("context_files")
            if context_files:
                for file in context_files:
                    workspace += f"File: {file.get('path', '')}\n```cpp\n{file.get('content', '')}\n```\n"
                    
            test_files = row.get("test_files")
            if test_files:
                for file in test_files:
                    workspace += f"Test File: {file.get('path', '')}\n```cpp\n{file.get('content', '')}\n```\n"
            
            workspace = workspace.strip()
            if not workspace:
                workspace = "None"
                    
            cuda_tk = row.get('min_cuda_toolkit')
            cuda_tk = "Any" if cuda_tk is None else str(cuda_tk)
            build_cmd = row.get('build_command')
            build_cmd = "N/A" if build_cmd is None else str(build_cmd)
            test_cmd = row.get('test_command')
            test_cmd = "N/A" if test_cmd is None else str(test_cmd)
            
            environment = f"CUDA Toolkit: {cuda_tk}\nBuild Command: {build_cmd}\nTest Command: {test_cmd}"
            
            prompt = row.get('prompt')
            prompt = "" if prompt is None else str(prompt)
            
            solution_content = ""
            baseline = row.get("baseline_solution")
            if isinstance(baseline, dict) and baseline.get("files"):
                solution_content = baseline["files"][0].get("content", "")

            # --- 2. Build the Model Context for Parquet ---
            user_content = f"### WORKSPACE CONTEXT ###\n{workspace}\n\n### ENVIRONMENT ###\n{environment}\n\n### INSTRUCTION ###\n{prompt}\n"

            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert GPU C++ developer. Write optimized code to solve the user's problem. <|think|>"
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "model",
                        "content": solution_content
                    }
                ]
            }
            output_records.append(training_example)
            
            # --- 3. Write Human Readable Text ---
            f_out.write(f"Task / Record: {line_idx}\n")
            f_out.write("="*60 + "\n\n")
            
            f_out.write("### PROMPT / INPUT ###\n")
            f_out.write(f"{prompt}\n\n")
            f_out.write(f"Environment:\n{environment}\n\n")
            
            f_out.write("### CONTEXT / THOUGHT ###\n")
            f_out.write("WORKSPACE PROVIDED:\n")
            f_out.write(f"{workspace}\n\n")
            

            f_out.write("### OUTPUT ###\n")
            f_out.write(f"{solution_content}\n\n")
            
            f_out.write("#"*80 + "\n\n\n")

    # --- 4. Write Parquet ---
    out_table = pa.Table.from_pylist(output_records)
    pq.write_table(out_table, output_parquet)

    print(f"Successfully formatted {len(records)} records for Gemma 4 training to {output_parquet}.")
    print(f"Human-readable dataset generated successfully to {output_txt}.")

if __name__ == "__main__":
    main()