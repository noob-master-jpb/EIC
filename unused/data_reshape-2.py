import pyarrow.parquet as pq
import pyarrow as pa
import os

def main():
    input_file = os.path.join("Datasets", "openhermes-coding-tasks.parquet")
    output_parquet = os.path.join("Datasets", "openhermes-coding-tasks_new.parquet")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Loading dataset from: {input_file}")
    table = pq.read_table(input_file)
    records = table.to_pylist()
    
    output_records = []
    
    print(f"Processing {len(records)} records...")
    
    for row in records:
        problem = ""
        solution = ""
        
        # If there is a system prompt, prepend it to the problem
        system_prompt = row.get("system_prompt")
        if system_prompt:
            problem += f"System: {system_prompt}\n\n"
            
        conversations = row.get("conversations", [])
        
        for msg in conversations:
            role = msg.get("from")
            value = msg.get("value", "")
            
            if role == "human":
                if problem: problem += "\n\n"
                problem += value
            elif role == "gpt":
                if solution: solution += "\n\n"
                solution += value
                
        # Only add valid entries that have both a problem and solution
        if problem and solution:
            output_records.append({
                "problem": problem.strip(),
                "solution": solution.strip()
            })

    # Save to parquet
    out_table = pa.Table.from_pylist(output_records)
    pq.write_table(out_table, output_parquet)

    print(f"Successfully formatted {len(output_records)} records into 2-column format to {output_parquet}.")

if __name__ == "__main__":
    main()
