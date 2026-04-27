import pyarrow.parquet as pq
import pyarrow as pa
import os

# 1. Load the parquet file
file_name = os.path.join("Datasets", "codefeedback_filtered.parquet")
CHUNK_SIZE = 52000

print(f"Loading {file_name}...")
# Load the parquet dataset using PyArrow for massive speedup
table = pq.read_table(file_name)
records = table.to_pylist()
total_records = len(records)

print(f"Total records found: {total_records}")
print(f"Chunking into sets of {CHUNK_SIZE} rows...")

# Split records into chunks
chunks = [records[i:i + CHUNK_SIZE] for i in range(0, total_records, CHUNK_SIZE)]

for chunk_index, chunk_records in enumerate(chunks, start=1):
    output_parquet = os.path.join("Datasets", f"codefeedback_filtered_part{chunk_index}.parquet")
    output_text = os.path.join("Datasets", f"codefeedback_part{chunk_index}.txt")
    
    print(f"\n--- Processing Chunk {chunk_index} ({len(chunk_records)} rows) ---")
    
    # 1. Write the text file for this chunk
    print(f"Writing text chunk to {output_text}...")
    with open(output_text, 'w', encoding='utf-8') as f:
        for row_idx, row in enumerate(chunk_records, start=1):
            query = row.get('query')
            query_str = str(query).strip() if query is not None else "N/A"
            
            answer = row.get('answer')
            answer_str = str(answer).strip() if answer is not None else "N/A"
            
            f.write(f"Task / Record: {row_idx + ((chunk_index-1)*CHUNK_SIZE)}\n")
            f.write("="*60 + "\n")
            f.write(f"query:\n{query_str}\n\n")
            f.write(f"answer:\n{answer_str}\n")
            f.write("="*60 + "\n\n\n")
            
    # 2. Write the parquet file for this chunk
    print(f"Writing parquet chunk to {output_parquet}...")
    out_table = pa.Table.from_pylist(chunk_records)
    pq.write_table(out_table, output_parquet)

print("\nDone! All chunks processed and saved.")