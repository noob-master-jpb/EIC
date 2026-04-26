import pyarrow.parquet as pq
import os
from collections import defaultdict

# 1. Load the parquet file 
file_name = os.path.join("Datasets", "codefeedback_filtered.parquet")
output_file = os.path.join("Datasets", "codefeedback.txt")

print(f"Loading {file_name}...")
# Load the parquet dataset using PyArrow for massive speedup
table = pq.read_table(file_name)
records = table.to_pylist()

print("Categorizing by language...")
# 2. Group the entire dataset by the 'lang' column
grouped_data = defaultdict(list)
for row in records:
    lang = row.get('lang')
    lang_str = str(lang) if lang is not None else "N/A"
    grouped_data[lang_str].append(row)

print(f"Writing to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    # Iterate through each language group sorted alphabetically
    for lang in sorted(grouped_data.keys()):
        group = grouped_data[lang]
        
        # Create a massive header for the new language section
        f.write("#" * 60 + "\n")
        f.write(f"### LANGUAGE: {lang.upper()} ###\n")
        f.write("#" * 60 + "\n\n")
        
        # Write every row for this language
        for row in group:
            query = row.get('query')
            query_str = str(query).strip() if query is not None else "N/A"
            
            answer = row.get('answer')
            answer_str = str(answer).strip() if answer is not None else "N/A"
            
            resource = row.get('resource')
            resource_str = str(resource).strip() if resource is not None else "N/A"
            
            f.write(f"query:\n{query_str}\n\n")
            f.write(f"answer:\n{answer_str}\n\n")
            f.write(f"resource: {resource_str}\n")
            f.write("-" * 60 + "\n\n")

print(f"Done! Check your folder for {output_file}")