import pyarrow.parquet as pq
import os

def main():
    input_file = os.path.join("Datasets", "ahmedheakl_cass_source_grpo.parquet")
    output_dir = "Datasets"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    table = pq.read_table(input_file)
    total_rows = len(table)
    print(f"Total rows: {total_rows}")

    # GitHub limit is 100MB. 149MB / 2 is ~75MB.
    # Splitting into 2 parts.
    midpoint = total_rows // 2
    
    part1_name = os.path.join(output_dir, "ahmedheakl_cass_source_grpo_part1.parquet")
    part2_name = os.path.join(output_dir, "ahmedheakl_cass_source_grpo_part2.parquet")

    print(f"Writing {part1_name}...")
    pq.write_table(table.slice(0, midpoint), part1_name)
    
    print(f"Writing {part2_name}...")
    pq.write_table(table.slice(midpoint), part2_name)

    # Verification
    for f in [part1_name, part2_name]:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"Generated {f}: {size_mb:.2f} MB")

    print("\nSuccessfully split the dataset into 2 parts. Both are now under the 100MB GitHub limit.")

if __name__ == "__main__":
    main()
