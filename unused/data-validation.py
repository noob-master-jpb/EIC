import pyarrow.parquet as pq
import pyarrow as pa
import os
import re

# ---------------------------------------------------------------------------
# ALLOWED UNICODE RANGES
# We allow everything EXCEPT foreign natural language scripts
# (i.e. Chinese, Japanese, Korean, Arabic, Cyrillic, Hindi, etc.)
# ---------------------------------------------------------------------------

# Build one big regex pattern for DISALLOWED characters.
# A character is "foreign" if it matches this pattern.
# We allow:
#   - All ASCII (0x00-0x7F)
#   - Latin Extended A/B (accented chars: é, ñ, ö, etc.) → U+0080-U+024F
#   - Greek/Coptic (math: α, β, Σ, π, μ, etc.)         → U+0370-U+03FF
#   - General Punctuation (smart quotes, dashes, …)     → U+2000-U+206F
#   - Letterlike Symbols (™, ©, etc.)                   → U+2100-U+214F
#   - Mathematical Operators (→, √, ≠, ±, etc.)         → U+2200-U+22FF
#   - Supplemental Math Operators                        → U+2A00-U+2AFF
#   - Box Drawing (tables: ─, │, ┌, etc.)               → U+2500-U+257F
#   - Block Elements (progress bars: █)                  → U+2580-U+259F
#   - Arrows (→, ⇒, etc.)                                → U+2190-U+21FF
#   - Superscripts/Subscripts (², ³, ₁, ₂)              → U+2070-U+209F
#   - Currency Symbols (€, £, ¥)                         → U+20A0-U+20CF
#   - Enclosed Alphanumerics                              → U+2460-U+24FF
#   - Miscellaneous Technical                             → U+2300-U+23FF
#   - Dingbats (✓, ✗)                                    → U+2700-U+27BF
#   - Emoji (🤖, 😊, etc.)                               → U+1F300-U+1FAFF
#   - Misc Symbols and Pictographs                        → U+1F000-U+1F2FF

DISALLOWED_PATTERN = re.compile(
    r"[^\x00-\x7F"        # ASCII
    r"\u0080-\u024F"      # Latin Extended
    r"\u0370-\u03FF"      # Greek
    r"\u2000-\u206F"      # General Punctuation
    r"\u2070-\u209F"      # Superscripts/Subscripts
    r"\u20A0-\u20CF"      # Currency
    r"\u2100-\u214F"      # Letterlike Symbols
    r"\u2190-\u21FF"      # Arrows
    r"\u2200-\u22FF"      # Math Operators
    r"\u2300-\u23FF"      # Misc Technical
    r"\u2460-\u24FF"      # Enclosed Alphanumerics
    r"\u2500-\u257F"      # Box Drawing
    r"\u2580-\u259F"      # Block Elements
    r"\u2600-\u27BF"      # Misc Symbols, Dingbats
    r"\u2A00-\u2AFF"      # Supplemental Math
    r"\U0001F000-\U0001FAFF"  # Emoji
    r"]+"
)

NON_ENGLISH_THRESHOLD = 0.005  # 0.5%


def get_non_english_ratio(text: str) -> tuple[float, list[str]]:
    """
    Returns the ratio of disallowed (foreign-language) chars to total chars,
    along with the list of matched blocks.
    """
    if not text:
        return 0.0, []
    
    matches = DISALLOWED_PATTERN.findall(text)
    if not matches:
        return 0.0, []
    
    total_foreign_chars = sum(len(m) for m in matches)
    ratio = total_foreign_chars / len(text)
    return ratio, matches


def get_row_text(row: dict) -> str:
    """Concatenates all string-valued columns in a row into a single string for ratio checking."""
    parts = []
    for value in row.values():
        if isinstance(value, str):
            parts.append(value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts)


def main():
    datasets_dir = "Datasets"
    if not os.path.exists(datasets_dir):
        print(f"Error: Directory '{datasets_dir}' not found.")
        return

    # 1. List all parquet datasets, ignoring text files
    dataset_files = sorted([f for f in os.listdir(datasets_dir) if f.endswith(".parquet")])
    print(f"Found {len(dataset_files)} dataset(s) to validate.\n")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("VALIDATION & CLEANING REPORT")
    report_lines.append(f"Threshold: non-English chars > {NON_ENGLISH_THRESHOLD*100:.1f}% of row → DROP")
    report_lines.append("=" * 70)

    for file_name in dataset_files:
        file_path = os.path.join(datasets_dir, file_name)
        print(f"Processing {file_name}...")

        try:
            table = pq.read_table(file_path)
            records = table.to_pylist()
            total = len(records)

            kept_rows = []
            dropped_rows = []  # (row_index, ratio, samples)

            for idx, row in enumerate(records):
                full_text = get_row_text(row)
                ratio, matches = get_non_english_ratio(full_text)

                if ratio > NON_ENGLISH_THRESHOLD:
                    # Collect a few sample characters for the report
                    samples = list(set(matches))[:5]
                    dropped_rows.append((idx, ratio, samples))
                else:
                    kept_rows.append(row)

            # Overwrite the file with the cleaned dataset
            cleaned_table = pa.Table.from_pylist(kept_rows, schema=table.schema)
            pq.write_table(cleaned_table, file_path)

            dropped_count = len(dropped_rows)
            status = "[OK] CLEAN" if dropped_count == 0 else f"[!] DROPPED {dropped_count} rows"
            print(f"  {status}  ({total - dropped_count}/{total} rows kept)")

            report_lines.append(f"\nFile: {file_name}")
            report_lines.append(f"  Total rows:   {total}")
            report_lines.append(f"  Rows kept:    {total - dropped_count}")
            report_lines.append(f"  Rows dropped: {dropped_count}")

            if dropped_rows:
                report_lines.append("  Samples of dropped rows:")
                for row_idx, ratio, samples in dropped_rows[:10]:
                    sample_str = ", ".join(samples)
                    report_lines.append(f"    Row {row_idx}: {ratio*100:.2f}% foreign  →  [{sample_str}]")
                if len(dropped_rows) > 10:
                    report_lines.append(f"    ... and {len(dropped_rows) - 10} more dropped rows")

        except Exception as e:
            print(f"  [ERROR] {e}")
            report_lines.append(f"\nFile: {file_name}")
            report_lines.append(f"  [ERROR] {e}")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("Done.")

    report_path = "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nAll done! Full report saved to {report_path}")


if __name__ == "__main__":
    main()
