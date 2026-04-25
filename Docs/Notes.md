# Project Notes

---

## 25 Apr 2026, 5:30 AM · @Ankan

### Test Script

Added `test.py` to run the model locally for inference testing.

### Checklist

1. **Download the model** — Ensure it is saved to the correct local directory *(automate with AI)*
2. **Clear background processes** — Close all running apps and IDEs; run via terminal only
3. **Python version** — Use **Python 3.12**
4. **Install dependencies** — Use `pip3` to install required packages *(automate with AI)*


## 26 Apr 2026, 1:53 AM · @Ankan

### Dataset Building

| # | Dataset | File | Contributor |
|---|---------|------|-------------|
| 1 | Best code generation datasets (curated list) | `Dataset.md` | @Arya |
| 2 | Magicoder OSS-Instruct-75k (filtered) | `oss-ins-75k.parquet` | @Ankan |
| 3 | NVIDIA ComputeEval (filtered & combined) | `nvidia_compute_eval.parquet` | @Ayush |

#### Details

**1. `Dataset.md` — Curated Code Generation Datasets**
- **Code Generation** — Instruction to Code
- **Code Debugging** — Conversational "Thinking" & Code Fixing
- **Code Conversion** — CUDA → HIP

**2. `oss-ins-75k.parquet` — Magicoder OSS-Instruct-75k**
- **Format:** `columns[problems, solutions]`
- **Problems:** User query inputs
- **Solutions:** Model responses

**3. `nvidia_compute_eval.parquet` — NVIDIA ComputeEval**
- **Format:** `columns[task_id, prompt, tags, min_cuda_toolkit, build_command, test_command, context_files, test_files, baseline_solution]`
- **Prompts:** To be constructed from the dataset fields
- **Solutions:** `baseline_solution`

> [!NOTE]
> The datasets have **not yet been checked for compatibility** with Gemma-4 (Training Model) or GLM-5 (Teacher/Distillation Model).