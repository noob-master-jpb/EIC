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

---

## 26 Apr 2026, 11:24 PM · @Ayush

### Dataset Optimization & Standardization

1.  **Standardized `nvidia_compute_eval`**:
    *   Merged the complex 9-parameter schema into a unified 2-column format (`problem`, `solution`) compatible with the training pipeline.
    *   Prepended expert GPU C++ system instructions to all problem statements.
    *   Generated `nvidia_compute_eval_new.parquet` and a matching human-readable `.txt` file.
2.  **Streamlined Data Pipeline**:
    *   Refactored `data_reshape.py` as the primary engine for conversion (switched to PyArrow for significant speed gains).
    *   Updated `data_loader.py` to automate downloading and Parquet-exporting the CodeFeedback dataset.
    *   Created `load.py` utility for rapid dataset previewing.
3.  **Language Analysis**:
    *   Performed frequency analysis on CodeFeedback dataset (150+ languages; Python/JS leading).
4.  **Workspace Cleanup**:
    *   Standardized all outputs to the `Datasets/` directory and updated `.gitignore`.

---

## 27 Apr 2026, 7:56 PM · @Ankan

### 🚀 Pipeline Validation: `ins-oss-75k`
Successfully completed end-to-end stress testing and validation of the primary training pipeline.

| Attribute | Details |
| :--- | :--- |
| **Model** | Gemma-4 2B |
| **Hardware** | 1x AMD Instinct™ MI300X Accelerator |
| **Dataset** | `ins-oss-75k.parquet` |
| **Status** | ✅ Verified Stable |

**Observations:**
*   Confirmed high-throughput performance within the **ROCm** stack.
*   Loss convergence behavior is consistent with synthetic benchmark expectations.

---

### 🛠️ Infrastructure Tasking
> [!IMPORTANT]
> **Environment Automation Script** — Assigned to **@Ayush**
>
> **Objective:** Develop a robust shell script to handle Python environment initialization during GPU droplet instantiation.
> - **Scope:** ROCm drivers check, dependency injection, and `venv` isolation.
> - **Goal:** Reduce setup time for new cloud nodes to < 2 minutes.

---
