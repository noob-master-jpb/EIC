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
> 1.  Installs `uv` and initializes a fresh Python virtual environment.
> 2. Configures `unsloth[amd]` using `uv` for immediate GPU-accelerated training.
> 3.  Performs a headless installation of PyTorch (ROCm-enabled), `pandas`, and `datasets`.

---

## 28 Apr 2026, 01:33 AM · @Ayush

### Dataset Cleaning & Chunking

1.  **Advanced Validation & Row Pruning**:
    *   Refactored `data-validation.py` with a precise Unicode allowlist (Math, Greek, Emojis, Box-drawing).
    *   Implemented a **0.5% foreign language threshold**; rows exceeding this are now automatically dropped.
    *   Permanently cleaned all datasets, removing foreign natural language while preserving technical symbols.
2.  **GitHub Compatibility (Splitting)**:
    *   Splitted `codefeedback_filtered.parquet` (163MB) into 4 smaller chunks using `data_reshaper-1.py`.
    *   Splitted `ahmedheakl_cass_source_grpo.parquet` (149MB) into 2 chunks using `data_reshaper-3.py`.
    *   Updated `.gitignore` to prevent tracking of original massive files.
3.  **OpenHermes Standardisation**:
    *   Created `data_reshape-2.py` to map OpenHermes `conversations` (human/gpt) into the unified 2-column `problem`/`solution` schema.
4.  **Training Coordination**:
    *   Collaborated with **@Ankan** on the implementation of `train.py` using Unsloth's `FastLanguageModel`.

### Environment Automation & Droplet Setup

1. **Automated Setup Script (`setup_venv.sh`)**:
   * Developed a bash script to automate the full environment initialization on DigitalOcean GPU droplets.
   * Integrated `uv` for lightning-fast package installation and virtual environment management.
2. **ROCm 7.2 Optimization**:
   * Diagnosed and fixed ROCm version mismatches on the MI300X platform.
   * Standardized the pipeline to use **ROCm 7.2** specific PyTorch wheels, ensuring full GPU hardware acceleration.

---

## 28 Apr 2026, 03:30 AM · @Arya

### Dataset research and findings.      [Status:Done]

 * Founded the datasets present in file `Dataset.md` 

### Model Serve via vLLM.      [Status:Failed]

 *   **Model Used:** `unsloth/gemma-4-E2B-it-GGUF`.
 *   **Served via vLLM:** No.
 *   **Problem with vLLM:** The `gemma4` architecture is too new; `vLLM` (and `transformers`) currently lacks support for parsing this specific architecture from GGUF files.
 *   **Current Solution:** Served via a local **Ollama** instance (llama.cpp), which has specialized support for the `gemma4` architecture and its reasoning/thinking tokens.
    
### 28 Apr 2026, 03:50 PM · @Arya 
 > Proposed Solution: Fine-Tuning Gemma 4 for CUDA to ROCm Conversion, Given in `Solutions.md`. [Status:Done]

---

## 29 Apr 2026, 12:02 AM · @Ayush

### Infrastructure & Environment Perfection
1. **Persistence & Portability**: Updated `setup_venv.sh` to permanently configure `PATH` and `ROCM_PATH` in `~/.bashrc`. This ensures that `bitsandbytes` and `unsloth` always detect the ROCm hardware correctly upon login.
2. **Setup Automation**: Verified the full end-to-end automation of the DigitalOcean droplet initialization, resolving the `rocminfo` and Flash Attention 2 detection warnings.

### Dataset Analytics & Tokenizer Planning
1. **Large-Scale Data Counting**: Developed `data_counter.py` and `data_counter-1.py` to analyze the entire cleaned corpus.
2. **Corpus Statistics**:
   - **Combined Output**: Analyzed 301,129 rows totaling **1.001 Billion characters** (~250M estimated tokens).
   - **Remaining Parts**: Analyzed 64,602 rows (cass & nvidia parts) totaling **470 Million characters** (~117M estimated tokens).
3. **Tokenizer Insights**: Extracted critical metrics including character vocabulary size (1,949 unique chars) and max sequence lengths (76k chars) to guide the `gemma4` fine-tuning configuration.

---



