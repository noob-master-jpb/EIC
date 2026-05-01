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
 * Proposed Solution: Fine-Tuning Gemma 4 for CUDA to ROCm Conversion, Given in `Solutions.md`. [Status:Done]

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

## 29 Apr 2026, 01:13 AM · @Arya

### Model Serve via vLLM.      [Status:Done]
 *   **Model Used:** `Qwen 3.5 (0.8B)`.
 *   **File/Code-base Location:** `vLLM/qwen-vLLM`.

---

## 30 Apr 2026, 01:50 AM · @Ayush

### Dataset Diversity Validation & Scoring
1. **Mathematically Optimized Scorer**: Developed `diversity_scorer.py` using a high-performance linear algebra trick ($O(N)$ complexity) to calculate average pairwise cosine distance without requiring a memory-heavy $O(N^2)$ matrix.
2. **Success Metrics (Subsampling Proof)**:
   - Successfully cross-verified the **11,723 row diverse subset** against the **64,036 row original dataset**.
   - **Result**: The selected subset (K-Means selection performed by **@Ankan**) achieved a **2.69% increase in Total Variance** while being **82% smaller**. This proves the K-Means selection effectively pruned redundant data while increasing the overall semantic spread.
   - **Stability Fix**: Resolved `float16` precision overflow issues in the scoring script by implementing strategic `float32` upscaling for squared-magnitude calculations.

### Extended Dataset Analytics
1. **NVIDIA Compute Evaluation**: Profiled `nvidia_compute_eval.parquet`, identifying **1.65 Million tokens** (6.6M characters) of high-quality compute logic for the training pool.
2. **Environmental Cleanup**: Suppressed `transformers` load warnings in the diagnostic tools for cleaner terminal output and faster automated reporting.

---

## 30 Apr 2026, 09:08 PM · @Arya

## Distillation Part: Batch Dataset Generator.  [Status:Done]

- **Simplified Script:** Refactored to a variable-driven config (no terminal args needed).
- **Clean Output:** Formatted JSON to strictly use `id`, `input`, `output`, and `raw_response`.
- **Git Security:** Updated `.gitignore` to exclude large model files and local logs.
- **Optimized:** Enabled concurrent processing and added input validation.
- **Ready to Run:** Updated README with a simple one-command workflow.
- **Working Folder:** `Batch_test`.
---

## 30 Apr 2026, 11:45 PM · @Ayush

## LLM Binarization Research: BiLLM vs PB-LLM

- **Comparative Analysis:** Conducted a deep-dive comparison of BiLLM and PB-LLM implementations based on technical papers.
- **Saliency Strategies:** Contrasted BiLLM's **Binary Residual Approximation** (salient weights as two 1-bit components) against PB-LLM's **Partial Binarization** (salient weights as INT8).
- **Metric Evaluation:** Analyzed selection metrics—**Hessian-based structural selection** (BiLLM) vs. **Magnitude-based element-wise selection** (PB-LLM).
- **Performance Benchmarking:** Identified BiLLM as the "Fast and Accurate" choice for ultra-low bit-widths (~1.1 bit), outperforming PB-LLM (~1.7 bit) in both perplexity and inference potential.
- **Strategic Advantages:** Documented PB-LLM's strengths in **Quantization-Aware Training (QAT)** and native hardware support for its INT8 components.
- **Docs Updated:** Summarized trade-offs between PTQ efficiency and surgical precision.

## 1 May 2026, 4:15 AM · @Ankan

### Training Pipeline Optimization & Validation `[Status: Done]`

The training pipeline has been optimized for the **Gemma 4 31B** model on the `cass` dataset, including full validation runs across all model sizes.

#### Loss Metrics

| Metric | 31B-Dense | 26B-MoE | 4B | 2B |
| :--- | :---: | :---: | :---: | :---: |
| **Training Loss** | ~1.7 | ~3.3 | ~15 | ~17 |
| **Validation Loss** | ~1.7 | ~3.3 | ~0.9 | ~0.9 |

> [!NOTE]
> The unusually low validation loss on the 4B and 2B models is suspected to be caused by a difference in loss calculation methodology between dense/MoE and smaller models.

#### 1-Epoch Training Time

**Config:** Batch size 45 · Dataset 60k rows · 1k tokens/row (input + output) · Hardware: 1× MI300X GPU · Refer to `cass` dataset.

| Model | 31B-Dense | 26B-MoE | 4B | 2B |
| :--- | :---: | :---: | :---: | :---: |
| **Est. Time** | > 20h | > 8h | > 2h | < 2h |

---

### Distillation Pipeline Validation `[Status: Ongoing]`

- **Phase 1** — GLM-5 distillation for 500 rows of `nvidia_compute_eval` is at **60%** completion. Total estimated time: **10–12 hours** for 500 datapoints via DigitalOcean Serverless API (bottlenecked by rate limits and latency).
- **Phase 2** — Conversion of the retrieved 500 datapoints to ROCm/HIP. Estimated time: **~10 hours**.
- **Final Phase (Revised)** — Changed from GLM-5 distillation to **Qwen 3 72B** on **12,000 datapoints** running directly on the MI300X GPU.

---

### Model Binarization Research `[Status: Upcoming]`

**Selected Method: PB-LLM**

1. **PTQ Efficiency** — PB-LLM's magnitude-based weight selection enables a more straightforward and computationally efficient Post-Training Quantization (PTQ) process, which is crucial given our resource constraints.
2. **QAT Compatibility** — PB-LLM's use of INT8 for salient weights is more compatible with Quantization-Aware Training (QAT) techniques, enabling better fine-tuning and optimization.
3. **Ease of Implementation** — PB-LLM's partial binarization approach is generally simpler to implement and integrate into existing training pipelines compared to BiLLM's more complex binary residual approximation.
---

## 1 May 2026, 11:48 PM · @Ayush

### Distillation Dataset Preparation
1. **CUDA-to-ROCm Prompt Engineering**: Developed `create_distill_dataset.py` to transform the `nvidia_compute_eval_glm5.jsonl` dataset into a specialized distillation format.
2. **Dataset Transformation**:
   - Prepended a task-specific instruction: *"Convert this CUDA kernal to ROCm/HIP kernal"* to all 565 entries.
   - Initialized a blank `Response` column to facilitate structured output generation during the distillation process.
3. **Storage Strategy**: Exported the processed dataset to **`cuda_to_rocm_distill.parquet`** for high-performance loading and compatibility with the Unsloth/Hugging Face ecosystem.

### 📊 Project Status Overview (Current)
*   **Accomplished**:
    *   **Environment**: Full automation of MI300X/ROCm 7.2 droplets via `setup_venv.sh`.
    *   **Data**: 1B+ character corpus cleaned; 11.7k diverse subset selected and verified.
    *   **Training**: Pipeline verified for Gemma 4 (Dense/MoE); `Qwen 3.5` and `Gemma 4` successfully served.
    *   **Research**: PB-LLM selected as the primary binarization strategy.
*   **Remaining**:
    *   **Distillation**: Finish GLM-5 Phase 1 (40% left); execute ROCm/HIP conversion (Phase 2).
    *   **Binarization**: Implementation of PB-LLM (PTQ/QAT phases).
    *   **Training**: Final 1-epoch fine-tune of **Gemma 4 31B** (Est. >20 hours).


---

## 2 May 2026, 12:35 AM · @Arya 

### PB-LLM Quantizer.  `[Status: Done]`
The model ran successfully on my local GPU, but some changes may be needed in the "Variables" section of the Python script, along with a few tweaks.
 * Folder: `PB-LLM`
 * Model: `google/gemma-4-E2B-it`
 * `Math.md` and `README.md` contains proper instructions.

