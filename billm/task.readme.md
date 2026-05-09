# BiLLM Project Task Overview

## 1. billm.py (The Engine)
*   **Purpose**: A high-efficiency binarization library for Large Language Models (specifically Qwen 3.5).
*   **Working**:
    *   **Calibration**: Collects Hessian-diagonal data to identify "salient" (important) weights.
    *   **Quantization**: Keeps salient weights in FP16/BF16 (2-6%) and binarizes everything else to ±$\alpha$.
    *   **Fallback**: Automatically detects VRAM and falls back to CPU on Out-Of-Memory (OOM) errors.
*   **Goal**: Compress models into binary weights to save memory while maintaining mathematical accuracy (low perplexity).
*   **Output**: 
    *   Quantized model weights saved to disk.
    *   Perplexity score (mathematical sanity check).
    *   `--test` flag: Runs 22 unit tests for core math and device stability.

## 2. billm_eval.py (The Vibe Check)
*   **Purpose**: A functional smoke-testing script to evaluate model language capabilities.
*   **Working**:
    *   Loads a local model (quantized or unquantized).
    *   Runs four distinct prompt categories through the `generate()` loop.
*   **Goal**: Verify if the model can still communicate coherently after quantization.
*   **Output**:
    *   **Basic Greeting**: Checks for polite, natural interaction.
    *   **Reasoning**: Tests multi-step math/logic solving.
    *   **Complex**: Explains difficult concepts (e.g., Quantum Entanglement) with analogies.
    *   **Jailbreak**: Tests safety alignment and roleplay (DAN prompt).

---

### How to Run
| Task | Command |
| :--- | :--- |
| **Quantize Model** | `python billm.py` |
| **Run Core Tests** | `python billm.py --test` |
| **Test AI Brain** | `python billm_eval.py [model_path]` |
