# PB-LLM Quantizer and Ollama Manager (Cloud & Local)

A standalone Python pipeline designed to demonstrate the **Partially Binarized LLM (PB-LLM) 1-Bit** algorithm alongside **BitsAndBytes 4-Bit QLoRA**, interacting seamlessly with local or cloud-based deployments. It supports converting Hugging Face models (like Qwen, Llama, and Gemma) into extremely memory-efficient formats and allows for Quantization-Aware Training (QAT) to recover degraded language abilities.

Everything you need to configure and run the application is contained within a single file: `pb_llm_pipeline.py`, which is fully configurable via an easy-to-use variable dashboard at the top of the script.

## Features

1. **Dual-Mode Quantization Pipeline:** 
   - **`pb-llm-1bit`:** Pure PyTorch mathematical simulation of 1-bit quantization. Identifies salient weights (kept in high precision) and binarizes the non-salient weights (-1 or 1). Because it uses pure PyTorch, this method natively supports **AMD ROCm** servers without compiling custom C++ kernels.
   - **`bnb-4bit`:** Uses Hugging Face's `bitsandbytes` library with QLoRA to physically compress model memory to 4-bit. Highly optimized for NVIDIA GPUs (CUDA) running on consumer 8GB-12GB VRAM cards.
2. **Quantization-Aware Training (QAT) & QLoRA:**
   - Automatically fine-tunes the compressed model using QAT or QLoRA to mathematically heal the degradation caused by extreme quantization.
3. **Cloud & Multi-GPU Scaling:**
   - Support for `device_map="auto"`, allowing massive models to automatically span across 8x H100 or MI300X cloud servers.
4. **Ollama Manager:**
   - Automatically download missing models.
   - Run existing local models.
   - Create custom models dynamically from local GGUF manual paths.

## Requirements

If you plan to use the quantization pipeline, install the required ecosystem libraries:

```bash
pip install torch transformers accelerate bitsandbytes peft
```

If you plan to use the **Ollama Manager**, ensure [Ollama](https://ollama.com) is installed and available in your system `PATH`.

## How to Use

Everything is controlled by simply editing the variables at the top of the `pb_llm_pipeline.py` script. **No terminal arguments are required.**

Open `pb_llm_pipeline.py` and modify the Configuration Dashboard:

```python
# ==========================================
# CLOUD & GLOBAL CONFIGURATION
# ==========================================

# --- Task Toggles ---
RUN_QUANTIZATION_PIPELINE = True  # Run the Python quantization/training pipeline
RUN_OLLAMA = False                # Manage models via Ollama

# --- Quantization Algorithm ---
# "pb-llm-1bit" : Pure PyTorch 1-bit quantization (Best for AMD ROCm servers & Cloud)
# "bnb-4bit"    : BitsAndBytes 4-bit + QLoRA (Requires NVIDIA/CUDA, best for 8GB Local GPUs)
QUANTIZATION_METHOD = "pb-llm-1bit"

# --- Hardware Scaling ---
# "auto" : Automatically spans across multiple GPUs on high-end cloud servers
# {"": 0}: Forces the model onto GPU 0 (Useful for strict single-GPU local testing)
DEVICE_MAP = "auto" 

# --- Model & Pipeline Settings ---
INPUT_MODEL_ID = "google/gemma-4-E2B-it"    

PERFORM_QAT = True                 # Fine-tune the model to fix quantization degradation
QAT_EPOCHS = 10                    # Number of training loops
SALIENT_RATIO = 0.50               # For PB-LLM: Protect 50% of weights in high precision
```

### Running the Script

Once you have set your desired variables (depending on if you are on an 8GB local laptop or a massive AMD/NVIDIA cloud server), simply run the script from your terminal:

```bash
python pb_llm_pipeline.py
```
