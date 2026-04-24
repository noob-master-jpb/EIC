# ROCm-Bridge (Internal)

## What we’re building

LLM-based CUDA → HIP transpiler.
Goal is not just conversion, but **performance-aware translation** for AMD GPUs.

---

## Plan

**Phase 1 — Prototype (Gemma 4 2B)**

* Fast iteration
* Build pipeline + dataset
* Basic finetuning(for tech stack testing)

**Phase 2 — Brain (Gemma 4 26B MoE)**

* Real reasoning
* Basic CUDA → HIP mapping
* Multi-file understanding
* Use MI300X full VRAM
* Distil Training (Parent Model -> Not Selected)
  
**Phase 3 — Future (BitNet 1.58)**

* Conversion to Bitnet 1.58 for finetuned model

---

## Dataset (Parallel Corpus)

1. CUDA kernels (.cu from real libs)
2. NL → GPU code pairs
3. Broken HIP → Fixed HIP (debug learning)
4. CUDA intrinsics → ROCm mappings

---

## Stack

* MI300X (192GB)
* ROCm 7+
* PyTorch + Unsloth
* Gemma 4 (2B / 26B), BitNet (goal)

---

## References

* Gemma 4 Model Card - https://ai.google.dev/gemma/docs/core/model_card_4
* Gemma Fine-Tunning Docs - https://gemma-llm.readthedocs.io/en/latest/colab_finetuning.html
* Gemma Data Pipeline Docs - https://kauldron.readthedocs.io/en/latest/data_py.html

---
