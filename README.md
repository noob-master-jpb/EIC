# ROCm-Bridge (Internal)

## What we’re building

LLM-based CUDA → HIP transpiler.
Goal is not just conversion, but **performance-aware translation** for AMD GPUs.

---

## Core idea

Most tools do syntax swap.
We’re training a model to understand:

* Warp (CUDA) vs Wavefront (AMD)
* Memory layout differences
* Intrinsics + kernel-level optimizations

End goal: code that actually runs well on ROCm, not just compiles.

---

## Plan

**Phase 1 — Prototype (Gemma 4 2B)**

* Fast iteration
* Build pipeline + dataset
* Basic CUDA → HIP mapping

**Phase 2 — Brain (Gemma 4 26B MoE)**

* Real reasoning
* Multi-file understanding
* Use MI300X full VRAM

**Phase 3 — Future (BitNet 1.58)**

* Try distilling 26B → ultra-light model
* If it works: huge efficiency win

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
* Gemma Data Pipeline Docs - data pipeline documentation 
https://kauldron.readthedocs.io/en/latest/data_py.html

---