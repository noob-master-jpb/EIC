# 1. Code Generation (Instruction to Code)

* **Magicoder OSS-Instruct**
  * **Link:** https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
  * **Size:** ~75,000 examples (filter for C++/CUDA)
  * **Fit:** Features conversational instructions where an AI agent generates code based on user prompts. Good baseline for natural language to code generation.

* **NVIDIA ComputeEval**
  * **Link:** https://github.com/NVIDIA/compute-eval
  * **Size:** ~100 curated tasks
  * **Fit:** High-quality natural language prompts mapping directly to optimized CUDA kernels.

# 2. Code Debugging (Conversational "Thinking" & Code Fixing)

* **CodeFeedback-Filtered-Instruction**
  * **Link:** https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction
  * **Size:** ~136,000 multi-turn conversations
  * **Fit:** A perfect match for your friend's requirement. This dataset contains multi-turn user-agent chats where a user provides broken code, execution logs, or compiler errors, and the AI agent explicitly reasons through the problem ("thinking") to provide the fixed code. 

* **OpenHermes 2.5 (Coding Subset)**
  * **Link:** https://huggingface.co/datasets/teknium/OpenHermes-2.5
  * **Size:** ~1 million multi-turn chats (filter for coding/C++)
  * **Fit:** Contains highly conversational user-agent interactions. The AI acts as a coding assistant, utilizing step-by-step reasoning (Chain-of-Thought) to analyze user code, explain vulnerabilities, and debug.

* **HPC Repository Mined Commits (vLLM, Triton, etc.)**
  * **Link:** https://github.com/vllm-project/vllm
  * **Size:** Dynamically mined (varies based on pull requests)
  * **Fit:** Real-world performance bugs. You take these real fixes and format them into the conversational template your friend wants (e.g., User: *"Why is this HIP kernel failing?"* -> AI: *"[Thinking: Wavefront size is 64 on AMD, not 32...] Here is the optimized fix."*).

# 3. Code Conversion (CUDA → HIP)

* **CASS (CUDA-AMD ASsembly and Source Mapping)**
  * **Link:** https://huggingface.co/datasets/ahmedheakl/cass-source-grpo
  * **Size:** ~70,000 verified code pairs
  * **Fit:** The core dataset for cross-architecture translation. It provides the exact performance-aware conversions needed so the AI knows what a successful translation looks like.

* **ROCm HIPIFY API Mappings**
  * **Link:** https://github.com/ROCm/HIPIFY/tree/amd-staging/doc/markdown
  * **Size:** Thousands of exact API mappings
  * **Fit:** The ground-truth dictionary for API swaps (e.g., cuBLAS to rocBLAS), ensuring the AI does not hallucinate library calls during its reasoning steps.