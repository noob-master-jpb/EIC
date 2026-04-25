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

### Report

1. Dataset.md - Compiled a list of the best code generation datasets. (by Arya)
   - Code Generation (Instruction to Code)
   - Code Debugging (Conversational "Thinking" & Code Fixing)
   - Code Conversion (CUDA → HIP)
2. Filtered Magicoder OSS-Instruct-75k dataset to *oss-ins-75k.parquet* (by Ankan)
    - *Format* (columns[problems, solutions,])
    - *Problems:* User query inputs
    - *Solutions:* Model responses
3. Filtered And Combined NVIDIA ComputeEval to *nvidia_compute_eval.parquet* (by Ayush)
    - *Format* (columns[task_id,  prompt, tags, min_cuda_toolkit, build_command, test_command, context_files,test_files, baseline_solution])
    - *Prompts:* Needs to build from the given dataset
    - *Solutions:* baseline_solution
    
