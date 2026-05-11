**Loaded 66 prompts — 16 stress | 50 unit tests**



**Stress prompts:**

&#x20; **\[01] WARP MATRIX MULTIPLY-ACCUMULATE (WMMA / Tensor Cores)**

&#x20; **\[02] COOPERATIVE GROUPS -- GRID-LEVEL SYNC**

&#x20; **\[03] PTX INLINE ASSEMBLY (LDMATRIX + ASYNC COPY)**

&#x20; **\[04] WARP-LEVEL PRIMITIVES: \_\_shfl\_sync with non-trivial masks**

&#x20; **\[05] DYNAMIC PARALLELISM (CHILD KERNEL LAUNCH FROM GPU)**

&#x20; **\[06] PERSISTENT THREADS + CLOCK-BASED SPIN WAIT**

&#x20; **\[07] TEXTURE OBJECTS WITH LAYERED 2D ARRAYS**

&#x20; **\[08] NANOSLEEP + MEMORY ORDERING WITH ACQUIRE/RELEASE FENCES**

&#x20; **\[09] MULTI-CAST MEMORY + NVLink PEER ACCESS (MULTI-GPU)**

&#x20; **\[10] CUTLASS-Style Template GEMM (Double-Buffered Software Pipeline)**

&#x20; **\[11] Multi-File Project: cp.async 3-Stage Software Pipeline GEMM**

&#x20; **\[12] Warp Specialization + Named PTX Barriers (Hopper/GH200-Style)**

&#x20; **\[13] CUB BlockScan + Thrust Transform-Reduce with Custom Functors**

&#x20; **\[14] NCCL Ring All-Reduce with Stream/Event Ordering (maps to RCCL)**

&#x20; **\[15] Structured Sparse GEMM (cuSPARSELt 2:4 Sparsity, maps to rocSPARSE)**

&#x20; **\[16] EXTREME: Full Flash-Attention SDPA (4-File Project: vec\_math.h + attention.h + attention.cu + main.cu) \[max\_tokens=4096]**



**Unit test categories: {'Sync', 'Memory', 'Math', 'Atomics', 'Warp'}**



**################################################################################**

**# BASE MODEL**

**################################################################################**



**================================================================================**

&#x20;**STRESS: BASE MODEL | Thinking DISABLED | Batch=15 | MaxTok=2048**

**================================================================================**

&#x20; **Stress BASE MODEL: 100%|████████████████████████| 30720/30720 tok  \[03:49<00:00  133.77tok/s]**

**Total Time:         229.65s**

**Tokens Generated:   34433 (Batch=15)**

**Speed:              149.94 tok/s**

**Base VRAM:          58.83 GB**

**KV Overhead:        22.20 GB**

**Peak VRAM:          81.03 GB**



&#x20; **✓ 149.9 TPS | 81.0 GB VRAM**



**================================================================================**

&#x20;**STRESS: BASE MODEL | Thinking ENABLED | Batch=15 | MaxTok=2048**

**================================================================================**

&#x20; **Stress BASE MODEL: 100%|████████████████████████| 30720/30720 tok  \[04:39<00:00  110.08tok/s]**

**Total Time:         279.06s**

**Tokens Generated:   43058 (Batch=15)**

**Speed:              154.30 tok/s**

**Base VRAM:          58.83 GB**

**KV Overhead:        22.22 GB**

**Peak VRAM:          81.05 GB**



&#x20; **✓ 154.3 TPS | 81.1 GB VRAM**



**================================================================================**

&#x20;**STRESS: BASE MODEL | Thinking DISABLED | Batch=1 | MaxTok=4096**

**================================================================================**

&#x20; **Stress BASE MODEL: 100%|███████████████████████████| 4096/4096 tok  \[06:04<00:00  11.24tok/s]**

**Total Time:         364.46s**

**Tokens Generated:   3380 (Batch=1)**

**Speed:              9.27 tok/s**

**Base VRAM:          58.83 GB**

**KV Overhead:        2.60 GB**

**Peak VRAM:          61.43 GB**



&#x20; **✓ 9.3 TPS | 61.4 GB VRAM**



**================================================================================**

&#x20;**STRESS: BASE MODEL | Thinking ENABLED | Batch=1 | MaxTok=4096**

**================================================================================**

&#x20; **Stress BASE MODEL: 100%|███████████████████████████| 4096/4096 tok  \[06:54<00:00   9.88tok/s]**

**Total Time:         414.74s**

**Tokens Generated:   4066 (Batch=1)**

**Speed:              9.80 tok/s**

**Base VRAM:          58.83 GB**

**KV Overhead:        2.60 GB**

**Peak VRAM:          61.43 GB**



&#x20; **✓ 9.8 TPS | 61.4 GB VRAM**



**================================================================================**

&#x20;**UNIT TESTS: BASE MODEL | 50 snippets**

**================================================================================**

&#x20; **Unit BASE MODEL batch 1/5: 100%|██████████████████| 5120/5120 tok  \[00:10<00:00  482.31tok/s]**

&#x20; **Batch 1/5 | 118.7 TPS**

&#x20; **Unit BASE MODEL batch 2/5: 100%|██████████████████| 5120/5120 tok  \[00:09<00:00  515.19tok/s]**

&#x20; **Batch 2/5 | 122.2 TPS**

&#x20; **Unit BASE MODEL batch 3/5: 100%|██████████████████| 5120/5120 tok  \[00:06<00:00  757.41tok/s]**

&#x20; **Batch 3/5 | 119.8 TPS**

&#x20; **Unit BASE MODEL batch 4/5: 100%|██████████████████| 5120/5120 tok  \[00:10<00:00  483.47tok/s]**

&#x20; **Batch 4/5 | 138.6 TPS**

&#x20; **Unit BASE MODEL batch 5/5: 100%|██████████████████| 5120/5120 tok  \[00:11<00:00  459.71tok/s]**

&#x20; **Batch 5/5 | 139.0 TPS**



**──────────────────────────────────────────────────────────────────────────────**

&#x20;**UNIT TEST SCORE — BASE MODEL**

**──────────────────────────────────────────────────────────────────────────────**

&#x20; **Total   : 50**

&#x20; **✅ PASS  :  44  (88.0%)**

&#x20; **⚠️  WARN  :   4  (8.0%)**

&#x20; **❌ FAIL  :   2  (4.0%)**

&#x20; **Score   : 92.0 / 100**



&#x20; **Category        PASS  WARN  FAIL   Score**

&#x20; **────────────────────────────────────**

&#x20; **Memory            10     0     0  100.0%**

&#x20; **Math               8     0     2   80.0%**

&#x20; **Atomics           10     0     0  100.0%**

&#x20; **Warp               6     4     0   80.0%**

&#x20; **Sync              10     0     0  100.0%**



&#x20; **❌ Failed:**

&#x20;    **- MATH-01: \_\_fmaf\_rn fused multiply-add**

&#x20;    **- MATH-04: \_\_powf**





**################################################################################**

**# LORA MODEL**

**################################################################################**



**================================================================================**

&#x20;**STRESS: LORA MODEL | Thinking DISABLED | Batch=15 | MaxTok=2048**

**================================================================================**

&#x20; **Stress LORA MODEL: 100%|████████████████████████| 30720/30720 tok  \[04:07<00:00  124.14tok/s]**

**Total Time:         247.47s**

**Tokens Generated:   34133 (Batch=15)**

**Speed:              137.93 tok/s**

**Base VRAM:          58.83 GB**

**KV Overhead:        22.20 GB**

**Peak VRAM:          81.03 GB**



&#x20; **✓ 137.9 TPS | 81.0 GB VRAM**



**================================================================================**

&#x20;**STRESS: LORA MODEL | Thinking ENABLED | Batch=15 | MaxTok=2048**

**================================================================================**

&#x20; **Stress LORA MODEL: 100%|█████████████████████████| 30720/30720 tok  \[05:55<00:00  86.31tok/s]**

**Total Time:         355.93s**

**Tokens Generated:   43718 (Batch=15)**

**Speed:              122.83 tok/s**

**Base VRAM:          58.83 GB**

**KV Overhead:        22.22 GB**

**Peak VRAM:          81.05 GB**



&#x20; **✓ 122.8 TPS | 81.1 GB VRAM**



**================================================================================**

&#x20;**STRESS: LORA MODEL | Thinking DISABLED | Batch=1 | MaxTok=4096**

**================================================================================**

&#x20; **Stress LORA MODEL: 100%|███████████████████████████| 4096/4096 tok  \[09:20<00:00   7.30tok/s]**

**Total Time:         560.77s**

**Tokens Generated:   4089 (Batch=1)**

**Speed:              7.29 tok/s**

**Base VRAM:          58.83 GB**

**KV Overhead:        2.60 GB**

**Peak VRAM:          61.43 GB**



&#x20; **✓ 7.3 TPS | 61.4 GB VRAM**



**================================================================================**

&#x20;**STRESS: LORA MODEL | Thinking ENABLED | Batch=1 | MaxTok=4096**

**================================================================================**

&#x20; **Stress LORA MODEL: 100%|███████████████████████████| 4096/4096 tok  \[09:18<00:00   7.33tok/s]**

**Total Time:         558.54s**

**Tokens Generated:   4096 (Batch=1)**

**Speed:              7.33 tok/s**

**Base VRAM:          58.83 GB**

**KV Overhead:        2.60 GB**

**Peak VRAM:          61.43 GB**



&#x20; **✓ 7.3 TPS | 61.4 GB VRAM**



**================================================================================**

&#x20;**UNIT TESTS: LORA MODEL | 50 snippets**

**================================================================================**

&#x20; **Unit LORA MODEL batch 1/5: 100%|██████████████████| 5120/5120 tok  \[00:12<00:00  410.42tok/s]**

&#x20; **Batch 1/5 | 101.0 TPS**

&#x20; **Unit LORA MODEL batch 2/5: 100%|██████████████████| 5120/5120 tok  \[00:13<00:00  373.45tok/s]**

&#x20; **Batch 2/5 | 88.5 TPS**

&#x20; **Unit LORA MODEL batch 3/5: 100%|██████████████████| 5120/5120 tok  \[00:10<00:00  491.32tok/s]**

&#x20; **Batch 3/5 | 85.4 TPS**

&#x20; **Unit LORA MODEL batch 4/5: 100%|██████████████████| 5120/5120 tok  \[00:15<00:00  325.38tok/s]**

&#x20; **Batch 4/5 | 98.4 TPS**

&#x20; **Unit LORA MODEL batch 5/5: 100%|██████████████████| 5120/5120 tok  \[00:13<00:00  381.74tok/s]**

&#x20; **Batch 5/5 | 98.3 TPS**



**──────────────────────────────────────────────────────────────────────────────**

&#x20;**UNIT TEST SCORE — LORA MODEL**

**──────────────────────────────────────────────────────────────────────────────**

&#x20; **Total   : 50**

&#x20; **✅ PASS  :  44  (88.0%)**

&#x20; **⚠️  WARN  :   6  (12.0%)**

&#x20; **❌ FAIL  :   0  (0.0%)**

&#x20; **Score   : 94.0 / 100**



&#x20; **Category        PASS  WARN  FAIL   Score**

&#x20; **────────────────────────────────────**

&#x20; **Memory            10     0     0  100.0%**

&#x20; **Math              10     0     0  100.0%**

&#x20; **Atomics           10     0     0  100.0%**

&#x20; **Warp               4     6     0   70.0%**

&#x20; **Sync              10     0     0  100.0%**





**================================================================================**

&#x20;**FINAL UNIT TEST SCORES**

**================================================================================**

&#x20; **BASE MODEL      ██████████████████   92.0/100**

&#x20; **LORA MODEL      ██████████████████   94.0/100**

**================================================================================**



**✓ Complete. Results saved to: benchmark\_output.txt**



