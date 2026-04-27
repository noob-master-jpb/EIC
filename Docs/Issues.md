# Issues

---

## ISSUE 001 · *25 Apr 2026, 5:32 AM* · @Ankan

**Improper Generation of Block Tokens via Prompt Injection — Gemma 4 2B Model**

---

### Sample Responses

| # | Response | Parser |
|---|----------|--------|
| Sample 1 | ✅ ok | ❌ broken |
| Sample 2 | ❌ broken | ❌ broken |
| Sample 3 | ✅ ok | ✅ ok |

---

### Observations & Debugging

> [!NOTE]
> Possible causes:
> - Low parameter count and quantization
> - Need to test on larger models (e.g., 26B MoE or 31B)

---

### Action Items

> [!IMPORTANT]
> The following tasks have been assigned and must be completed after the next test run.

1. **Dataset Guardrails** — Add input validation/guardrails to the dataset. *(Assigned to: Ayush)*
2. **Post-Processing** — Implement regex or token validation after model output. *(Assigned to: Arya)*
3. **Prompt Engineering** — Refine prompts used during hosting. *(Note: "k korbi pore dekhchi ota")*

---

### Follow up
> [!NOTE]
> - Issue persists on 26B MoE, ruling out parameter count as the sole cause


## ISSUE 002: Training Efficiency & Convergence Optimization
**Date:** 27 Apr 2026, 7:46 PM  
**Author:** @Ankan

### 📊 Performance Benchmarks (75k Samples)
| Model | Size | Epochs | Training Time | Observations |
| :--- | :--- | :--- | :--- | :--- |
| Gemma 4 | 2B | 1.0 | ~1 hour | Converges (Loss < 1.0) at **~0.2 Epochs**. |
| Gemma 4 | 31B | 1.0 | ~13 hours | Starts with Loss < 1.0; potentially over-parameterized for task. |

---

### 🔍 Technical Interpretation
The current training pipeline is hitting convergence milestones much faster than expected. 
- **Over-Convergence:** The 2B model achieves target loss within 20% of the first epoch, suggesting that further training may lead to overfitting or "memorization" rather than generalization.
- **Initialization & Rank:** The 31B model's low starting loss indicates that the current LoRA rank (`r=16/32`) may be providing too much capacity for this specific dataset, or the base model is already highly aligned with the target distribution.

---

### 💡 Optimization Recommendations
> [!TIP]
> **For Gemma 4 2B:**
> - Cap training at **0.2 - 0.3 epochs** to prevent overfitting.
> - Alternatively, reduce LoRA complexity to **Rank 4 / Alpha 8** for better regularization.
>
> **For Gemma 4 31B:**
> - To significantly reduce the **13-hour training window**, implement a "Micro-LoRA" approach.
> - Reduce **Rank to 2-4** and **Alpha to 4-8**. This will lower memory overhead and focus the model on high-level feature adjustment.
