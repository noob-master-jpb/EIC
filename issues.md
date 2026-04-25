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


