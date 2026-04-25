# Project Notes

### 25.04.2026 5:49AM (Ankan)

## Sample Responses

- **Sample 1** – Response: `ok`, Parser: `broken`
- **Sample 2** – Response: `broken`, Parser: `broken`
- **Sample 3** – Response: `ok`, Parser: `ok`

## Observations & Debugging

```markdown
- Possible causes:
  - Low parameter count and quantization
  - Need to test on larger models (e.g., 26B MoE or 31B)
```

## Action Items

1. **Dataset Guardrails** – Add validation checks (assigned to *Ayush*).
2. **Post‑Processing Improvements** – Implement regex or token validation (assigned to *Arya*).
3. **Prompt Engineering** – Refine prompts for hosting (note: "k korbi pore dekhchi ota").

## Test Script

Added `test.py` to run the model.

## Test Script Checklist

```markdown
1. Ensure the model is downloaded to the specified location (assign to AI).
2. Close all background processes, including IDEs; run via terminal.
3. Use Python **3.12**.
4. Install required packages with `pip3` (automate with AI).
```

