

### 25.04.2026 5:32AM (Ankan)

##\#ISSUE 001
Improper Generation of Block Tokens through prompt injection in Gemme 4 2B Model

>[!NOTE]
>**Sample Responses**<br>
> **Sample 1** – Response: `ok`, Parser: `broken`
> **Sample 2** – Response: `broken`, Parser: `broken`
> **Sample 3** – Response: `ok`, Parser: `ok`

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
