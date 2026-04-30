
# Batch Dataset Generator

## Short Explanation

This folder turns a small list of prompts into one JSON dataset file.

- What it does: sends your prompts to the local `qwen3.5:0.8b` model and saves the answers in `polishing_dataset.json`.
- How it works: the script starts Ollama if needed, loads prompts from `prompts.json`, calls the model for each prompt, cleans the reply, and writes structured records.
- Why it does it: the output is easier to reuse later for polishing, reviewing, filtering, or feeding into another pipeline.

## File Guide

| File | What it does |
| --- | --- |
| `batch_dataset_ollama.py` | Main entrypoint. Starts Ollama if needed, registers the model if missing, sends the batch prompts, cleans the replies, and writes `polishing_dataset.json`. |
| `prompts.json` | Input prompt list. Edit this file when you want different batch prompts. |
| `polishing_dataset.json` | Final output dataset. This is the reusable JSON file you can use later for polishing or other processing. |
| `Modelfile` | Ollama model definition used to register the local Qwen model from the copied files in this folder. |
| `qwen3.5-0.8b-local/` | Local model snapshot files that Ollama uses to build/register `qwen3.5:0.8b` without downloading again. |
| `.env` | Optional local settings file. It can hold environment values like `OLLAMA_HOST` or `OLLAMA_MODEL` if you want to override defaults. |
| `README.md` | This documentation file. |

## Main Script

This folder has one main script:

```bash
batch_dataset_ollama.py
```

It does the full workflow:

1. Starts the bundled Ollama server if no Ollama server is already running.
2. Registers the local `qwen3.5:0.8b` model if it is missing.
3. Reads batch prompts from `prompts.json`.
4. Sends the prompts to the model.
5. Saves one reusable dataset JSON file.

## Run

From this folder:

```bash
cd "/home/aryarakshit/Documents/AMD Hackathon/Arya-Files/Batch_test"
python batch_dataset_ollama.py
```

Output:

```bash
polishing_dataset.json
```

## Prompt Input

Edit this file to change the batch prompts:

```bash
prompts.json
```

It should be a JSON array:

```json
[
  "Prompt one",
  "Prompt two",
  "Prompt three"
]
```

## Dataset Output Format

The output JSON has this structure:

```json
{
  "dataset_name": "ollama_batch_polishing_dataset",
  "model": "qwen3.5:0.8b",
  "record_count": 5,
  "created_at": "2026-04-30T00:00:00+00:00",
  "records": [
    {
      "id": "batch-0001",
      "instruction": "Original prompt",
      "input": "",
      "output": "Model response",
      "raw_response": "Uncleaned model response",
      "metadata": {
        "index": 1,
        "model": "qwen3.5:0.8b",
        "source": "ollama_batch_dataset",
        "ok": true,
        "seconds": 1.23,
        "created_at": "2026-04-30T00:00:00+00:00"
      }
    }
  ]
}
```

Use `output` as the polished dataset answer. Keep `raw_response` for debugging.

## Options

Run only three prompts:

```bash
python batch_dataset_ollama.py --batch-size 3
```

Change response length:

```bash
python batch_dataset_ollama.py --max-tokens 80
```

Write to another JSON file:

```bash
python batch_dataset_ollama.py --output my_dataset.json
```

## Support Files

- `prompts.json` - input prompts.
- `Modelfile` - local Ollama model registration file.
- `qwen3.5-0.8b-local/` - local model files used by Ollama.

---

> I have installed and used the model in LOCAL MACHINE.