from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bit import MODEL_ID

REPO_DIR = Path(__file__).resolve().parent
MODEL_DIR = REPO_DIR / "models" / f"{MODEL_ID.split('/')[-1]}-ternary-blockwise"

PREFER_CUDA = True
MAX_NEW_TOKENS = 64
DO_SAMPLE = False

SMOKE_PROMPTS = [
    "Hello! Briefly explain what quantization does in neural networks.",
]


def runtime_device() -> str:
    if PREFER_CUDA and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    if not MODEL_DIR.exists():
        print(f"Error: model path does not exist: {MODEL_DIR}")
        return

    device = runtime_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model from: {MODEL_DIR}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    for index, prompt in enumerate(SMOKE_PROMPTS, start=1):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n=== Smoke Prompt {index} ===")
        print(prompt)
        print("--- Response ---")
        print(text)


if __name__ == "__main__":
    main()
