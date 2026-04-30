#!/usr/bin/env python3
"""One-command Ollama batch prompt dataset generator."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --- CONFIGURATION ---
# Script directory setup
SCRIPT_DIR = Path(__file__).resolve().parent

# Model and Server Settings
MODEL = "qwen3.5:0.8b"
HOST = "http://127.0.0.1:11434"
BATCH_SIZE = 5
MAX_TOKENS = 120
TEMPERATURE = 0.2
CONCURRENCY = 5
TIMEOUT = 240

# File Paths
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"
OUTPUT_PATH = SCRIPT_DIR / "polishing_dataset.json"
MODELFILE_PATH = SCRIPT_DIR / "Modelfile"

# Ollama Backend Paths (Relative to script dir)
OLLAMA_BIN = (SCRIPT_DIR / "../amd_hckn/backend/ollama").resolve()
OLLAMA_LIB = (SCRIPT_DIR / "../amd_hckn/backend/lib/ollama").resolve()
OLLAMA_LOG = SCRIPT_DIR / "ollama_server.log"
# ---------------------

def load_dotenv() -> None:
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_prompts() -> list[str]:
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Prompts file not found: {PROMPTS_PATH}")
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"{PROMPTS_PATH} must contain a JSON array of prompt strings.")
    if not data:
        raise ValueError(f"{PROMPTS_PATH} does not contain any prompts.")
    return data


def clean_response(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"\s*<think>.*$", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("</think>", "")
    cleaned = re.sub(r"<\|[^>]+?\|>", "", cleaned)
    cleaned = re.split(r"\n\s*Final answer\s*:?", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = re.split(r"\n\s*Answer\s*:?", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = cleaned.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    parts = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    if parts and all(part == parts[0] for part in parts):
        return parts[0]
    if len(parts) % 2 == 0 and parts[: len(parts) // 2] == parts[len(parts) // 2 :]:
        return "\n\n".join(parts[: len(parts) // 2])

    return cleaned.strip()


def ollama_is_running() -> bool:
    try:
        with urllib.request.urlopen(f"{HOST.rstrip('/')}/api/tags", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def list_models() -> list[str]:
    with urllib.request.urlopen(f"{HOST.rstrip('/')}/api/tags", timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    return [item["name"] for item in body.get("models", []) if "name" in item]


def start_ollama_if_needed() -> subprocess.Popen[str] | None:
    if ollama_is_running():
        print(f"Using existing Ollama server at {HOST}")
        return None

    env = os.environ.copy()
    lib_path = str(OLLAMA_LIB)
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else lib_path
    env.setdefault("OLLAMA_NUM_PARALLEL", str(CONCURRENCY))

    print(f"Starting Ollama server at {HOST}")
    log_file = OLLAMA_LOG.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(OLLAMA_BIN), "serve"],
        cwd=SCRIPT_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for _ in range(30):
        if ollama_is_running():
            return process
        if process.poll() is not None:
            break
        time.sleep(1)

    log_file.close()
    raise RuntimeError(f"Ollama did not start. Check log: {OLLAMA_LOG}")


def register_model_if_needed() -> None:
    models = list_models()
    if MODEL in models:
        print(f"Model already available: {MODEL}")
        return

    env = os.environ.copy()
    lib_path = str(OLLAMA_LIB)
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else lib_path

    print(f"Registering model: {MODEL}")
    subprocess.run(
        [str(OLLAMA_BIN), "create", MODEL, "-f", str(MODELFILE_PATH)],
        cwd=SCRIPT_DIR,
        env=env,
        check=True,
    )


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            "stop": ["<|endoftext|>", "<|im_start|>", "\nFinal answer:", "\nAnswer:"],
        },
    }
    request = urllib.request.Request(
        f"{HOST.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {HOST}") from exc

    if "response" not in body:
        raise RuntimeError(f"Ollama response did not include text: {body}")
    return str(body["response"]).strip()


async def generate_one(
    index: int,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        raw_response = ""
        try:
            raw_response = await asyncio.to_thread(call_ollama, prompt=prompt)
            output = clean_response(raw_response)
        except Exception:
            output = "ERROR: Failed to generate response"

    return {
        "id": f"batch-{index:04d}",
        "input": prompt,
        "output": output,
        "raw_response": raw_response,
    }


async def generate_dataset() -> list[dict[str, Any]]:
    prompts = read_prompts()[: BATCH_SIZE]
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        generate_one(index=index, prompt=prompt, semaphore=semaphore)
        for index, prompt in enumerate(prompts, start=1)
    ]
    return await asyncio.gather(*tasks)


def main() -> int:
    load_dotenv()
    try:
        # Simple validation
        if not OLLAMA_BIN.exists():
            raise FileNotFoundError(f"Ollama binary not found: {OLLAMA_BIN}")
        if not MODELFILE_PATH.exists():
            raise FileNotFoundError(f"Modelfile not found: {MODELFILE_PATH}")

        ollama_process = start_ollama_if_needed()
        register_model_if_needed()
        print("Generating reusable JSON dataset...")
        records = asyncio.run(generate_dataset())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if "ollama_process" in locals() and ollama_process is not None:
            ollama_process.terminate()
            try:
                ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_process.kill()

    # Save as a clean list of records with only input and output
    OUTPUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nGenerated {len(records)} records.")
    print(f"JSON dataset saved to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


