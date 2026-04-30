#!/usr/bin/env python3
"""One-command Ollama batch prompt dataset generator."""

from __future__ import annotations

import argparse
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


DEFAULT_MODEL = "qwen3.5:0.8b"
DEFAULT_BATCH_SIZE = 5
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MAX_TOKENS = 120


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_prompts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"{path} must contain a JSON array of prompt strings.")
    if not data:
        raise ValueError(f"{path} does not contain any prompts.")
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


def ollama_is_running(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def list_models(host: str) -> list[str]:
    with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    return [item["name"] for item in body.get("models", []) if "name" in item]


def start_ollama_if_needed(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if ollama_is_running(args.host):
        print(f"Using existing Ollama server at {args.host}")
        return None

    env = os.environ.copy()
    lib_path = str(args.ollama_lib)
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else lib_path
    env.setdefault("OLLAMA_NUM_PARALLEL", str(args.concurrency))

    print(f"Starting Ollama server at {args.host}")
    log_file = args.ollama_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(args.ollama_bin), "serve"],
        cwd=args.script_dir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for _ in range(30):
        if ollama_is_running(args.host):
            return process
        if process.poll() is not None:
            break
        time.sleep(1)

    log_file.close()
    raise RuntimeError(f"Ollama did not start. Check log: {args.ollama_log}")


def register_model_if_needed(args: argparse.Namespace) -> None:
    models = list_models(args.host)
    if args.model in models:
        print(f"Model already available: {args.model}")
        return

    env = os.environ.copy()
    lib_path = str(args.ollama_lib)
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else lib_path

    print(f"Registering model: {args.model}")
    subprocess.run(
        [str(args.ollama_bin), "create", args.model, "-f", str(args.modelfile)],
        cwd=args.script_dir,
        env=env,
        check=True,
    )


def call_ollama(
    *,
    host: str,
    model: str,
    prompt: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop": ["<|endoftext|>", "<|im_start|>", "\nFinal answer:", "\nAnswer:"],
        },
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {host}") from exc

    if "response" not in body:
        raise RuntimeError(f"Ollama response did not include text: {body}")
    return str(body["response"]).strip()


async def generate_one(
    *,
    index: int,
    prompt: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    started = time.perf_counter()
    async with semaphore:
        try:
            raw_response = await asyncio.to_thread(
                call_ollama,
                host=args.host,
                model=args.model,
                prompt=prompt,
                timeout=args.timeout,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            ok = True
            error = None
            output = clean_response(raw_response)
        except Exception as exc:  # noqa: BLE001 - keep failed records inspectable.
            ok = False
            error = str(exc)
            raw_response = ""
            output = ""

    metadata: dict[str, Any] = {
        "index": index,
        "model": args.model,
        "source": "ollama_batch_dataset",
        "ok": ok,
        "seconds": round(time.perf_counter() - started, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        metadata["error"] = error

    return {
        "id": f"batch-{index:04d}",
        "instruction": prompt,
        "input": "",
        "output": output,
        "raw_response": raw_response,
        "metadata": metadata,
    }


async def generate_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompts = read_prompts(args.prompts)[: args.batch_size]
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        generate_one(index=index, prompt=prompt, args=args, semaphore=semaphore)
        for index, prompt in enumerate(prompts, start=1)
    ]
    return await asyncio.gather(*tasks)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env")

    parser = argparse.ArgumentParser(
        description="Start Ollama if needed, run batch prompts, and write one JSON dataset."
    )
    parser.add_argument("--prompts", type=Path, default=script_dir / "prompts.json")
    parser.add_argument("--output", type=Path, default=script_dir / "polishing_dataset.json")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", DEFAULT_HOST))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--ollama-bin", type=Path, default=script_dir / "../amd_hckn/backend/ollama")
    parser.add_argument("--ollama-lib", type=Path, default=script_dir / "../amd_hckn/backend/lib/ollama")
    parser.add_argument("--modelfile", type=Path, default=script_dir / "Modelfile")
    parser.add_argument("--ollama-log", type=Path, default=Path("/tmp/batch_test_ollama.log"))

    args = parser.parse_args()
    args.script_dir = script_dir
    args.ollama_bin = args.ollama_bin.resolve()
    args.ollama_lib = args.ollama_lib.resolve()
    args.modelfile = args.modelfile.resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if not args.ollama_bin.exists():
        raise FileNotFoundError(f"Ollama binary not found: {args.ollama_bin}")
    if not args.ollama_lib.exists():
        raise FileNotFoundError(f"Ollama library path not found: {args.ollama_lib}")
    if not args.modelfile.exists():
        raise FileNotFoundError(f"Modelfile not found: {args.modelfile}")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        ollama_process = start_ollama_if_needed(args)
        register_model_if_needed(args)
        print("Generating reusable JSON dataset...")
        records = asyncio.run(generate_dataset(args))
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

    dataset = {
        "dataset_name": "ollama_batch_polishing_dataset",
        "model": args.model,
        "record_count": len(records),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    ok_count = sum(1 for record in records if record["metadata"]["ok"])
    for record in records:
        status = "ok" if record["metadata"]["ok"] else "failed"
        print(f"[{record['id']}] {status}: {record['instruction']}")

    print(f"\nGenerated {ok_count}/{len(records)} records.")
    print(f"JSON dataset saved to: {args.output}")
    return 0 if ok_count == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
