"""Batch prompt Ollama with parallel requests."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple


MODEL = "qwen3.5:0.8b"
HOST = "http://127.0.0.1:11434"
PROMPTS_PATH = "Batch_test/prompts.json"
OUTPUT_PATH = "Batch_test/polishing_dataset.json"
MAX_WORKERS = 0
TIMEOUT = 120.0
RETRIES = 2
BACKOFF = 0.5
TEMPERATURE: Optional[float] = None
TOP_P: Optional[float] = None
TOP_K: Optional[int] = None
NUM_PREDICT: Optional[int] = None
STOP_TOKENS: Optional[List[str]] = None
DRY_RUN = False


def _parse_stop_tokens(value: Optional[object]) -> Optional[List[str]]:
	if not value:
		return None
	if isinstance(value, list):
		return [str(item) for item in value if str(item)]
	if isinstance(value, str):
		text = value.strip()
		if not text:
			return None
		if text.startswith("["):
			try:
				parsed = json.loads(text)
			except json.JSONDecodeError:
				parsed = None
			if isinstance(parsed, list):
				return [str(item) for item in parsed if str(item)]
		return [part for part in (p.strip() for p in text.split(",")) if part]
	return [str(value)]


def _load_prompts(path: str) -> List[Any]:
	with open(path, "r", encoding="utf-8") as handle:
		data = json.load(handle)
	if isinstance(data, list):
		return data
	if isinstance(data, dict):
		if "prompts" in data and isinstance(data["prompts"], list):
			return data["prompts"]
		return [data]
	return [data]


def _normalize_messages(item: Any) -> Tuple[List[Dict[str, str]], Any]:
	if isinstance(item, list) and all(isinstance(m, dict) for m in item):
		return item, item
	if isinstance(item, str):
		return [{"role": "user", "content": item}], item
	if isinstance(item, dict):
		if "messages" in item and isinstance(item["messages"], list):
			return item["messages"], item
		system = item.get("system")
		user = item.get("prompt") or item.get("input") or item.get("text")
		if user is None:
			raise ValueError("Prompt item missing 'prompt', 'input', or 'text'.")
		messages: List[Dict[str, str]] = []
		if system:
			messages.append({"role": "system", "content": str(system)})
		messages.append({"role": "user", "content": str(user)})
		return messages, item
	raise ValueError("Unsupported prompt item type.")


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
	body = json.dumps(payload).encode("utf-8")
	request = urllib.request.Request(
		url,
		data=body,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	with urllib.request.urlopen(request, timeout=timeout) as response:
		raw = response.read().decode("utf-8")
	return json.loads(raw)


def _call_ollama_chat(
	host: str,
	payload: Dict[str, Any],
	timeout: float,
	retries: int,
	backoff: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
	url = host.rstrip("/") + "/api/chat"
	last_error: Optional[str] = None
	for attempt in range(1, retries + 1):
		try:
			return _post_json(url, payload, timeout), None
		except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
			last_error = f"{type(exc).__name__}: {exc}"
		except Exception as exc:  # noqa: BLE001
			last_error = f"{type(exc).__name__}: {exc}"
		if attempt < retries:
			time.sleep(backoff * attempt)
	return None, last_error


def _build_options(stop_tokens: Optional[List[str]]) -> Dict[str, Any]:
	options: Dict[str, Any] = {}
	if TEMPERATURE is not None:
		options["temperature"] = TEMPERATURE
	if TOP_P is not None:
		options["top_p"] = TOP_P
	if TOP_K is not None:
		options["top_k"] = TOP_K
	if NUM_PREDICT is not None:
		options["num_predict"] = NUM_PREDICT
	if stop_tokens:
		options["stop"] = stop_tokens
	return options


def _build_record(
	index: int,
	item: Any,
	messages: List[Dict[str, str]],
	response: Optional[Dict[str, Any]],
	error: Optional[str],
	model: str,
) -> Dict[str, Any]:
	record_id = index
	if isinstance(item, dict) and "id" in item:
		record_id = item["id"]

	record: Dict[str, Any] = {
		"id": record_id,
		"model": model,
		"messages": messages,
	}
	if response is not None:
		message = response.get("message")
		if isinstance(message, dict):
			record["output"] = message.get("content", "")
		else:
			record["output"] = response.get("response", "")
		record["metadata"] = {
			key: response.get(key)
			for key in (
				"created_at",
				"prompt_eval_count",
				"prompt_eval_duration",
				"eval_count",
				"eval_duration",
				"total_duration",
				"load_duration",
			)
			if key in response
		}
	if error:
		record["error"] = error
	return record


def _generate_one(
	index: int,
	item: Any,
	host: str,
	model: str,
	options: Dict[str, Any],
	timeout: float,
	retries: int,
	backoff: float,
) -> Dict[str, Any]:
	messages, _ = _normalize_messages(item)
	payload: Dict[str, Any] = {
		"model": model,
		"messages": messages,
		"stream": False,
	}
	if options:
		payload["options"] = options
	response, error = _call_ollama_chat(host, payload, timeout, retries, backoff)
	return _build_record(index, item, messages, response, error, model)


def _print_diagnostics(
	count: int,
	max_workers: int,
	stop_tokens: Optional[List[str]],
) -> None:
	print("Diagnostics:", flush=True)
	print(f"  MODEL: {MODEL}", flush=True)
	print(f"  HOST: {HOST}", flush=True)
	print(f"  PROMPTS_PATH: {PROMPTS_PATH}", flush=True)
	print(f"  OUTPUT_PATH: {OUTPUT_PATH}", flush=True)
	print(f"  MAX_WORKERS: {max_workers}", flush=True)
	if stop_tokens:
		print(f"  STOP_TOKENS: {stop_tokens}", flush=True)
	print(f"  PROMPT_COUNT: {count}", flush=True)


def main() -> int:
	stop_tokens = _parse_stop_tokens(STOP_TOKENS)

	max_workers = MAX_WORKERS
	if max_workers <= 0:
		cpu_count = os.cpu_count() or 4
		max_workers = min(32, cpu_count * 5)

	prompts = _load_prompts(PROMPTS_PATH)
	_print_diagnostics(len(prompts), max_workers, stop_tokens)

	if DRY_RUN:
		print("Dry run complete. No requests sent.", flush=True)
		return 0

	options = _build_options(stop_tokens)
	results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

	print("Generating JSON dataset with parallel workers...", flush=True)
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = {
			executor.submit(
				_generate_one,
				index,
				item,
				HOST,
				MODEL,
				options,
				TIMEOUT,
				RETRIES,
				BACKOFF,
			): index
			for index, item in enumerate(prompts)
		}
		for future in as_completed(futures):
			index = futures[future]
			try:
				results[index] = future.result()
			except Exception as exc:  # noqa: BLE001
				results[index] = {
					"id": index,
					"model": MODEL,
					"messages": [],
					"error": f"{type(exc).__name__}: {exc}",
				}

	output_records = [record for record in results if record is not None]
	with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
		json.dump(output_records, handle, ensure_ascii=True, indent=2)

	print(f"Generated {len(output_records)} records.", flush=True)
	print(f"JSON dataset saved to: {OUTPUT_PATH}", flush=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
