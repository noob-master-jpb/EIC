import os
import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH    = "./Datasets/nvidia_compute_eval.parquet"
OUTPUT_PATH     = "./Datasets/nvidia_compute_eval_glm5.jsonl"
MODEL           = "glm-5"
MAX_TOKENS      = 4096
ENABLE_THINKING = False
RETRY_DELAY     = 5    # seconds to wait before retrying a failed request
START_ROW       = 1   # skip rows with index < this (already collected)
BATCH_SIZE      = 2   # number of concurrent requests per batch
BATCH_WAIT_MS   = 1000  # milliseconds to wait between batches
# ─────────────────────────────────────────────────────────────────────────────

URL = "https://inference.do-ai.run/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('DO_TOKEN')}"
}



def query_glm5_until_success(idx: int, problem: str) -> dict:
    """
    Query GLM-5 and retry indefinitely until a valid (non-error) response
    is received. Handles both HTTP errors and API-level errors embedded
    in a 200 OK response body (e.g. looping content flags, missing choices).
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
        "stream_options": {"include_usage": True},
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
            resp.raise_for_status()          # catches 4xx / 5xx HTTP errors
            data = resp.json()

            # ── Detect API-level errors inside a 200 OK body ──────────────
            # 1. Top-level "error" key (e.g. looping content flag)
            if "error" in data:
                err_msg = data["error"].get("message", str(data["error"]))
                tqdm.write(f"[idx={idx} attempt={attempt}] API error: {err_msg}  → retrying")
                time.sleep(RETRY_DELAY)
                continue

            # 2. No choices returned
            choices = data.get("choices") or []
            if not choices:
                tqdm.write(f"[idx={idx} attempt={attempt}] Empty choices  → retrying")
                time.sleep(RETRY_DELAY)
                continue

            choice  = choices[0]
            finish  = choice.get("finish_reason", "")
            content = (choice.get("message") or {}).get("content", "") or ""

            # 3. Bad finish reason or empty content
            if finish not in ("stop", "length") or not content.strip():
                tqdm.write(f"[idx={idx} attempt={attempt}] Bad finish='{finish}' or empty content  → retrying")
                time.sleep(RETRY_DELAY)
                continue

            # ── Valid response ────────────────────────────────────────────
            usage = data.get("usage", {})
            return {"index": idx, "problem": problem, "response": content, "usage": usage}

        except Exception as exc:
            tqdm.write(f"[idx={idx} attempt={attempt}] Exception: {exc}  → retrying")

        # Failed — wait and retry
        time.sleep(RETRY_DELAY)


def main():
    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded {len(df)} rows from {DATASET_PATH}")

    # Load already-saved indices so we never re-query them
    done_indices = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_indices.add(rec["index"])
                except json.JSONDecodeError:
                    pass
        print(f"Resuming — {len(done_indices)} rows already saved.")

    # Filter: skip rows below START_ROW and already-done indices
    pending = [
        (int(idx), row["problem"])
        for idx, row in df.iterrows()
        if int(idx) >= START_ROW and int(idx) not in done_indices
    ]
    print(f"Rows to query: {len(pending)}  (START_ROW={START_ROW}, BATCH_SIZE={BATCH_SIZE})")

    batch_queue = []   # holds results for the current batch only

    with tqdm(total=len(pending), desc="Querying GLM-5") as pbar:
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start : batch_start + BATCH_SIZE]

            # ── Fire all requests in this batch concurrently ──────────────
            # Each worker retries itself until it gets a valid response.
            # The batch is NOT considered done until every item succeeds.
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                futures = [
                    executor.submit(query_glm5_until_success, idx, prob)
                    for idx, prob in batch
                ]
                for f in futures:
                    batch_queue.append(f.result())   # blocks until this item is done

            # ── Batch complete: sort queue by index, write, then clear ────
            batch_queue.sort(key=lambda r: r["index"])
            with open(OUTPUT_PATH, "a", encoding="utf-8") as out_file:
                for record in batch_queue:
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            batch_no = batch_start // BATCH_SIZE + 1
            tqdm.write(f"  ✓ Batch {batch_no}: {len(batch_queue)} rows saved (idx {batch_queue[0]['index']}–{batch_queue[-1]['index']})")
            batch_queue.clear()   # reset queue for next batch
            pbar.update(len(batch))

            # Wait between batches to avoid hammering the API
            if BATCH_WAIT_MS > 0 and batch_start + BATCH_SIZE < len(pending):
                time.sleep(BATCH_WAIT_MS / 1000)

    # Final count across the whole file
    total = 0
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
                total += 1
            except json.JSONDecodeError:
                pass
    print(f"\nDone! Total rows in file: {total}")


if __name__ == "__main__":
    main()
