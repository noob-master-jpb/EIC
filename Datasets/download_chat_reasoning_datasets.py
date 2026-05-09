import os
import random
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset


REPO_DIR = Path(__file__).resolve().parent
DATASETS_DIR = REPO_DIR / "Datasets"
OUTPUT_PATH = DATASETS_DIR / "chat_reasoning_qat_mix.parquet"
SEED = 3407

DATASET_LIMITS = {
    "open_thoughts": 50_000,
    "smoltalk": 30_000,
    "numina_math": 10_000,
    "openhermes": 10_000,
}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def first_text_turn(conversations: list[dict], user_roles: set[str]) -> str | None:
    for turn in conversations or []:
        if turn.get("from") in user_roles or turn.get("role") in user_roles:
            value = turn.get("value", turn.get("content", ""))
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def last_text_turn(conversations: list[dict], assistant_roles: set[str]) -> str | None:
    for turn in reversed(conversations or []):
        if turn.get("from") in assistant_roles or turn.get("role") in assistant_roles:
            value = turn.get("value", turn.get("content", ""))
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def collect_rows(name: str, rows: Iterable[dict], limit: int, source_prefix: str) -> pd.DataFrame:
    records = []
    for idx, example in enumerate(islice(rows, limit), start=1):
        record = normalize_example(name, example, source_prefix)
        if record is not None:
            records.append(record)
        if idx % 1_000 == 0:
            print(f"{source_prefix}: scanned {idx:,}/{limit:,}, kept {len(records):,}", flush=True)

    df = pd.DataFrame.from_records(records, columns=["problem", "solution", "source"])
    print(f"{source_prefix}: kept {len(df):,}/{limit:,} rows")
    return df


def normalize_example(name: str, example: dict, source_prefix: str) -> dict | None:
    if name in {"open_thoughts", "openhermes"}:
        user = first_text_turn(example.get("conversations"), {"user", "human"})
        assistant = last_text_turn(example.get("conversations"), {"assistant", "gpt"})
    elif name == "smoltalk":
        messages = example.get("messages") or []
        user = first_text_turn(messages, {"user"})
        assistant = last_text_turn(messages, {"assistant"})
    elif name == "numina_math":
        user = example.get("problem")
        assistant = example.get("solution")
    else:
        raise ValueError(f"Unsupported dataset key: {name}")

    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    user = user.strip()
    assistant = assistant.strip()
    if not user or not assistant:
        return None

    source = example.get("source") or source_prefix
    return {
        "problem": user,
        "solution": assistant,
        "source": f"{source_prefix}:{source}",
    }


def save_part(df: pd.DataFrame, filename: str) -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASETS_DIR / filename
    df.to_parquet(path, index=False)
    print(f"saved {len(df):,} rows -> {path}")
    return path


def main() -> None:
    random.seed(SEED)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    limits = {
        key: env_int(f"{key.upper()}_ROWS", default)
        for key, default in DATASET_LIMITS.items()
    }

    parts: list[pd.DataFrame] = []

    open_thoughts = load_dataset(
        "open-thoughts/OpenThoughts2-1M",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    df = collect_rows("open_thoughts", open_thoughts, limits["open_thoughts"], "open_thoughts")
    save_part(df, "open_thoughts_reasoning.parquet")
    parts.append(df)

    smoltalk = load_dataset(
        "HuggingFaceTB/smoltalk",
        "all",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    df = collect_rows("smoltalk", smoltalk, limits["smoltalk"], "smoltalk")
    save_part(df, "smoltalk_chat.parquet")
    parts.append(df)

    numina_math = load_dataset(
        "AI-MO/NuminaMath-1.5",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    df = collect_rows("numina_math", numina_math, limits["numina_math"], "numina_math")
    save_part(df, "numina_math_reasoning.parquet")
    parts.append(df)

    openhermes = load_dataset(
        "teknium/OpenHermes-2.5",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    df = collect_rows("openhermes", openhermes, limits["openhermes"], "openhermes")
    save_part(df, "openhermes_chat.parquet")
    parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"saved mixed QAT dataset: {len(combined):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
