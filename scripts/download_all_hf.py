"""Качает множество HF датасетов, не более 10 ГБ суммарно."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ.get("HF_TOKEN")

OUT_DIR = Path("data/clean_dialogs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOTAL_BYTES = 10_000_000_000  # 10 GB
MAX_PER_FILE_BYTES = 2_000_000_000  # 2 GB

print(f"TOKEN: {TOKEN[:10] if TOKEN else 'None'}")
print(f"MAX TOTAL: {MAX_TOTAL_BYTES/1e9:.1f} GB\n")


def get_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6 if path.exists() else 0


def total_size_mb() -> float:
    return sum(get_size_mb(p) for p in OUT_DIR.glob("*.jsonl"))


def write_pair(f, item, role_keys: dict, source: str) -> int:
    """Универсальная запись пары prompt/response. role_keys: {user: [...], assistant: [...]}"""
    written = 0
    convs = None
    if source == "conversations":
        convs = item.get("conversations", [])
    elif source == "conversation":
        convs = item.get("conversation", [])
    elif source == "messages":
        convs = item.get("messages", [])

    if not convs:
        return 0

    for j in range(0, len(convs) - 1, 2):
        if j + 1 >= len(convs):
            break
        m1 = convs[j]
        m2 = convs[j + 1]
        if not isinstance(m1, dict) or not isinstance(m2, dict):
            continue

        # Определяем роль
        role1 = m1.get("from", m1.get("role", ""))
        role2 = m2.get("from", m2.get("role", ""))

        is_user1 = role1 in role_keys.get("user", ["user", "human", "prompter"])
        is_asst1 = role1 in role_keys.get("assistant", ["assistant", "gpt"])
        is_user2 = role2 in role_keys.get("user", ["user", "human", "prompter"])
        is_asst2 = role2 in role_keys.get("assistant", ["assistant", "gpt"])

        # Ищем user → assistant пару
        prompt = ""
        response = ""
        if is_user1 and is_asst2:
            prompt = m1.get("value", m1.get("content", ""))
            response = m2.get("value", m2.get("content", ""))
        elif is_asst1 and is_user2:
            prompt = m2.get("value", m2.get("content", ""))
            response = m1.get("value", m1.get("content", ""))

        if 5 < len(prompt) < 1000 and 5 < len(response) < 1000:
            pair = {"prompt": prompt, "response": response}
            line = json.dumps(pair, ensure_ascii=False) + "\n"
            f.write(line)
            written += len(line)
            break
    return written


def fetch_dataset(name: str, hf_id: str, source: str, role_keys: dict, out_name: str, max_bytes: int):
    """Скачивает один датасет."""
    if total_size_mb() * 1e6 >= MAX_TOTAL_BYTES:
        print(f"  SKIP {name}: total >= 10GB")
        return

    out_path = OUT_DIR / out_name
    if out_path.exists() and out_path.stat().st_size > 100_000:
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"  ✓ {out_name} уже есть ({n} пар, {get_size_mb(out_path):.1f} МБ)")
        return

    print(f"  → {out_name} ← {hf_id}")
    try:
        ds = load_dataset(hf_id, split="train", streaming=True, token=TOKEN)
    except Exception as e:
        print(f"  ⚠ {name}: {str(e)[:120]}")
        return

    written = 0
    pairs = 0
    start_total = total_size_mb() * 1e6
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if total_size_mb() * 1e6 >= MAX_TOTAL_BYTES:
                print(f"  STOP {name}: total >= 10GB")
                break
            w = write_pair(f, ex, role_keys, source)
            written += w
            if w > 0:
                pairs += 1
            if i % 10000 == 0 and i > 0:
                print(f"  {name}: {i}, пар={pairs}, written={written/1e6:.1f} МБ, total={total_size_mb():.1f} МБ")
            if written >= max_bytes or written >= MAX_PER_FILE_BYTES:
                break
    print(f"  ✓ {name}: {pairs} пар, {get_size_mb(out_path):.1f} МБ\n")


# Список датасетов (с приоритетами)
DATASETS = [
    # (name, hf_id, source, role_keys, out_name, max_mb)
    ("UltraChat-200k", "HuggingFaceH4/ultrachat_200k", "conversations",
     {"user": ["human"], "assistant": ["gpt"]}, "ultrachat.jsonl", 500),
    ("SmolTalk", "HuggingFaceTB/smoltalk", "messages",
     {"user": ["user"], "assistant": ["assistant"]}, "smoltalk.jsonl", 1500),
    ("EvolKit", "arcee-ai/EvolKit-20k", "conversations",
     {"user": ["human"], "assistant": ["gpt"]}, "evolkit.jsonl", 50),
    ("no_robots", "HuggingFaceH4/no_robots", "conversations",
     {"user": ["user"], "assistant": ["assistant"]}, "no_robots.jsonl", 30),
    ("alpaca", "tatsu-lab/alpaca", "messages",
     {"user": ["user"], "assistant": ["assistant"]}, "alpaca_orig.jsonl", 200),
    ("oasst2", "OpenAssistant/oasst2", "messages",
     {"user": ["prompter"], "assistant": ["assistant"]}, "oasst2.jsonl", 300),
    ("tulu", "allenai/tulu-v2-sft-mixture", "messages",
     {"user": ["user"], "assistant": ["assistant"]}, "tulu_v2.jsonl", 1000),
    ("dolly_clean", "argilla/databricks-dolly-15k-curated", "conversations",
     {"user": ["user"], "assistant": ["assistant"]}, "dolly_clean.jsonl", 30),
]

for name, hf_id, source, rk, out, max_mb in DATASETS:
    if total_size_mb() * 1e6 >= MAX_TOTAL_BYTES:
        print(f"GLOBAL STOP: total >= 10GB ({total_size_mb():.1f} МБ)")
        break
    fetch_dataset(name, hf_id, source, rk, out, max_mb * 1_000_000)

print(f"\n✓ итого: {total_size_mb():.1f} МБ скачано")