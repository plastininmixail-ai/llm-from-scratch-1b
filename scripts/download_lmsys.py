"""Скачивает LMSYS-Chat-1M используя HF токен."""
from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ.get("HF_TOKEN")

OUT_PATH = Path("data/clean_dialogs/lmsys.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"TOKEN_PREFIX: {TOKEN[:8] if TOKEN else 'None'}")
print(f"→ {OUT_PATH.name} ← lmsys/lmsys-chat-1m")

written = 0
pairs = 0
with OUT_PATH.open("w", encoding="utf-8") as f:
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True, token=TOKEN)
        for i, ex in enumerate(ds):
            convs = ex.get("conversation", [])
            for j in range(0, len(convs) - 1, 2):
                if j + 1 < len(convs):
                    item1 = convs[j]
                    item2 = convs[j+1]
                    if not isinstance(item1, dict) or not isinstance(item2, dict):
                        continue
                    if item1.get("role") == "user" and item2.get("role") == "assistant":
                        prompt = item1.get("content", "")
                        response = item2.get("content", "")
                        if prompt and response and 5 < len(prompt) < 1000 and 5 < len(response) < 1000:
                            pair = {"prompt": prompt, "response": response}
                            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            pairs += 1
                            written += len(prompt) + len(response)
                            break
            if i % 10000 == 0 and i > 0:
                print(f"  LMSYS: {i}, пар={pairs}, written={written/1e6:.1f} МБ")
            if written >= 200_000_000:  # 200 MB
                break
    except Exception as e:
        print(f"  ERR: {e}")

print(f"\n✓ итого: {pairs} пар → {OUT_PATH}")