"""
Скачивает чистые диалоги (без системных промптов).

Источники:
- OpenAssistant/oasst1 — реальные диалоги (human-AI)
- ShareGPT_Vicuna_unfiltered — реальные разговоры с ChatGPT
- lmsys/lmsys-chat-1m — 1M реальных диалогов
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset


def fetch_oasst1(out_path: Path, target_bytes: int) -> int:
    """OASST1: messages table → prompt/response пары."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"  ✓ {out_path.name} уже есть ({n} пар)")
        return n
    print(f"  → {out_path.name} ← OpenAssistant/oasst1")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    except Exception as e:
        print(f"  ⚠ OASST1: {e}")
        return 0

    msg_index: dict = {}
    written = 0
    pairs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            mid = ex.get("message_id")
            text = ex.get("text", "")
            role = ex.get("role", "")
            parent = ex.get("parent_id")
            lang = ex.get("lang", "en")
            deleted = ex.get("deleted", False)
            if not mid or not text or role not in ("prompter", "assistant"):
                continue
            if deleted:
                continue
            if lang != "en":  # только EN
                continue
            msg_index[mid] = {"text": text, "role": role, "parent": parent}

            if role == "assistant" and parent and parent in msg_index:
                parent_msg = msg_index[parent]
                if parent_msg["role"] == "prompter":
                    pair = {"prompt": parent_msg["text"], "response": text}
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    pairs += 1
                    written += len(parent_msg["text"]) + len(text)
            if i % 5000 == 0 and i > 0:
                print(f"  OASST1: {i}, пар={pairs}, written={written/1e6:.1f} МБ")
            if written >= target_bytes:
                break
    return pairs


def fetch_sharegpt(out_path: Path, target_bytes: int) -> int:
    """ShareGPT_Vicuna_unfiltered: conversations → multi-turn."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"  ✓ {out_path.name} уже есть ({n} пар)")
        return n
    print(f"  → {out_path.name} ← ShareGPT")
    try:
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True, data_files=["ShareGPT_V3_unfiltered_cleaned_split.json"])
    except Exception as e:
        try:
            # Попробуем альтернативный конфиг
            ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        except Exception as e2:
            print(f"  ⚠ ShareGPT: {e2}")
            return 0

    written = 0
    pairs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            convs = ex.get("conversations", [])
            # Берём первую user-assistant пару
            for j in range(0, len(convs) - 1, 2):
                if j + 1 < len(convs):
                    # ShareGPT формат: [{from: "human", value: "..."}, {from: "gpt", value: "..."}]
                    item1 = convs[j]
                    item2 = convs[j+1]
                    if not isinstance(item1, dict) or not isinstance(item2, dict):
                        continue
                    role1 = item1.get("from", "")
                    role2 = item2.get("from", "")
                    if role1 == "human" and role2 == "gpt":
                        prompt = item1.get("value", "")
                        response = item2.get("value", "")
                        if prompt and response and 5 < len(prompt) < 1000 and 5 < len(response) < 1000:
                            pair = {"prompt": prompt, "response": response}
                            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            pairs += 1
                            written += len(prompt) + len(response)
                            break
            if i % 1000 == 0 and i > 0:
                print(f"  ShareGPT: {i}, пар={pairs}, written={written/1e6:.1f} МБ")
            if written >= target_bytes:
                break
    return pairs


def fetch_lmsys(out_path: Path, target_bytes: int) -> int:
    """lmsys/lmsys-chat-1m: 1M реальных диалогов."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"  ✓ {out_path.name} уже есть ({n} пар)")
        return n
    print(f"  → {out_path.name} ← lmsys/lmsys-chat-1m")
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    except Exception as e:
        print(f"  ⚠ LMSYS: {e}")
        return 0

    written = 0
    pairs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            convs = ex.get("conversation", [])
            # Берём первую user-assistant пару
            for j in range(0, len(convs) - 1, 2):
                if j + 1 < len(convs):
                    if convs[j].get("role") == "user" and convs[j+1].get("role") == "assistant":
                        prompt = convs[j].get("content", "")
                        response = convs[j+1].get("content", "")
                        if prompt and response and 5 < len(prompt) < 1000 and 5 < len(response) < 1000:
                            pair = {"prompt": prompt, "response": response}
                            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            pairs += 1
                            written += len(prompt) + len(response)
                            break
            if i % 5000 == 0 and i > 0:
                print(f"  LMSYS: {i}, пар={pairs}, written={written/1e6:.1f} МБ")
            if written >= target_bytes:
                break
    return pairs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/clean_dialogs"))
    p.add_argument("--target-mb", type=int, default=300)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Скачиваю ЧИСТЫЕ диалоги (без системных промптов)")
    print("=" * 60)

    total = 0
    for fn, name in [
        (fetch_oasst1, "oasst1_clean.jsonl"),
        (fetch_sharegpt, "sharegpt.jsonl"),
        (fetch_lmsys, "lmsys.jsonl"),
    ]:
        out = args.out_dir / name
        n = fn(out, args.target_mb * 1_000_000 // 3)
        total += n
        print(f"  ✓ {n} пар → {out.name}\n")

    print(f"✓ итого: {total} пар")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())