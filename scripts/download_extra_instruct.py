"""
Скачивает дополнительные instruct-датасеты с HuggingFace.

Источники:
- HuggingFaceH4/ultrachat_200k — реальные chat-диалоги
- Open-Orca/OpenOrca — GPT-4 distilled
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def fetch_one(name: str, config, split: str, out_path: Path, target_bytes: int):
    """Скачивает один датасет streaming до target_bytes."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"  ✓ {out_path.name} уже есть ({out_path.stat().st_size/1e6:.1f} МБ)")
        return 0
    print(f"  → {out_path.name} ← {name}")
    try:
        if config:
            ds = load_dataset(name, config, split=split, streaming=True)
        else:
            ds = load_dataset(name, split=split, streaming=True)
    except Exception as e:
        print(f"  ⚠ {name}: {e}")
        return 0

    written = 0
    pairs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            # Разные форматы
            prompt = ""
            response = ""
            if "messages" in ex:
                # Chat формат (Ultrachat, OpenOrca)
                msgs = ex["messages"]
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                prompt = user_msg
                response = asst_msg
            elif "instruction" in ex and "response" in ex:
                prompt = ex["instruction"]
                response = ex["response"]
            elif "question" in ex and "answer" in ex:
                prompt = ex["question"]
                response = ex["answer"]
            elif "prompt" in ex and "response" in ex:
                prompt = ex["prompt"]
                response = ex["response"]

            if not prompt or not response:
                continue
            if len(prompt) > 1000 or len(response) > 1000:
                continue
            if len(prompt) < 5 or len(response) < 5:
                continue
            pair = {"prompt": prompt, "response": response}
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pairs += 1
            written += len(prompt) + len(response)
            if i % 5000 == 0 and i > 0:
                print(f"  {name}: {i}, пар={pairs}, written={written/1e6:.1f} МБ")
            if written >= target_bytes:
                break
    return pairs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/extra"))
    p.add_argument("--target-mb-per-source", type=int, default=15)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        # UltraChat — реальные chat-диалоги между людьми
        ("HuggingFaceH4/ultrachat_200k", None, "train_sft"),
        # OpenOrca — GPT-4 distilled
        ("Open-Orca/OpenOrca", None, "train"),
        # Если что-то не работает — добавим ещё
    ]

    target = args.target_mb_per_source * 1_000_000
    total = 0
    for name, config, split in sources:
        out = args.out_dir / f"{name.split('/')[-1]}.jsonl"
        n = fetch_one(name, config, split, out, target)
        total += n
        print(f"  ✓ {n} пар → {out}")

    print(f"\n✓ итого: {total} пар")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())