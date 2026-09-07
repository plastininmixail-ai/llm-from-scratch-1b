"""
Скачивает OpenAssistant/oasst1 — реальные человеческие диалоги с AI.

Использование:
    python scripts/download_oasst.py --target-mb 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/oasst1.jsonl"))
    p.add_argument("--target-mb", type=int, default=50)
    p.add_argument("--ru-ratio", type=float, default=0.30,
                   help="доля RU диалогов (0.0-1.0)")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # OASST1 в плоском формате (messages)
    print("Загружаю OpenAssistant/oasst1 (messages)...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    target_bytes = args.target_mb * 1_000_000
    written = 0
    pairs = 0
    pairs_ru = 0
    pairs_en = 0

    # Словарь для построения Q&A пар
    msg_index: dict = {}

    with args.out.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            mid = ex.get("message_id")
            text = ex.get("text", "")
            role = ex.get("role", "")
            parent = ex.get("parent_id")
            lang = ex.get("lang", "en")
            if not mid or not text or role not in ("prompter", "assistant"):
                continue
            msg_index[mid] = {"text": text, "role": role, "parent": parent, "lang": lang}

            # Если это assistant и есть parent (prompter) — записываем Q&A пару
            if role == "assistant" and parent and parent in msg_index:
                parent_msg = msg_index[parent]
                if parent_msg["role"] == "prompter" and parent_msg["lang"] == lang:
                    pair = {
                        "prompt": parent_msg["text"][:1000],
                        "response": text[:1000],
                    }
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    pairs += 1
                    if lang == "ru":
                        pairs_ru += 1
                    else:
                        pairs_en += 1
                    written += len(pair["prompt"]) + len(pair["response"])

            if i % 5000 == 0 and i > 0:
                ru_pct = 100 * pairs_ru / max(pairs, 1)
                print(f"  обработано {i}, пар={pairs} (RU={pairs_ru}/{ru_pct:.0f}%, EN={pairs_en}), written={written/1e6:.1f} МБ")

            # Достигли ли лимита
            if written >= target_bytes:
                # Если мало RU, продолжаем
                if pairs_ru < target_bytes * args.ru_ratio * 0.5 / 1e6:
                    target_bytes += 30_000_000  # даём ещё 30 МБ для RU
                    print(f"  мало RU, продлеваю до {target_bytes/1e6:.0f} МБ")
                else:
                    break

    print(f"\n✓ готово: {pairs} пар Q&A ({pairs_ru} RU + {pairs_en} EN) → {args.out}")
    print(f"  размер: {args.out.stat().st_size/1e6:.2f} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())