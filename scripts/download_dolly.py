"""
Скачивает дополнительные диалоги из Dolly (databricks) — открытый instruction dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/dolly.jsonl"))
    p.add_argument("--target-mb", type=int, default=20)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sources = [
        ("databricks/databricks-dolly-15k", None, "train"),
        ("nvidia/OpenMathInstruct-2", None, "train"),
    ]

    for name, config, split in sources:
        try:
            print(f"Загружаю {name}...")
            if config:
                ds = load_dataset(name, config, split=split, streaming=True)
            else:
                ds = load_dataset(name, split=split, streaming=True)
        except Exception as e:
            print(f"  ⚠ {name}: {e}")
            continue

        target = args.target_mb * 1_000_000 // len(sources)
        written = 0
        pairs = 0
        with args.out.open("a", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                # Разные форматы
                prompt = ex.get("instruction") or ex.get("question") or ex.get("prompt") or ""
                response = ex.get("response") or ex.get("answer") or ex.get("answer") or ex.get("output") or ""
                if not prompt or not response:
                    continue
                if len(prompt) > 1000 or len(response) > 1000:
                    continue
                pair = {"prompt": prompt, "response": response}
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                pairs += 1
                written += len(prompt) + len(response)
                if i % 1000 == 0 and i > 0:
                    print(f"  {name}: {i}, пар={pairs}, written={written/1e6:.1f} МБ")
                if written >= target:
                    break
        print(f"  ✓ {name}: {pairs} пар")

    print(f"\n✓ итого → {args.out}")
    print(f"  размер: {args.out.stat().st_size/1e6:.2f} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())