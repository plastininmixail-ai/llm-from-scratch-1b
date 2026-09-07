"""
Качает Alpaca-style instruct-датасеты (разнообразие тем).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def fetch(name: str, config, split: str, out_path: Path, target_bytes: int) -> int:
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"  ✓ {out_path.name} уже есть")
        return 0
    print(f"  → {out_path.name} ← {name}")
    try:
        if config:
            ds = load_dataset(name, config, split=split, streaming=True)
        else:
            ds = load_dataset(name, split=split, streaming=True)
    except Exception as e:
        print(f"  ⚠ {e}")
        return 0

    written = 0
    pairs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            # Разные форматы датасетов
            prompt = (
                ex.get("instruction")
                or ex.get("question")
                or ex.get("prompt")
                or ex.get("input")
                or ex.get("query")
                or ""
            )
            response = (
                ex.get("output")
                or ex.get("answer")
                or ex.get("response")
                or ex.get("completion")
                or ex.get("text")
                or ""
            )
            if not prompt or not response:
                continue
            if len(prompt) > 800 or len(response) > 800:
                continue
            if len(prompt) < 5 or len(response) < 5:
                continue
            pair = {"prompt": prompt, "response": response}
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pairs += 1
            written += len(prompt) + len(response)
            if i % 3000 == 0 and i > 0:
                print(f"  {name}: {i}, пар={pairs}, written={written/1e6:.1f} МБ")
            if written >= target_bytes:
                break
    return pairs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/alpaca"))
    p.add_argument("--target-mb", type=int, default=20)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        # Alpaca-style
        ("yahma/alpaca-cleaned", None, "train"),
        # WizardLM
        ("WizardLMTeam/WizardLM_evol_instruct_70k", None, "train"),
        # Tulu (open)
        ("allenai/tulu-v2-sft-mixture", None, "train"),
    ]

    target = args.target_mb * 1_000_000 // len(sources)
    total = 0
    for name, config, split in sources:
        out = args.out_dir / f"{name.split('/')[-1]}.jsonl"
        n = fetch(name, config, split, out, target)
        total += n
        print(f"  ✓ {n} пар → {out.name}")

    print(f"\n✓ итого: {total} пар")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())