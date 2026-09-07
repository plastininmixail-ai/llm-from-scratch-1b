"""
Качает дополнительные instruct-датасеты (×2 данных).
Источники: SlimOrca, OpenAssistant-2, Magpie, Tulu-3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def fetch(name: str, config, split: str, out_path: Path, target_bytes: int) -> int:
    if out_path.exists() and out_path.stat().st_size > 1000:
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"  ✓ {out_path.name} уже есть ({out_path.stat().st_size/1e6:.1f} МБ, {n} пар)")
        return n
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
            prompt = (
                ex.get("instruction")
                or ex.get("question")
                or ex.get("prompt")
                or ex.get("input")
                or ex.get("problem")  # math
                or ""
            )
            response = (
                ex.get("output")
                or ex.get("answer")
                or ex.get("response")
                or ex.get("completion")
                or ex.get("text")
                or ex.get("solution")
                or ""
            )
            # SlimOrca/SystemChat format
            if not prompt and "conversations" in ex:
                convs = ex["conversations"]
                for j in range(0, len(convs) - 1, 2):
                    if j + 1 < len(convs):
                        p = convs[j].get("value", "").replace("USER:", "").replace("ASSISTANT:", "").strip()
                        r = convs[j + 1].get("value", "").replace("USER:", "").replace("ASSISTANT:", "").strip()
                        if p and r and 5 < len(p) < 1000 and 5 < len(r) < 1000:
                            pair = {"prompt": p, "response": r}
                            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            pairs += 1
                            written += len(p) + len(r)
                continue

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
    p.add_argument("--out-dir", type=Path, default=Path("data/instruct_v3"))
    p.add_argument("--target-mb", type=int, default=150)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        # SlimOrca — высококачественный subset OpenOrca
        ("Open-Orca/SlimOrca", None, "train"),
        # Magpie — Llama 3.1 self-instruct
        ("argilla/magpie-ultra-v0.1", None, "train"),
        # Tulu-3 — AllenAI новейший
        ("allenai/tulu-3-sft-mixture", None, "train"),
        # Capybara — multi-turn dialog
        ("argilla/Capybara-Preferences", None, "train"),
    ]

    target = args.target_mb * 1_000_000 // len(sources)
    total = 0
    for name, config, split in sources:
        out = args.out_dir / f"{name.split('/')[-1]}.jsonl"
        n = fetch(name, config, split, out, target)
        total += n
        print(f"  ✓ {n} пар → {out.name}\n")

    print(f"✓ итого: {total} пар")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())