"""
Скачивает дополнительный корпус ~2 ГБ из HuggingFace и кладёт в data/raw/.

Использование:
    python scripts/download_extra_corpus.py --target-gb 2
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--target-gb", type=float, default=2.0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. WikiText-103 (English Wikipedia, ~180 МБ) ---
    wikitext_path = args.out_dir / "wikitext103.txt"
    if not wikitext_path.exists():
        print(f"[1/2] скачиваю WikiText-103 → {wikitext_path}")
        try:
            ds = load_dataset(
                "Salesforce/wikitext",
                "wikitext-103-raw-v1",
                split="train",
                trust_remote_code=True,
            )
            with wikitext_path.open("w", encoding="utf-8") as f:
                for ex in ds:
                    txt = ex["text"].strip()
                    if txt:
                        f.write(txt + "\n")
            size_mb = wikitext_path.stat().st_size / 1e6
            print(f"   ✓ wikitext103 сохранён ({size_mb:.1f} МБ)")
        except Exception as e:
            print(f"   ⚠ ошибка: {e}")
    else:
        print(f"[1/2] wikitext103 уже есть ({wikitext_path.stat().st_size/1e6:.1f} МБ)")

    # --- 2. BookCorpus (English books, ~1 ГБ) ---
    bc_path = args.out_dir / "bookcorpus_part.txt"
    target_bytes = int(args.target_gb * 1e9)
    current_total = sum(
        p.stat().st_size
        for p in [
            args.out_dir / "corpus.txt",
            args.out_dir / "wikitext103.txt",
            args.out_dir / "bookcorpus_part.txt",
        ]
        if p.exists()
    )

    if current_total < target_bytes:
        need = target_bytes - current_total
        print(f"[2/2] нужно ещё ~{need/1e9:.2f} ГБ BookCorpus → {bc_path}")
        try:
            ds = load_dataset(
                "bookcorpus/bookcorpus",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            with bc_path.open("w", encoding="utf-8") as f:
                written = 0
                for i, ex in enumerate(ds):
                    txt = ex["text"].strip()
                    if not txt:
                        continue
                    f.write(txt + "\n\n")
                    written += len(txt) + 2
                    if i % 5000 == 0:
                        print(f"   итерация {i}, написано {written/1e6:.1f} МБ")
                    if written >= need:
                        break
            print(f"   ✓ bookcorpus_part сохранён ({bc_path.stat().st_size/1e9:.2f} ГБ)")
        except Exception as e:
            print(f"   ⚠ ошибка bookcorpus: {e}")
            # fallback: streaming из C4
            print("   fallback: использую C4 streaming...")
            try:
                ds = load_dataset(
                    "allenai/c4",
                    "en",
                    split="train",
                    streaming=True,
                    trust_remote_code=True,
                )
                with bc_path.open("w", encoding="utf-8") as f:
                    written = 0
                    for i, ex in enumerate(ds):
                        txt = ex["text"].strip()
                        if not txt:
                            continue
                        f.write(txt + "\n\n")
                        written += len(txt) + 2
                        if i % 5000 == 0:
                            print(f"   c4 итерация {i}, написано {written/1e6:.1f} МБ")
                        if written >= need:
                            break
            except Exception as e2:
                print(f"   ⚠ ошибка c4: {e2}")
    else:
        print(f"[2/2] уже {current_total/1e9:.2f} ГБ, доп. скачивание не нужно")

    # --- Итог ---
    print("\n=== итоговые размеры ===")
    total = 0
    for p in sorted(args.out_dir.glob("*.txt")):
        sz = p.stat().st_size
        total += sz
        print(f"  {p.name:40s} {sz/1e6:8.1f} МБ")
    print(f"  {'TOTAL':40s} {total/1e9:8.2f} ГБ")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
