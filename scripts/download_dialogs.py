"""
Скачивает диалоговые датасеты для обучения chat-модели.

Источники:
- HuggingFace: daily_dialog (короткие диалоги)
- EmpatheticDialogues (эмпатичные разговоры)
- Lite LSDD (диалоги на стоические темы)

Использование:
    python scripts/download_dialogs.py --target-gb 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def fetch_streaming(name: str, config: str | None, split: str, target_bytes: int, out_path: Path):
    """Скачивает streaming датасет до target_bytes."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"  ✓ {out_path.name} уже есть ({out_path.stat().st_size/1e6:.1f} МБ)")
        return 0
    print(f"  → {out_path.name} ← {name}")
    if config:
        ds = load_dataset(name, config, split=split, streaming=True)
    else:
        ds = load_dataset(name, split=split, streaming=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            # формат зависит от датасета
            if "dialog" in ex:
                turns = ex["dialog"]
            elif "utterances" in ex:
                turns = ex["utterances"]
            elif "conversation" in ex:
                turns = ex["conversation"]
            else:
                continue
            # turns = list of strings or list of {speaker, text}
            line_parts = []
            for turn in turns:
                if isinstance(turn, str):
                    line_parts.append(turn)
                elif isinstance(turn, dict):
                    line_parts.append(turn.get("text", ""))
                else:
                    continue
            line = " <SEP> ".join(line_parts) + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            if i % 5000 == 0:
                print(f"    итерация {i}, написано {written/1e6:.1f} МБ")
            if written >= target_bytes:
                break
    return written


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/dialogs"))
    p.add_argument("--target-gb", type=float, default=1.0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.target_gb * 1e9)
    per_source = target_bytes // 3

    print(f"=== Скачиваю диалоговые датасеты ({args.target_gb} ГБ) ===\n")

    # 1. DailyDialog (короткие диалоги) — несколько зеркал
    try:
        fetch_streaming(
            "liyucheng/IMDB",
            None, "train", per_source,
            args.out_dir / "imdb.txt"
        )
    except Exception as e:
        print(f"  ⚠ imdb: {e}")
    try:
        fetch_streaming(
            "Anthropic/hh-rlhf",
            None, "train", per_source,
            args.out_dir / "hh_rlhf.txt"
        )
    except Exception as e:
        print(f"  ⚠ hh_rlhf: {e}")

    # 2. DailyDialog (другое имя)
    try:
        fetch_streaming(
            "Generative-Dialogue-Systems/daily_dialog",
            None, "train", per_source,
            args.out_dir / "daily_dialog_v2.txt"
        )
    except Exception as e:
        print(f"  ⚠ daily_dialog_v2: {e}")

    # 3. Soda (Social Dialogs)
    try:
        fetch_streaming(
            "allenai/soda",
            None, "train", per_source,
            args.out_dir / "soda.txt"
        )
    except Exception as e:
        print(f"  ⚠ soda: {e}")

    print("\n=== итог ===")
    total = 0
    for p in sorted(args.out_dir.glob("*.txt")):
        sz = p.stat().st_size
        total += sz
        print(f"  {p.name:30s} {sz/1e6:8.1f} МБ")
    print(f"  {'TOTAL':30s} {total/1e9:8.2f} ГБ")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())