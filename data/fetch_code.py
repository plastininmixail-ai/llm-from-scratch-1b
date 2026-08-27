"""
Скачивает кодовые сэмплы через streaming-API, поддерживает разные открытые датасеты.

Использование:
    python -m data.fetch_code --sources codeparrot --max-mb 3
"""
from __future__ import annotations

import argparse
from pathlib import Path


def fetch_codeparrot(lang: str, max_mb: float, out_dir: Path) -> int:
    """Только Python из codeparrot-clean."""
    from datasets import load_dataset

    print(f"  load {lang} (codeparrot-clean)...", flush=True)
    ds = load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")

    out_path = out_dir / f"code_{lang}.txt"
    target_bytes = int(max_mb * 1024 * 1024)
    written = 0
    files_written = 0
    seen_paths = set()

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            content = row["content"]
            if len(content) < 200 or len(content) > 20000:
                continue
            key = (row.get("repo_name", ""), row.get("path", ""))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            f.write(content + "\n\n")
            written += len(content) + 2
            files_written += 1
            if written >= target_bytes:
                break

    return files_written


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["python"])
    p.add_argument("--max-mb", type=float, default=3.0)
    args = p.parse_args()

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for lang in args.languages:
        try:
            n = fetch_codeparrot(lang, args.max_mb, out_dir)
            size = (out_dir / f"code_{lang}.txt").stat().st_size / 1e6
            print(f"  {lang}: {n} файлов ({size:.1f} МБ)", flush=True)
            total += n
        except Exception as exc:
            print(f"  {lang}: ОШИБКА {exc}", flush=True)

    print(f"всего: {total} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())