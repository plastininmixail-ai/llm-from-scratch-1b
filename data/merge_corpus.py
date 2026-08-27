"""
Объединяет EN и RU тексты в один перемешанный корпус.

Использование:
    python -m data.merge_corpus
"""
from __future__ import annotations

import random
from pathlib import Path


def main() -> int:
    raw = Path("data/raw")

    en = (raw / "enwiki.txt").read_text(encoding="utf-8")
    ru = (raw / "ruwiki.txt").read_text(encoding="utf-8")

    articles: list[str] = []
    for big in (en, ru):
        for chunk in big.split("\n\n"):
            chunk = chunk.strip()
            if 100 < len(chunk) < 50000:
                articles.append(chunk)

    en_count = sum(1 for a in articles if not any(ord(c) > 1024 for c in a))
    ru_count = sum(1 for a in articles if any(ord(c) > 1024 for c in a))

    random.seed(42)
    random.shuffle(articles)

    out = "\n\n".join(articles)
    out_path = raw / "corpus.txt"
    out_path.write_text(out, encoding="utf-8")

    print(f"EN статей: {en_count}")
    print(f"RU статей: {ru_count}")
    print(f"всего статей: {len(articles)}")
    print(f"corpus.txt: {out_path.stat().st_size / 1e6:.1f} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())