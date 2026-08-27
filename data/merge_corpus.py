"""
Объединяет EN+RU Wiki + кодовые сэмплы в один перемешанный корпус.

Использование:
    python -m data.merge_corpus
"""
from __future__ import annotations

import random
from pathlib import Path


def main() -> int:
    raw = Path("data/raw")

    # Wiki (EN + RU)
    en = (raw / "enwiki.txt").read_text(encoding="utf-8") if (raw / "enwiki.txt").exists() else ""
    ru = (raw / "ruwiki.txt").read_text(encoding="utf-8") if (raw / "ruwiki.txt").exists() else ""

    # Code: code_<lang>.txt
    code_files = sorted(raw.glob("code_*.txt"))

    articles: list[str] = []

    for big in (en, ru):
        for chunk in big.split("\n\n"):
            chunk = chunk.strip()
            if 100 < len(chunk) < 50000:
                articles.append(chunk)

    for cf in code_files:
        text = cf.read_text(encoding="utf-8")
        for chunk in text.split("\n\n"):
            chunk = chunk.strip()
            if 100 < len(chunk) < 30000:  # код обычно покороче статей
                articles.append(chunk)

    en_count = sum(1 for a in articles if not any(ord(c) > 1024 for c in a if c.isalpha()))
    ru_count = sum(1 for a in articles if any(ord(c) > 1024 for c in a))
    code_count = len(articles) - en_count - ru_count

    random.seed(42)
    random.shuffle(articles)

    out = "\n\n".join(articles)
    out_path = raw / "corpus.txt"
    out_path.write_text(out, encoding="utf-8")

    print(f"EN wiki: {en_count}")
    print(f"RU wiki: {ru_count}")
    print(f"code:    {code_count}")
    print(f"всего:   {len(articles)}")
    print(f"corpus.txt: {out_path.stat().st_size / 1e6:.1f} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())