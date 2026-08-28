"""
Скачивает тексты стоических философов (древнегреческих и древнеримских).

Источники:
- Project Gutenberg (mirror через стандартные URL)
- Wikisource (для русских переводов)

Использование:
    python scripts/download_stoicism.py --out-dir data/raw/stoicism
"""
from __future__ import annotations

import argparse
import gzip
import time
import urllib.request
from pathlib import Path

# ============================================================
# Реестр текстов стоиков
# ============================================================
# Каждый — это список (имя_файла, URL, язык)
STOIC_TEXTS_EN = [
    # Марк Аврелий — Meditations (George Long translation)
    ("marcus_aurelius_meditations_long.txt",
     "https://www.gutenberg.org/cache/epub/7142/pg7142.txt"),
    # Марк Аврелий — Meditations (Hays translation)
    ("marcus_aurelius_meditations_hays.txt",
     "https://www.gutenberg.org/cache/epub/7142/pg7142.txt"),
    # Сенека — Moral Letters to Lucilius
    ("seneca_moral_letters.txt",
     "https://www.gutenberg.org/cache/epub/3794/pg3794.txt"),
    # Сенека — On the Shortness of Life (De Brevitate Vitae)
    ("seneca_shortness_of_life.txt",
     "https://www.gutenberg.org/cache/epub/65004/pg65004.txt"),
    # Эпиктет — Enchiridion (Handbook)
    ("epictetus_enchiridion.txt",
     "https://www.gutenberg.org/cache/epub/45109/pg45109.txt"),
    # Эпиктет — Discourses
    ("epictetus_discourses.txt",
     "https://www.gutenberg.org/cache/epub/75209/pg75209.txt"),
    # Cicero — De Officiis (на латыни, есть stoic references)
    # Cicero — Tusculan Disputations
    ("cicero_tusculan_disputations.txt",
     "https://www.gutenberg.org/cache/epub/14988/pg14988.txt"),
]

STOIC_TEXTS_RU = [
    # Марк Аврелий — Размышления (русский перевод)
    ("marcus_aurelius_meditations_ru.txt",
     "https://ru.wikisource.org/wiki/%D0%9C%D0%B5%D0%B4%D0%B8%D1%82%D0%B0%D1%86%D0%B8%D0%B8_(%D0%90%D0%B2%D1%80%D0%B5%D0%BB%D0%B8%D0%B9)"),
    # Сенека — Нравственные письма к Луцилию (русский перевод)
    ("seneca_moral_letters_ru.txt",
     "https://ru.wikisource.org/wiki/%D0%9D%D1%80%D0%B0%D0%B2%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5_%D0%BF%D0%B8%D1%81%D1%8C%D0%BC%D0%B0_%D0%BA_%D0%9B%D1%83%D1%86%D0%B8%D0%BB%D0%B8%D1%8E"),
    # Эпиктет — Внутренний покой (русский перевод)
    ("epictetus_enchiridion_ru.txt",
     "https://ru.wikisource.org/wiki/%D0%AD%D0%BD%D1%85%D0%B8%D1%80%D0%B8%D0%B4%D0%B8%D0%BE%D0%BD"),
]


def fetch(url: str, retries: int = 3) -> bytes | None:
    """Скачивает URL с retries."""
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            print(f"  retry {i+1}/{retries}: {e}")
            time.sleep(2)
    return None


def strip_gutenberg(raw: bytes) -> str:
    """Удаляет Gutenberg-обёрку (header + footer)."""
    text = raw.decode("utf-8", errors="replace")
    start = text.find("*** START OF THE PROJECT GUTENBERG")
    end = text.find("*** END OF THE PROJECT GUTENBERG")
    if start != -1:
        # пропускаем саму строку-маркер
        start = text.find("\n", start) + 1
    if end != -1:
        text = text[start:end]
    return text.strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/stoicism"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    downloaded = 0

    print("=" * 60)
    print("скачиваю стоические тексты (EN)")
    print("=" * 60)
    for fname, url in STOIC_TEXTS_EN:
        out = args.out_dir / fname
        if out.exists() and out.stat().st_size > 1000:
            print(f"  ✓ {fname} ({out.stat().st_size/1e3:.1f} КБ) — уже есть")
            total_bytes += out.stat().st_size
            continue
        print(f"  → {fname} ← {url}")
        raw = fetch(url)
        if raw is None:
            print(f"  ⚠ пропускаю {fname}")
            continue
        # Очищаем от Gutenberg headers
        if b"PROJECT GUTENBERG" in raw:
            text = strip_gutenberg(raw)
            out.write_text(text, encoding="utf-8")
        else:
            out.write_bytes(raw)
        total_bytes += out.stat().st_size
        downloaded += 1
        print(f"  ✓ {fname} ({out.stat().st_size/1e3:.1f} КБ)")
        time.sleep(1)  # polite pause

    print()
    print("=" * 60)
    print("скачиваю русские переводы (RU, через Wikisource)")
    print("=" * 60)
    for fname, url in STOIC_TEXTS_RU:
        out = args.out_dir / fname
        if out.exists() and out.stat().st_size > 1000:
            print(f"  ✓ {fname} ({out.stat().st_size/1e3:.1f} КБ) — уже есть")
            total_bytes += out.stat().st_size
            continue
        print(f"  → {fname} ← {url}")
        raw = fetch(url)
        if raw is None:
            print(f"  ⚠ пропускаю {fname}")
            continue
        out.write_bytes(raw)
        total_bytes += out.stat().st_size
        downloaded += 1
        print(f"  ✓ {fname} ({out.stat().st_size/1e3:.1f} КБ)")
        time.sleep(1)

    print()
    print("=" * 60)
    print(f"итого:")
    print(f"  скачано файлов: {downloaded}")
    print(f"  общий размер: {total_bytes/1e6:.2f} МБ / {total_bytes/1e9:.3f} ГБ")
    print(f"  директория: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())