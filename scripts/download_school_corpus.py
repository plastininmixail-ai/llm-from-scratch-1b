"""
Скачивает школьные программы: RU + EN классическая литература.
Источники: Lib.ru (RU), Project Gutenberg (EN).

Использование:
    python scripts/download_school_corpus.py --target-mb 200
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path


# === RU школьная программа (Lib.ru / Wikisource) ===
# Канонические тексты из школьной программы 7-11 класс
RU_SCHOOL_TEXTS = [
    # Пушкин
    ("pushkin_eugene_onegin.txt", "https://ilibrary.ru/text/1513/p.1/index.html"),
    ("pushkin_captain_daughter.txt", "https://ilibrary.ru/text/1300/p.1/index.html"),
    # Гоголь
    ("gogol_dead_souls.txt", "https://ilibrary.ru/text/1370/p.1/index.html"),
    ("gogol_overcoat.txt", "https://ilibrary.ru/text/1382/p.1/index.html"),
    # Толстой
    ("tolstoy_war_and_peace_1.txt", "https://ilibrary.ru/text/1098/p.1/index.html"),
    # Достоевский
    ("dostoevsky_crime.txt", "https://ilibrary.ru/text/1166/p.1/index.html"),
    # Чехов
    ("chekov_lady_dog.txt", "https://ilibrary.ru/text/1300/p.1/index.html"),
    # Тургенев
    ("turgenev_mumu.txt", "https://ilibrary.ru/text/1241/p.1/index.html"),
    # Лермонтов
    ("lermontov_hero_our_time.txt", "https://ilibrary.ru/text/1241/p.1/index.html"),
    # Булгаков
    ("bulgakov_master_margarita.txt", "https://ilibrary.ru/text/1181/p.1/index.html"),
]


# === EN школьная программа (Project Gutenberg) ===
EN_SCHOOL_TEXTS = [
    # Американская классика
    ("twain_huckleberry_finn.txt", "https://www.gutenberg.org/files/76/76-0.txt"),
    ("twain_tom_sawyer.txt", "https://www.gutenberg.org/files/74/74-0.txt"),
    # Британская классика
    ("austen_pride_prejudice.txt", "https://www.gutenberg.org/files/1342/1342-0.txt"),
    ("austen_sense_sensibility.txt", "https://www.gutenberg.org/files/161/161-0.txt"),
    ("dickens_tale_two_cities.txt", "https://www.gutenberg.org/files/98/98-0.txt"),
    ("dickens_oliver_twist.txt", "https://www.gutenberg.org/files/730/730-0.txt"),
    ("bronte_jane_eyre.txt", "https://www.gutenberg.org/files/1260/1260-0.txt"),
    # Школьная обязательная программа
    ("mellersh_wonder_book.txt", "https://www.gutenberg.org/files/7227/7227-0.txt"),
    ("sir_oliver_walks_back.txt", "https://www.gutenberg.org/files/7025/7025-0.txt"),
    # Грамматика и короткие тексты
    ("aesop_fables.txt", "https://www.gutenberg.org/files/11339/11339-0.txt"),
    ("grimms_fairy_tales.txt", "https://www.gutenberg.org/files/2591/2591-0.txt"),
    ("andersen_fairy_tales.txt", "https://www.gutenberg.org/files/1597/1597-0.txt"),
    # Поэзия
    ("poe_poems.txt", "https://www.gutenberg.org/files/10031/10031-0.txt"),
    ("whitman_leaves.txt", "https://www.gutenberg.org/files/1322/1322-0.txt"),
]


def fetch(url: str, retries: int = 3) -> bytes | None:
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
    text = raw.decode("utf-8", errors="replace")
    start = text.find("*** START OF THE PROJECT GUTENBERG")
    end = text.find("*** END OF THE PROJECT GUTENBERG")
    if start != -1:
        start = text.find("\n", start) + 1
    if end != -1:
        text = text[start:end]
    return text.strip()


def fetch_one(name: str, url: str, out_dir: Path) -> bool:
    out = out_dir / name
    if out.exists() and out.stat().st_size > 1000:
        print(f"  ✓ {name} ({out.stat().st_size/1e6:.2f} МБ) — уже есть")
        return True
    print(f"  → {name} ← {url}")
    raw = fetch(url)
    if raw is None:
        print(f"  ⚠ не удалось скачать {name}")
        return False
    if b"PROJECT GUTENBERG" in raw:
        text = strip_gutenberg(raw)
    else:
        text = raw.decode("utf-8", errors="replace")
    out.write_text(text, encoding="utf-8")
    print(f"  ✓ {name} ({out.stat().st_size/1e6:.2f} МБ)")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/school"))
    p.add_argument("--target-mb", type=int, default=200)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Скачиваю RU школьную программу (Lib.ru / Wikisource)")
    print("=" * 60)
    ru_ok = 0
    for name, url in RU_SCHOOL_TEXTS:
        if fetch_one(name, url, args.out_dir):
            ru_ok += 1
        time.sleep(0.5)

    print()
    print("=" * 60)
    print("Скачиваю EN школьную программу (Project Gutenberg)")
    print("=" * 60)
    en_ok = 0
    for name, url in EN_SCHOOL_TEXTS:
        if fetch_one(name, url, args.out_dir):
            en_ok += 1
        time.sleep(0.5)

    print()
    print("=" * 60)
    print(f"Итого: RU={ru_ok}/{len(RU_SCHOOL_TEXTS)}, EN={en_ok}/{len(EN_SCHOOL_TEXTS)}")
    total = 0
    for p_file in sorted(args.out_dir.glob("*.txt")):
        sz = p_file.stat().st_size
        total += sz
        print(f"  {p_file.name:40s} {sz/1e6:6.1f} МБ")
    print(f"  {'TOTAL':40s} {total/1e6:6.1f} МБ ({total/1e9:.2f} ГБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())