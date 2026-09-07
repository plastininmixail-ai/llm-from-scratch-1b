"""Парсер Wikipedia dump (XML Multistream).

Извлекает title → первый параграф (summary) для каждой статьи.
Сохраняет в JSONL для дальнейшей загрузки в TF-IDF KB.

Формат входа: enwiki-latest-pages-articles-multistream.xml.bz2
~25 GB bz2 → ~95 GB raw XML → ~100 GB extracted text
Обработка: ~30-60 минут на CPU.

Использование:
  python -m scripts.parse_wiki_dump
"""
from __future__ import annotations

import bz2
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
WIKI_PATH = ROOT / "data/wiki/enwiki-latest-pages-articles-multistream.xml.bz2"
OUTPUT_PATH = ROOT / "data/wiki_summaries.jsonl"

# Regex для извлечения title
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
# Regex для извлечения text (статья целиком, не только intro)
TEXT_RE = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL)
# Wikipedia разметка: убираем templates, infoboxes
INFOBOX_RE = re.compile(r"\{\{[^}]*\}\}", re.DOTALL)
TEMPLATE_RE = re.compile(r"\{\{[^}]*\}\}", re.DOTALL)
WIKI_LINK_RE = re.compile(r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def clean_text(raw: str) -> str:
    """Очистка Wikipedia разметки → plain text."""
    text = raw

    # Убираем комментарии
    text = COMMENT_RE.sub("", text)
    # Убираем HTML tags
    text = HTML_TAG_RE.sub(" ", text)
    # Убираем infobox
    text = INFOBOX_RE.sub("", text)
    # Убираем ссылки [[X|Y]] → X
    text = WIKI_LINK_RE.sub(r"\1", text)
    # Убираем refs
    text = REF_RE.sub("", text)
    # Схлопываем whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_summary(text: str, max_len: int = 500) -> str:
    """Извлечь первый параграф (summary)."""
    # Берём первое предложение (до точки)
    text = clean_text(text)
    # Берём первые N chars или до первой точки + пробел
    sentences = re.split(r"\. (?=[A-ZА-Я])", text)
    summary = ""
    for s in sentences[:3]:
        if len(summary) + len(s) > max_len:
            break
        summary += s + ". "
    return summary.strip()[:max_len]


def should_skip_title(title: str) -> bool:
    """Пропускаем служебные статьи."""
    if title.startswith(("Wikipedia:", "Help:", "Category:", "File:", "Template:", "Portal:", "Special:", "User:", "Talk:")):
        return True
    if ":" in title and not title.startswith(("List of",)):
        return True
    if title.startswith("List of"):
        return False  # Lists интересны
    return False


def parse_dump() -> int:
    """Парсит Wikipedia dump и сохраняет в JSONL."""
    if not WIKI_PATH.exists():
        print(f"ERROR: {WIKI_PATH} не найден")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"📖 Парсю {WIKI_PATH.name}")
    print(f"   Размер: {WIKI_PATH.stat().st_size / 1e9:.2f} GB")

    n_processed = 0
    n_skipped = 0
    n_saved = 0
    t0 = time.time()
    last_log = t0

    with bz2.open(WIKI_PATH, "rt", encoding="utf-8") as f, \
         OUTPUT_PATH.open("w", encoding="utf-8") as out:

        # Идём постранично (XML Wikipedia Multistream использует <page> теги)
        buffer = ""
        for line in f:
            buffer += line

            # Конец страницы
            if "</page>" in buffer:
                # Извлекаем title
                title_match = TITLE_RE.search(buffer)
                if not title_match:
                    buffer = ""
                    continue

                title = title_match.group(1).strip()
                n_processed += 1

                if should_skip_title(title):
                    n_skipped += 1
                else:
                    # Извлекаем text
                    text_match = TEXT_RE.search(buffer)
                    if text_match:
                        raw_text = text_match.group(1)
                        summary = extract_summary(raw_text)
                        if summary and len(summary) > 50:
                            out.write(json.dumps({
                                "title": title,
                                "summary": summary,
                                "lang": "en",
                            }, ensure_ascii=False) + "\n")
                            n_saved += 1

                buffer = ""

                # Логируем каждые 30 сек
                if time.time() - last_log > 30:
                    elapsed = time.time() - t0
                    rate = n_processed / elapsed if elapsed > 0 else 0
                    print(f"  [{elapsed/60:.1f}min] processed={n_processed}, saved={n_saved}, skipped={n_skipped}, rate={rate:.0f}/s")
                    last_log = time.time()

    elapsed = time.time() - t0
    print(f"\n✅ Готово за {elapsed/60:.1f} мин")
    print(f"   Processed: {n_processed}")
    print(f"   Saved: {n_saved}")
    print(f"   Skipped: {n_skipped}")
    print(f"   Output: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(parse_dump())
