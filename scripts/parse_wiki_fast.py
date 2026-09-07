"""Быстрый парсер: обрабатывает только первые N страниц Wikipedia.

Использует bytes mode для скорости (вместо text mode).
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

# Лимит: обработать только первые ~1 GB (быстро, ~50K статей)
MAX_BYTES = 1 * 1024**3  # 1 GB
# Лимит количества статей
MAX_ARTICLES = 50_000

INFOBOX_RE = re.compile(r"\{\{[^}]*\}\}", re.DOTALL)
TEMPLATE_RE = re.compile(r"\{\{[^}]*\}\}", re.DOTALL)
WIKI_LINK_RE = re.compile(r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def clean_text(raw: str) -> str:
    text = raw
    text = COMMENT_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = INFOBOX_RE.sub("", text)
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = REF_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_summary(text: str, max_len: int = 500) -> str:
    text = clean_text(text)
    sentences = re.split(r"\. (?=[A-Z])", text)
    summary = ""
    for s in sentences[:3]:
        if len(summary) + len(s) > max_len:
            break
        summary += s + ". "
    return summary.strip()[:max_len]


def should_skip_title(title: str) -> bool:
    if title.startswith(("Wikipedia:", "Help:", "Category:", "File:", "Template:", "Portal:", "Special:", "User:", "Talk:", "MediaWiki:", "Draft:", "Module:")):
        return True
    if ":" in title and not title.startswith("List of"):
        return True
    return False


def main() -> int:
    if not WIKI_PATH.exists():
        print(f"ERROR: {WIKI_PATH} не найден")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"📖 Быстрый парс {WIKI_PATH.name}")
    print(f"   Лимит: {MAX_BYTES / 1e9:.1f} GB, {MAX_ARTICLES} статей")

    n_processed = 0
    n_saved = 0
    n_skipped = 0
    bytes_read = 0
    t0 = time.time()
    last_log = t0

    # Bytes mode - быстрее
    with bz2.open(WIKI_PATH, "rb") as f, \
         OUTPUT_PATH.open("w", encoding="utf-8") as out:

        chunk_size = 1024 * 1024  # 1 MB chunks
        buffer = b""

        while bytes_read < MAX_BYTES and n_saved < MAX_ARTICLES:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            bytes_read += len(chunk)
            buffer += chunk

            # Находим все <page>...</page> в buffer
            while b"</page>" in buffer:
                start = buffer.find(b"<page>")
                end = buffer.find(b"</page>") + len(b"</page>")
                if start == -1:
                    buffer = buffer[end:]
                    continue

                page_xml = buffer[start:end].decode("utf-8", errors="ignore")
                buffer = buffer[end:]

                # Извлекаем title
                title_match = re.search(r"<title>(.*?)</title>", page_xml, re.DOTALL)
                if not title_match:
                    continue

                title = title_match.group(1).strip()
                n_processed += 1

                if should_skip_title(title):
                    n_skipped += 1
                    continue

                # Извлекаем text
                text_match = re.search(r"<text[^>]*>(.*?)</text>", page_xml, re.DOTALL)
                if not text_match:
                    continue

                raw_text = text_match.group(1)
                summary = extract_summary(raw_text)
                if summary and len(summary) > 50:
                    out.write(json.dumps({
                        "title": title,
                        "summary": summary,
                        "lang": "en",
                    }, ensure_ascii=False) + "\n")
                    n_saved += 1

            # Лог каждые 30 сек
            if time.time() - last_log > 30:
                elapsed = time.time() - t0
                mb_read = bytes_read / 1e6
                gb_total = WIKI_PATH.stat().st_size / 1e9
                pct = (bytes_read / WIKI_PATH.stat().st_size) * 100
                rate = n_saved / elapsed * 60 if elapsed > 0 else 0
                print(f"  [{elapsed/60:.1f}min] {mb_read:.0f}MB / {gb_total:.1f}GB ({pct:.1f}%), saved={n_saved}, rate={rate:.0f}/min")
                last_log = time.time()

    elapsed = time.time() - t0
    print(f"\n✅ Готово за {elapsed/60:.1f} мин")
    print(f"   Processed: {n_processed}")
    print(f"   Saved: {n_saved}")
    print(f"   Skipped: {n_skipped}")
    print(f"   Output: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
