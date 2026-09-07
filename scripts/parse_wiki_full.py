"""Парсит полный Wiki dump bz2 в KB.

Использует быстрый bytes-mode парсер.
"""
import bz2
import re
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
BZ2 = ROOT / "data/wiki/enwiki-latest-pages-articles-multistream.xml.bz2"
OUT = ROOT / "data/wiki_summaries_full.jsonl"

MAX_BYTES = 5 * 1024**3  # 5 GB (после распаковки ~50 GB)
MAX_ARTICLES = 100_000


def extract_text(xml: str) -> str:
    """Извлекает plain text из wiki XML."""
    # Remove templates
    xml = re.sub(r"\{\{[^{}]*\}\}", "", xml)
    # Remove tags
    xml = re.sub(r"<[^>]+>", " ", xml)
    # Decode entities
    xml = xml.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    xml = xml.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Normalize whitespace
    xml = re.sub(r"\s+", " ", xml)
    return xml.strip()


print("=" * 70)
print("Wiki dump parser (быстрый, bytes mode)")
print("=" * 70)
print(f"Input: {BZ2} ({BZ2.stat().st_size / 1024**3:.2f} GB)")
print(f"Output: {OUT}")
print(f"Max articles: {MAX_ARTICLES:,}")
print()

t0 = time.time()
articles_parsed = 0
articles_saved = 0
bytes_read = 0
last_status = time.time()

with bz2.open(BZ2, "rb") as f:
    with OUT.open("w", encoding="utf-8") as out:
        # Read in chunks
        chunk_size = 1 << 20  # 1 MB
        buf = b""

        while articles_parsed < MAX_ARTICLES and bytes_read < MAX_BYTES:
            try:
                chunk = f.read(chunk_size)
            except (OSError, EOFError):
                break
            if not chunk:
                break
            bytes_read += len(chunk)
            buf += chunk

            # Find pages
            while b"<page>" in buf and articles_parsed < MAX_ARTICLES:
                start = buf.index(b"<page>")
                end = buf.find(b"</page>", start)
                if end < 0:
                    break

                page_xml = buf[start:end + 7].decode("utf-8", errors="replace")
                buf = buf[end + 7:]

                # Extract title
                title_match = re.search(r"<title>([^<]+)</title>", page_xml)
                title = title_match.group(1) if title_match else "Unknown"
                articles_parsed += 1

                # Skip non-articles
                if ":" in title or "Wikipedia:" in title:
                    continue

                # Extract text
                text = extract_text(page_xml)
                if len(text) < 200:
                    continue

                # Truncate to first 2000 chars
                text = text[:2000]
                if "." in text[1500:2000]:
                    cut = text[1500:2000].rfind(".")
                    if cut > 100:
                        text = text[:1500 + cut + 1]

                # Save
                import json
                out.write(json.dumps({
                    "prompt": f"Расскажи про: {title}",
                    "response": text,
                }, ensure_ascii=False) + "\n")
                articles_saved += 1

                # Status every 5 sec
                if time.time() - last_status > 5:
                    elapsed = time.time() - t0
                    rate = articles_parsed / elapsed * 60
                    print(f"  [{elapsed:.0f}s] {articles_parsed:,} parsed, "
                          f"{articles_saved:,} saved, {rate:.0f}/min", flush=True)
                    last_status = time.time()

elapsed = time.time() - t0
print()
print("=" * 70)
print(f"✅ Done in {elapsed:.0f}s")
print(f"   Parsed: {articles_parsed:,}")
print(f"   Saved: {articles_saved:,}")
print(f"   Skipped: {articles_parsed - articles_saved:,}")
print(f"   Output: {OUT}")
print(f"   Size: {OUT.stat().st_size / 1024 / 1024:.1f} MB")
