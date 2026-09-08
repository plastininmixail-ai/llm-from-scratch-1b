"""Парсинг Wiki XML bz2 → jsonl."""
import sys
import bz2
import re
import json
from pathlib import Path

sys.path.insert(0, ".")


def parse_wiki_dump(bz2_path: Path, out_path: Path, max_articles: int = None):
    """Парсит Wiki dump в jsonl.

    Формат MediaWiki XML:
    <page>
      <title>...</title>
      <id>...</id>
      <revision>
        <text>...длинная статья...</text>
      </revision>
    </page>
    """
    print(f"Parsing {bz2_path.name}...")
    print(f"  size: {bz2_path.stat().st_size/1e9:.2f} GB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    text_count = 0

    # Regex для извлечения страниц
    page_pattern = re.compile(
        rb"<page>(.*?)</page>", re.DOTALL
    )
    title_pattern = re.compile(rb"<title>(.*?)</title>", re.DOTALL)
    text_pattern = re.compile(rb"<text[^>]*>(.*?)</text>", re.DOTALL)

    with bz2.open(bz2_path, "rb") as f, out_path.open("w", encoding="utf-8") as out:
        # Читаем блоками (XML большой)
        buffer = b""
        chunk_size = 10 * 1024 * 1024  # 10 MB

        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            buffer += chunk

            # Находим все page в buffer
            pages = page_pattern.findall(buffer)

            # Оставляем только последний неполный page в buffer
            last_page_end = buffer.rfind(b"</page>")
            if last_page_end != -1:
                buffer = buffer[last_page_end + len(b"</page>"):]
            else:
                buffer = b""

            for page in pages:
                if max_articles and count >= max_articles:
                    print(f"  Reached max_articles={max_articles}")
                    return count

                title_match = title_pattern.search(page)
                text_match = text_pattern.search(page)

                if not title_match or not text_match:
                    count += 1
                    continue

                title = title_match.group(1).decode("utf-8", errors="ignore").strip()
                text = text_match.group(1).decode("utf-8", errors="ignore").strip()

                # Пропускаем короткие
                if len(text) < 200:
                    count += 1
                    continue

                # Убираем Wiki markup (минимально)
                text = re.sub(r"\[\[[^|\]]*\|", "", text)  # [[link|text]] → text
                text = re.sub(r"\[\[", "", text)  # [[link]] → link
                text = re.sub(r"\{\{[^}]*\}\}", "", text)  # {{template}}
                text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"'''(.+?)'''", r"\1", text)  # bold
                text = re.sub(r"''(.+?)''", r"\1", text)  # italic
                text = re.sub(r"==+ *(.+?) *==+", r"\1", text)  # headers
                text = re.sub(r"\s+", " ", text).strip()

                if len(text) < 200:
                    count += 1
                    continue

                out.write(json.dumps({
                    "title": title,
                    "text": text,
                }, ensure_ascii=False) + "\n")

                count += 1
                text_count += 1

                if count % 10000 == 0:
                    print(f"  {count} pages, {text_count} with text")

    print(f"\nDone: {count} pages, {text_count} with text")
    print(f"Output: {out_path}")
    return text_count


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to .xml.bz2 file")
    p.add_argument("--output", required=True, help="Output .jsonl file")
    p.add_argument("--max-articles", type=int, default=None)
    args = p.parse_args()

    parse_wiki_dump(
        Path(args.input),
        Path(args.output),
        args.max_articles,
    )


if __name__ == "__main__":
    main()
