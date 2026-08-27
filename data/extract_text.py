"""
Извлечение чистого текста из Wikipedia XML multistream dump.

Использование:
    python -m data.extract_text data/raw/enwiki-50mb.xml.bz2 data/raw/enwiki.txt
    python -m data.extract_text data/raw/ruwiki-50mb.xml.bz2 data/raw/ruwiki.txt

Файл формата multistream XML содержит <page>...</page> блоки;
внутри каждого есть <title> и <text>. Мы оставляем только текст,
отбрасываем шаблоны, разметку, ссылки.

Это самая примитивная версия — для серьёзного пайплайна нужен
WikiExtractor (https://github.com/attardi/wikiextractor), но для
50 МБ корпуса хватает такого.
"""
from __future__ import annotations

import argparse
import bz2
import re
import sys
from pathlib import Path


# Разметка Wikipedia, которую мы вырезаем
WIKI_PATTERNS = [
    re.compile(r"\{\{[^{}]*\}\}"),                 # {{шаблоны}}
    re.compile(r"\[\[([^|\]]*?\|)?([^\]]*?)\]\]"),  # [[ссылки]], [[link|text]]
    re.compile(r"'''([^']*?)'''"),                 # '''жирный-курсив'''
    re.compile(r"''([^']*?)''"),                     # ''курсив''
    re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL),  # <ref>...</ref>
    re.compile(r"<ref[^/]*/>"),                    # <ref ... />
    re.compile(r"<[^>]+>"),                        # любые HTML-теги
    re.compile(r"^\s*==+\s*.*?\s*==+\s*$", re.MULTILINE),  # == заголовки ==
    re.compile(r"^\s*\*+\s*", re.MULTILINE),       # маркеры списков
    re.compile(r"^\s*#+\s*", re.MULTILINE),        # нумерованные списки
    re.compile(r"&[a-z]+;"),                       # HTML entities
]


def clean_wiki_text(text: str) -> str:
    """Удаляет wiki-разметку, оставляет только plain text."""
    for pat in WIKI_PATTERNS:
        text = pat.sub("", text)
    # схлопываем пустые строки
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_from_file(xml_path: Path, out_path: Path) -> int:
    """Парсит bz2-файл с XML-дампами, пишет plain text построчно."""
    pages = 0
    bytes_written = 0

    with bz2.open(xml_path, mode="rt", encoding="utf-8") as src, \
         out_path.open("w", encoding="utf-8") as dst:
        # multistream — это конкатенация XML, читаем потоково
        buffer = []
        in_text = False
        in_page = False
        current_text_lines: list[str] = []

        for line in src:
            if "<page>" in line:
                in_page = True
                current_text_lines = []
                continue
            if "</page>" in line and in_page:
                in_page = False
                # собрали статью — обрабатываем и пишем
                article = "".join(current_text_lines)
                cleaned = clean_wiki_text(article)
                if len(cleaned) > 200:  # выкидываем слишком короткие
                    dst.write(cleaned + "\n\n")
                    pages += 1
                    bytes_written += len(cleaned) + 2
                continue
            if in_page and "<text" in line and ">" in line:
                in_text = True
                # пропускаем сам тег <text ...>
                line = line.split(">", 1)[1] if ">" in line else ""
            if in_page and "</text>" in line:
                in_text = False
                line = line.split("</text>")[0]
            if in_text and in_page:
                current_text_lines.append(line)

            if bytes_written > 50 * 1024 * 1024:  # 50 МБ лимит
                print(f"  [stop] достигнут лимит 50 МБ, {pages} страниц", file=sys.stderr)
                break

    return pages


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="путь к *.xml.bz2")
    p.add_argument("output", type=Path, help="куда писать .txt")
    args = p.parse_args()

    if not args.input.exists():
        print(f"нет файла: {args.input}", file=sys.stderr)
        return 1

    print(f"extract: {args.input} → {args.output}")
    n = extract_from_file(args.input, args.output)
    print(f"готово: {n} страниц → {args.output} ({args.output.stat().st_size / 1e6:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())