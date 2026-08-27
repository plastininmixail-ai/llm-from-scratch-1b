"""
Извлечение чистого текста из Wikipedia XML multistream dump.

Использование:
    python -m data.extract_text data/raw/enwiki-50mb.xml.bz2 data/raw/enwiki.txt
    python -m data.extract_text data/raw/ruwiki-50mb.xml.bz2 data/raw/ruwiki.txt

Парсит bz2-файл с конкатенацией <page>...</page> блоков, оставляет только
plain text без wiki-разметки.
"""
from __future__ import annotations

import argparse
import bz2
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------- #
# Рекурсивные регулярки для вложенных шаблонов
# ---------------------------------------------------------------------------- #
TEMPLATE_RE = re.compile(
    r"\{\{(?:[^{}]|\{\{[^{}]*\}\})*\}\}",
    re.DOTALL,
)
WIKI_LINK_RE = re.compile(r"\[\[([^|\]\n]*?\|)?([^\]\n]*?)\]\]")
EXT_LINK_RE = re.compile(r"\[(https?|ftp)://[^\]\s]+\s+[^\]]+\]")
BARE_URL_RE = re.compile(r"https?://[^\s<>\"'\}\)\]]+")
REF_TAG_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
REF_SELF_RE = re.compile(r"<ref[^/]*/>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?[a-z][^>]*>", re.IGNORECASE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_RE = re.compile(r"^=+\s*[^=\n]+\s*=+$", re.MULTILINE)
LIST_RE = re.compile(r"^\s*[*#:;]+\s*", re.MULTILINE)
TABLE_RE = re.compile(r"\{\|[^{}]*\|\}", re.DOTALL)
BOLD_ITAL_RE = re.compile(r"'''([^']*?)'''|''([^']*?)''")
ENTITY_RE = re.compile(r"&[a-z]+;|&#\d+;")
MAGIC_WORD_RE = re.compile(r"__[A-Z]+__")
NOWIKI_RE = re.compile(r"<nowiki>.*?</nowiki>", re.DOTALL | re.IGNORECASE)


def clean_wiki_text(text: str) -> str:
    """Полная очистка wiki-разметки → plain text."""
    # 1) комментарии и nowiki
    text = COMMENT_RE.sub("", text)
    text = NOWIKI_RE.sub("", text)

    # 2) таблицы (до шаблонов, чтобы не ломать баланс скобок)
    text = TABLE_RE.sub("", text)

    # 3) рекурсивные шаблоны {{...}} — нужно несколько проходов
    prev = None
    while prev != text:
        prev = text
        text = TEMPLATE_RE.sub("", text)

    # 4) ссылки и внешние URL
    text = EXT_LINK_RE.sub("", text)
    text = WIKI_LINK_RE.sub(r"\2", text)
    text = BARE_URL_RE.sub("", text)

    # 5) ref-теги и HTML
    text = REF_TAG_RE.sub("", text)
    text = REF_SELF_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)

    # 6) markdown wiki-разметка
    text = HEADING_RE.sub("", text)
    text = BOLD_ITAL_RE.sub(r"\1\2", text)
    text = LIST_RE.sub("", text)

    # 7) magic words и entities
    text = MAGIC_WORD_RE.sub("", text)
    text = ENTITY_RE.sub(" ", text)

    # 8) финальная уборка
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def extract_from_file(xml_path: Path, out_path: Path, max_bytes: int = 50 * 1024 * 1024) -> int:
    """Парсит bz2-файл, пишет очищенные статьи."""
    pages = 0
    bytes_written = 0
    current_text: list[str] = []
    in_text = False
    in_page = False

    with bz2.open(xml_path, mode="rt", encoding="utf-8") as src, \
         out_path.open("w", encoding="utf-8") as dst:

        for line in src:
            if "<page>" in line:
                in_page = True
                current_text = []
                continue

            if "</page>" in line:
                if current_text:
                    article = "".join(current_text)
                    cleaned = clean_wiki_text(article)
                    if 200 < len(cleaned) < 50000:
                        dst.write(cleaned + "\n\n")
                        pages += 1
                        bytes_written += len(cleaned) + 2
                in_page = False
                in_text = False
                current_text = []
                if bytes_written >= max_bytes:
                    break
                continue

            if not in_page:
                continue

            # начало <text ...> тега
            if "<text" in line and ">" in line and not in_text:
                in_text = True
                idx = line.find(">")
                line = line[idx + 1:]

            # конец </text>
            if in_text and "</text>" in line:
                idx = line.find("</text>")
                line = line[:idx]
                in_text = False

            if in_text:
                current_text.append(line)

    return pages


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-mb", type=int, default=50)
    args = p.parse_args()

    if not args.input.exists():
        print(f"нет файла: {args.input}", file=sys.stderr)
        return 1

    print(f"extract: {args.input} → {args.output}")
    n = extract_from_file(args.input, args.output, max_bytes=args.max_mb * 1024 * 1024)
    print(f"готово: {n} страниц → {args.output} ({args.output.stat().st_size / 1e6:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())