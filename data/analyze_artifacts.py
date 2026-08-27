"""Анализ артефактов в очищенном тексте."""
import re
from collections import Counter
from pathlib import Path

text = Path("data/raw/corpus.txt").read_text(encoding="utf-8")
articles = text.split("\n\n")

# Ищем артефакты
patterns = [
    (r"</?ref[^>]*>", "ref теги"),
    (r"''+", "множественные '' (курсив)"),
    (r"\{\{[^{}]*\}\}", "двойные {{ }}"),
    (r"\{\{[^{}]*\}\}", "одиночные { }"),
    (r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", "wiki-ссылки"),
    (r"\[https?://[^\]]+\]", "URL-скобки"),
    (r"&[a-z]+;", "HTML entities"),
    (r"<[^>]+>", "HTML-теги"),
    (r"={2,}[^=]+={2,}", "заголовки ===="),
    (r"\[\s*[a-z]+\s*\|\s*[^\]]+\]", "pipe-ссылки"),
    (r"://[^\s]+", "URL без скобок"),
    (r"\.mw-[a-z-]+", "mediawiki-классы"),
    (r"\b[A-Z]{2,}[A-Z_0-9]+\b", "длинные акронимы"),
]

print(f"Всего статей: {len(articles)}")
total_chars = sum(len(a) for a in articles)
print(f"Всего символов: {total_chars:,}\n")

for pat, name in patterns:
    matches = re.findall(pat, " ".join(articles[:5000]))  # sample 5000
    cnt = len(matches)
    examples = list(set(matches))[:5]
    if cnt > 0:
        print(f"{name}: {cnt} в первых 5000 статей")
        for ex in examples:
            print(f"  '{ex[:60]}'")