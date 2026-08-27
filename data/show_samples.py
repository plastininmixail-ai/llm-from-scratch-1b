"""Sanity-check: показать первые примеры статей EN и RU из корпуса."""
from pathlib import Path

text = Path("data/raw/corpus.txt").read_text(encoding="utf-8")
articles = [a for a in text.split("\n\n") if a.strip()]

print(f"всего статей: {len(articles)}\n")
print("--- пример EN ---")
for a in articles:
    if all(ord(c) < 1024 for c in a if c.isalpha()):
        print(a[:300])
        break

print("\n--- пример RU ---")
for a in articles:
    if any(ord(c) > 1024 for c in a):
        print(a[:300])
        break