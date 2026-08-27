"""Sanity-check: показать первые примеры статей EN, RU и кода из корпуса."""
from pathlib import Path
import random

random.seed(42)

text = Path("data/raw/corpus.txt").read_text(encoding="utf-8")
articles = [a for a in text.split("\n\n") if a.strip()]

print(f"всего статей: {len(articles)}\n")

# EN sample
for a in articles:
    if all(ord(c) < 1024 for c in a if c.isalpha()):
        print("--- пример EN (wiki) ---")
        print(a[:300])
        break

# RU sample
for a in articles:
    if any(ord(c) > 1024 for c in a):
        print("\n--- пример RU (wiki) ---")
        print(a[:300])
        break

# CODE sample
for a in articles:
    if "def " in a or "class " in a or "function " in a:
        print("\n--- пример CODE (Python) ---")
        print(a[:300])
        break

print(f"\n--- итог ---")
print(f"EN: {sum(1 for a in articles if all(ord(c) < 1024 for c in a if c.isalpha()))}")
print(f"RU: {sum(1 for a in articles if any(ord(c) > 1024 for c in a))}")
print(f"code: {sum(1 for a in articles if 'def ' in a or 'class ' in a or 'function ' in a or 'import ' in a[:200])}")