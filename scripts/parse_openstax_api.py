"""Парсит OpenStax API JSON для списка книг."""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SRC = ROOT / "data/sources/openstax/books/openstax_books.json"
OUT = ROOT / "data/kb_openstax.jsonl"

data = json.loads(SRC.read_text(encoding="utf-8"))

pairs = []
# OpenStax API возвращает dict с ключом "results" или list
items = data.get("results", data) if isinstance(data, dict) else data
if isinstance(items, dict):
    items = items.get("items", [])

for book in items[:30]:  # Top-30 книг
    title = book.get("title", "")
    slug = book.get("slug", "")
    subjects = book.get("subjects", [])
    desc = book.get("description", "")

    if not title:
        continue

    if isinstance(subjects, list):
        subjects = ", ".join([s.get("name", "") if isinstance(s, dict) else str(s) for s in subjects])

    response = f"{title}\n\nПредмет: {subjects}\n\n{desc[:1500]}"
    pairs.append({
        "prompt": f"Расскажи про книгу OpenStax: {title}",
        "response": response,
    })

print(f"📊 OpenStax книг: {len(pairs)}")

OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
print(f"✅ Сохранено в {OUT}")
