"""Парсит OpenStax API JSON — books внутри."""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SRC = ROOT / "data/sources/openstax/books/openstax_books.json"
OUT = ROOT / "data/kb_openstax.jsonl"

data = json.loads(SRC.read_text(encoding="utf-8"))
books = data.get("books", [])

pairs = []
for book in books:
    title = book.get("title", "")
    slug = book.get("slug", "")
    subjects = book.get("subjects", [])
    cover_color = book.get("cover_color", "")
    publish_date = book.get("publish_date", "")

    if not title:
        continue

    if isinstance(subjects, list):
        subjects = ", ".join(subjects)

    # Используем book_state или другие поля для контента
    book_state = book.get("book_state", "")
    response_parts = [f"Название: {title}"]
    if subjects:
        response_parts.append(f"Предметы: {subjects}")
    if publish_date:
        response_parts.append(f"Дата публикации: {publish_date}")
    if slug:
        response_parts.append(f"URL: https://openstax.org/books/{slug}/pages/1")

    pairs.append({
        "prompt": f"Расскажи про книгу OpenStax: {title}",
        "response": "\n".join(response_parts),
    })

print(f"📊 OpenStax книг: {len(pairs)}")

OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
print(f"✅ Сохранено в {OUT}")

# Топ-10 по алфавиту
print("\n📚 Топ-10 книг:")
for p in pairs[:10]:
    print(f"  - {p['prompt'].replace('Расскажи про книгу OpenStax: ', '')}")
