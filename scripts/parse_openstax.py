"""Парсит OpenStax PDF файлы и добавляет в KB."""
import json
import re
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
BOOKS_DIR = ROOT / "data/sources/openstax/books"
OUT = ROOT / "data/kb_openstax.jsonl"

# Проверяем pypdf
try:
    import pypdf
except ImportError:
    print("Installing pypdf...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pypdf", "--quiet"])
    import pypdf

print("=" * 60)
print("Парсинг OpenStax книг")
print("=" * 60)

pairs = []
pdf_files = list(BOOKS_DIR.glob("*.pdf"))
print(f"Найдено PDF: {len(pdf_files)}")

for pdf_path in pdf_files:
    try:
        size_mb = pdf_path.stat().st_size / 1024 / 1024
        if size_mb < 0.5:
            # Слишком маленький — наверное ошибка
            print(f"  ! {pdf_path.name}: {size_mb:.1f} MB (skip)")
            continue

        reader = pypdf.PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        print(f"  📖 {pdf_path.name}: {size_mb:.1f} MB, {num_pages} pages")

        # Берём первые 50 страниц (оглавление + введение)
        text_parts = []
        for i in range(min(50, num_pages)):
            try:
                page_text = reader.pages[i].extract_text()
                if page_text and len(page_text) > 100:
                    text_parts.append(page_text)
            except Exception:
                pass

        full_text = "\n\n".join(text_parts)[:15000]

        if len(full_text) > 500:
            title = pdf_path.stem
            pairs.append({
                "prompt": f"Расскажи про книгу {title}",
                "response": full_text,
            })
            print(f"    ✓ Извлечено {len(full_text)} chars")
    except Exception as e:
        print(f"  ! Error {pdf_path.name}: {e}")

print(f"\n📊 OpenStax записей: {len(pairs)}")

OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
print(f"✅ Сохранено в {OUT}")
