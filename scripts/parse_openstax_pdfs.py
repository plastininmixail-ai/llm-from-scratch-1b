"""Парсит OpenStax PDF → текст → Q&A пары.

Берёт первые 50 страниц каждой книги (оглавление + введение).
"""
import json
import re
import sys
from pathlib import Path

try:
    import pypdf
except ImportError:
    print("Installing pypdf...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pypdf", "--quiet"])
    import pypdf

PDF_DIR = Path("C:/Users/mixai/Desktop/llm-from-scratch/data/sources/openstax_pdfs")
OUT = Path("C:/Users/mixai/Desktop/llm-from-scratch/data/openstax_textbooks.jsonl")

print(f"Scanning {PDF_DIR}...")
pdfs = sorted(PDF_DIR.glob("*.pdf"))
print(f"Found {len(pdfs)} PDFs")

pairs = []
total_pages = 0

for pdf_path in pdfs:
    title = pdf_path.stem
    print(f"\n📖 {title}...")
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        n_pages = len(reader.pages)
        total_pages += n_pages

        # Берём первые 30 страниц (оглавление + введение)
        text_parts = []
        for i in range(min(30, n_pages)):
            try:
                p = reader.pages[i].extract_text()
                if p and len(p) > 100:
                    text_parts.append(p)
            except Exception:
                pass

        full_text = "\n\n".join(text_parts)[:15000]
        if len(full_text) < 500:
            print(f"   Skip: too short ({len(full_text)} chars)")
            continue

        # Split into paragraphs (разделы учебника)
        paragraphs = re.split(r'\n\n+', full_text)
        for p_text in paragraphs:
            p_text = p_text.strip()
            if len(p_text) < 200 or len(p_text) > 3000:
                continue

            # Ищем заголовок (первая строка короткая)
            lines = p_text.split("\n")
            if len(lines) < 2:
                continue
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()

            if 5 < len(heading) < 100 and len(body) > 100:
                pairs.append({
                    "prompt": f"Расскажи про: {heading}",
                    "response": body[:1500],
                })

        print(f"   ✓ {n_pages} pages, {len(pairs)} total sections")

    except Exception as e:
        print(f"   ✗ Error: {e}")

print(f"\n✅ Total pairs: {len(pairs)}")
print(f"   Total pages processed: {total_pages}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"✅ Saved to {OUT}")
