"""Парсит Arxiv XML, извлекает title/abstract."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
ARXIV = ROOT / "data/sources/arxiv/transformer_attention.xml"
OUT = ROOT / "data/kb_arxiv.jsonl"

print("=" * 60)
print("Парсинг Arxiv метаданных")
print("=" * 60)

ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
tree = ET.parse(ARXIV)
root = tree.getroot()

pairs = []
for entry in root.findall("atom:entry", ns):
    title_el = entry.find("atom:title", ns)
    summary_el = entry.find("atom:summary", ns)
    if title_el is None or summary_el is None:
        continue

    title = "".join(title_el.itertext()).strip()
    summary = "".join(summary_el.itertext()).strip()

    if not title or not summary:
        continue

    # Берём первый абзац аннотации
    summary_short = summary[:800].strip()
    if len(summary) > 800:
        last_period = summary_short.rfind(".")
        if last_period > 400:
            summary_short = summary_short[:last_period + 1]

    pairs.append({
        "prompt": f"Расскажи про статью: {title}",
        "response": f"{title}\n\n{summary_short}",
    })

print(f"📊 Найдено {len(pairs)} статей")

OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
print(f"✅ Сохранено в {OUT}")
