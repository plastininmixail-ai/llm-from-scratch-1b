"""Парсит Wikipedia API JSON и добавляет в KB."""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
WIKI_DIR = ROOT / "data/sources/wiki_api"
OUT = ROOT / "data/kb_wiki_api.jsonl"

pairs = []
for f in WIKI_DIR.glob("*.json"):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if "extract" not in data:
            continue
        title = data.get("title", f.stem.replace("_", " ").title())
        extract = data.get("extract", "")[:2000]
        pairs.append({
            "prompt": f"Что такое {title}?",
            "response": f"{title}\n\n{extract}",
        })
    except Exception as e:
        print(f"Error: {f.name}: {e}")

OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
print(f"✅ Wiki API: {len(pairs)} записей → {OUT}")
