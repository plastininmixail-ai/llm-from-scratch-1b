"""Парсит HTML файлы misc_tools и добавляет в KB."""
import json
import re
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
DIR = ROOT / "data/sources/misc_tools"
OUT = ROOT / "data/kb_misc_tools.jsonl"


def clean_html(html: str) -> str:
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    html = re.sub(r"<[^>]+>", " ", html)
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", html).strip()


pairs = []
for f in DIR.glob("*.html"):
    try:
        text = clean_html(f.read_text(encoding="utf-8", errors="ignore"))
        if len(text) < 200:
            continue
        text = text[:2000]
        # Извлекаем title из slug
        title = f.stem.replace("_", ".").replace("..", ".")
        pairs.append({
            "prompt": f"Расскажи про сайт {title}",
            "response": text,
        })
    except Exception as e:
        print(f"  ! {f.name}: {e}")

OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
print(f"✅ Misc tools: {len(pairs)} записей → {OUT}")
