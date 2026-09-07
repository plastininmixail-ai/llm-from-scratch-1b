"""Парсит SEP articles2 + wiki API + crash course transcripts + openstax."""
import json
import re
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SOURCES = ROOT / "data/sources"


def clean_html(html: str) -> str:
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    html = re.sub(r"<[^>]+>", " ", html)
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", html).strip()


def extract_paragraphs(text: str, max_chars: int = 2500) -> str:
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
        last = text.rfind(".")
        if last > max_chars * 0.7:
            text = text[:last + 1]
    return text


all_pairs = []

# 1. SEP articles2
sep2_dir = SOURCES / "sep/articles2"
if sep2_dir.exists():
    for f in sep2_dir.glob("*.html"):
        try:
            text = clean_html(f.read_text(encoding="utf-8", errors="ignore"))
            text = extract_paragraphs(text, 2500)
            title = f.stem.replace("-", " ").title()
            all_pairs.append({
                "prompt": f"Что такое {title}?",
                "response": text,
            })
        except Exception as e:
            print(f"  ! {f.name}: {e}")
print(f"SEP articles2: добавлено")

# 2. Crash Course transcripts (если есть)
cc_file = ROOT / "data/kb_crashcourse_transcripts.jsonl"
if cc_file.exists():
    with cc_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                all_pairs.append(json.loads(line))
            except Exception:
                pass
    print(f"Crash Course: добавлено")

# 3. Wiki API
wiki_file = ROOT / "data/kb_wiki_api.jsonl"
if wiki_file.exists():
    with wiki_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                all_pairs.append(json.loads(line))
            except Exception:
                pass
    print(f"Wiki API: добавлено")

# Сохраняем объединённый
OUT = ROOT / "data/kb_round2.jsonl"
OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in all_pairs), encoding="utf-8")
print(f"\n📊 Всего Round 2: {len(all_pairs)} записей")
print(f"✅ Сохранено в {OUT}")
