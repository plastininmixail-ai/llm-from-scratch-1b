"""Парсит остальные источники: Crash Course, D2L, Teachyourselfcs, Roadmap.sh."""
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


def extract_videos(html: str) -> list[dict]:
    """Извлекает видео из YouTube playlist HTML."""
    pairs = []
    # Ищем заголовки видео в playlist
    titles = re.findall(r'"title":\s*\{\s*"runs":\s*\[\s*\{\s*"text":\s*"([^"]+)"', html)
    for i, t in enumerate(titles[:20]):
        pairs.append({
            "prompt": f"Видео: {t}",
            "response": f"Видео из плейлиста Crash Course по AI: {t}",
        })
    return pairs


def extract_text(html: str, max_chars: int = 2000) -> str:
    text = clean_html(html)
    return text[:max_chars]


all_pairs = []

# 1. Crash Course
cc_html = (SOURCES / "crashcourse/ai_playlist.html").read_text(encoding="utf-8", errors="ignore")
videos = extract_videos(cc_html)
all_pairs.extend(videos)
print(f"✓ Crash Course: {len(videos)} видео")

# 2. D2L
d2l_html = (SOURCES / "d2l/d2l_index.html").read_text(encoding="utf-8", errors="ignore")
text = extract_text(d2l_html, 3000)
all_pairs.append({
    "prompt": "Что такое Dive into Deep Learning?",
    "response": text,
})
print(f"✓ D2L: 1 запись")

# 3. Roadmap.sh
roadmap_dir = SOURCES / "roadmap"
for f in roadmap_dir.glob("*.html"):
    title = f.stem.replace("_", " ").title()
    text = extract_text(f.read_text(encoding="utf-8", errors="ignore"), 1500)
    all_pairs.append({
        "prompt": f"Расскажи про roadmap: {title}",
        "response": text,
    })
print(f"✓ Roadmap.sh: {len(list(roadmap_dir.glob('*.html')))} записей")

# 4. Teachyourselfcs
tycs = (SOURCES / "teachyourselfcs/index.html").read_text(encoding="utf-8", errors="ignore")
text = extract_text(tycs, 3000)
all_pairs.append({
    "prompt": "Какие книги по CS рекомендуются?",
    "response": text,
})
print(f"✓ Teachyourselfcs: 1 запись")

# 5. Distill
distill_html = (SOURCES / "distill/attention.html").read_text(encoding="utf-8", errors="ignore")
text = extract_text(distill_html, 3000)
all_pairs.append({
    "prompt": "Что такое attention в нейросетях?",
    "response": text,
})
print(f"✓ Distill: 1 запись")

# Сохраняем
OUT = ROOT / "data/kb_misc.jsonl"
OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in all_pairs), encoding="utf-8")
print(f"\n📊 Всего: {len(all_pairs)} пар")
print(f"✅ Сохранено в {OUT}")
