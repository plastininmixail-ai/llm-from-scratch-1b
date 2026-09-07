"""Парсит SEP и IEP HTML файлы, извлекает текст и конвертирует в Q&A пары для KB."""
import json
import re
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SOURCES = ROOT / "data/sources"
OUT = ROOT / "data/kb_philosophy.jsonl"


def clean_html(html: str) -> str:
    """Убирает HTML теги, оставляет текст."""
    # Убираем скрипты и стили
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    # Убираем HTML теги
    html = re.sub(r"<[^>]+>", " ", html)
    # Декодируем HTML entities
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Убираем множественные пробелы
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def extract_first_paragraphs(text: str, max_chars: int = 2000) -> str:
    """Берёт первые абзацы статьи (без меню)."""
    # Берём текст после первого большого блока
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
        # Обрезаем по последней точке
        last_period = text.rfind(".")
        if last_period > max_chars * 0.7:
            text = text[:last_period + 1]
    return text


def make_qa(title: str, content: str) -> dict:
    """Создаёт Q&A пару."""
    # Question: "Что такое {title}?" или "Расскажи о {title}"
    question = f"Что такое {title}? Расскажи о {title}."
    response = content[:1500]  # Ограничиваем длину
    return {
        "prompt": question,
        "response": response,
    }


def parse_sep():
    """Парсит SEP статьи."""
    sep_dir = SOURCES / "sep/articles"
    pairs = []
    if not sep_dir.exists():
        return pairs

    for f in sep_dir.glob("*.html"):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            text = clean_html(content)
            text = extract_first_paragraphs(text, 3000)
            # Title — имя файла без расширения
            title = f.stem.replace("-", " ").title()
            pairs.append(make_qa(title, text))
        except Exception as e:
            print(f"  ! Error parsing {f.name}: {e}")
    return pairs


def parse_iep():
    """Парсит IEP статьи."""
    iep_dir = SOURCES / "iep"
    pairs = []
    if not iep_dir.exists():
        return pairs

    for f in iep_dir.glob("*.html"):
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
            text = clean_html(content)
            text = extract_first_paragraphs(text, 3000)
            title = f.stem.replace("-", " ").title()
            pairs.append(make_qa(title, text))
        except Exception as e:
            print(f"  ! Error parsing {f.name}: {e}")
    return pairs


print("=" * 60)
print("Парсинг философских источников → KB")
print("=" * 60)

all_pairs = []

print("\n[1/2] Парсинг SEP...")
sep_pairs = parse_sep()
print(f"  ✓ SEP: {len(sep_pairs)} статей")
all_pairs.extend(sep_pairs)

print("\n[2/2] Парсинг IEP...")
iep_pairs = parse_iep()
print(f"  ✓ IEP: {len(iep_pairs)} статей")
all_pairs.extend(iep_pairs)

print(f"\n📊 Всего: {len(all_pairs)} Q&A пар")

# Сохраняем
OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in all_pairs), encoding="utf-8")
print(f"✅ Сохранено в {OUT}")
