"""Распаковывает Fast.ai курс и парсит notebooks → Q&A."""
import json
import re
import zipfile
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
ZIP = ROOT / "data/sources/fastai/course22.zip"
OUT = ROOT / "data/sft_fastai.jsonl"
EXTRACT_DIR = ROOT / "data/sources/fastai/course22"

if not EXTRACT_DIR.exists():
    print("Распаковываю...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"✓ Распаковано в {EXTRACT_DIR}")


def parse_notebook(ipynb_path: Path) -> str:
    """Парсит .ipynb файл, извлекает markdown + code."""
    try:
        with ipynb_path.open(encoding="utf-8", errors="ignore") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        parts = []
        for cell in cells:
            ctype = cell.get("cell_type", "")
            source = cell.get("source", [])
            if isinstance(source, list):
                source = "".join(source)
            if ctype == "markdown":
                parts.append(f"\n## {source[:500]}\n")
            elif ctype == "code":
                parts.append(f"\n```python\n{source[:500]}\n```\n")
        return "\n".join(parts)[:3000]
    except Exception as e:
        return f"[Error: {e}]"


def parse_markdown(md_path: Path) -> str:
    """Парсит .md файл."""
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        return text[:3000]
    except Exception:
        return ""


print("=" * 60)
print("Парсинг Fast.ai курса → SFT")
print("=" * 60)

pairs = []
files_processed = 0

for nb_path in EXTRACT_DIR.rglob("*.ipynb"):
    title = nb_path.stem.replace("_", " ").replace("-", " ").title()
    content = parse_notebook(nb_path)
    if len(content) < 100:
        continue
    pairs.append({
        "prompt": f"Объясни тему: {title}",
        "response": content,
    })
    files_processed += 1
    if files_processed >= 30:
        break

for md_path in EXTRACT_DIR.rglob("*.md"):
    title = md_path.stem.replace("_", " ").replace("-", " ").title()
    content = parse_markdown(md_path)
    if len(content) < 100:
        continue
    pairs.append({
        "prompt": f"Расскажи про {title}",
        "response": content,
    })
    files_processed += 1
    if files_processed >= 50:
        break

print(f"✓ Обработано файлов: {files_processed}")
print(f"📊 SFT пар: {len(pairs)}")

# Сохраняем
OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
print(f"✅ Сохранено в {OUT}")
