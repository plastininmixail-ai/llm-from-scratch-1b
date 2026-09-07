"""Собирает большой корпус из всех доступных текстов для BPE обучения."""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
OUT = ROOT / "data/raw/big_corpus.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

texts = []

# 1. SFT combined
fpath = ROOT / "data/sft_combined.jsonl"
if fpath.exists():
    with fpath.open(encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
                texts.append(p.get("prompt", ""))
                texts.append(p.get("response", ""))
            except Exception:
                pass

# 2. Chat v23 (большой, берём только часть)
fpath = ROOT / "data/chat_v23.jsonl"
if fpath.exists():
    with fpath.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50000:  # 50K пар хватит
                break
            try:
                p = json.loads(line)
                texts.append(p.get("prompt", ""))
                texts.append(p.get("response", ""))
            except Exception:
                pass

# 3. Round 3 corpus
fpath = ROOT / "data/raw/round3_corpus.txt"
if fpath.exists():
    texts.append(fpath.read_text(encoding="utf-8"))

# 4. OpenStax
fpath = ROOT / "data/kb_openstax.jsonl"
if fpath.exists():
    with fpath.open(encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
                texts.append(p.get("prompt", ""))
                texts.append(p.get("response", ""))
            except Exception:
                pass

# 5. Wiki summaries (10K)
fpath = ROOT / "data/wiki_summaries.jsonl"
if fpath.exists():
    with fpath.open(encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
                texts.append(p.get("prompt", ""))
                texts.append(p.get("response", ""))
            except Exception:
                pass

# Сохраняем
text = "\n\n".join(t for t in texts if t)
OUT.write_text(text, encoding="utf-8")
print(f"✅ Corpus: {len(text)} chars, {len(texts)} документов")
print(f"   Path: {OUT}")
