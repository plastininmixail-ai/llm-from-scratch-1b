"""Объединяет все новые источники в один большой файл и добавляет в KB."""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")

NEW_FILES = [
    "data/kb_philosophy.jsonl",  # 36
    "data/kb_arxiv.jsonl",  # 50
    "data/kb_misc.jsonl",  # 11
    "data/sft_fastai.jsonl",  # 27
]

OUT_SFT = ROOT / "data/sft_sources_combined.jsonl"
OUT_KB = ROOT / "data/kb_sources_combined.jsonl"

sft_pairs = []
kb_pairs = []

for fname in NEW_FILES:
    fpath = ROOT / fname
    if not fpath.exists():
        continue
    with fpath.open(encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
                prompt = p.get("prompt", "")
                response = p.get("response", "")
                if prompt and response:
                    # В SFT — для обучения
                    sft_pairs.append({"prompt": prompt, "response": response})
                    # В KB — для поиска
                    kb_pairs.append({"prompt": prompt, "response": response})
            except Exception:
                pass

# Сохраняем
OUT_SFT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in sft_pairs), encoding="utf-8")
OUT_KB.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in kb_pairs), encoding="utf-8")

print(f"📊 SFT пар: {len(sft_pairs)} → {OUT_SFT}")
print(f"📊 KB пар: {len(kb_pairs)} → {OUT_KB}")
