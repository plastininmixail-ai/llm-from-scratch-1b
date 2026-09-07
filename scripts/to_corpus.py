"""Конвертирует jsonl в один txt файл для chat_train_xlarge."""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")

INPUT_FILES = [
    "data/kb_philosophy.jsonl",
    "data/kb_arxiv.jsonl",
    "data/kb_misc.jsonl",
    "data/sft_fastai.jsonl",
    "data/kb_round2.jsonl",
]

OUT_TXT = ROOT / "data/raw/sources_corpus.txt"
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

lines = []
for fname in INPUT_FILES:
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
                    # ChatML формат
                    lines.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n")
            except Exception:
                pass

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
print(f"✅ Создан {OUT_TXT}")
print(f"   Строк: {len(lines)}")
print(f"   Размер: {len(OUT_TXT.read_text(encoding='utf-8'))} chars")
