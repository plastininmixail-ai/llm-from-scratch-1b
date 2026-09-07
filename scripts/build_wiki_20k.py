"""Берёт первые 20K Wiki статей для быстрого обучения."""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
INPUT = ROOT / "data/wiki_summaries_full.jsonl"
OUT_TXT = ROOT / "data/raw/wiki_corpus_20k.txt"

lines = []
with INPUT.open(encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20000:
            break
        try:
            p = json.loads(line)
            prompt = p.get("prompt", "")
            response = p.get("response", "")
            if prompt and response:
                lines.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n")
        except Exception:
            pass

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
print(f"✅ Wiki 20K corpus: {len(lines)} строк, {len(OUT_TXT.read_text(encoding='utf-8'))} chars")
