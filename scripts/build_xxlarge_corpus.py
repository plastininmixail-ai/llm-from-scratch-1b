"""Объединяет все данные для обучения xxlarge_v2 (300M)."""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
OUT = ROOT / "data/raw/xxlarge_corpus.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Все источники
SOURCES = [
    "data/chat_v23.jsonl",  # 700K пар
    "data/wiki_summaries_full.jsonl",  # 94K Wiki
    "data/openstax_textbooks.jsonl",  # 145 OpenStax
    "data/sft_fastai.jsonl",
    "data/kb_round2.jsonl",
    "data/kb_openstax.jsonl",
    "data/kb_misc_tools.jsonl",
    "data/kb_philosophy.jsonl",
    "data/kb_arxiv.jsonl",
    "data/kb_misc.jsonl",
]

lines = []
for fname in SOURCES:
    fpath = ROOT / fname
    if not fpath.exists():
        print(f"Skip: {fname}")
        continue
    count = 0
    with fpath.open(encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
                prompt = p.get("prompt", "").strip()
                response = p.get("response", "").strip()
                if prompt and response:
                    lines.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n")
                    count += 1
            except Exception:
                pass
    print(f"{fname}: {count}")

print(f"\nTotal: {len(lines)} pairs")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved to {OUT}: {len(OUT.read_text(encoding='utf-8'))} chars")
