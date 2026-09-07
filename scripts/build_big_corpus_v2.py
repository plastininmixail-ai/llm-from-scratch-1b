"""Собирает большой corpus из всех доступных данных.

Приоритет:
1. chat_v23 (700K пар, реальные диалоги) — ОСНОВА
2. Wiki 20K (факты)
3. SFT sources (instruction-following)
4. Hardcoded QA (точные ответы)
"""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
OUT = ROOT / "data/raw/big_corpus_v2.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

lines = []
total_chars = 0

# === 1. Chat v23 (700K пар, реальные диалоги) — ГЛАВНЫЙ источник ===
fpath = ROOT / "data/chat_v23.jsonl"
print(f"[1] Чтение {fpath.name}...")
count = 0
if fpath.exists():
    with fpath.open(encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
                prompt = p.get("prompt", "").strip()
                response = p.get("response", "").strip()
                if prompt and response and len(prompt) > 5 and len(response) > 5:
                    # ChatML формат
                    lines.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n")
                    count += 1
                    if count >= 500_000:  # Берём первые 500K
                        break
            except Exception:
                pass
print(f"   Добавлено: {count:,} пар")

# === 2. Wiki 20K (факты, длинный контекст) ===
fpath = ROOT / "data/wiki_summaries_full.jsonl"
print(f"[2] Чтение {fpath.name}...")
count = 0
if fpath.exists():
    with fpath.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 30_000:  # 30K Wiki статей
                break
            try:
                p = json.loads(line)
                prompt = p.get("prompt", "").strip()
                response = p.get("response", "").strip()
                if prompt and response:
                    lines.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n")
                    count += 1
            except Exception:
                pass
print(f"   Добавлено: {count:,} пар")

# === 3. SFT sources (instruction-following) ===
SFT_FILES = [
    "data/kb_philosophy.jsonl",
    "data/kb_arxiv.jsonl",
    "data/kb_misc.jsonl",
    "data/sft_fastai.jsonl",
    "data/kb_round2.jsonl",
    "data/kb_openstax.jsonl",
    "data/kb_misc_tools.jsonl",
]
print(f"[3] Чтение SFT sources...")
for fname in SFT_FILES:
    fpath = ROOT / fname
    count = 0
    if fpath.exists():
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
        print(f"   {fname}: {count} пар")

# Сохраняем
print(f"\n📝 Записываю в {OUT}...")
OUT.write_text("\n".join(lines), encoding="utf-8")

total_chars = len(OUT.read_text(encoding="utf-8"))
print(f"\n✅ Готово!")
print(f"   Всего пар: {len(lines):,}")
print(f"   Размер: {total_chars / 1024 / 1024:.1f} MB")
print(f"   Path: {OUT}")
