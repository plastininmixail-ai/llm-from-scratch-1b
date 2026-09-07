"""Сравнительный тест v94 vs v95."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.generate import generate_text, load_model_from_checkpoint

CKPT_V94 = ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt"
CKPT_V95 = ROOT / "checkpoints/chat-xlarge-v95-sft/best.pt"
TOK = ROOT / "tokenizer/vocab.json"

# Загружаем обе модели
print("Loading v94...")
m94, t94, _ = load_model_from_checkpoint(CKPT_V94, TOK)
print("Loading v95...")
m95, t95, _ = load_model_from_checkpoint(CKPT_V95, TOK)
print("✓ Both loaded\n")

PROMPTS = [
    ("RU", "Сколько будет 2+2?"),
    ("RU", "Что такое гравитация?"),
    ("RU", "Расскажи о себе"),
    ("EN", "What is gravity?"),
    ("EN", "Tell me about yourself"),
    ("EN", "How are you?"),
]

FEW_SHOT = (
    "You are a calm AI. Reply briefly (1-2 sentences).\n"
    "Q: Hello!\nA: Hello!\n\n"
    "Q: "
)

for lang, p in PROMPTS:
    print(f"\n{'='*60}")
    print(f"[{lang}] Q: {p}")
    print(f"{'='*60}")

    # v94
    t0 = time.time()
    r94 = generate_text(m94, t94, prompt=FEW_SHOT + p,
                         max_new_tokens=60, temperature=0.3,
                         top_k=40, top_p=0.95, repetition_penalty=1.3,
                         no_repeat_ngram_size=4, device="cpu")
    t94e = time.time() - t0
    r94 = r94.split("\n")[0].strip()[:200]

    # v95
    t0 = time.time()
    r95 = generate_text(m95, t95, prompt=FEW_SHOT + p,
                         max_new_tokens=60, temperature=0.3,
                         top_k=40, top_p=0.95, repetition_penalty=1.3,
                         no_repeat_ngram_size=4, device="cpu")
    t95e = time.time() - t0
    r95 = r95.split("\n")[0].strip()[:200]

    print(f"\n[v94] [{t94e:.1f}s]: {r94}")
    print(f"\n[v95] [{t95e:.1f}s]: {r95}")
