"""Тест 302M модели на качество генерации."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.generate import generate_text, load_model_from_checkpoint

CKPT_XXL = ROOT / "checkpoints/chat-xxlarge-v1/best.pt"
CKPT_XL = ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt"
TOK = ROOT / "tokenizer/vocab.json"

print("Loading xxlarge (302M)...")
m_xxl, t_xxl, _ = load_model_from_checkpoint(CKPT_XXL, TOK)
print("Loading xlarge (114M)...")
m_xl, t_xl, _ = load_model_from_checkpoint(CKPT_XL, TOK)
print("✓ Both loaded\n")

PROMPTS = [
    ("RU", "Сколько будет 2+2?"),
    ("RU", "Что такое гравитация?"),
    ("RU", "Расскажи о себе"),
    ("EN", "What is gravity?"),
    ("EN", "Tell me about yourself"),
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

    # xxlarge
    t0 = time.time()
    r_xxl = generate_text(m_xxl, t_xxl, prompt=FEW_SHOT + p,
                          max_new_tokens=60, temperature=0.3,
                          top_k=40, top_p=0.95, repetition_penalty=1.3,
                          no_repeat_ngram_size=4, device="cpu")
    t_xxle = time.time() - t0
    r_xxl = r_xxl.split("\n")[0].strip()[:200]

    # xlarge
    t0 = time.time()
    r_xl = generate_text(m_xl, t_xl, prompt=FEW_SHOT + p,
                         max_new_tokens=60, temperature=0.3,
                         top_k=40, top_p=0.95, repetition_penalty=1.3,
                         no_repeat_ngram_size=4, device="cpu")
    t_xle = time.time() - t0
    r_xl = r_xl.split("\n")[0].strip()[:200]

    print(f"\n[xxl 302M] [{t_xxle:.1f}s]: {r_xxl}")
    print(f"\n[xL  114M] [{t_xle:.1f}s]: {r_xl}")
