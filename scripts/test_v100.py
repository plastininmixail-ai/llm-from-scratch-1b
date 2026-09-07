"""Тест v100 — Wiki pretrained model."""
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.generate import generate_text, load_model_from_checkpoint

print("Loading v100 (Wiki pretrained)...")
model, tokenizer, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v100-wiki-20k/best.pt",
    ROOT / "tokenizer/vocab.json"
)
print("✓ Loaded\n")

QUESTIONS = [
    ("RU", "Привет!"),
    ("RU", "Что такое гравитация?"),
    ("RU", "Расскажи про космос"),
    ("RU", "Кто такой Эйнштейн?"),
    ("RU", "Что такое нейронная сеть?"),
    ("EN", "Hello!"),
    ("EN", "What is machine learning?"),
    ("EN", "Tell me about Einstein"),
]

print("=" * 70)
print("v100 TEST (Wiki pretrained)")
print("=" * 70)

for lang, q in QUESTIONS:
    print(f"\n[{lang}] Q: {q}")
    t0 = time.time()
    try:
        r = generate_text(
            model, tokenizer, prompt=q,
            max_new_tokens=80, temperature=0.3,
            top_k=40, top_p=0.95, repetition_penalty=1.3,
            no_repeat_ngram_size=4, device="cpu"
        )
        r = r.strip()[:200]
    except Exception as e:
        r = f"[ERR: {e}]"
    elapsed = time.time() - t0
    print(f"[{elapsed:.1f}s] A: {r}")
