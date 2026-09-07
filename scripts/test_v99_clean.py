"""Чистый тест v99 — без few-shot промпта."""
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.generate import generate_text, load_model_from_checkpoint

print("Loading v99...")
v99, t99, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v99-openstax/best.pt",
    ROOT / "tokenizer/vocab.json"
)
print("✓ Loaded\n")

# Без few-shot, простой промпт
QUESTIONS = [
    "Привет!",
    "Что такое гравитация?",
    "Кто ты?",
    "Сколько будет 2+2?",
]

print("=" * 70)
print("v99 без few-shot")
print("=" * 70)

for q in QUESTIONS:
    print(f"\nQ: {q}")
    t0 = time.time()
    try:
        response = generate_text(
            v99, t99,
            prompt=q,  # Без few-shot!
            max_new_tokens=60,
            temperature=0.3,
            top_k=40, top_p=0.95,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            device="cpu",
        )
        response = response.strip()[:200]
    except Exception as e:
        response = f"[ERR: {e}]"
    elapsed = time.time() - t0
    print(f"[{elapsed:.1f}s] A: {response}")
