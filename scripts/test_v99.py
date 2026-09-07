"""Тест v99 — лучший чекпойнт. 15 вопросов разных категорий."""
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.generate import generate_text, load_model_from_checkpoint

print("Loading v99 (best model)...")
model, tokenizer, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v99-openstax/best.pt",
    ROOT / "tokenizer/vocab.json"
)
print(f"✓ Model loaded\n")

QUESTIONS = [
    ("RU", "Что такое искусственный интеллект?"),
    ("RU", "Расскажи про трансформеры"),
    ("RU", "Как работает градиентный спуск?"),
    ("RU", "Кто такой Пушкин?"),
    ("RU", "Что такое Вселенная?"),
    ("RU", "Как устроен Python?"),
    ("RU", "Что такое нейронная сеть?"),
    ("RU", "Расскажи про Большой взрыв"),
    ("RU", "Что такое OpenStax?"),
    ("RU", "Привет!"),
    ("EN", "What is machine learning?"),
    ("EN", "Tell me about Einstein"),
    ("EN", "How does a computer work?"),
    ("EN", "What is philosophy?"),
    ("EN", "Hi!"),
]

FEW_SHOT = (
    "You are a calm AI. Reply briefly (1-2 sentences).\n"
    "Q: Hello!\nA: Hello!\n\n"
    "Q: "
)

print("=" * 70)
print("v99 TEST: 15 вопросов")
print("=" * 70)

for lang, q in QUESTIONS:
    print(f"\n{'─'*70}")
    print(f"[{lang}] Q: {q}")

    t0 = time.time()
    try:
        response = generate_text(
            model, tokenizer,
            prompt=FEW_SHOT + q,
            max_new_tokens=80,
            temperature=0.3,
            top_k=40, top_p=0.95,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            device="cpu",
        )
        response = response.split("\n")[0].strip()[:250]
    except Exception as e:
        response = f"[ERROR: {e}]"

    elapsed = time.time() - t0
    print(f"\n[{elapsed:.1f}s] A: {response}")
