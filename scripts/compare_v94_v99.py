"""Сравнительное тестирование v94 vs v99."""
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.generate import generate_text, load_model_from_checkpoint

print("Loading models...")
v94, t94, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt",
    ROOT / "tokenizer/vocab.json"
)
v99, t99, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v99-openstax/best.pt",
    ROOT / "tokenizer/vocab.json"
)
print("✓ Both loaded\n")

QUESTIONS = [
    "Что такое нейронная сеть?",
    "Кто такой Пушкин?",
    "Расскажи про Большой взрыв",
    "Привет!",
    "Что такое ИИ?",
]

FEW_SHOT = (
    "You are a calm AI. Reply briefly (1-2 sentences).\n"
    "Q: Hello!\nA: Hello!\n\n"
    "Q: "
)

print("=" * 70)
print("v94 vs v99")
print("=" * 70)

for q in QUESTIONS:
    print(f"\n{'─'*70}")
    print(f"Q: {q}")

    for name, m, t in [("v94", v94, t94), ("v99", v99, t99)]:
        t0 = time.time()
        try:
            response = generate_text(
                m, t,
                prompt=FEW_SHOT + q,
                max_new_tokens=80,
                temperature=0.4,
                top_k=40, top_p=0.95,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                device="cpu",
            )
            response = response.split("\n")[0].strip()[:180]
        except Exception as e:
            response = f"[ERR: {e}]"
        elapsed = time.time() - t0
        print(f"\n[{name}] [{elapsed:.1f}s] {response}")
