"""Тест DPO модели: 10 вопросов НЕ из датасета.

Сравнивает v94 vs DPO.
"""
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.generate import generate_text, load_model_from_checkpoint

print("Loading v94...")
v94, t94, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt",
    ROOT / "tokenizer/vocab.json"
)
print("Loading DPO...")
dpo, t_dpo, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v94-dpo/best.pt",
    ROOT / "tokenizer/vocab.json"
)
print("✓ Both models loaded\n")

# Вопросы, которых НЕТ в DPO датасете
TEST_QUERIES = [
    "Привет, как дела?",
    "Что ты думаешь о погоде?",
    "Расскажи анекдот",
    "Какой твой любимый цвет?",
    "Что такое счастье?",
    "Как ты провёл выходные?",
    "Что бы ты посоветовал мне почитать?",
    "Опиши идеальный день",
    "Что тебя вдохновляет?",
    "Как справиться со стрессом?",
]

FEW_SHOT = (
    "You are a calm AI. Reply briefly (1-2 sentences).\n"
    "Q: Hello!\nA: Hello!\n\n"
    "Q: "
)

print("=" * 70)
print("DPO Test: 10 вопросов не из датасета")
print("=" * 70)

for q in TEST_QUERIES:
    print(f"\n❓ Q: {q}")
    print("-" * 50)

    for name, m, t in [("v94 ", v94, t94), ("DPO ", dpo, t_dpo)]:
        t0 = time.time()
        r = generate_text(m, t, prompt=FEW_SHOT + q,
                         max_new_tokens=80, temperature=0.7,
                         top_k=50, top_p=0.95, repetition_penalty=1.2,
                         no_repeat_ngram_size=3, device="cpu")
        elapsed = time.time() - t0
        r = r.split("\n")[0].strip()[:200]
        print(f"  [{name}] [{elapsed:.1f}s] {r}")
