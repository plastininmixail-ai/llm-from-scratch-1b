"""Тест v97 (302M SFT) vs v94 (114M SFT) vs v82 (114M base)."""
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.generate import generate_text, load_model_from_checkpoint

# Загружаем все 3 модели
print("Loading v97 (302M SFT)...")
m97, t97, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xxlarge-v97-sft/best.pt",
    ROOT / "tokenizer/vocab.json"
)
print("Loading v94 (114M SFT)...")
m94, t94, _ = load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt",
    ROOT / "tokenizer/vocab.json"
)
print("✓ Models loaded\n")

PROMPTS = [
    ("RU", "Что такое стоицизм?"),
    ("RU", "Кто такой Марк Аврелий?"),
    ("RU", "Расскажи о себе"),
    ("EN", "What is stoicism?"),
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

    for name, m, t in [("v97-302M", m97, t97), ("v94-114M", m94, t94)]:
        t0 = time.time()
        r = generate_text(m, t, prompt=FEW_SHOT + p,
                         max_new_tokens=60, temperature=0.3,
                         top_k=40, top_p=0.95, repetition_penalty=1.3,
                         no_repeat_ngram_size=4, device="cpu")
        elapsed = time.time() - t0
        r = r.split("\n")[0].strip()[:200]
        print(f"\n[{name}] [{elapsed:.1f}s]: {r}")
