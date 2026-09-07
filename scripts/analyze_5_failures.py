"""Анализ 5 провалов автотеста Bridge Bot v11."""
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

import importlib
import inference.bridge_bot
importlib.reload(inference.bridge_bot)
bb = inference.bridge_bot

# Загружаем модель и KB
print("Loading...")
model, tokenizer, _ = bb.load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt",
    ROOT / "tokenizer/vocab.json"
)
kb = bb.KnowledgeBase(ROOT / "data/chat_merged.jsonl")
print(f"✓ Model loaded")
print(f"✓ KB: {len(kb.entries)} entries\n")

# 5 вопросов которые провалились в прошлом тесте
FAILED = [
    "100 умножить на 8",
    "15 процентов от 200",
    "Когда была вторая мировая война",
    "Кто такой Пушкин?",
    "Кто такой Бетховен?",
]

for q in FAILED:
    print(f"\n{'='*70}")
    print(f"❓ Q: {q}")
    print(f"{'='*70}")

    # ШАГ 1: hardcoded
    hardcoded = bb._try_hardcoded(q)
    if hardcoded:
        print(f"✅ [hardcoded]: {hardcoded}")
        continue

    # ШАГ 2: TF-IDF KB
    tfidf = bb._state.get("tfidf_kb") if hasattr(bb, "_state") else None
    if tfidf is None:
        # Создаём TF-IDF
        from inference.tfidf_kb import TfidfKB
        tfidf = TfidfKB.from_jsonl(ROOT / "data/chat_merged.jsonl")

    tfidf_result = tfidf.find(q, threshold=0.3)
    print(f"  [tfidf]: {tfidf_result[:200] if tfidf_result else 'NOT FOUND'}")

    # ШАГ 3: KB word overlap
    kb_result = kb.find(q, threshold=0.5)
    print(f"  [kb]: {kb_result[:200] if kb_result else 'NOT FOUND'}")

    # ШАГ 4: v94 (fallback)
    t0 = time.time()
    try:
        from inference.generate import generate_text
        few_shot = (
            "You are a calm AI. Reply briefly (1-2 sentences).\n"
            "Q: Hello!\nA: Hello!\n\n"
            "Q: "
        )
        v94_response = generate_text(
            model, tokenizer, prompt=few_shot + q,
            max_new_tokens=60, temperature=0.3,
            top_k=40, top_p=0.95, repetition_penalty=1.3,
            no_repeat_ngram_size=4, device="cpu"
        )
        v94_response = v94_response.split("\n")[0].strip()[:200]
    except Exception as e:
        v94_response = f"ERROR: {e}"
    elapsed = time.time() - t0
    print(f"  [v94] [{elapsed:.1f}s]: {v94_response}")

print("\n\n📊 Анализ провалов готов")
