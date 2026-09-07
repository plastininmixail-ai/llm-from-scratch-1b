"""Автотест Bridge Bot v11: 30 вопросов разных категорий."""
import json
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

import importlib
import inference.bridge_bot
importlib.reload(inference.bridge_bot)

# Загружаем модуль
bb = inference.bridge_bot

# Загружаем модель и KB (для теста без Telegram)
model, tokenizer, _ = bb.load_model_from_checkpoint(
    ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt",
    ROOT / "tokenizer/vocab.json"
)
kb = bb.KnowledgeBase(ROOT / "data/chat_merged.jsonl")

# 30 вопросов
QUESTIONS = [
    # Math
    ("Сколько будет 2+2?", "math"),
    ("100 умножить на 8", "math"),
    ("Корень из 144", "math"),
    ("15 процентов от 200", "math"),
    ("Сколько будет 2 в степени 10?", "math"),
    # Geo
    ("Столица Японии?", "geo"),
    ("Самый большой океан?", "geo"),
    ("Самое глубокое озеро?", "geo"),
    ("Какая самая длинная река?", "geo"),
    # History
    ("Когда Гагарин полетел в космос?", "history"),
    ("Когда была вторая мировая война?", "history"),
    ("Когда распался СССР?", "history"),
    # Biology
    ("Сколько костей у человека?", "biology"),
    ("Что такое ДНК?", "biology"),
    ("Что такое фотосинтез?", "biology"),
    # Tech
    ("Что такое API?", "tech"),
    ("Что такое Docker?", "tech"),
    ("Что такое Python?", "tech"),
    ("Что такое Git?", "tech"),
    # People
    ("Кто такой Эйнштейн?", "people"),
    ("Кто такой Пушкин?", "people"),
    ("Кто такой Бетховен?", "people"),
    # Quick chat
    ("Привет!", "chat"),
    ("Как дела?", "chat"),
    ("Что ты умеешь?", "chat"),
    # Time
    ("Сколько времени?", "time"),
    ("Какое сегодня число?", "time"),
    # Other
    ("Сколько элементов в таблице Менделеева?", "chemistry"),
    ("Сколько планет в Солнечной системе?", "astronomy"),
    ("Что такое H₂O?", "chemistry"),
]

results = {"hardcoded": 0, "kb": 0, "v94": 0, "fallback": 0, "fail": 0}
log = []

for q, category in QUESTIONS:
    norm = bb._normalize(q)
    t0 = time.time()

    # ШАГ 1: hardcoded
    hardcoded = bb._try_hardcoded(q)

    if hardcoded:
        results["hardcoded"] += 1
        tool = "hardcoded"
        response = hardcoded
    else:
        # ШАГ 2: TF-IDF KB
        # (skip TF-IDF for test, just go to fallback)
        tool = "v94"
        response = "(KB / v94 / fallback)"

    elapsed = time.time() - t0
    status = "✅" if (hardcoded or response != "(KB / v94 / fallback)") else "🟡"
    log.append({
        "q": q,
        "category": category,
        "tool": tool,
        "response": response[:120] if isinstance(response, str) else str(response),
        "elapsed": round(elapsed, 3)
    })
    print(f"{status} [{tool:10}] [{elapsed:.2f}s] {q}: {response[:80] if isinstance(response, str) else response}")

print()
print("=" * 60)
print(f"📊 Hardcoded coverage: {results['hardcoded']}/{len(QUESTIONS)} = "
      f"{results['hardcoded']*100//len(QUESTIONS)}%")
print(f"⏱️  Avg time: {sum(l['elapsed'] for l in log)/len(log):.3f}s")

# Save log
LOG = ROOT / "data/test_log.jsonl"
with LOG.open("w", encoding="utf-8") as f:
    for entry in log:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"\n📝 Log saved to {LOG}")
