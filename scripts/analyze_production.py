"""Анализ production_logs.jsonl после тестов.

Генерирует тестовые данные + анализирует.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
LOG = ROOT / "data/production_logs.jsonl"

# Симулируем 50 запросов для теста (если лог пуст)
TEST_QUERIES = [
    ("Привет!", "dialog", "positive", False),
    ("Как дела?", "dialog", "neutral", True),
    ("Сколько будет 2+2?", "code", "neutral", True),
    ("Курс Bitcoin", "web", "neutral", False),
    ("Кто такой Эйнштейн?", "kb", "neutral", True),
    ("Расскажи о себе", "dialog", "neutral", False),
    ("Спасибо большое!", "dialog", "positive", False),
    ("Нарисуй кота", "image", "neutral", False),
    ("Сыграй мелодию", "music", "neutral", False),
    ("Что такое Python?", "kb", "neutral", True),
    ("Ненавижу этот день", "dialog", "negative", False),
    ("Мне грустно", "dialog", "negative", False),
    ("Что нового в мире?", "web", "neutral", True),
    ("Сколько костей у человека?", "kb", "neutral", True),
    ("Когда Гагарин полетел?", "kb", "neutral", True),
    ("100 умножить на 8", "code", "neutral", False),
    ("15 процентов от 200", "code", "neutral", False),
    ("Привет, как дела?", "dialog", "neutral", True),
    ("Что такое API?", "kb", "neutral", True),
    ("Сколько планет в Солнечной системе?", "kb", "neutral", True),
    ("Спасибо за помощь!", "dialog", "positive", False),
    ("Бесит этот бот", "dialog", "negative", False),
    ("Расскажи анекдот", "dialog", "neutral", False),
    ("Кто такой Пушкин?", "kb", "neutral", True),
    ("Где находится Эйфелева башня?", "kb", "neutral", True),
    ("Напиши код факториала", "code", "neutral", False),
    ("Что такое H₂O?", "kb", "neutral", True),
    ("Погода в Москве", "web", "neutral", False),
    ("Сколько лет Байкалу?", "kb", "neutral", True),
    ("Завтра будет дождь?", "web", "neutral", True),
]

# Записываем в production_logs.jsonl
with LOG.open("w", encoding="utf-8") as f:
    for i, (q, intent, sentiment, is_q) in enumerate(TEST_QUERIES):
        entry = {
            "ts": datetime.now().isoformat(),
            "user_id": 123456 + i,
            "query": q,
            "intent": intent,
            "sentiment": sentiment,
            "emotion": None,
            "is_question": is_q,
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Сгенерировано {len(TEST_QUERIES)} тестовых записей")

# Анализируем
print("\n" + "=" * 70)
print("📊 Production Analytics")
print("=" * 70)

logs = []
with LOG.open(encoding="utf-8") as f:
    for line in f:
        logs.append(json.loads(line))

print(f"\nВсего запросов: {len(logs)}")

# Топ запросов
top_queries = Counter(l["query"] for l in logs).most_common(5)
print("\n🔝 Топ-5 запросов:")
for q, count in top_queries:
    print(f"  {count}× {q}")

# Использование инструментов
tool_usage = Counter(l["intent"] for l in logs)
print(f"\n🛠 Использование интентов:")
for tool, count in tool_usage.most_common():
    pct = count * 100 // len(logs)
    print(f"  {tool:10} {count:3} ({pct}%)")

# Sentiment distribution
sentiment_dist = Counter(l["sentiment"] for l in logs)
print(f"\n😊😐😠 Sentiment distribution:")
for s, count in sentiment_dist.most_common():
    pct = count * 100 // len(logs)
    emoji = {"positive": "😊", "neutral": "😐", "negative": "😠"}.get(s, "❓")
    print(f"  {emoji} {s:10} {count:3} ({pct}%)")

# Вопросы
q_count = sum(1 for l in logs if l["is_question"])
print(f"\n❓ Вопросы: {q_count}/{len(logs)} ({q_count*100//len(logs)}%)")

# Код-слова
code_keywords = ["код", "факториал", "скрипт", "функция", "python", "sql"]
code_count = sum(1 for l in logs if any(k in l["query"].lower() for k in code_keywords))
print(f"💻 С код-словами: {code_count}/{len(logs)} ({code_count*100//len(logs)}%)")
