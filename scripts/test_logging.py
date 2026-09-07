"""Генерирует тестовые логи queries_log.jsonl через симуляцию _log."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

import importlib
import inference.bridge_bot
importlib.reload(inference.bridge_bot)

from inference.bridge_bot import _try_hardcoded, LOG_PATH, _is_garbage

# 30 тестовых запросов разных типов
TEST_QUERIES = [
    "Привет!",
    "Сколько будет 2+2?",
    "Сколько рук у человека?",
    "Какая столица Франции?",
    "Когда вышел GPT-5?",
    "Кто такой Эйнштейн?",
    "Напиши код для факториала 5",
    "Расскажи о себе",
    "Что такое гравитация?",
    "Сколько спутников у Юпитера?",
    "Когда день космонавтики?",
    "Кто написал Войну и мир?",
    "Что больше 0.9 или 0.99?",
    "Что такое чёрная дыра?",
    "Сколько будет 5+5?",
    "100 км в мили",
    "32F to C",
    "Что такое Python?",
    "Когда Гагарин полетел в космос?",
    "Символ кислорода",
    "Формула воды",
    "Сколько элементов в таблице Менделеева?",
    "Когда началась Вторая мировая война",
    "Сколько костей в теле человека?",
    "Сколько длится беременность у слона?",
    "Когда 8 марта",
    "Что такое H₂O",
    "Какая самая длинная река?",
    "Самое большое животное?",
    "Что такое API?",
]

# Очищаем старый лог
if LOG_PATH.exists():
    LOG_PATH.unlink()

# Симулируем логирование
for q in TEST_QUERIES:
    t0 = time.time()
    result = _try_hardcoded(q)
    elapsed = time.time() - t0
    tool = "hardcoded" if result else "none"

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(),
            "user_id": 0,
            "query": q[:200],
            "tool": tool,
            "elapsed": round(elapsed, 4),
            "response_len": len(result or ""),
            "is_garbage": _is_garbage(result or ""),
        }, ensure_ascii=False) + "\n")

print(f"✅ Сгенерировано {len(TEST_QUERIES)} логов в {LOG_PATH}")
print(f"   Размер: {LOG_PATH.stat().st_size} байт")
