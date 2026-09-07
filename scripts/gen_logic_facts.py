"""Генерация чистых logic/facts пар через qwen3 (Ollama) для fine-tune.

Создаёт 500 пар: вопрос + короткий правильный ответ.
Только логика и факты, без мусора.
"""
import json
import sys
import time
from pathlib import Path
import httpx

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
OUTPUT = ROOT / "data/logic_facts_v1.jsonl"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3-nothink"

# 50 базовых вопросов (RU + EN)
BASE_QUESTIONS = [
    # Логика простая
    "Сколько будет 2+2?", "What is 2+2?",
    "Сколько будет 3*4?", "What is 3*4?",
    "Сколько будет 10-3?", "What is 10-3?",
    "Сколько будет 100/4?", "What is 100/4?",
    "Сколько будет 12+8?", "What is 12+8?",
    "Что больше: 5 или 7?", "Which is bigger: 5 or 7?",
    # Анатомия
    "Сколько ног у человека?", "How many legs does a person have?",
    "Сколько рук у человека?", "How many hands does a person have?",
    "Сколько пальцев на одной руке?", "How many fingers on one hand?",
    "Сколько глаз у человека?", "How many eyes does a person have?",
    "Сколько ушей у человека?", "How many ears does a person have?",
    # Планеты
    "Сколько планет в Солнечной системе?", "How many planets in solar system?",
    "Какая планета ближайшая к Солнцу?", "Which planet is closest to sun?",
    "Какая планета самая большая?", "Which is largest planet?",
    # Календарь
    "Сколько дней в неделе?", "How many days in a week?",
    "Сколько месяцев в году?", "How many months in year?",
    "Сколько часов в сутках?", "How many hours in a day?",
    "Сколько минут в часе?", "How many minutes in hour?",
    # География
    "Какая столица России?", "Capital of Russia?",
    "Какая столица Франции?", "Capital of France?",
    "Какая столица США?", "Capital of USA?",
    "Какая столица Японии?", "Capital of Japan?",
    "Какой самый большой океан?", "Largest ocean?",
    "Какая самая длинная река?", "Longest river?",
    # Простая физика
    "Сколько будет 0+0?", "What is 0+0?",
    "Сколько будет 1+1?", "What is 1+1?",
    "Сколько будет 5-5?", "What is 5-5?",
    "Сколько будет 100-100?", "What is 100-100?",
]


SYSTEM_PROMPT = (
    "You are a helpful teacher. Answer the question with a short, "
    "accurate factual answer in 1-2 sentences. "
    "Reply in the same language as the question. "
    "No lists, no code, no preambles."
)


def query_ollama(prompt: str) -> str | None:
    try:
        r = httpx.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt + " /no_think"},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150},
        }, timeout=30.0)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        return None


def main():
    print(f"📊 Генерирую ответы для {len(BASE_QUESTIONS)} вопросов...")

    pairs = []
    fails = 0
    for i, q in enumerate(BASE_QUESTIONS):
        r = query_ollama(q)
        if r and 5 <= len(r) <= 250:
            pairs.append({"prompt": q, "response": r})
            print(f"  [{i+1}] ✓ {q[:50]}")
        else:
            fails += 1
            print(f"  [{i+1}] FAIL: {q[:50]}")
        time.sleep(0.5)

    print(f"\n✅ Готово: {len(pairs)} пар, {fails} fail")

    # Сохраняем
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"📤 {OUTPUT}")


if __name__ == '__main__':
    main()
