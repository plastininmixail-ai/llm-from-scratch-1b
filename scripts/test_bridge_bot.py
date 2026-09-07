"""Тест Bridge Bot — имитирует отправку сообщений в Telegram и собирает ответы.

Использует Bot API напрямую (не требует живого пользователя).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
ENV = ROOT / ".env"

# Загружаем токен
token = None
with ENV.open() as f:
    for line in f:
        if line.startswith("BOT_TOKEN="):
            token = line.strip().split("=", 1)[1].strip('"').strip("'")
            break

if not token:
    print("❌ BOT_TOKEN не найден в .env")
    sys.exit(1)

import httpx

API = f"https://api.telegram.org/bot{token}"

# Тестовые вопросы — 20 штук по категориям
TEST_QUESTIONS = [
    # Hardcoded (должны быть мгновенные правильные ответы)
    ("RU", "Сколько будет 2+2?"),
    ("RU", "Сколько рук у человека?"),
    ("RU", "Какая столица Франции?"),
    ("RU", "Сколько планет в Солнечной системе?"),
    ("RU", "Сколько дней в неделе?"),
    ("EN", "What is 2+2?"),
    ("EN", "How many hands does a person have?"),
    ("EN", "Capital of France?"),

    # Translation Bridge (свободные вопросы)
    ("RU", "Привет!"),
    ("RU", "Что такое счастье?"),
    ("RU", "Как научиться вставать рано?"),
    ("RU", "Что делать если тревожно?"),
    ("EN", "Hello!"),
    ("EN", "What is happiness?"),
    ("EN", "How to wake up early?"),
    ("EN", "How to deal with anxiety?"),

    # Logic (hardcoded)
    ("RU", "Что больше: 5 или 7?"),
    ("RU", "Сколько будет 100/4?"),
    ("EN", "Which is bigger: 5 or 7?"),
    ("EN", "What is 100/4?"),
]


def get_updates(offset=None, timeout=30):
    """Long polling — ждём сообщения."""
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    try:
        r = httpx.get(f"{API}/getUpdates", params=params, timeout=timeout + 10)
        return r.json().get("result", [])
    except Exception as e:
        print(f"  [getUpdates error] {e}")
        return []


def send_message(chat_id, text):
    try:
        r = httpx.post(f"{API}/sendMessage", json={
            "chat_id": chat_id,
            "text": text,
        }, timeout=30)
        return r.json()
    except Exception as e:
        print(f"  [send error] {e}")
        return None


def main():
    # Шаг 1: получаем свой chat_id
    print("📋 Получаю chat_id (@Gopcaninebot)...")
    updates = get_updates(timeout=5)
    chat_id = None
    if updates:
        # Берём последний update
        last = updates[-1]
        chat_id = last["message"]["chat"]["id"]
        print(f"  ✓ chat_id = {chat_id}")
    else:
        print("  ⚠ нет сообщений в очереди, нужно сначала отправить боту сообщение в Telegram")
        print("  Пробую через getMe...")
        r = httpx.get(f"{API}/getMe", timeout=10)
        print(f"  Bot: {r.json()}")
        return

    # Шаг 2: отправляем вопросы и собираем ответы
    print(f"\n📤 Отправляю {len(TEST_QUESTIONS)} тестовых вопросов...\n")
    results = []
    offset = None
    if updates:
        offset = updates[-1]["update_id"] + 1

    for i, (lang, q) in enumerate(TEST_QUESTIONS):
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {lang}: {q}")
        # Отправляем
        r = send_message(chat_id, q)
        if not r or not r.get("ok"):
            print(f"  ❌ send error: {r}")
            continue

        # Ждём ответ (long polling)
        print(f"  ⏳ жду ответ...")
        waited = 0
        while waited < 60:
            time.sleep(2)
            waited += 2
            ups = get_updates(offset=offset, timeout=5)
            for u in ups:
                if "message" in u and u["message"].get("chat", {}).get("id") == chat_id:
                    text = u["message"].get("text", "")
                    if text and not text.startswith("/") and text != q:
                        # Нашли ответ
                        results.append({
                            "lang": lang,
                            "question": q,
                            "response": text,
                        })
                        print(f"  ✓ ответ получен ({len(text)} chars)")
                        offset = u["update_id"] + 1
                        break
            else:
                continue
            break

    # Шаг 3: сохраняем результаты
    out = ROOT / "data/bridge_bot_test.jsonl"
    with out.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n✅ Готово: {len(results)} ответов")
    print(f"📤 {out}")


if __name__ == '__main__':
    main()
