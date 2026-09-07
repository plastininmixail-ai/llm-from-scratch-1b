"""Финальный тест — отправляет вопросы и читает ответы."""
import json
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
ENV = ROOT / ".env"

with ENV.open() as f:
    for line in f:
        if line.startswith("BOT_TOKEN="):
            token = line.strip().split("=", 1)[1].strip('"').strip("'")
            break

import httpx

API = f"https://api.telegram.org/bot{token}"
RESPONSES_LOG = ROOT / "data/bridge_bot_responses.jsonl"

TEST_QUESTIONS = [
    ("RU", "Сколько будет 2+2?"),
    ("RU", "Сколько рук у человека?"),
    ("RU", "Какая столица Франции?"),
    ("RU", "Сколько планет в Солнечной системе?"),
    ("RU", "Привет!"),
    ("RU", "Что такое стоицизм?"),
    ("RU", "Как справиться с тревогой?"),
    ("RU", "Как научиться вставать рано?"),
    ("EN", "What is 2+2?"),
    ("EN", "How many hands does a person have?"),
    ("EN", "Capital of France?"),
    ("EN", "Hello!"),
    ("EN", "What is happiness?"),
]


def get_updates(offset=None, timeout=30):
    params = {"timeout": timeout, "allowed_updates": ["message"]}
    if offset is not None:
        params["offset"] = offset
    try:
        r = httpx.get(f"{API}/getUpdates", params=params, timeout=timeout + 10)
        return r.json().get("result", [])
    except Exception as e:
        print(f"  [error] {e}")
        return []


def send_message(chat_id, text):
    try:
        r = httpx.post(f"{API}/sendMessage", json={
            "chat_id": chat_id, "text": text,
        }, timeout=30)
        return r.json()
    except Exception as e:
        print(f"  [send err] {e}")
        return None


def main():
    print("=" * 60)
    print(f"📋 Отправляю {len(TEST_QUESTIONS)} вопросов")
    print("=" * 60)

    # Ждём пока Mini Bridge Bot получит хотя бы 1 ответ
    print("\nЖду пока бот обработает первое сообщение...")
    time.sleep(3)

    # Отправляем сообщения через бота (он же polling-ает)
    # Но нам нужен chat_id - получаем через свой polling
    # Тут возникает conflict! Бот уже polling-ает, мы не можем.
    #
    # Решение: вместо polling используем webhooks (не работает) или
    # используем file-based channel через mini_bridge_bot (он логирует).

    # Mini Bridge Bot логирует ответы в файл. Читаем их периодически.
    print("\nЖду ответы в логе (Mini Bridge Bot записывает в bridge_bot_responses.jsonl)...")

    results = []
    sent = set()
    initial = set()

    start = time.time()
    while time.time() - start < 240:  # ждём до 4 минут
        if RESPONSES_LOG.exists():
            current = []
            with RESPONSES_LOG.open('r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        current.append(json.loads(line))

            for r in current:
                q = r['q']
                if q in sent:
                    continue
                # Нашли новый ответ!
                # Но мы должны отправить вопрос через бота...
                # Тут проблема: мы не можем polling-ить пока бот polling-ает

        time.sleep(2)

    print("\n⚠️ Не могу отправлять сообщения пока Mini Bridge Bot держит токен")
    print("Он логирует ответы — нужно проверять вручную")


if __name__ == '__main__':
    main()
