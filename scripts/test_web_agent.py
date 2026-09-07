"""Автономный тест Web Agent — без Telegram.

Тестирует пайплайн search → fetch → summarize на 10 свежих вопросах.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.web_agent import web_answer

TEST_LOG = ROOT / "data/web_agent_test.jsonl"

TEST_QUESTIONS = [
    # RU: новости
    "Что нового в AI 2026?",
    "Курс Bitcoin сегодня",
    "Погода в Москве сейчас",
    # RU: факты
    "Когда вышел iPhone 16?",
    "Сколько длится беременность у слона?",
    # RU: история
    "Последний чемпион мира по шахматам 2025",
    # EN
    "Latest AI news 2026",
    "Bitcoin price today",
    "Weather in New York now",
    "When was GPT-5 released?",
]


def main():
    print("=" * 60)
    print("🧪 ТЕСТ WEB AGENT (автономный, без Telegram)")
    print("=" * 60)
    print(f"\nТестовых вопросов: {len(TEST_QUESTIONS)}\n")

    results = []
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] Q: {q}")
        t0 = time.time()
        answer = web_answer(q, max_results=2, max_chars_per_page=1500)
        elapsed = time.time() - t0

        if answer:
            # Классифицируем качество
            no_info = "нет информации" in answer.lower() or "no information" in answer.lower()
            garbage = (
                "какиш" in answer.lower() or
                "поме то на" in answer.lower() or
                len(answer) < 30
            )
            status = "FAIL_NO_INFO" if no_info else ("GARBAGE" if garbage else "OK")
            print(f"  [{elapsed:.1f}s] [{status}]")
            print(f"  >>> {answer[:300]}")
        else:
            status = "NO_ANSWER"
            print(f"  [{elapsed:.1f}s] [{status}]")

        results.append({
            "i": i,
            "q": q,
            "a": answer,
            "elapsed": round(elapsed, 2),
            "status": status,
        })

    # Сохраняем
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TEST_LOG.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Сводка
    print("\n" + "=" * 60)
    print("📊 СВОДКА")
    print("=" * 60)
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "OK")
    no_info = sum(1 for r in results if r["status"] == "FAIL_NO_INFO")
    no_answer = sum(1 for r in results if r["status"] == "NO_ANSWER")
    garbage = sum(1 for r in results if r["status"] == "GARBAGE")
    avg_time = sum(r["elapsed"] for r in results) / total

    print(f"Всего: {total}")
    print(f"  ✅ OK (свежий ответ): {ok} ({ok/total*100:.0f}%)")
    print(f"  ⚠️ No info: {no_info}")
    print(f"  ❌ No answer (DuckDuckGo пуст): {no_answer}")
    print(f"  ❌ Garbage: {garbage}")
    print(f"  Среднее время: {avg_time:.1f}s/q")
    print(f"\n📤 {TEST_LOG}")


if __name__ == "__main__":
    main()
