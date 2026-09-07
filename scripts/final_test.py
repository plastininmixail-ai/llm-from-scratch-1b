"""Финальный автономный тест: Web Agent + Code Agent + Hardcoded."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.web_agent import web_answer
from inference.code_agent import code_task
from inference.router import route
from inference.bridge_bot import _try_hardcoded, _normalize

TEST_LOG = ROOT / "data/final_test.jsonl"

# 20 тестовых вопросов разных категорий
TEST_QUESTIONS = [
    # Hardcoded (0ms)
    ("Привет!", "hardcoded", "приветствие"),
    ("Сколько будет 2+2?", "hardcoded", "математика"),
    ("Сколько рук у человека?", "hardcoded", "анатомия"),
    ("Какая столица Франции?", "hardcoded", "столицы"),
    ("Сколько планет в Солнечной системе?", "hardcoded", "планеты"),
    # Code (Qwen3 + sandbox)
    ("Напиши код для вычисления факториала 5", "code", "code"),
    ("Посчитай сумму чисел от 1 до 100", "code", "code"),
    # Web (DDG + trafilatura + Qwen3)
    ("Курс Bitcoin сегодня", "web", "web"),
    ("Какая погода в Москве сейчас", "web", "web"),
    ("Когда вышел GPT-5?", "web", "web"),
    # TF-IDF KB / Wikipedia
    ("Кто такой Эйнштейн?", "kb", "kb"),
    ("Что такое фотосинтез?", "kb", "kb"),
    ("Кто изобрёл лампочку?", "kb", "kb"),
    ("Какой самый большой океан?", "kb", "kb"),
    # Dialog (v94 или qwen3 fallback)
    ("Как дела?", "text", "text"),
    ("Расскажи о себе", "text", "text"),
    # Edge cases
    ("Что такое смысл жизни?", "text", "edge"),
    ("Почему небо голубое?", "kb", "edge"),
    ("Как стать умнее?", "text", "edge"),
    ("Что такое квантовый компьютер?", "kb", "edge"),
]


def main():
    print("=" * 60)
    print("🧪 ФИНАЛЬНЫЙ ТЕСТ BRIDGE BOT v3")
    print("=" * 60)
    print(f"Тестовых вопросов: {len(TEST_QUESTIONS)}\n")

    results = []
    for i, (q, expected_path, category) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] Q: {q}")
        t0 = time.time()

        # 1. Hardcoded
        response = None
        path = ""
        hc = _try_hardcoded(q)
        if hc:
            response = hc
            path = "hardcoded"
            elapsed = time.time() - t0
            print(f"  [{elapsed:.1f}s] [hardcoded] {response[:100]}")
            results.append({"i": i, "q": q, "a": response, "path": path, "elapsed": round(elapsed, 2), "expected": expected_path})
            continue

        # 2. Web Agent (если web keyword)
        web_kw = ["новост", "сегодня", "сейчас", "курс", "погода", "когда"]
        if any(k in q.lower() for k in web_kw):
            try:
                response = web_answer(q, max_results=2, max_chars_per_page=1500)
                if response:
                    path = "web-agent"
                    elapsed = time.time() - t0
                    print(f"  [{elapsed:.1f}s] [web-agent] {response[:100]}")
                    results.append({"i": i, "q": q, "a": response, "path": path, "elapsed": round(elapsed, 2), "expected": expected_path})
                    continue
            except Exception as e:
                print(f"  [web err] {e}")

        # 3. Code Agent (если code keyword)
        code_kw = ["напиши код", "код", "посчитай сумму", "факториал"]
        if any(k in q.lower() for k in code_kw):
            try:
                cr = code_task(q, max_attempts=2)
                if cr and cr.get("stdout"):
                    response = cr["stdout"][:200]
                    path = f"code-agent ({cr['attempts']} attempts)"
                    elapsed = time.time() - t0
                    print(f"  [{elapsed:.1f}s] [code-agent] {response[:100]}")
                    results.append({"i": i, "q": q, "a": response, "path": path, "elapsed": round(elapsed, 2), "expected": expected_path})
                    continue
            except Exception as e:
                print(f"  [code err] {e}")

        # 4. Router → TF-IDF / KB / v94
        intent = route(q)
        path = f"router:{intent}"
        elapsed = time.time() - t0
        print(f"  [{elapsed:.1f}s] [{path}] (no concrete answer - would use KB/v94)")

        results.append({"i": i, "q": q, "a": None, "path": path, "elapsed": round(elapsed, 2), "expected": expected_path})

    # Сохраняем
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TEST_LOG.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Сводка
    print("\n" + "=" * 60)
    print("📊 СВОДКА ФИНАЛЬНОГО ТЕСТА")
    print("=" * 60)
    total = len(results)
    hardcoded = sum(1 for r in results if r["path"] == "hardcoded")
    web = sum(1 for r in results if "web-agent" in r["path"])
    code = sum(1 for r in results if "code-agent" in r["path"])
    router = sum(1 for r in results if "router" in r["path"])
    avg_time = sum(r["elapsed"] for r in results) / total

    print(f"Всего вопросов: {total}")
    print(f"  ✅ Hardcoded (мгновенно): {hardcoded}")
    print(f"  🌐 Web Agent: {web}")
    print(f"  💻 Code Agent: {code}")
    print(f"  🔀 Router (→KB/v94): {router}")
    print(f"  Среднее время: {avg_time:.1f}s/q")
    print(f"\n📤 {TEST_LOG}")


if __name__ == "__main__":
    main()
