"""Финальный тест bridge_bot через импорт напрямую."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

# Импортируем обновлённую функцию
from inference.bridge_bot import _try_hardcoded, _normalize, HARDCODED_ANSWERS

TEST_CASES = [
    # (вопрос, ожидаемое hardcoded или None)
    ("Сколько рук у человека?", "hardcoded"),
    ("Capital of France?", "hardcoded"),
    ("What is 2+2?", "hardcoded"),
    ("How many hands does a person have?", "hardcoded"),
    ("Какая столица Франции?", "hardcoded"),
    ("Привет!", "hardcoded"),
    ("Сколько планет в Солнечной системе?", "hardcoded"),
    ("Сколько будет 2+2?", "hardcoded"),
    ("Что больше: 5 или 7?", "hardcoded"),
    ("Сколько будет 100/4?", "hardcoded"),
    ("Сколько дней в неделе?", "hardcoded"),
    ("Что такое стоицизм?", "hardcoded"),
    ("How to deal with anxiety?", "hardcoded"),
    ("Сколько месяцев в году?", "hardcoded"),
    ("Сколько будет 3*4?", "hardcoded"),
    ("Какого цвета небо?", "hardcoded"),
    ("Why is the sky blue?", "hardcoded"),
    ("Сколько сторон у квадрата?", "hardcoded"),
    ("Сколько будет 7+8?", "hardcoded"),
    ("What is 9*9?", "hardcoded"),
    ("Сколько будет 5-5?", "hardcoded"),
    ("Сколько будет 0+0?", "hardcoded"),
    ("How many planets?", "hardcoded"),
    ("How many days in a week?", "hardcoded"),
    ("Сколько часов в сутках?", "hardcoded"),
    ("How many hours in a day?", "hardcoded"),
    ("Capital of Japan?", "hardcoded"),
    ("Какая столица Германии?", "hardcoded"),
    ("Сколько сторон у треугольника?", "hardcoded"),
    ("How many sides does a triangle have?", "hardcoded"),
    ("Расскажи анекдот", "v94"),
    ("Как ты?", "v94"),
]

print("=" * 60)
print("🧪 ФИНАЛЬНЫЙ ТЕСТ ОБНОВЛЁННОГО BRIDGE_BOT")
print("=" * 60)
print(f"\nHARDCODED_ANSWERS содержит {len(HARDCODED_ANSWERS)} паттернов\n")

hardcoded_count = 0
v94_count = 0
fails = []

for q, expected in TEST_CASES:
    result = _try_hardcoded(q)
    if expected == "hardcoded":
        status = "✅" if result else "❌ FAIL"
        if result:
            hardcoded_count += 1
        else:
            fails.append(q)
    else:
        status = "✅ (v94)" if not result else "⚠ unexpectedly hardcoded"
        if not result:
            v94_count += 1
    print(f"[{status}] Q: {q!r}")
    if result:
        print(f"    → {result[:100]}")

total_expected_hardcoded = sum(1 for q, e in TEST_CASES if e == "hardcoded")
total_expected_v94 = sum(1 for q, e in TEST_CASES if e == "v94")
print("\n" + "=" * 60)
print(f"📊 РЕЗУЛЬТАТ:")
print(f"  HARDCODED поймано: {hardcoded_count} / {total_expected_hardcoded} ожидалось")
print(f"  v94 fallback: {v94_count} / {total_expected_v94} ожидалось")
print(f"  Покрытие: {hardcoded_count}/{len(TEST_CASES)} = {hardcoded_count/len(TEST_CASES)*100:.0f}%")
print(f"  FAILs: {len(fails)}")
if fails:
    print(f"  Failed queries: {fails}")
print("=" * 60)
