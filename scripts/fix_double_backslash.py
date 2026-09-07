"""Фикс HARDCODED_ANSWERS: убирает двойные бэкслеши в паттернах.

При импорте V4 escape-последовательности стали двойными.
"""
import re
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

# Импортируем
import importlib
import inference.bridge_bot
importlib.reload(inference.bridge_bot)
bb = inference.bridge_bot

HA = bb.HARDCODED_ANSWERS
print(f"Before: {len(HA)} patterns")
broken = sum(1 for k in HA if '\\\\\\\\' in k)
print(f"Broken: {broken}")

# Чиним — заменяем \\s на \s, \\d на \d и т.д.
new_ha = {}
for pattern, answer in HA.items():
    # Заменяем \\ на \
    fixed = pattern.replace('\\\\', '\\')
    new_ha[fixed] = answer

bb.HARDCODED_ANSWERS = new_ha
print(f"After fix: {len(new_ha)} patterns")

# Тестируем
test_qs = [
    "Кто такой Пушкин?",
    "Когда Гагарин полетел в космос?",
    "Сколько будет 2+2?",
    "Самое глубокое озеро?",
]
ok = 0
for q in test_qs:
    r = bb._try_hardcoded(q)
    if r:
        ok += 1
    print(f"  {q}: {r[:80] if r else 'NONE'}")

print(f"\n{ok}/{len(test_qs)} OK")

# Сохраняем исправленный модуль
print("\n⚠️ ВАЖНО: исправления нужно применить в файле через patch.")
