"""Применяет фикс двойных бэкслешей к bridge_bot.py напрямую."""
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SRC = ROOT / "inference/bridge_bot.py"

content = SRC.read_text(encoding="utf-8")
print(f"File size before: {len(content)} chars")

# Считаем сколько раз встречается \\\\
broken_count = content.count('\\\\')
print(f"\\\\ occurrences: {broken_count}")

# Заменяем \\s на \s, \\d на \d, \\w на \w (regex паттерны)
# В Python string literal: "r\"\\s*\"" — это "r'\s*'" в исходнике
# В файле это выглядит как r"\\\\s*" (4 бэкслеша = 2 в строке)
# Нужно заменить на r"\\s*" (2 бэкслеша = 1 в строке = regex \s)

fixed = content.replace('\\\\\\\\', '\\')
print(f"After fix: {len(fixed)} chars")

# Сохраняем
SRC.write_text(fixed, encoding="utf-8")
print(f"✅ Saved: {SRC}")
