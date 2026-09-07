"""Импорт hardcoded_v4 в bridge_bot.py."""
import json
import re
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
BRIDGE = ROOT / "inference/bridge_bot.py"
SRC = ROOT / "data/hardcoded_v4.json"

# Загружаем
patterns = json.loads(SRC.read_text(encoding="utf-8"))
print(f"Loaded {len(patterns)} patterns from {SRC}")

# Читаем файл
content = BRIDGE.read_text(encoding="utf-8")

# Находим место перед "# Стоические цитаты" или анатомией
# Вставляем в HARDCODED_ANSWERS сразу после определения dict
# Используем первое вхождение "    r"\\d+\\s*(?:мил..." и вставляем ПЕРЕД ним

marker = "    r\"(?:стоическ"
if marker in content:
    print("⚠️ Стоические паттерны найдены (не должны быть, но есть маркер)")
    # Вставляем перед маркером
    insert_point = content.index(marker)
else:
    # Вставляем после конца dist_convert
    insert_point = content.find('"DIST_CONVERT"')
    if insert_point < 0:
        # Fallback: перед анатомией
        insert_point = content.find('r"сколько\\s*зуб')

# Генерируем Python код
lines = []
for pattern, answer in patterns.items():
    # Escape кавычки
    pat_escaped = pattern.replace("\\", "\\\\").replace('"', '\\"')
    ans_escaped = answer.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'    r"{pat_escaped}": "{ans_escaped}",')

new_block = "\n".join(lines)
print(f"Generated {len(lines)} lines")

# Сохраняем
out = content[:insert_point] + new_block + "\n" + content[insert_point:]
BRIDGE.write_text(out, encoding="utf-8")
print(f"✅ Written to {BRIDGE}")
print(f"File size: {len(out)} chars")
