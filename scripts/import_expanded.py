"""Импортирует expanded_hardcoded.json в bridge_bot.py."""
import json
import re
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
EXPANDED = ROOT / "data/expanded_hardcoded.json"
BRIDGE_BOT = ROOT / "inference/bridge_bot.py"

# Загружаем expanded patterns
with EXPANDED.open(encoding="utf-8") as f:
    expanded = json.load(f)

print(f"Загружено {len(expanded)} паттернов")

# Читаем bridge_bot.py
content = BRIDGE_BOT.read_text(encoding="utf-8")

# Генерируем новые строки
new_lines = []
for pattern, answer in expanded.items():
    ans_escaped = answer.replace('"', '\\"')
    new_lines.append(f'    r"{pattern}": "{ans_escaped}",\n')

print(f"Готово {len(new_lines)} строк")

# Ищем "    # Спорт"
sport_pos = content.find("    # Спорт")
if sport_pos == -1:
    print("Не могу найти # Спорт")
    sys.exit(1)

# Найдём позицию последнего pattern перед # Спорт
last_pattern_pos = -1
for m in re.finditer(r'    r"[^"]+":\s*"[^"]+",\n', content[:sport_pos]):
    last_pattern_pos = m.end()

if last_pattern_pos == -1:
    print("Не могу найти patterns перед # Спорт")
    sys.exit(1)

# Вставляем новые patterns ПЕРЕД # Спорт
insert_pos = last_pattern_pos

new_content = (
    content[:insert_pos] +
    "\n" +
    "".join(new_lines) +
    content[insert_pos:]
)

BRIDGE_BOT.write_text(new_content, encoding="utf-8")

print(f"✅ Записано {len(new_lines)} patterns в bridge_bot.py")
print(f"   Размер файла: {BRIDGE_BOT.stat().st_size / 1024:.1f} KB")
