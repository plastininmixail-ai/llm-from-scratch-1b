"""Поиск реальных проблемных паттернов в файле."""
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SRC = ROOT / "inference/bridge_bot.py"
content = SRC.read_text(encoding="utf-8")

# Поищем строки где 'кто' и 's*' рядом
import re
problems = []
for m in re.finditer(r'r"кто[^"]*s\*[^"]*такой[^"]*"', content):
    problems.append((m.start(), m.group()))

print(f"Found {len(problems)} 'кто...такой' patterns:")
for pos, txt in problems[:10]:
    print(f"  pos={pos}: {repr(txt)}")
