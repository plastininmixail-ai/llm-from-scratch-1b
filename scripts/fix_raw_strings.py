"""Фикс raw strings: \\\\s → \\s (только в raw string patterns)."""
import re
from pathlib import Path

SRC = Path("C:/Users/mixai/Desktop/llm-from-scratch/inference/bridge_bot.py")
content = SRC.read_text(encoding="utf-8")

# В raw strings r"..." \\s должно быть \s
# Паттерн: r"[^"]*\\\\s[^"]*"
# Заменяем \\s на \s внутри r"..."

def fix_raw(match):
    s = match.group(0)
    # В raw string \\s → \s
    s = s.replace('\\\\s', '\\s')
    s = s.replace('\\\\d', '\\d')
    s = s.replace('\\\\w', '\\w')
    s = s.replace('\\\\.', '\\.')
    s = s.replace('\\\\(', '\\(')
    s = s.replace('\\\\)', '\\)')
    s = s.replace('\\\\|', '\\|')
    s = s.replace('\\\\?', '\\?')
    return s

# Ищем только raw strings
fixed = re.sub(r'r"[^"]*"', fix_raw, content)
print(f"Original: {len(content)} chars")
print(f"Fixed: {len(fixed)} chars")
print(f"Reduced by: {len(content) - len(fixed)} chars")

SRC.write_text(fixed, encoding="utf-8")
print("✅ Saved")

# Test
import sys
sys.path.insert(0, str(SRC.parent.parent))
import importlib
import inference.bridge_bot
importlib.reload(inference.bridge_bot)
print(f"HARDCODED_ANSWERS: {len(inference.bridge_bot.HARDCODED_ANSWERS)}")

# Quick test
test = ["Кто такой Пушкин?", "Когда Гагарин?", "2+2", "Самое глубокое озеро?"]
for q in test:
    r = inference.bridge_bot._try_hardcoded(q)
    print(f"  {q}: {r[:80] if r else 'NONE'}")
