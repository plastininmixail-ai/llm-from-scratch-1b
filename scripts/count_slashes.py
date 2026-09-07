"""Считает бэкслеши в файле."""
from pathlib import Path

SRC = Path("C:/Users/mixai/Desktop/llm-from-scratch/inference/bridge_bot.py")
content = SRC.read_text(encoding="utf-8")

# Считаем разные варианты
print(f"len: {len(content)}")

# \\\\: 2 бэкслеша в строке = 4 в repr
quadruple = content.count("\\\\")  # raw 2 бэкслеша
print(f"2 backslashes (\\\\\\\\): {quadruple}")

# Одинарные бэкслеши в raw string r"..."
import re
rstr = re.findall(r'r"[^"]*"', content)
single_in_raw = sum(1 for s in rstr if "\\s" in s and "\\\\s" not in s)
print(f"Raw strings with single \\\\s: {single_in_raw}")
