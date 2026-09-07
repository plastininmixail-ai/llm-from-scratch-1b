"""Расширенные hardcoded regex — математика, время, цитаты.

Генерирует словарь HARDCODED_EXT для добавления в bridge_bot.py.
"""
import re
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")

# Импорт существующих паттернов
import sys
sys.path.insert(0, str(ROOT))
from inference.bridge_bot import _normalize, HARDCODED_ANSWERS

print(f"Existing patterns: {len(HARDCODED_ANSWERS)}")
