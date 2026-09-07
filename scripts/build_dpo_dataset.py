"""Создаёт DPO датасет: chosen vs rejected пары для v94.

Источники chosen:
- hardcoded_answers (правильные)
- KB ответы
- Web Agent v5 ответы

Источники rejected:
- v94 garbage (с предыдущих запусков)
- generic "I don't know"
- Стохастический шум
"""
import json
import random
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")

# Загружаем hardcoded (chosen)
HARDCODED = []
with open(ROOT / "inference/bridge_bot.py", encoding="utf-8") as f:
    content = f.read()

import re
pattern_blocks = re.findall(r'r"([^"]+)":\s*"([^"]+)"', content)
for p, a in pattern_blocks:
    if not p.startswith('\\d') and not a.startswith('STOIC'):
        HARDCODED.append((p, a))

print(f"Loaded {len(HARDCODED)} hardcoded pairs")

# Шаблоны rejected ответов
REJECTED_TEMPLATES = [
    "Я не знаю.",
    "Извините, я не могу ответить на этот вопрос.",
    "Это очень сложный вопрос для меня.",
    "Я плохо понимаю такие темы.",
    "У меня нет информации по этому вопросу.",
    "Извини, я ещё учусь.",
    "Я AI-модель, у меня ограниченные знания.",
    "Переформулируй вопрос, пожалуйста.",
    "Это выходит за рамки моих возможностей.",
    "Не знаю, попробуй спросить у поисковика.",
]

# Стохастические garbage ответы
GARBAGE_TEMPLATES = [
    "ываыва прывыфыв jhkjhkjhkj",
    "The quick brown fox jumps over the lazy dog 1234567890",
    "— ... — ... — ... — ... —",
    "💀💀💀",
    "asdfghjkl; qwertyuiop[]",
    "Я не уверен что правильно понимаю.",
    "Вопросы про {topic} мне сложно воспринимать.",
    "Это философский вопрос без однозначного ответа.",
    "Иногда я ошибаюсь, извини.",
    "Может быть поговорим о чём-то другом?",
]

dpo_pairs = []

# Пары из hardcoded (chosen) vs rejected
for prompt, chosen in HARDCODED[:1500]:  # берём все
    # Превращаем regex pattern в человеческий prompt
    human_prompt = re.sub(r'\\s\*', ' ', prompt)
    human_prompt = re.sub(r'\(\?:\|\|\|', '', human_prompt)
    human_prompt = re.sub(r'\\w\+\?\)', '', human_prompt)
    human_prompt = re.sub(r'\\.', '', human_prompt)
    human_prompt = human_prompt.replace('\\', '').replace('?', '').strip()
    if not human_prompt or len(human_prompt) < 3:
        continue

    # 3 rejected варианта на каждый chosen
    rejected_options = [
        random.choice(REJECTED_TEMPLATES),
        random.choice(GARBAGE_TEMPLATES),
        "Мне нужно больше контекста для ответа.",
    ]

    for rejected in rejected_options:
        dpo_pairs.append({
            "prompt": human_prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

random.shuffle(dpo_pairs)
print(f"DPO pairs: {len(dpo_pairs)}")

# Сохраняем
out = ROOT / "data/dpo_dataset.jsonl"
with out.open("w", encoding="utf-8") as f:
    for p in dpo_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"✅ Saved {len(dpo_pairs)} pairs to {out}")
