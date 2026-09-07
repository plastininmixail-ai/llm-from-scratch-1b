"""RU/EN парные вопросы — одна тема, два языка.

100 вопросов: 50 пар (RU + EN), одна тема.
Категории: facts, logic, dialog, abstract.
"""
import json
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
OUTPUT = ROOT / "data/ru_en_pairs.jsonl"

# 50 пар (одна тема, два языка)
PAIRS = [
    # Facts (1-10)
    ("Сколько планет в Солнечной системе?", "How many planets are in the solar system?"),
    ("Сколько ног у человека?", "How many legs does a person have?"),
    ("Какая столица России?", "What is the capital of Russia?"),
    ("Какая столица Франции?", "What is the capital of France?"),
    ("Что такое ДНК?", "What is DNA?"),
    ("Кто написал Войну и мир?", "Who wrote War and Peace?"),
    ("Что такое гравитация?", "What is gravity?"),
    ("Когда была вторая мировая война?", "When was World War II?"),
    ("Кто открыл Америку?", "Who discovered America?"),
    ("Сколько костей в теле человека?", "How many bones are in the human body?"),
    # Logic (11-20)
    ("Сколько будет 2+2?", "What is 2+2?"),
    ("Сколько будет 7*8?", "What is 7*8?"),
    ("Сколько будет 100/4?", "What is 100/4?"),
    ("Сколько будет 12*12?", "What is 12*12?"),
    ("Что больше: 7 или 8?", "Which is bigger: 7 or 8?"),
    ("Сколько будет 9*9?", "What is 9*9?"),
    ("Сколько будет 25*4?", "What is 25*4?"),
    ("Что больше: 0.5 или 0.05?", "Which is bigger: 0.5 or 0.05?"),
    ("Сколько будет 144/12?", "What is 144/12?"),
    ("Сколько будет 5 в квадрате?", "What is 5 squared?"),
    # Dialog (21-30)
    ("Привет!", "Hello!"),
    ("Как дела?", "How are you?"),
    ("Как тебя зовут?", "What is your name?"),
    ("Что ты умеешь?", "What can you do?"),
    ("Расскажи о себе", "Tell me about yourself"),
    ("Кто ты?", "Who are you?"),
    ("Как настроение?", "How is your mood?"),
    ("Чем занят?", "What are you doing?"),
    ("Скучаешь?", "Do you miss anyone?"),
    ("Что нового?", "What's new?"),
    # Abstract (31-40)
    ("Что такое счастье?", "What is happiness?"),
    ("В чём смысл жизни?", "What is the meaning of life?"),
    ("Что важнее: ум или доброта?", "What is more important: intelligence or kindness?"),
    ("Стоит ли быть идеалистом?", "Should one be an idealist?"),
    ("Что такое свобода?", "What is freedom?"),
    ("Что такое дружба?", "What is friendship?"),
    ("Что такое любовь?", "What is love?"),
    ("Что такое справедливость?", "What is justice?"),
    ("В чём смысл страдания?", "What is the meaning of suffering?"),
    ("Что есть истина?", "What is truth?"),
    # Lifestyle (41-50)
    ("Как научиться вставать рано?", "How to wake up early?"),
    ("Как перестать прокрастинировать?", "How to stop procrastinating?"),
    ("Как бросить курить?", "How to quit smoking?"),
    ("Как справиться с тревогой?", "How to deal with anxiety?"),
    ("Что делать если скучно?", "What to do if bored?"),
    ("Как найти друзей?", "How to make friends?"),
    ("Как стать увереннее?", "How to be more confident?"),
    ("Как научиться говорить нет?", "How to learn to say no?"),
    ("Что делать если всё бесит?", "What to do if everything annoys?"),
    ("Как перестать бояться?", "How to stop being afraid?"),
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pairs = []
    for ru, en in PAIRS:
        pairs.append({"lang": "ru", "prompt": ru, "pair_id": len(pairs)//2 + 1})
        pairs.append({"lang": "en", "prompt": en, "pair_id": len(pairs)//2 + 1})

    print(f"✅ Создано {len(pairs)} вопросов ({len(PAIRS)} пар)")
    with OUTPUT.open('w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"📤 {OUTPUT}")


if __name__ == '__main__':
    main()
