"""Объективная диагностика моделей v82, v93, v94.

Создаёт 100 тестовых вопросов:
  30 фактов (RU+EN)
  30 диалогов (привет, как дела)
  20 логика (2+2, сколько рук)
  20 язык (русский/английский понимание)

Запуск:
  python -m scripts.diagnose_models
"""
import json
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
OUTPUT = ROOT / "data/diagnostic_questions.jsonl"

# 100 вопросов: 30 facts + 30 dialog + 20 logic + 20 language
QUESTIONS = {
    "facts_ru": [
        "Сколько планет в Солнечной системе?",
        "Сколько ног у человека?",
        "Какая столица России?",
        "Какая столица Франции?",
        "Какая самая длинная река в мире?",
        "Что такое ДНК?",
        "Кто написал Войну и мир?",
        "Что такое гравитация?",
        "Сколько лет вселенной?",
        "Что такое чёрная дыра?",
        "Когда была вторая мировая война?",
        "Кто открыл Америку?",
        "Сколько костей в теле человека?",
        "Какой самый большой океан?",
        "Что такое эволюция?",
        "Что такое фотосинтез?",
        "Кто такой Наполеон?",
        "Что такое HTML?",
        "Сколько цветов у радуги?",
        "Что такое ДНК?",
        "Сколько зубов у взрослого человека?",
        "Кто построил пирамиды?",
        "Когда распался СССР?",
        "Что такое Wi-Fi?",
        "Какой самый высокий водопад?",
        "Сколько букв в русском алфавите?",
        "Что такое гравитация?",
        "Кто такой Пушкин?",
        "Сколько длится год на Земле?",
        "Что такое фотосинтез?",
    ],
    "facts_en": [
        "What is the capital of France?",
        "How many planets are in the solar system?",
        "What is DNA?",
        "Who discovered America?",
        "How many legs does a person have?",
        "What is gravity?",
        "What is the largest ocean?",
        "How many bones in human body?",
        "What is photosynthesis?",
        "Who wrote War and Peace?",
    ],
    "dialog_ru": [
        "Привет!",
        "Как дела?",
        "Что нового?",
        "Как тебя зовут?",
        "Что ты умеешь?",
        "Расскажи о себе",
        "Кто ты?",
        "Как настроение?",
        "Чем занят?",
        "Скучаешь?",
    ],
    "dialog_en": [
        "Hello!",
        "How are you?",
        "What's up?",
        "Who are you?",
        "What is your name?",
        "Tell me about yourself",
        "What can you do?",
        "Nice to meet you",
        "How old are you?",
        "Are you a robot?",
    ],
    "logic_ru": [
        "Сколько будет 2+2?",
        "Сколько будет 7*8?",
        "Сколько будет 100/4?",
        "Сколько будет 12*12?",
        "Что больше: 7 или 8?",
        "Сколько будет 9*9?",
        "Сколько будет 25*4?",
        "Что больше: 0.5 или 0.05?",
        "Сколько будет 144/12?",
        "Сколько будет 5 в квадрате?",
    ],
    "logic_en": [
        "What is 2+2?",
        "What is 7*8?",
        "What is 100/4?",
        "What is 12*12?",
        "Which is bigger: 7 or 8?",
        "What is 9*9?",
        "What is 25*4?",
        "Which is bigger: 0.5 or 0.05?",
        "What is 144/12?",
        "What is 5 squared?",
    ],
    "language_ru": [
        "Сколько слов в русском языке?",
        "Что такое родительный падеж?",
        "Как пишется 'в течение'?",
        "Что такое деепричастие?",
        "В чем разница между 'также' и 'так же'?",
        "Что такое синоним?",
        "Как пишется 'неужели'?",
        "Что такое прилагательное?",
        "Как пишется 'вследствие'?",
        "Что такое омоним?",
    ],
    "language_en": [
        "What is a noun?",
        "What is a verb?",
        "What is an adjective?",
        "What is an adverb?",
        "What is a pronoun?",
        "What is a preposition?",
        "What is a conjunction?",
        "What is an interjection?",
        "What is a synonym?",
        "What is an antonym?",
    ],
}


def main():
    print(f"📊 Создаю 100 диагностических вопросов...")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with OUTPUT.open('w', encoding='utf-8') as f:
        for cat, qs in QUESTIONS.items():
            for q in qs:
                f.write(json.dumps({"category": cat, "prompt": q}, ensure_ascii=False) + '\n')
                total += 1
    print(f"✅ Создано {total} вопросов в {OUTPUT}")
    print(f"   Категории: {[(c, len(q)) for c, q in QUESTIONS.items()]}")


if __name__ == '__main__':
    main()
