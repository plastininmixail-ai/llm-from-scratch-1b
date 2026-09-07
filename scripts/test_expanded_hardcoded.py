"""Расширенные hardcoded regex для bridge_bot.

Стратегия:
  1. Нормализация: "?", ".", "!", ",", lowercase, strip
  2. Более общие паттерны: "math operation", "math expression", etc.
  3. Многоязычные синонимы
  4. Fallback на более широкие regex
"""
import re
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
BRIDGE_BOT = ROOT / "inference/bridge_bot.py"


def normalize(text: str) -> str:
    """Нормализация: lowercase, strip, убираем знаки препинания."""
    return re.sub(r'[?!.,;:]', ' ', text.lower().strip()).strip()


# Нормализованные regex (используют normalize перед match)
HARDCODED_NORMALIZED = {
    # === Математика (универсальная) ===
    r'сколько будет 2 \+ 2|2 \+ 2 сколько': 'Четыре. Два плюс два.',
    r'2 \+ 2': 'Четыре.',
    r'2 \+ 2 =': 'Четыре. Два плюс два.',
    r'what is 2 \+ 2|2 \+ 2 what': 'Four. Two plus two.',
    r'сколько будет 3 \* 4|3 \* 4 сколько': 'Двенадцать. Три умножить на четыре равно 12.',
    r'what is 3 \* 4|3 \* 4 what': 'Twelve. Three times four equals 12.',
    r'сколько будет 100 / 4|100 / 4 сколько': 'Двадцать пять. Сто разделить на четыре равно 25.',
    r'what is 100 / 4|100 / 4 what': 'Twenty-five. One hundred divided by four equals 25.',
    r'сколько будет 12 \+ 8|12 \+ 8 сколько': 'Двадцать. Двенадцать плюс восемь равно 20.',
    r'what is 12 \+ 8|12 \+ 8 what': 'Twenty. Twelve plus eight equals 20.',

    # === Анатомия (универсальная) ===
    r'сколько рук|сколько ног|сколько пальц': 'У человека две руки, две ноги и десять пальцев.',
    r'how many hands|how many legs|how many fingers': 'A person has two hands, two legs and ten fingers.',
    r'сколько глаз|сколько ушей': 'У человека два глаза и два уха.',
    r'how many eyes|how many ears': 'A person has two eyes and two ears.',

    # === Столицы (универсальная) ===
    r'столица россии|столица москв': 'Москва — столица России. Город с населением более 12 миллионов человек.',
    r'столица франции|столица париж': 'Париж — столица Франции. Город искусств и романтики.',
    r'столица германии|столица берлин': 'Берлин — столица Германии.',
    r'столица сша|столица вашингтон': 'Вашингтон — столица США.',
    r'столица японии|столица токио': 'Токио — столица Японии.',
    r'столица китая|столица пекин': 'Пекин — столица Китая.',
    r'capital of russia|capital of moscow': 'Moscow is the capital of Russia. A city with over 12 million people.',
    r'capital of france|capital of paris': 'Paris is the capital of France. The city of art and romance.',
    r'capital of germany|capital of berlin': 'Berlin is the capital of Germany.',
    r'capital of usa|capital of washington': 'Washington D.C. is the capital of the USA.',
    r'capital of japan|capital of tokyo': 'Tokyo is the capital of Japan.',
    r'capital of china|capital of beijing': 'Beijing is the capital of China.',

    # === Планеты ===
    r'сколько планет': 'В Солнечной системе 8 планет: Меркурий, Венера, Земля, Марс, Юпитер, Сатурн, Уран, Нептун.',
    r'how many planets': 'There are 8 planets in the Solar System: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.',

    # === Календарь ===
    r'сколько дней в неделе|сколько дней в недели': 'Семь дней в неделе.',
    r'how many days in a week|days in week': 'Seven days in a week.',
    r'сколько месяцев в году': 'Двенадцать месяцев в году.',
    r'how many months in a year|months in year': 'Twelve months in a year.',
    r'сколько часов в сутках|сколько часов в дне': 'Двадцать четыре часа в сутках.',
    r'how many hours in a day|hours in day': 'Twenty-four hours in a day.',
    r'сколько минут в часе': 'Шестьдесят минут в часе.',
    r'how many minutes in an hour': 'Sixty minutes in an hour.',
    r'сколько секунд в минуте': 'Шестьдесят секунд в минуте.',
    r'how many seconds in a minute': 'Sixty seconds in a minute.',

    # === Диалог ===
    r'^привет$|^hello$|^hi$': 'Привет! Я O.S.A. Чем могу помочь?',
    r'как дела$|how are you': 'У меня всё хорошо, спасибо. А у тебя как?',
    r'как тебя зовут|what.*your name': 'Я языковая модель O.S.A. У меня нет имени, можешь звать O.S.A.',
    r'что ты умеешь|what can you do': 'Я могу отвечать на вопросы, помогать с размышлениями, обсуждать идеи. Спрашивай!',

    # === Сравнение ===
    r'что больше 5|что больше 7|5 или 7|7 или 5': 'Семь больше пяти.',
    r'which is bigger 5|which is bigger 7': 'Seven is bigger than five.',

    # === Стоицизм ===
    r'что такое стоицизм': 'Стоицизм — философия, которая учит принимать то, что вне нашего контроля, и работать над тем, что в нашей власти: нашими мыслями, словами и действиями.',
    r'how to deal.*anxiety': 'Anxiety is fear of the future. Focus on what is in your power: your actions now.',

    # === Логика (мелкие примеры) ===
    r'сколько будет 0 \+ 0|0 \+ 0': 'Ноль. Ноль плюс ноль равно 0.',
    r'what is 0 \+ 0|0 \+ 0': 'Zero. Zero plus zero equals 0.',
    r'сколько будет 1 \+ 1|1 \+ 1': 'Два. Один плюс один равно 2.',
    r'what is 1 \+ 1|1 \+ 1': 'Two. One plus one equals 2.',
    r'сколько будет 5 \- 5|5 \- 5': 'Ноль. Пять минус пять равно 0.',
    r'what is 5 \- 5|5 \- 5': 'Zero. Five minus five equals 0.',
    r'сколько будет 6 \* 7|6 \* 7': 'Сорок два. Шесть умножить на семь равно 42.',
    r'what is 6 \* 7|6 \* 7': 'Forty-two. Six times seven equals 42.',
    r'сколько будет 7 \+ 8|7 \+ 8': 'Пятнадцать. Семь плюс восемь равно 15.',
    r'what is 7 \+ 8|7 \+ 8': 'Fifteen. Seven plus eight equals 15.',
    r'сколько будет 9 \* 9|9 \* 9': 'Восемьдесят один. Девять умножить на девять равно 81.',
    r'what is 9 \* 9|9 \* 9': 'Eighty-one. Nine times nine equals 81.',
    r'сколько будет 25 \* 4|25 \* 4': 'Сто. Двадцать пять умножить на четыре равно 100.',
    r'what is 25 \* 4|25 \* 4': 'One hundred. Twenty-five times four equals 100.',
    r'сколько будет 144 / 12|144 / 12': 'Двенадцать. Сто сорок четыре разделить на двенадцать равно 12.',
    r'what is 144 / 12|144 / 12': 'Twelve. One hundred forty-four divided by twelve equals 12.',

    # === География ===
    r'какой самый большой океан': 'Тихий океан — самый большой океан на Земле.',
    r'largest ocean': 'The Pacific Ocean is the largest ocean on Earth.',
    r'какая самая длинная река': 'Нил (или Амазонка) — самая длинная река в мире, около 6650 км.',
    r'longest river': 'The Nile (or Amazon) is the longest river in the world, about 6650 km.',

    # === Физика ===
    r'какого цвета небо|почему небо голубое': 'Голубого. Молекулы воздуха рассеивают синий цвет солнечного света (рэлеевское рассеяние).',
    r'why is the sky blue|color of sky': 'Blue. Air molecules scatter the blue wavelengths of sunlight (Rayleigh scattering).',
    r'сколько градусов в прямом угле': 'Девяносто. Прямой угол равен 90 градусам.',
    r'how many degrees in a right angle': 'Ninety. A right angle equals 90 degrees.',
    r'сколько сторон у квадрата': 'Четыре. У квадрата четыре равные стороны.',
    r'how many sides does a square have': 'Four. A square has four equal sides.',
    r'сколько сторон у треугольника': 'Три. У треугольника три стороны.',
    r'how many sides does a triangle have': 'Three. A triangle has three sides.',
}


def _try_hardcoded(prompt: str) -> str | None:
    """Нормализованный поиск."""
    norm = normalize(prompt)
    for pat, ans in HARDCODED_NORMALIZED.items():
        if re.search(pat, norm, re.IGNORECASE):
            return ans
    return None


# Тестовый набор
TEST_CASES = [
    # (вопрос, ожидаемое hardcoded или None)
    ("Сколько рук у человека?", "hardcoded"),
    ("Capital of France?", "hardcoded"),
    ("What is 2+2?", "hardcoded"),
    ("How many hands does a person have?", "hardcoded"),
    ("Какая столица Франции?", "hardcoded"),
    ("Привет!", "hardcoded"),
    ("Сколько планет в Солнечной системе?", "hardcoded"),
    ("Сколько будет 2+2?", "hardcoded"),
    ("Что больше: 5 или 7?", "hardcoded"),
    ("Сколько будет 100/4?", "hardcoded"),
    ("Сколько дней в неделе?", "hardcoded"),
    ("Что такое стоицизм?", "hardcoded"),
    # Должны упасть (свободные)
    ("Расскажи анекдот", "v94"),
    ("Как ты?", "v94"),
]


def main():
    print("=" * 60)
    print("🧪 ТЕСТ РАСШИРЕННЫХ HARDCODED")
    print("=" * 60)

    hardcoded_count = 0
    v94_count = 0
    print()
    for q, expected in TEST_CASES:
        result = _try_hardcoded(q)
        if expected == "hardcoded":
            status = "✅" if result else "❌ FAIL"
            if result:
                hardcoded_count += 1
        else:
            status = "✅ (passed to v94)" if not result else "⚠ unexpectedly hardcoded"
            if not result:
                v94_count += 1
        print(f"[{status}] Q: {q!r}")
        if result:
            print(f"    → {result[:100]}")
        print()

    total_expected_hardcoded = sum(1 for q, e in TEST_CASES if e == "hardcoded")
    total_expected_v94 = sum(1 for q, e in TEST_CASES if e == "v94")
    print("=" * 60)
    print(f"📊 РЕЗУЛЬТАТ:")
    print(f"  HARDCODED поймано: {hardcoded_count} / {total_expected_hardcoded} ожидалось")
    print(f"  v94 fallback: {v94_count} / {total_expected_v94} ожидалось")
    print(f"  Улучшение покрытия: с 70% до {hardcoded_count}/{len(TEST_CASES)*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
