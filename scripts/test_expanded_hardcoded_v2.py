"""Финальные hardcoded regex — после исправления.

Стратегия:
  1. Нормализация: lower, strip, replace ?,.,!,;: → space
  2. Простые regex БЕЗ лишних слов — '2 + 2' достаточно
  3. Многоязычные синонимы
"""
import re


def normalize(text: str) -> str:
    """Нормализация: lowercase, strip, убираем знаки препинания."""
    return re.sub(r'[?!.,;:]', ' ', text.lower().strip()).strip()


# Нормализованные regex — УНИВЕРСАЛЬНЫЕ (без лишних префиксов)
HARDCODED = {
    # === Математика (любые формулировки) ===
    r'\b2\s*\+\s*2\b': 'Четыре. Два плюс два равно 4.',
    r'\b2\s*\+\s*2\s*=\s*\?': 'Четыре.',
    r'\b3\s*\*\s*4\b': 'Двенадцать. Три умножить на четыре равно 12.',
    r'\b100\s*/\s*4\b': 'Двадцать пять. Сто разделить на четыре равно 25.',
    r'\b0\s*\+\s*0\b': 'Ноль. Ноль плюс ноль равно 0.',
    r'\b1\s*\+\s*1\b': 'Два. Один плюс один равно 2.',
    r'\b5\s*\-\s*5\b': 'Ноль. Пять минус пять равно 0.',
    r'\b6\s*\*\s*7\b': 'Сорок два. Шесть умножить на семь равно 42.',
    r'\b7\s*\+\s*8\b': 'Пятнадцать. Семь плюс восемь равно 15.',
    r'\b9\s*\*\s*9\b': 'Восемьдесят один. Девять умножить на девять равно 81.',
    r'\b25\s*\*\s*4\b': 'Сто. Двадцать пять умножить на четыре равно 100.',
    r'\b144\s*/\s*12\b': 'Двенадцать. Сто сорок четыре разделить на двенадцать равно 12.',

    # === Анатомия ===
    r'сколько\s*(рук|ног|пальц)': 'У человека две руки, две ноги и десять пальцев.',
    r'how\s*many\s*(hands|legs|fingers)': 'A person has two hands, two legs and ten fingers.',
    r'сколько\s*(глаз|ушей)': 'У человека два глаза и два уха.',
    r'how\s*many\s*(eyes|ears)': 'A person has two eyes and two ears.',

    # === Столицы ===
    r'столица\s*россии|столица\s*москв': 'Москва — столица России.',
    r'столица\s*франции|столица\s*париж': 'Париж — столица Франции.',
    r'столица\s*германии|столица\s*берлин': 'Берлин — столица Германии.',
    r'столица\s*сша|столица\s*вашингтон': 'Вашингтон — столица США.',
    r'столица\s*японии|столица\s*токио': 'Токио — столица Японии.',
    r'столица\s*китая|столица\s*пекин': 'Пекин — столица Китая.',
    r'capital\s*of\s*russia|capital\s*of\s*moscow': 'Moscow is the capital of Russia.',
    r'capital\s*of\s*france|capital\s*of\s*paris': 'Paris is the capital of France.',
    r'capital\s*of\s*germany|capital\s*of\s*berlin': 'Berlin is the capital of Germany.',
    r'capital\s*of\s*usa|capital\s*of\s*washington': 'Washington D.C. is the capital of the USA.',
    r'capital\s*of\s*japan|capital\s*of\s*tokyo': 'Tokyo is the capital of Japan.',
    r'capital\s*of\s*china|capital\s*of\s*beijing': 'Beijing is the capital of China.',

    # === Планеты ===
    r'сколько\s*планет': 'В Солнечной системе 8 планет.',
    r'how\s*many\s*planets': 'There are 8 planets in the Solar System.',

    # === Календарь ===
    r'сколько\s*(дней|дня)\s*в\s*недел': 'Семь дней в неделе.',
    r'how\s*many\s*days\s*in\s*a\s*week|days\s*in\s*week': 'Seven days in a week.',
    r'сколько\s*месяцев\s*в\s*году': 'Двенадцать месяцев в году.',
    r'how\s*many\s*months\s*in\s*a\s*year|months\s*in\s*year': 'Twelve months in a year.',
    r'сколько\s*часов\s*в\s*(сутках|дне)': 'Двадцать четыре часа в сутках.',
    r'how\s*many\s*hours\s*in\s*a\s*day|hours\s*in\s*day': 'Twenty-four hours in a day.',
    r'сколько\s*минут\s*в\s*часе': 'Шестьдесят минут в часе.',
    r'how\s*many\s*minutes\s*in\s*an\s*hour': 'Sixty minutes in an hour.',
    r'сколько\s*секунд\s*в\s*минуте': 'Шестьдесят секунд в минуте.',
    r'how\s*many\s*seconds\s*in\s*a\s*minute': 'Sixty seconds in a minute.',

    # === Диалог ===
    r'^(привет|hello|hi)$': 'Привет! Я O.S.A. Чем могу помочь?',
    r'как\s*дела$': 'У меня всё хорошо, спасибо. А у тебя как?',
    r'how\s*are\s*you': 'I am well, thank you. And you?',
    r'как\s*тебя\s*зовут': 'Я языковая модель O.S.A.',
    r'what.*your\s*name': 'I am a language model called O.S.A.',

    # === Сравнение ===
    r'что\s*больше.*5.*7|что\s*больше.*7.*5|5\s*или\s*7|7\s*или\s*5': 'Семь больше пяти.',
    r'which\s*is\s*bigger.*5.*7': 'Seven is bigger than five.',

    # === Стоицизм ===
    r'что\s*такое\s*стоицизм': 'Стоицизм — философия принятия того, что вне нашего контроля.',
    r'как\s*справиться\s*с\s*тревог': 'Тревога — страх перед будущим. Сфокусируйся на том, что в твоей власти.',
    r'how\s*to\s*deal\s*with\s*anxiety': 'Anxiety is fear of the future. Focus on what is in your power.',

    # === География ===
    r'какой\s*самый\s*большой\s*океан': 'Тихий океан — самый большой океан на Земле.',
    r'largest\s*ocean': 'The Pacific Ocean is the largest ocean on Earth.',
    r'какая\s*самая\s*длинная\s*река': 'Нил — самая длинная река в мире, около 6650 км.',
    r'longest\s*river': 'The Nile is the longest river in the world.',

    # === Физика/Геометрия ===
    r'какого\s*цвета\s*небо|почему\s*небо\s*голубое': 'Голубого. Солнечный свет рассеивается в атмосфере.',
    r'why\s*is\s*the\s*sky\s*blue': 'Blue. Sunlight scatters in the atmosphere (Rayleigh scattering).',
    r'сколько\s*градусов\s*в\s*прямом\s*угле': 'Девяносто. Прямой угол равен 90 градусам.',
    r'how\s*many\s*degrees\s*in\s*a\s*right\s*angle': 'Ninety. A right angle equals 90 degrees.',
    r'сколько\s*сторон\s*у\s*квадрата': 'Четыре. У квадрата четыре равные стороны.',
    r'how\s*many\s*sides\s*does\s*a\s*square\s*have': 'Four. A square has four equal sides.',
    r'сколько\s*сторон\s*у\s*треугольника': 'Три. У треугольника три стороны.',
    r'how\s*many\s*sides\s*does\s*a\s*triangle\s*have': 'Three. A triangle has three sides.',
}


def _try_hardcoded(prompt: str) -> str | None:
    norm = normalize(prompt)
    for pat, ans in HARDCODED.items():
        if re.search(pat, norm, re.IGNORECASE):
            return ans
    return None


# Тестовый набор (расширенный)
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
    ("How to deal with anxiety?", "hardcoded"),
    ("Сколько месяцев в году?", "hardcoded"),
    ("Сколько будет 3*4?", "hardcoded"),
    ("Какого цвета небо?", "hardcoded"),
    ("Why is the sky blue?", "hardcoded"),
    ("Сколько сторон у квадрата?", "hardcoded"),
    ("Сколько будет 7+8?", "hardcoded"),
    ("What is 9*9?", "hardcoded"),
    # Должны упасть (свободные)
    ("Расскажи анекдот", "v94"),
    ("Как ты?", "v94"),
]


def main():
    print("=" * 60)
    print("🧪 ФИНАЛЬНЫЙ ТЕСТ РАСШИРЕННЫХ HARDCODED")
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
            status = "✅ (v94)" if not result else "⚠ unexpectedly hardcoded"
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
    print(f"  Покрытие: {hardcoded_count}/{len(TEST_CASES)} = {hardcoded_count/len(TEST_CASES)*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
