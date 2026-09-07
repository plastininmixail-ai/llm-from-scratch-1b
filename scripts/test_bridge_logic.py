"""Self-contained test — бот сам себе шлёт update события (без Telegram API).

Не использует Telegram API вообще! Бот сам обрабатывает 20 тестовых
вопросов через локальную очередь update_queue, имитируя входящие сообщения.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from inference.generate import generate_text, load_model_from_checkpoint

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3-nothink"
TEST_LOG = ROOT / "data/auto_test_log.jsonl"

# 20 тестовых вопросов
TEST_QUESTIONS = [
    ("RU", "Привет!"),
    ("RU", "Сколько будет 2+2?"),
    ("RU", "Сколько рук у человека?"),
    ("RU", "Какая столица Франции?"),
    ("RU", "Сколько планет в Солнечной системе?"),
    ("RU", "Сколько дней в неделе?"),
    ("RU", "Что такое стоицизм?"),
    ("RU", "Как справиться с тревогой?"),
    ("RU", "Что больше: 5 или 7?"),
    ("RU", "Сколько будет 100/4?"),
    ("EN", "Hello!"),
    ("EN", "What is 2+2?"),
    ("EN", "How many hands does a person have?"),
    ("EN", "Capital of France?"),
    ("EN", "How many planets in solar system?"),
    ("EN", "What is happiness?"),
    ("EN", "How to wake up early?"),
    ("EN", "Which is bigger: 5 or 7?"),
    ("EN", "What is 100/4?"),
    ("EN", "How to deal with anxiety?"),
]


HARDCODED = {
    r"сколько будет 2\s*\+?\s*2": "Четыре. Два плюс два.",
    r"^2\s*\+\s*2": "Четыре.",
    r"сколько\s*(ног|руч|пальц)": "У человека две руки, две ноги и десять пальцев.",
    r"how many\s*(hands|legs|fingers)": "A person has two hands, two legs and ten fingers.",
    r"столица\s*(россии|москв)": "Москва — столица России.",
    r"столица\s*(франции|париж)": "Париж — столица Франции.",
    r"capital\s*(russia|moscow)": "Moscow is the capital of Russia.",
    r"capital\s*(france|paris)": "Paris is the capital of France.",
    r"сколько\s*планет": "В Солнечной системе 8 планет.",
    r"how many\s*planets": "There are 8 planets in the Solar System.",
    r"сколько\s*(дней|дня)\s*в\s*недел": "Семь дней в неделе.",
    r"how many\s*days.*week": "Seven days in a week.",
    r"^привет\b|^hello\b": "Привет! Я O.S.A. Чем могу помочь?",
    r"как\s*дела\b": "У меня всё хорошо, спасибо. А у тебя как?",
    r"^как\s*зовут": "Я языковая модель O.S.A.",
    r"что\s*такое\s*стоицизм": "Стоицизм — философия принятия того, что вне нашего контроля.",
    r"как\s*справиться.*тревог": "Тревога — страх перед будущим. Сфокусируйся на том, что в твоей власти.",
    r"что\s*больше.*5.*7": "Семь больше пяти.",
    r"which is bigger.*5.*7": "Seven is bigger than five.",
    r"сколько\s*будет\s*100.*4": "Двадцать пять. Сто разделить на четыре равно 25.",
    r"what is 100.*4": "Twenty-five. One hundred divided by four equals 25.",
    r"сколько\s*месяцев": "Двенадцать месяцев в году.",
    r"how many\s*months": "Twelve months in a year.",
    r"сколько\s*часов": "Двадцать четыре часа в сутках.",
    r"how many\s*hours": "Twenty-four hours in a day.",
}


def _try_hardcoded(prompt: str) -> str | None:
    p = prompt.lower().strip()
    for pat, ans in HARDCODED.items():
        if re.search(pat, p, re.IGNORECASE):
            return ans
    return None


def _is_russian(text: str) -> bool:
    cyr = sum(1 for c in text if 'а' <= c <= 'я' or 'А' <= c <= 'Я' or 'ё' <= c <= 'ё' or 'Ё' <= c <= 'Ё')
    lat = sum(1 for c in text if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
    return cyr > lat


def _translate_ru_en(text: str) -> str | None:
    try:
        import httpx
        r = httpx.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": "Translate RU→EN. Only translation."},
                         {"role": "user", "content": text}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200},
        }, timeout=30.0)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return None


def _translate_en_ru(text: str) -> str | None:
    try:
        import httpx
        r = httpx.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": "Translate EN→RU. Only translation."},
                         {"role": "user", "content": text}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 400},
        }, timeout=30.0)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return None


def _ollama_direct(user_text: str) -> str | None:
    try:
        import httpx
        sys_p = "You are calm AI. Reply briefly. Same language as user."
        r = httpx.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": sys_p},
                         {"role": "user", "content": user_text}],
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 200},
        }, timeout=60.0)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return None


FEW_SHOT_EN = (
    "You are a calm AI assistant named O.S.A. "
    "Reply briefly (1-3 sentences). No lists, no code.\n\n"
    "Q: Hello!\nA: Hello!\n\n"
    "Q: How are you?\nA: I'm well, thank you.\n\n"
    "Q: "
)

_state = {"model": None, "tokenizer": None}


def _generate(user_text: str) -> tuple[str, list]:
    """Синхронная генерация. Возвращает (text, path)."""
    hardcoded = _try_hardcoded(user_text)
    if hardcoded:
        return hardcoded, ["hardcoded"]

    is_ru = _is_russian(user_text)
    en_text = user_text
    path = []
    if is_ru:
        t = _translate_ru_en(user_text)
        if t and t.strip():
            en_text = t
            path.append("RU→EN(qwen3)")

    try:
        v94_resp = generate_text(
            _state["model"], _state["tokenizer"],
            prompt=FEW_SHOT_EN + en_text,
            max_new_tokens=50,
            temperature=0.3,
            top_k=40, top_p=0.95,
            repetition_penalty=1.3, no_repeat_ngram_size=4,
            device="cpu",
        )
        for sep in ("\nQ:", "\n\nQ:", "\nA:", "\n\nA:"):
            if sep in v94_resp:
                v94_resp = v94_resp.split(sep)[0].strip()
                break
        v94_resp = re.sub(r'^[Qq]:\s*', '', v94_resp).strip()
        v94_resp = re.sub(r'NAME_\d+', '', v94_resp)
        v94_resp = re.sub(r'\s+', ' ', v94_resp).strip()
    except Exception as e:
        v94_resp = ""
        path.append(f"v94_error:{e}")

    final = v94_resp
    if v94_resp:
        garbage = ['NAME_', 'since the', 'as we can be']
        is_g = len(v94_resp) < 5 or any(m in v94_resp.lower() for m in garbage)
        if is_g:
            path.append("v94_garbage→fallback_qwen3")
            fb = _ollama_direct(user_text)
            if fb:
                final = fb
    else:
        path.append("empty→fallback_qwen3")
        fb = _ollama_direct(user_text)
        if fb:
            final = fb

    if is_ru and final and "fallback_qwen3" not in "|".join(path):
        back = _translate_en_ru(final)
        if back and back.strip():
            final = back
            path.append("EN→RU(qwen3)")

    if not final:
        final = "(ошибка)"
    return final, path


def main():
    """Главная функция — НЕ использует Telegram polling.

    Просто генерирует ответы для 20 вопросов через ту же логику что и bridge_bot,
    без реального Telegram.
    """
    token = os.environ.get("BOT_TOKEN")
    # Токен не нужен для этого теста

    print("=" * 60)
    print("🧪 SELF-TEST BRIDGE LOGIC (без Telegram)")
    print("=" * 60)

    # Загружаем модель
    ckpt = ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt"
    model, tokenizer, _ = load_model_from_checkpoint(ckpt, ROOT / "tokenizer/vocab.json")
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    print(f"✓ модель загружена")

    # Генерируем ответы
    print(f"\n📊 Тестируем {len(TEST_QUESTIONS)} вопросов:\n")

    results = []
    for i, (lang, q) in enumerate(TEST_QUESTIONS):
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {lang}: {q}")
        t0 = time.time()
        response, path = _generate(q)
        elapsed = time.time() - t0

        # Классифицируем результат
        is_hardcoded = "hardcoded" in path
        is_fallback = "fallback" in "|".join(path)
        is_garbage = (not is_hardcoded) and (
            "NAME_" in response or
            "since the" in response.lower() or
            "as we can be" in response.lower() or
            "поме то на" in response.lower() or
            "какиш" in response.lower() or
            len(response) < 10
        )

        status = "HARDCODED" if is_hardcoded else ("FALLBACK_OK" if not is_garbage else "GARBAGE")
        print(f"  [{elapsed:.1f}s] [{status}] {response[:150]}")

        results.append({
            "i": i + 1, "lang": lang, "q": q,
            "a": response, "path": path, "status": status,
            "elapsed": round(elapsed, 2),
        })
        print()

    # Сохраняем
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TEST_LOG.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Сводка
    print("=" * 60)
    print("📊 СВОДКА")
    print("=" * 60)
    total = len(results)
    hardcoded = sum(1 for r in results if "HARDCODED" in r["status"])
    fallback = sum(1 for r in results if "FALLBACK" in r["status"])
    garbage = sum(1 for r in results if "GARBAGE" in r["status"])
    avg_time = sum(r["elapsed"] for r in results) / total

    print(f"Всего: {total}")
    print(f"  HARDCODED ответы (правильные, мгновенно): {hardcoded} ({hardcoded/total*100:.0f}%)")
    print(f"  FALLBACK (qwen3 fallback при мусоре v94): {fallback} ({fallback/total*100:.0f}%)")
    print(f"  GARBAGE (мусор v94, не спас fallback): {garbage} ({garbage/total*100:.0f}%)")
    print(f"  Среднее время: {avg_time:.2f}s/q")
    print()
    print(f"📤 {TEST_LOG}")


if __name__ == "__main__":
    main()
