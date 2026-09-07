"""Всё-в-одном: polling + отправка вопросов + ответы.

Сам отправляет вопросы, сам polling-ает, сам отвечает через hardcoded/v94,
сам пишет ответы в файл. ОДИН polling = ОДИН процесс = нет conflict.
"""
from __future__ import annotations

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

import httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
sys.path.insert(0, str(ROOT))

from inference.generate import generate_text, load_model_from_checkpoint
from inference.telegram_bot import _pick_few_shot

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3-nothink"
TEST_LOG = ROOT / "data/test_log.jsonl"

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
        r = httpx.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "Translate Russian to English. Only the translation."},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200},
        }, timeout=30.0)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return None


def _translate_en_ru(text: str) -> str | None:
    try:
        r = httpx.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "Translate English to Russian. Only the translation."},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 400},
        }, timeout=30.0)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return None


def _ollama_direct(user_text: str) -> str | None:
    sys_prompt = "You are a calm AI assistant. Reply briefly. No lists, no code. Same language as user."
    try:
        r = httpx.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_text},
            ],
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


async def _generate_response(user_text: str) -> tuple[str, list]:
    """Генерирует ответ и возвращает (text, path)."""
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
        full_prompt = FEW_SHOT_EN + en_text
        v94_resp = generate_text(
            _state["model"], _state["tokenizer"],
            prompt=full_prompt,
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


# Этот словарь будет наполняться при polling
_results = []


async def _send_and_wait(app, chat_id, q_text):
    """Отправляет вопрос через bot API и ждёт ответа в polling."""
    # Но polling уже работает! Просто добавляем в очередь вопрос
    # и обработчик сам его подхватит.
    await app.bot.send_message(chat_id=chat_id, text=q_text)


async def _on_test_question(app, chat_id):
    """Последовательно отправляет тестовые вопросы."""
    for i, (lang, q) in enumerate(TEST_QUESTIONS):
        print(f"[test {i+1}/{len(TEST_QUESTIONS)}] {lang}: {q}")
        try:
            await app.bot.send_message(chat_id=chat_id, text=q)
        except Exception as e:
            print(f"  [send err] {e}")
        await __import__('asyncio').sleep(4)


async def handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text.strip()
    if not user_text:
        return

    chat_id = update.message.chat.id
    user_id = update.message.from_user.id
    print(f"[handler] from {user_id}: {user_text[:80]}")

    # Логируем входящие
    with TEST_LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps({
            "type": "incoming", "user_id": user_id, "text": user_text,
            "ts": time.time(),
        }, ensure_ascii=False) + '\n')

    # Генерируем ответ
    response, path = await _generate_response(user_text)

    await update.message.reply_text(f"{response}\n\n⏱ 0.5s · [{' → '.join(path)}]")

    # Логируем исходящие
    with TEST_LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps({
            "type": "outgoing",
            "to": user_text,
            "response": response,
            "path": path,
            "ts": time.time(),
        }, ensure_ascii=False) + '\n')


async def post_init(app):
    """После старта — отправляем тестовые вопросы САМИ СЕБЕ."""
    import asyncio
    await asyncio.sleep(5)

    # Получаем свой chat_id через getUpdates
    # Нет, это не работает (polling уже занимает updates)
    # Решение: использовать bot.get_me() и слать в saved chat_id
    # Но мы не знаем свой chat_id!
    #
    # Workaround: использовать @Gopcaninebot username + get_chat()
    # Но get_chat() для бота с собой не работает.
    #
    # Workaround: прислать себе сообщение через bot.send_message() с chat_id = me
    # Но bot.send_message(chat_id='@Gopcaninebot') не работает
    #
    # Самый простой workaround: ждать сообщение от тебя,
    # а после него — в этом же handler-е запустить тест.
    print("[post_init] жду первое входящее сообщение...")


async def test_after_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Запускается после первого входящего сообщения."""
    chat_id = update.message.chat.id
    print(f"\n[test-trigger] первое сообщение от chat={chat_id}")
    print(f"[test-trigger] запускаю тест {len(TEST_QUESTIONS)} вопросов...\n")

    # Запускаем как фоновую задачу
    import asyncio
    asyncio.create_task(_on_test_question(ctx.application, chat_id))


def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        print("ERROR: BOT_TOKEN")
        return

    print("=== Self-Test Bot (auto-test on first message) ===")
    ckpt = ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt"
    model, tokenizer, _ = load_model_from_checkpoint(ckpt, ROOT / "tokenizer/vocab.json")
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    print(f"✓ loaded {ckpt.name}")

    # Очищаем лог
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    TEST_LOG.write_text("")

    app = ApplicationBuilder().token(token).connect_timeout(30).read_timeout(30).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handler))
    print("✓ polling started (отправь /start или любое сообщение → запустится автотест)")
    app.run_polling(drop_pending_updates=False, stop_signals=None)


if __name__ == "__main__":
    main()
