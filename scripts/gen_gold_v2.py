"""Генерация Gold Dataset v2: 5000 пар через Ollama qwen3.

Использует локальный Ollama (http://127.0.0.1:11434) напрямую,
без LiteLLM прокси.

System prompt: тот же что в gold_v1, но обновлённый:
  - Указывает что ответы должны быть КРАТКИЕ и ПОЛЕЗНЫЕ
  - Без списков и кода
  - Стиль дружелюбного ассистента

Используется новый датасет промптов: data/gold_v2_prompts.jsonl
"""
import json
import time
from pathlib import Path
import httpx

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
PROMPTS = ROOT / "data/gold_v2_prompts.jsonl"
OUTPUT = ROOT / "data/gold_dataset_v2.jsonl"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3-nothink"

SYSTEM_PROMPT = """Ты — дружелюбный ИИ-ассистент. Отвечай кратко, по делу, на русском или английском (на том же языке, что и вопрос).

Правила:
- 30-150 слов
- Без списков, без markdown, без нумерации
- Без преамбул типа "Хороший вопрос" или "Конечно!"
- Только живой человеческий текст
- Если вопрос фактический (кто/что/где/сколько) — отвечай конкретно
- Если вопрос философский — дай мудрый ответ
- Говори как дружелюбный собеседник, не как энциклопедия"""


def query_ollama(prompt: str, max_retries: int = 3) -> str:
    """Запрос к Ollama qwen3 напрямую (без LiteLLM)."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 350,
            "num_ctx": 2048,
        },
    }
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(OLLAMA_URL, json=payload)
                r.raise_for_status()
                data = r.json()
                content = data["message"]["content"].strip()
                return content
        except Exception as e:
            print(f"  [retry {attempt+1}] error: {e}")
            time.sleep(2)
    return None


def main():
    if not PROMPTS.exists():
        print(f"❌ {PROMPTS} не найден")
        return

    prompts = []
    with PROMPTS.open('r', encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"📖 Загружено {len(prompts)} промптов")

    existing = set()
    if OUTPUT.exists():
        with OUTPUT.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing.add(obj.get('prompt', ''))
                except:
                    pass
    print(f"📦 Уже готово: {len(existing)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUTPUT.open('a', encoding='utf-8')

    n_new = 0
    n_fail = 0
    start_time = time.time()
    for i, item in enumerate(prompts):
        prompt = item['prompt']
        category = item.get('category', 'unknown')

        if prompt in existing:
            continue

        response = query_ollama(prompt)
        if response is None or len(response) < 10:
            n_fail += 1
            print(f"  [{i+1}] FAIL: {prompt[:50]}")
            continue

        # Пост-обработка
        response = response.replace("**", "").replace("##", "").replace("```", "")
        lines = response.split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('-', '*', '1.', '2.', '3.', '4.', '5.')):
                cleaned.append(line)
        response = ' '.join(cleaned).strip()

        if len(response) < 15:
            n_fail += 1
            continue

        out.write(json.dumps({
            "category": category,
            "prompt": prompt,
            "response": response,
        }, ensure_ascii=False) + '\n')
        out.flush()
        n_new += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = n_new / max(elapsed / 60, 0.01)
            remaining = (len(prompts) - i - 1) / max(rate, 0.1)
            print(f"  [{i+1}/{len(prompts)}] {n_new} new, {n_fail} fail, {rate:.1f}/min, ETA {remaining:.0f} min")

    out.close()
    elapsed = time.time() - start_time
    print(f"\n✅ Готово: +{n_new} новых пар, {n_fail} fail, {elapsed/60:.1f} мин")
    print(f"📤 {OUTPUT}")


if __name__ == '__main__':
    main()
