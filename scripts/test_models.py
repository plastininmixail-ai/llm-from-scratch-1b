"""Local Multi-Checkpoint Tester — сравнение ответов разных моделей.

Не использует Telegram, токены или сеть. Только локальные чекпойнты.

Использование:
    python -m scripts.test_models "Привет, как дела?"
    python -m scripts.test_models --checkpoints v82,qwen "Что такое стоицизм?"
    python -m scripts.test_models --temperature 0.5 --max-tokens 100 "Расскажи анекдот"

Аргументы:
    prompt              — текст промпта (или --prompt)
    --checkpoints       — список чекпойнтов через запятую (v82,v93,qwen)
                          По умолчанию все доступные
    --max-tokens N      — длина генерации (default 80)
    --temperature T     — температура (default 0.2)
    --raw               — выключить few-shot (если поддерживается)
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")

# Реестр доступных чекпойнтов
# (имя, путь к best.pt, описание)
CHECKPOINTS = [
    ("v82", ROOT / "checkpoints/chat-xlarge-v82/best.pt", "pre-train baseline (ЛУЧШИЙ по val 2.6)"),
    ("v89", ROOT / "checkpoints/chat-xlarge-v89/best.pt", "warm-start от v82 на chat_v26"),
    ("v93", ROOT / "checkpoints/chat-xlarge-v93-sft-oasst1/best.pt", "SFT v93 на OASST1 (val 3.69)"),
    ("v94", ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt", "SFT v94 OASST1 + qwen3 (ЛУЧШИЙ val 3.62)"),
]


def _check_ollama() -> bool:
    """Проверить, доступен ли Ollama."""
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def _has_checkpoint(name: str) -> bool:
    for ckpt_name, path, _ in CHECKPOINTS:
        if ckpt_name == name:
            return path.exists()
    return False


def generate_local(checkpoint_path: Path, prompt: str,
                   max_tokens: int = 80, temperature: float = 0.2,
                   few_shot: bool = True) -> dict:
    """Генерация через локальный PyTorch чекпойнт."""
    # Добавляем корень проекта в path для импортов
    sys.path.insert(0, str(ROOT))

    from inference.generate import generate_text, load_model_from_checkpoint
    from inference.telegram_bot import _pick_few_shot

    t0 = time.time()
    model, tokenizer, cfg = load_model_from_checkpoint(
        checkpoint_path, ROOT / "tokenizer/vocab.json"
    )
    t_load = time.time() - t0

    # Собираем промпт
    if few_shot:
        prefix = _pick_few_shot(prompt)
        full_prompt = prefix + prompt
    else:
        full_prompt = prompt

    t0 = time.time()
    text = generate_text(
        model, tokenizer,
        prompt=full_prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        device="cpu",
    )
    t_gen = time.time() - t0

    # Чистим артефакты
    for sep in ("\nQ:", "\n\nQ:", "\nA:", "\n\nA:", "ref /ref", "<ref", "</ref"):
        if sep in text:
            text = text.split(sep)[0].strip()
            break

    return {
        "response": text.strip() if text else "(пустой ответ)",
        "load_time": t_load,
        "gen_time": t_gen,
    }


def generate_ollama(prompt: str, max_tokens: int = 80,
                    temperature: float = 0.2) -> dict:
    """Генерация через Ollama qwen3 (без токенов)."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx не установлен"}

    system = (
        "Ты — дружелюбный ИИ-ассистент. Отвечай кратко, по делу. "
        "Без списков, без кода, без преамбул."
    )
    payload = {
        "model": "qwen3-nothink",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    try:
        t0 = time.time()
        r = httpx.post("http://127.0.0.1:11434/api/chat",
                       json=payload, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        return {
            "response": data["message"]["content"].strip(),
            "gen_time": time.time() - t0,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default=None,
                   help="Текст промпта для тестирования")
    p.add_argument("--checkpoints", default="auto",
                   help="Список моделей: v82,v89,qwen (default: auto = все)")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--raw", action="store_true",
                   help="Выключить few-shot префикс")
    args = p.parse_args()

    prompt = args.prompt
    if not prompt:
        p.error("не указан prompt. Пример: python -m scripts.test_models 'Привет!'")

    # Какие модели тестировать
    if args.checkpoints == "auto":
        selected = ["v82"]  # дефолт — v82 (ЛУЧШИЙ)
        if _check_ollama():
            selected.append("qwen")
    else:
        selected = [s.strip() for s in args.checkpoints.split(",")]

    print(f"\n📊 Тестируем {len(selected)} модель(ей): {', '.join(selected)}")
    print(f"💬 Промпт: {prompt!r}\n")

    for name in selected:
        print("=" * 60)
        if name in ("qwen", "qwen3"):
            print(f"🤖 qwen3 (Ollama, 8B)")
        else:
            # Найти путь к чекпойнту
            ckpt_path = None
            for ckpt_name, path, desc in CHECKPOINTS:
                if ckpt_name == name:
                    ckpt_path = path
                    break
            if ckpt_path is None or not ckpt_path.exists():
                print(f"❌ {name}: чекпойнт не найден")
                continue
            n_params = sum(1 for _ in open(ckpt_path.parent / "train_log.jsonl")) if (ckpt_path.parent / "train_log.jsonl").exists() else 0
            print(f"🤖 {name} ({ckpt_path.parent.name})")
        print("=" * 60)

        try:
            if name in ("qwen", "qwen3"):
                result = generate_ollama(prompt, args.max_tokens, args.temperature)
            else:
                result = generate_local(
                    ckpt_path, prompt, args.max_tokens, args.temperature,
                    few_shot=not args.raw
                )
            if "error" in result:
                print(f"⚠ ошибка: {result['error']}")
            else:
                print(result["response"])
                timing = []
                if "load_time" in result:
                    timing.append(f"load={result['load_time']:.1f}s")
                timing.append(f"gen={result['gen_time']:.1f}s")
                print(f"\n⏱ {' · '.join(timing)}")
        except Exception as e:
            print(f"⚠ исключение: {e}")
        print()


if __name__ == '__main__':
    main()
