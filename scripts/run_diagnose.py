"""Прогон 100 вопросов через все модели, сохранение в JSON.

Использование:
    python -m scripts.run_diagnose
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
QUESTIONS = ROOT / "data/diagnostic_questions.jsonl"
OUTPUT = ROOT / "data/diagnostic_results.jsonl"

CHECKPOINTS = [
    ("v82", ROOT / "checkpoints/chat-xlarge-v82/best.pt"),
    ("v93", ROOT / "checkpoints/chat-xlarge-v93-sft-oasst1/best.pt"),
    ("v94", ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt"),
]

# Добавляем корень проекта в path
sys.path.insert(0, str(ROOT))
from inference.generate import generate_text, load_model_from_checkpoint
from inference.telegram_bot import _pick_few_shot


def main():
    questions = []
    with QUESTIONS.open('r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"📊 {len(questions)} вопросов")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUTPUT.open('w', encoding='utf-8')

    # Загружаем каждую модель и гоняем
    for name, path in CHECKPOINTS:
        if not path.exists():
            print(f"⚠ {name}: {path} не найден")
            continue

        print(f"\n{'='*60}")
        print(f"🤖 {name}: загружаю {path.name}...")
        print(f"{'='*60}")

        t0 = time.time()
        model, tokenizer, cfg = load_model_from_checkpoint(path, ROOT / "tokenizer/vocab.json")
        t_load = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ {n_params/1e6:.2f}M параметров, загрузка {t_load:.1f}s")

        # Гоняем все вопросы
        for i, q in enumerate(questions):
            prompt = q['prompt']
            cat = q['category']

            # Few-shot prefix
            full_prompt = _pick_few_shot(prompt) + prompt

            t0 = time.time()
            try:
                text = generate_text(
                    model, tokenizer,
                    prompt=full_prompt,
                    max_new_tokens=80,
                    temperature=0.2,
                    top_k=40,
                    top_p=0.95,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=4,
                    device="cpu",
                )
                # Чистим
                for sep in ("\nQ:", "\n\nQ:", "\nA:", "\n\nA:"):
                    if sep in text:
                        text = text.split(sep)[0].strip()
                        break
            except Exception as e:
                text = f"ERROR: {e}"
            t_gen = time.time() - t0

            out.write(json.dumps({
                "model": name,
                "category": cat,
                "prompt": prompt,
                "response": text.strip() if text else "",
                "gen_time": round(t_gen, 2),
            }, ensure_ascii=False) + '\n')
            out.flush()

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] {name} avg {t_gen:.1f}s/q")

    out.close()
    print(f"\n✅ Готово: {OUTPUT}")


if __name__ == '__main__':
    main()
