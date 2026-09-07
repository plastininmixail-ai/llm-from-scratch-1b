"""Прогон RU/EN парных вопросов через v82 и v94.

Выходной формат: для каждой пары показываем ответы обеих моделей
на RU и EN, чтобы наглядно сравнить.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
INPUT = ROOT / "data/ru_en_pairs.jsonl"
OUTPUT = ROOT / "data/ru_en_results.jsonl"

CHECKPOINTS = [
    ("v82", ROOT / "checkpoints/chat-xlarge-v82/best.pt"),
    ("v94", ROOT / "checkpoints/chat-xlarge-v94-sft-final/best.pt"),
]

sys.path.insert(0, str(ROOT))
from inference.generate import generate_text, load_model_from_checkpoint
from inference.telegram_bot import _pick_few_shot


def main():
    questions = []
    with INPUT.open('r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"📊 {len(questions)} вопросов (RU/EN пары)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUTPUT.open('w', encoding='utf-8')

    for name, path in CHECKPOINTS:
        if not path.exists():
            print(f"⚠ {name}: не найден")
            continue

        print(f"\n{'='*60}")
        print(f"🤖 {name}: загружаю {path.name}...")
        print(f"{'='*60}")

        t0 = time.time()
        model, tokenizer, cfg = load_model_from_checkpoint(path, ROOT / "tokenizer/vocab.json")
        print(f"✓ загрузка {time.time()-t0:.1f}s, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        for i, q in enumerate(questions):
            prompt = q['prompt']
            lang = q['lang']
            pair_id = q['pair_id']

            # Few-shot prefix (auto-detect по языку)
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
                for sep in ("\nQ:", "\n\nQ:", "\nA:", "\n\nA:"):
                    if sep in text:
                        text = text.split(sep)[0].strip()
                        break
            except Exception as e:
                text = f"ERROR: {e}"
            t_gen = time.time() - t0

            out.write(json.dumps({
                "model": name,
                "lang": lang,
                "pair_id": pair_id,
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
