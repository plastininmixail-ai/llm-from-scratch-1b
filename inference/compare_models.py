"""
Сравнительный eval всех моделей: генерирует тексты по одинаковым промптам.

Использование:
    python -m inference.compare_models \
        --prompts en "The capital of France is" ru "Привет, мой друг" \
        --out results/comparison.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from inference.generate import generate_text, load_model_from_checkpoint

DEFAULT_CHECKPOINTS = [
    ("small-v2", "checkpoints/small-v2/best.pt"),
    ("small-v3", "checkpoints/small-v3/best.pt"),
    ("small-v4", "checkpoints/small-v4/best.pt"),
    ("medium-v2", "checkpoints/medium-v2/best.pt"),
]

DEFAULT_PROMPTS = [
    ("en", "The capital of France is"),
    ("en", "Mathematics is the"),
    ("en", "def hello():"),
    ("en", "The history of Russia begins"),
    ("ru", "Привет, меня зовут"),
    ("ru", "Вчера я пошёл в"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer/vocab.json"))
    p.add_argument("--prompts", nargs="+", help="lang + text, e.g. en 'The...'")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--out", type=Path, default=Path("results/comparison.txt"))
    args = p.parse_args()

    prompts = []
    if args.prompts:
        i = 0
        while i < len(args.prompts):
            lang = args.prompts[i]
            text = args.prompts[i + 1]
            prompts.append((lang, text))
            i += 2
    else:
        prompts = DEFAULT_PROMPTS

    print(f"=== сравнение {len(DEFAULT_CHECKPOINTS)} моделей × {len(prompts)} промптов ===\n")

    results = {}
    for name, ckpt in DEFAULT_CHECKPOINTS:
        if not Path(ckpt).exists():
            print(f"⚠ пропускаю {name}: {ckpt} не найден")
            continue
        print(f"▶ загружаю {name} ({ckpt})…")
        try:
            model = load_model_from_checkpoint(ckpt, args.tokenizer, device="cpu")
        except Exception as e:
            print(f"  ⚠ ошибка загрузки: {e}")
            continue
        results[name] = {}
        for lang, prompt in prompts:
            try:
                mdl, tok, _ = model
                txt = generate_text(
                    mdl,
                    tok,
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device="cpu",
                )
                results[name][prompt] = txt
            except Exception as e:
                results[name][prompt] = f"[ОШИБКА: {e}]"

    # Печать таблицы
    lines = []
    lines.append("=" * 80)
    lines.append(f"СРАВНИТЕЛЬНАЯ ГЕНЕРАЦИЯ — {len(results)} моделей")
    lines.append("=" * 80)
    for prompt_tuple in prompts:
        lang, prompt = prompt_tuple
        lines.append(f"\n[{lang.upper()}] prompt: {prompt!r}")
        lines.append("-" * 80)
        for name in results:
            txt = results[name].get(prompt, "[нет результата]")
            lines.append(f"[{name}] {txt}")

    text = "\n".join(lines)
    print("\n" + text)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"\n✓ сохранено в {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
