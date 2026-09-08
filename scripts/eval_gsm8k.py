"""GSM8K Evaluation Framework.

Оценивает математические способности модели на GSM8K test set.
Используется для Stage 2 (reasoning training).

Метрики:
- Accuracy (точное совпадение ответа)
- Format compliance (правильный формат вывода)
- CoT quality (наличие рассуждений)
"""
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, ".")


def extract_answer(response: str) -> Optional[str]:
    """Извлечь финальный ответ из GSM8K-style ответа.

    GSM8K format: "... рассуждение ... #### <число>"
    """
    # Попытка найти после ####
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", response)
    if match:
        return match.group(1).strip()

    # Попытка найти последнее число
    numbers = re.findall(r"[-+]?\d*\.?\d+", response)
    if numbers:
        return numbers[-1]

    return None


def normalize_number(s: str) -> Optional[float]:
    """Парсит строку в число."""
    try:
        # Удаляем запятые в числах (1,000 → 1000)
        s = s.replace(",", "").strip()
        return float(s)
    except (ValueError, AttributeError):
        return None


def check_format(response: str) -> dict:
    """Проверить форматирование ответа."""
    has_cot = bool(re.search(r"(step|step\s+\d|=|because|therefore|so)", response, re.IGNORECASE))
    has_number = bool(re.search(r"\d", response))
    has_marker = "####" in response

    return {
        "has_cot": has_cot,
        "has_number": has_number,
        "has_answer_marker": has_marker,
    }


def evaluate_single(model_response: str, ground_truth: str) -> dict:
    """Оценить один ответ модели."""
    predicted = extract_answer(model_response)
    gt_normalized = normalize_number(ground_truth)
    pred_normalized = normalize_number(predicted) if predicted else None

    correct = (
        gt_normalized is not None
        and pred_normalized is not None
        and abs(gt_normalized - pred_normalized) < 1e-6
    )

    fmt = check_format(model_response)

    return {
        "correct": correct,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "format": fmt,
    }


def load_gsm8k_test(max_samples: int = 100) -> list:
    """Загрузить GSM8K test set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test", streaming=True)
        questions = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            questions.append({
                "question": row["question"],
                "answer": row["answer"],
                "ground_truth": extract_answer(row["answer"]),
            })
        return questions
    except Exception as e:
        print(f"GSM8K load error: {e}")
        return []


def evaluate_model(checkpoint_path: Path, config_path: Path, max_samples: int = 100):
    """Запустить eval на GSM8K."""
    from framework.inference import generate, load_model_for_inference
    from framework.data.simple_tokenizer import SimpleBPETokenizer

    print(f"Loading model from {checkpoint_path}...")
    model = load_model_for_inference(checkpoint_path, config_path)
    tok = SimpleBPETokenizer("tokenizer/vocab_3k.json")

    print(f"Loading GSM8K test ({max_samples} samples)...")
    questions = load_gsm8k_test(max_samples)
    if not questions:
        print("No GSM8K data available")
        return None

    print(f"Evaluating {len(questions)} problems...")
    results = []

    for i, q in enumerate(questions):
        # Генерируем ответ
        ids = tok.encode(q["question"])
        if not ids:
            ids = [1]
        x = torch.tensor([ids], dtype=torch.long)
        out = generate(model, x, max_new_tokens=256, temperature=0.0)  # greedy
        gen_ids = out[0, len(ids):].tolist()
        response = tok.decode(gen_ids)

        # Оцениваем
        eval_result = evaluate_single(response, q["ground_truth"])
        eval_result["question"] = q["question"][:100]
        eval_result["response"] = response[:300]
        results.append(eval_result)

        if (i + 1) % 10 == 0:
            correct_so_far = sum(1 for r in results if r["correct"])
            acc = correct_so_far / len(results) * 100
            print(f"  [{i+1}/{len(questions)}] accuracy: {acc:.1f}%")

    # Summary
    n_correct = sum(1 for r in results if r["correct"])
    n_with_marker = sum(1 for r in results if r["format"]["has_answer_marker"])
    n_with_cot = sum(1 for r in results if r["format"]["has_cot"])

    summary = {
        "checkpoint": str(checkpoint_path),
        "n_samples": len(results),
        "accuracy": round(n_correct / len(results) * 100, 2),
        "format_compliance": round(n_with_marker / len(results) * 100, 2),
        "cot_presence": round(n_with_cot / len(results) * 100, 2),
        "results": results,
    }

    # Save
    out_path = Path(f"logs/gsm8k_eval_{checkpoint_path.parent.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GSM8K Results ({checkpoint_path.parent.name}):")
    print(f"  Accuracy:        {summary['accuracy']}%")
    print(f"  Format compliance: {summary['format_compliance']}%")
    print(f"  CoT presence:    {summary['cot_presence']}%")
    print(f"  Saved to:        {out_path}")
    print(f"{'='*60}")

    return summary


import torch  # в конце чтобы не дублировать

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/v0_1b.yaml")
    p.add_argument("--max-samples", type=int, default=50)
    args = p.parse_args()

    evaluate_model(
        Path(args.checkpoint),
        Path(args.config),
        args.max_samples,
    )
