"""
Генерация текста обученной моделью.

Использование (как модуль):
    from inference import load_model_from_checkpoint, generate_text
    model, tok, cfg = load_model_from_checkpoint(
        "checkpoints/small-v1/best.pt", "tokenizer/vocab.json"
    )
    text = generate_text(model, tok, "Once upon a time",
                          max_new_tokens=64, temperature=0.8, top_k=50)

Использование (CLI):
    python -m inference.generate \
        --checkpoint checkpoints/small-v1/best.pt \
        --tokenizer tokenizer/vocab.json \
        --prompt "Привет," \
        --max-new-tokens 50 \
        --temperature 0.8 \
        --top-k 50
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from model import GPT, ModelConfig
from tokenizer.bpe import BPETokenizer


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str = "cpu",
) -> tuple[GPT, BPETokenizer, ModelConfig]:
    """Загружает чекпойнт модели + BPE-токенизатор."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 1. Определяем ModelConfig:
    #    новые чекпойнты содержат 'model_config'
    #    старые — только TrainerConfig в 'config', либо вовсе ничего —
    #    тогда восстанавливаем из state_dict по размерностям
    if "model_config" in ckpt:
        cfg = ModelConfig(**ckpt["model_config"])
    elif "model_config_dict" in ckpt:
        cfg = ModelConfig(**ckpt["model_config_dict"])
    else:
        # восстанавливаем из state_dict
        sd = ckpt["model"]
        vocab_size = sd["tok_emb.weight"].shape[0]
        d_model = sd["tok_emb.weight"].shape[1]
        max_seq_len = sd["pos_emb.weight"].shape[0]
        # n_layers: считаем блоки DecoderBlock
        block_keys = [k for k in sd if k.startswith("blocks.")]
        # каждый блок содержит ~8 параметров (ln1.w/b, attn.qkv, attn.proj,
        # ln2.w/b, mlp.fc1, mlp.fc2 — bias=False в Linear)
        n_layers = max(1, len(block_keys) // 8)
        # n_heads: пробуем угадать по делителям d_model
        candidates = [h for h in (4, 6, 8, 12, 16)
                      if d_model % h == 0]
        n_heads = candidates[0] if candidates else 4
        cfg = ModelConfig(
            vocab_size=vocab_size, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers,
            max_seq_len=max_seq_len, dropout=0.0,
        )

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    tok = BPETokenizer.load(tokenizer_path)
    return model, tok, cfg


@torch.no_grad()
def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cpu",
) -> str:
    """Генерирует продолжение текста по prompt'у.

    Args:
        model: обученная GPT
        tokenizer: BPE-токенизатор
        prompt: начало текста
        max_new_tokens: сколько токенов сгенерировать
        temperature: >1 — креативнее, <1 — детерминированнее
        top_k: оставить топ-k токенов с самой высокой вероятностью
        top_p: nucleus sampling — оставить топ токенов с суммарной вероятностью p
    """
    # 1. Токенизируем prompt
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]  # fallback: id нулевого байта
    # обрезаем до max_seq_len (модель имеет фиксированный контекст)
    max_ctx = model.config.max_seq_len
    if len(ids) > max_ctx:
        ids = ids[-max_ctx:]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # 2. Генерируем по токену
    for _ in range(max_new_tokens):
        # forward
        with torch.no_grad():
            out = model(x)
            # GPT.forward возвращает (logits, loss) или просто logits
            logits = out[0] if isinstance(out, tuple) else out
        # берем последний токен
        logits = logits[0, -1, :] / max(temperature, 1e-5)  # (vocab,)

        # top-k фильтрация
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.size(-1)))
            mask = torch.full_like(logits, float("-inf"))
            mask[topk_idx] = topk_vals
            logits = mask

        # top-p (nucleus) фильтрация
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            # токены, у которых cumprob > top_p, исключаем (кроме первого)
            cutoff = cumprobs > top_p
            cutoff[0] = False
            sorted_logits[cutoff] = float("-inf")
            # возвращаем обратно
            logits = torch.full_like(logits, float("-inf"))
            logits[sorted_idx] = sorted_logits

        # softmax → сэмплируем
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        # приклеиваем к контексту
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        # обрезаем контекст
        if x.size(1) > max_ctx:
            x = x[:, -max_ctx:]

    # 3. Декодируем
    out_ids = x[0].tolist()
    return tokenizer.decode(out_ids)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--prompt", type=str, required=True,
                   help="начальный текст (в кавычках)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    print(f"загружаю чекпойнт {args.checkpoint}…")
    model, tok, cfg = load_model_from_checkpoint(
        args.checkpoint, args.tokenizer, device=args.device
    )
    print(f"модель: {model.info()}")
    print(f"prompt: {args.prompt!r}")
    print("—" * 40)
    out = generate_text(
        model, tok, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )
    print(out)
    print("—" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())