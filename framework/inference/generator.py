"""Свой Inference Engine — без HuggingFace, без vLLM."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


from ..model import GPT, ModelConfig, load_config


@torch.inference_mode()
def generate(
    model: GPT,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Генерация токенов по промпту.

    Args:
        model: GPT модель
        prompt_ids: (1, T) — тензор с токенами промпта
        max_new_tokens: сколько новых токенов сгенерировать
        temperature: 0=greedy, 1=normal, >1=creative
        top_k: оставить top-k токенов (0=off)
        top_p: nucleus sampling (1.0=off)
        eos_token_id: остановить на этом токене
        device: 'cpu' или 'cuda'

    Returns:
        (1, T + max_new_tokens) — тензор с промптом + генерацией
    """
    model.eval()
    idx = prompt_ids.to(device).clone()
    T = idx.shape[1]

    for _ in range(max_new_tokens):
        # Контекст — последние max_seq_len токенов
        idx_cond = idx if idx.shape[1] <= model.config.max_seq_len else idx[:, -model.config.max_seq_len:]

        # Forward
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)  # (1, vocab)

        # Top-k
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return idx


def load_model_for_inference(
    checkpoint_path: Path,
    config_path: Path,
    device: str = "cpu",
) -> GPT:
    """Загрузить модель из чекпойнта для инференса."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = load_config(str(config_path))
    model = GPT(cfg)

    # Поддержка разных форматов чекпойнта
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # Удаляем rope.* ключи если есть (они динамические)
    sd = {k: v for k, v in sd.items() if not k.endswith((".inv_freq",))}
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def load_tokenizer(tokenizer_path: Path):
    """Загрузить BPE tokenizer."""
    import sys
    sys.path.insert(0, ".")
    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer()
    tok.load(str(tokenizer_path))
    return tok
