"""Загрузка чекпойнтов и метаданных на HuggingFace Hub."""
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env():
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=True)


def get_hf_token() -> Optional[str]:
    load_env()
    return os.getenv("HF_TOKEN")


def upload_checkpoint(
    checkpoint_path: Path,
    repo_id: str,
    token: Optional[str] = None,
    commit_message: str = "Upload checkpoint",
):
    """Загрузить чекпойнт на HF Hub."""
    from huggingface_hub import HfApi, create_repo

    if token is None:
        token = get_hf_token()

    if not token:
        print("[error] HF_TOKEN not found in .env")
        return False

    api = HfApi(token=token)

    # Создать репо если не существует
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=False,
            exist_ok=True,
            repo_type="model",
        )
        print(f"[ok] Repo {repo_id} ready")
    except Exception as e:
        print(f"[warn] Repo create: {e}")

    # Загрузить чекпойнт
    if not checkpoint_path.exists():
        print(f"[error] Checkpoint not found: {checkpoint_path}")
        return False

    try:
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=checkpoint_path.name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        print(f"[ok] Uploaded {checkpoint_path.name} to {repo_id}")
        return True
    except Exception as e:
        print(f"[error] Upload failed: {e}")
        return False


def upload_model_card(repo_id: str, token: Optional[str] = None):
    """Загрузить README с описанием модели."""
    from huggingface_hub import HfApi

    if token is None:
        token = get_hf_token()

    if not token:
        print("[error] HF_TOKEN not found")
        return False

    card_content = f"""---
license: apache-2.0
tags:
- llm
- gpt
- from-scratch
- rmsnorm
- rope
language:
- en
- ru
---

# LLM from Scratch — 1B Model

GPT-style decoder-only transformer trained from scratch with custom framework.

## Architecture

- **Parameters**: ~1.07B
- **Layers**: 20
- **d_model**: 2048
- **Heads**: 32
- **Context**: 1024 tokens
- **Vocab**: 32K BPE
- **Norm**: RMSNorm
- **Position**: RoPE

## Training

- Warm-start from 113M base (v0)
- 5000 steps on FineWeb-Edu (30B tokens)
- AdamW, lr=1.5e-4, cosine schedule
- CPU-only training

## Framework

Custom multi-agent framework in `agents/`:
- Architect (model design)
- Data Engineer (corpus preparation)
- Trainer (training loop)
- Curator (checkpoint management)
- Orchestrator (coordination)

## Code

Full source code: https://github.com/plastininmixail-ai/llm-from-scratch-1b
"""

    api = HfApi(token=token)
    try:
        from huggingface_hub import hf_hub_download
        # Upload as README.md
        from io import BytesIO
        api.upload_file(
            path_or_fileobj=BytesIO(card_content.encode()),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Upload model card",
        )
        print(f"[ok] Model card uploaded")
        return True
    except Exception as e:
        print(f"[error] Card upload failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt file")
    p.add_argument("--repo", required=True, help="HF repo id (e.g., user/model)")
    p.add_argument("--card", action="store_true", help="Upload model card")
    args = p.parse_args()

    load_env()

    if args.card:
        upload_model_card(args.repo)

    success = upload_checkpoint(
        Path(args.checkpoint),
        args.repo,
    )
    sys.exit(0 if success else 1)
