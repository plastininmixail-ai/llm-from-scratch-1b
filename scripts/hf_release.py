"""Генерирует release notes для HF Hub при новой версии модели."""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

from scripts.upload_to_hf import load_env, get_hf_token
from huggingface_hub import HfApi


def generate_release_notes(checkpoint_path: Path, step: int, perplexity: float = None) -> str:
    notes = f"""# Release v{step}

**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Training Status

- **Checkpoint**: `{checkpoint_path.name}`
- **Step**: {step}
"""
    if perplexity:
        notes += f"- **Perplexity**: {perplexity:.4f}\n"
    notes += f"""
## Architecture

1.07B parameter GPT-style decoder-only transformer:
- 20 layers, d_model=2048, 32 heads
- RMSNorm + RoPE
- 32K BPE vocabulary
- 1024 token context

## Improvements over previous version

- Continued pretrain on FineWeb-Edu (educational web text)
- Warm-start from 113M Stage 0 base
- AdamW with cosine schedule, lr=1.5e-4

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from framework.model import GPT, load_config
from framework.inference import generate

# Download checkpoint
ckpt_path = hf_hub_download(
    repo_id="mixailplastinin/llm-from-scratch-1b",
    filename="{checkpoint_path.name}",
)

# Load model
cfg = load_config("configs/v1_fineweb.yaml")
model = GPT(cfg)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

# Generate
ids = [1, 2, 3]  # tokenize your prompt
x = torch.tensor([ids])
out = generate(model, x, max_new_tokens=50)
```
"""
    return notes


def create_hf_release(checkpoint_path: Path, step: int, perplexity: float = None) -> bool:
    """Создать release (tag) на HF Hub."""
    load_env()
    token = get_hf_token()
    if not token:
        print("ERROR: no HF_TOKEN")
        return False

    api = HfApi(token=token)
    repo_id = "mixailplastinin/llm-from-scratch-1b"

    try:
        # Upload as a tag (release)
        notes = generate_release_notes(checkpoint_path, step, perplexity)
        api.create_tag(
            repo_id=repo_id,
            tag=f"v{step}",
            message=notes[:200],
        )
        print(f"[ok] Created release v{step}")
        return True
    except Exception as e:
        print(f"[error] {e}")
        return False


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--perplexity", type=float, default=None)
    args = p.parse_args()

    success = create_hf_release(
        Path(args.checkpoint),
        args.step,
        args.perplexity,
    )
    sys.exit(0 if success else 1)
