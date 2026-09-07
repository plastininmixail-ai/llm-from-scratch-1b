"""Конвертирует DPO checkpoint в формат v94 (key='model')."""
import torch
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")
SRC = ROOT / "checkpoints/chat-xlarge-v94-dpo/best.pt"
DST = ROOT / "checkpoints/chat-xlarge-v94-dpo/best.pt"  # перезаписать

ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
print("Source keys:", list(ckpt.keys()))

new_ckpt = {
    "model": ckpt["model_state_dict"],
    "step": ckpt.get("step", 0),
    "loss": ckpt.get("loss", 0.0),
}
torch.save(new_ckpt, DST)
print(f"✅ Converted: {SRC}")
print(f"   Keys now: {list(new_ckpt.keys())}")
