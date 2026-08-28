"""
Строит loss/perplexity графики по всем runs проекта.

Использование:
    python scripts/plot_runs.py --out-dir results/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib не установлен. Установите: .venv/Scripts/pip install matplotlib"
    ) from e

DEFAULT_RUNS = [
    ("small-v1-2000steps", "checkpoints/small-v1-2000steps/train_log.jsonl", "v1 (3.3M, 2000)"),
    ("small-v2", "checkpoints/small-v2/train_log.jsonl", "v2 (3.3M, 8000)"),
    ("small-v3", "checkpoints/small-v3/train_log.jsonl", "v3 (3.3M warm-start)"),
    ("small-v4", "checkpoints/small-v4/train_log.jsonl", "v4 (3.3M warm-start)"),
    ("medium-v2", "checkpoints/medium-v2/train_log.jsonl", "medium-v2 (10.8M, 6000)"),
    ("medium-v3", "checkpoints/medium-v3/train_log.jsonl", "medium-v3 (10.8M warm-start)"),
]


def load_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    """Возвращает steps, train_losses, val_losses."""
    steps, train_losses, val_steps, val_losses = [], [], [], []
    if not path.exists():
        return steps, train_losses, [], []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "train":
                steps.append(rec["step"])
                train_losses.append(rec["loss"])
            elif rec.get("type") == "eval":
                val_steps.append(rec["step"])
                val_losses.append(rec["val_loss"])
    return steps, train_losses, val_steps, val_losses


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss-кривые (один график)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, log_path, label in DEFAULT_RUNS:
        steps, train_losses, val_steps, val_losses = load_log(Path(log_path))
        if not steps:
            print(f"⚠ пропускаю {name}: нет {log_path}")
            continue
        ax.plot(val_steps, val_losses, marker="o", markersize=3, label=f"{label} val")
    ax.set_xlabel("step")
    ax.set_ylabel("val_loss")
    ax.set_title("Validation loss по всем ранам (ниже = лучше)")
    ax.set_ylim(3.0, 5.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out1 = args.out_dir / "val_loss_curves.png"
    fig.savefig(out1, dpi=120)
    print(f"✓ сохранено {out1}")
    plt.close(fig)

    # 2. Bar-chart финальных результатов
    final = []
    for name, log_path, label in DEFAULT_RUNS:
        _, _, vs, vl = load_log(Path(log_path))
        if vl:
            final.append((label, min(vl)))
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{lbl}" for lbl, _ in final]
    vals = [v for _, v in final]
    bars = ax.bar(labels, vals, color=["#888"] * (len(labels) - 1) + ["#2a9d8f"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center")
    ax.set_ylabel("min val_loss")
    ax.set_title("Лучший val_loss по ранам")
    ax.set_ylim(3.0, 4.5)
    plt.xticks(rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out2 = args.out_dir / "best_val_loss_bar.png"
    fig.savefig(out2, dpi=120)
    print(f"✓ сохранено {out2}")
    plt.close(fig)

    # 3. Train loss curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, log_path, label in DEFAULT_RUNS:
        steps, train_losses, _, _ = load_log(Path(log_path))
        if not steps:
            continue
        ax.plot(steps, train_losses, label=label, alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("train_loss")
    ax.set_title("Train loss по всем ранам (smoothed)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out3 = args.out_dir / "train_loss_curves.png"
    fig.savefig(out3, dpi=120)
    print(f"✓ сохранено {out3}")
    plt.close(fig)

    print(f"\n✓ все графики в {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
