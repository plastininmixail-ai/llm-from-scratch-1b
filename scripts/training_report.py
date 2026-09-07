"""Генерирует красивый Markdown отчёт из training log."""
import sys
import json
from pathlib import Path
from datetime import datetime


def parse_log(log_path: Path):
    """Парсит train_log.jsonl."""
    if not log_path.exists():
        return None

    history = []
    for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
        try:
            history.append(json.loads(line))
        except Exception:
            pass
    return history


def make_loss_graph(history, width=60, height=12):
    """ASCII loss curve."""
    losses = [h.get("loss") for h in history if "loss" in h]
    if not losses:
        return "No loss data"

    min_l, max_l = min(losses), max(losses)
    if max_l - min_l < 0.01:
        max_l = min_l + 0.01

    # Sample
    if len(losses) > width:
        idx = [int(i * len(losses) / width) for i in range(width)]
        losses = [losses[i] for i in idx]

    lines = []
    lines.append(f"```")
    lines.append(f"Loss {min_l:.3f} ┐")
    for row in range(height, 0, -1):
        threshold = min_l + (max_l - min_l) * row / height
        line = ""
        for val in losses:
            line += "█" if val >= threshold else " "
        lines.append(f"│ {line} │")
    lines.append(f"└{'─' * (width + 2)}┘")
    lines.append(f"Step 0{' ' * (width - 8)}Step {len(history) - 1}")
    lines.append(f"```")
    return "\n".join(lines)


def main():
    log_path = Path("checkpoints/v1_fineweb/train_log.jsonl")
    if not log_path.exists():
        # Try older checkpoints
        for p in sorted(Path("checkpoints").rglob("train_log.jsonl"), reverse=True):
            log_path = p
            break

    history = parse_log(log_path)
    if not history:
        print("No training log found")
        return

    final_step = history[-1].get("step", "?")
    final_loss = history[-1].get("loss", "?")
    val_losses = [h.get("val_loss") for h in history if "val_loss" in h]
    best_val = min(val_losses) if val_losses else None

    # Loss progression
    checkpoints = [h for h in history if "step" in h]
    n = len(checkpoints)
    samples = [checkpoints[0], checkpoints[n // 4], checkpoints[n // 2],
               checkpoints[3 * n // 4], checkpoints[-1]] if n >= 5 else checkpoints

    report = []
    report.append(f"# Training Report\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append(f"**Log**: `{log_path}`\n\n")

    report.append(f"## Summary\n")
    report.append(f"- Total steps: **{final_step}**\n")
    report.append(f"- Final train loss: **{final_loss:.4f}**\n" if isinstance(final_loss, float) else f"- Final train loss: {final_loss}\n")
    if best_val:
        report.append(f"- Best val loss: **{best_val:.4f}**\n")
    report.append(f"\n")

    report.append(f"## Loss Curve\n")
    report.append(make_loss_graph(history))
    report.append(f"\n\n")

    report.append(f"## Loss Progression\n")
    report.append(f"| Step | Loss | Val Loss | LR |")
    report.append(f"|------|------|----------|-----|")
    for h in samples:
        step = h.get("step", "?")
        loss = h.get("loss", "")
        val = h.get("val_loss", "")
        lr = h.get("lr", "")
        report.append(f"| {step} | {loss:.4f} | {val if val else '-'} | {lr:.2e} |")
    report.append(f"\n")

    report_path = Path("logs/TRAINING_REPORT.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
