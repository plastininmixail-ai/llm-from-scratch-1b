"""Curator Agent — управляет чекпойнтами и метриками."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..common import (
    BaseAgent, AgentRole, AgentTask, AgentResult, ProjectSpec,
)


class CuratorAgent(BaseAgent):
    """Управляет артефактами: чекпойнты, логи, метрики."""

    def _define_role(self):
        return AgentRole.CURATOR

    def __init__(self, spec: ProjectSpec):
        super().__init__(spec)
        self.checkpoints_dir = Path("checkpoints")
        self.logs_dir = Path("logs")
        self.metrics_path = self.logs_dir / "metrics.jsonl"

    def execute(self, task: AgentTask) -> AgentResult:
        action = task.inputs.get("action", "report")

        if action == "report":
            return self._report()
        elif action == "list":
            return self._list_checkpoints()
        elif action == "cleanup":
            return self._cleanup(task.inputs)
        else:
            return AgentResult(
                task_id=task.task_id,
                role=self.role,
                success=False,
                message=f"Unknown action: {action}",
            )

    def _report(self) -> AgentResult:
        """Сводный отчёт по всем чекпойнтам + ASCII график loss."""
        if not self.checkpoints_dir.exists():
            return AgentResult(
                task_id="report",
                role=self.role,
                success=False,
                message="No checkpoints directory",
            )

        ckpts = []
        total_size = 0
        for d in self.checkpoints_dir.iterdir():
            if not d.is_dir():
                continue
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            total_size += size

            # Попытка прочитать метрики из train_log
            log_file = d / "train_log.jsonl"
            best_val = None
            final_step = None
            loss_history = []
            if log_file.exists():
                try:
                    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
                    best_val = float("inf")
                    for line in lines:
                        row = json.loads(line)
                        if "val_loss" in row:
                            if row["val_loss"] < best_val:
                                best_val = row["val_loss"]
                        final_step = row.get("step", final_step)
                        if "loss" in row:
                            loss_history.append((row["step"], row["loss"]))
                except Exception:
                    pass

            ckpts.append({
                "name": d.name,
                "size_gb": round(size / 1e9, 2),
                "best_val": best_val,
                "final_step": final_step,
                "loss_history": loss_history,
            })

        ckpts.sort(key=lambda c: c["best_val"] or float("inf"))

        # ASCII график для всех чекпойнтов с loss history
        graphs = []
        # Сортируем: сначала те у которых есть валидный val_loss
        with_val = [c for c in ckpts if c.get("best_val") and c["best_val"] != float("inf")]
        with_val.sort(key=lambda c: c["best_val"])
        for c in with_val[:3]:
            if c["loss_history"]:
                graph = self._ascii_graph(c["loss_history"])
                graphs.append({"name": c["name"], "graph": graph})

        report = {
            "total_checkpoints": len(ckpts),
            "total_size_gb": round(total_size / 1e9, 2),
            "best": ckpts[0] if ckpts else None,
            "all": [{k: v for k, v in c.items() if k != "loss_history"} for c in ckpts],
            "loss_graphs": graphs,
        }

        # Логируем
        self.log(f"Found {len(ckpts)} checkpoints, total {total_size/1e9:.1f} GB")
        if ckpts and ckpts[0].get("best_val") is not None:
            self.log(f"Best: {ckpts[0]['name']} val={ckpts[0]['best_val']:.4f}")

        return AgentResult(
            task_id="report",
            role=self.role,
            success=True,
            data=report,
            message=f"{len(ckpts)} checkpoints, best={ckpts[0]['name'] if ckpts else 'none'}",
        )

    def _ascii_graph(self, history: list, width: int = 40, height: int = 10) -> str:
        """ASCII-график loss history."""
        if not history:
            return ""
        losses = [h[1] for h in history]
        steps = [h[0] for h in history]

        min_l, max_l = min(losses), max(losses)
        if max_l - min_l < 0.01:
            max_l = min_l + 0.01

        # Семплируем N точек
        if len(losses) > width:
            indices = [int(i * len(losses) / width) for i in range(width)]
            sampled = [losses[i] for i in indices]
        else:
            sampled = losses

        lines = []
        for row in range(height, 0, -1):
            threshold = min_l + (max_l - min_l) * row / height
            line = ""
            for val in sampled:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            lines.append(line)

        # Header / footer
        first_step = steps[0] if steps else 0
        last_step = steps[-1] if steps else 0
        lines.insert(0, f"Loss {min_l:.2f} ──────────────────────────────── {max_l:.2f}")
        lines.append(f"Step {first_step} ──────────────────────────────── {last_step}")
        return "\n".join(lines)

    def _list_checkpoints(self) -> AgentResult:
        return self._report()

    def _cleanup(self, inputs: dict) -> AgentResult:
        """Очистка: удалить плохие чекпойнты."""
        keep_best = inputs.get("keep_best", 3)
        return AgentResult(
            task_id="cleanup",
            role=self.role,
            success=False,
            message="Cleanup disabled — too dangerous without explicit per-target approval",
        )
