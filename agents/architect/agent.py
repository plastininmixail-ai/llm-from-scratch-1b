"""Architect Agent: проектирует модель и конфиг."""
from pathlib import Path
from typing import Optional

import yaml

from ..common import (
    BaseAgent, AgentRole, AgentTask, AgentResult, ProjectSpec,
)


class ArchitectAgent(BaseAgent):
    """Создаёт config для модели."""

    def _define_role(self) -> AgentRole:
        return AgentRole.ARCHITECT

    def execute(self, task: AgentTask) -> AgentResult:
        """Спроектировать архитектуру и сохранить config."""
        self.log(f"Designing architecture for {self.spec.target_params/1e9:.1f}B model")

        # Расчёт архитектуры для 1B
        # d_model=2048, n_heads=32, n_layers=24 → ~1.28B params
        config = {
            "model": {
                "d_model": self.spec.d_model,
                "n_heads": self.spec.n_heads,
                "n_layers": self.spec.n_layers,
                "max_seq_len": self.spec.seq_len,
                "vocab_size": self.spec.vocab_size,
                "dropout": 0.0,
                "bias": False,
                "rope_enabled": True,
                "rope_base": 10000.0,
                "tie_weights": True,
            },
            "training": {
                "batch_size": 1,
                "seq_len": self.spec.seq_len,
                "lr": 3e-4,
                "warmup": 500,
                "min_lr_ratio": 0.1,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "beta1": 0.9,
                "beta2": 0.95,
                "grad_accum": 16,  # эффективный batch = 16
            },
            "tokenizer": {
                "vocab_size": self.spec.vocab_size,
                "bpe_merges": self.spec.vocab_size - 256,
            },
            "data": {
                "binary_format": "uint16",
                "shuffle_seed": 42,
            },
        }

        # Сохраняем config
        config_path = Path("configs") / f"v0_1b.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

        # Расчёт параметров
        d = self.spec.d_model
        L = self.spec.n_layers
        V = self.spec.vocab_size
        # Embedding (tied with head, считаем один раз)
        emb_params = V * d
        # Per layer:
        # Attention: QKV (3*d²) + Proj (d²) = 4*d²
        attn_params = 4 * d * d
        # MLP: fc1 (d * 4d) + fc2 (4d * d) = 8*d²
        mlp_params = 8 * d * d
        layer_params = attn_params + mlp_params
        # Total (без final_norm, это мелочь)
        total_params = emb_params + L * layer_params + d

        metrics = {
            "total_params": total_params,
            "total_params_B": total_params / 1e9,
            "target_B": self.spec.target_params / 1e9,
            "matches_target": abs(total_params - self.spec.target_params) / self.spec.target_params < 0.5,
            "embedding_params": emb_params,
            "per_layer_params": layer_params,
        }

        self.log(f"Total params: {total_params/1e9:.3f}B")
        self.log(f"Config saved to {config_path}")

        # Уведомляем Trainer
        self.send_message(
            AgentRole.TRAINER,
            "Architecture designed",
            {"config_path": str(config_path), "metrics": metrics}
        )

        return AgentResult(
            task_id=task.task_id,
            role=self.role,
            success=True,
            data=config,
            message=f"Architecture designed: {total_params/1e9:.2f}B params",
            artifacts=[config_path],
            metrics=metrics,
        )


if __name__ == "__main__":
    from ..common import ProjectSpec
    spec = ProjectSpec()
    agent = ArchitectAgent(spec)
    task = AgentTask(task_id="arch_001", role=AgentRole.ARCHITECT, description="Design 1B model")
    result = agent.execute(task)
    print(f"\nResult: {result.message}")
    print(f"Metrics: {result.metrics}")
