"""Trainer Agent: обучает модель, мониторит, не прерывает."""
from pathlib import Path
import subprocess
import time

from ..common import (
    BaseAgent, AgentRole, AgentTask, AgentResult, ProjectSpec,
)


class TrainerAgent(BaseAgent):
    """Управляет обучением модели. НЕ прерывает без серьёзной причины."""

    def _define_role(self) -> AgentRole:
        return AgentRole.TRAINER

    def execute(self, task: AgentTask) -> AgentResult:
        """Запустить обучение через свой framework."""
        config_path = task.inputs.get("config_path", "configs/v0_1b.yaml")
        data_path = task.inputs.get("data_path", "data/binary/v0")
        output_dir = task.inputs.get("output_dir", "checkpoints/v0_1b")
        max_steps = task.inputs.get("max_steps", self.spec.max_steps)
        lr = task.inputs.get("lr", 3e-4)
        batch_size = task.inputs.get("batch_size", 1)
        seq_len = task.inputs.get("seq_len", self.spec.seq_len)

        self.log(f"Starting training: {config_path}")
        self.log(f"  output: {output_dir}")
        self.log(f"  max_steps: {max_steps}, lr: {lr}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Запускаем через свой framework (используем venv python)
        import sys
        venv_python = sys.executable
        cmd = [
            venv_python, "-m", "framework.training.trainer",
            "--config", config_path,
            "--data", data_path,
            "--output", output_dir,
            "--max-steps", str(max_steps),
            "--lr", str(lr),
            "--batch-size", str(batch_size),
            "--seq-len", str(seq_len),
        ]

        self.log(f"Command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        last_status_time = time.time()
        train_log = Path(output_dir) / "train_output.log"
        with train_log.open("w", encoding="utf-8") as f:
            for line in process.stdout:
                f.write(line)
                f.flush()
                if time.time() - last_status_time > 60:
                    self.log(f"Training running... (last: {line.strip()[:100]})")
                    last_status_time = time.time()

        process.wait()
        success = process.returncode == 0

        self.log(f"Training {'completed' if success else 'FAILED'} code={process.returncode}")
        return AgentResult(
            task_id=task.task_id, role=self.role,
            success=success,
            data={"exit_code": process.returncode, "log": str(train_log)},
            message=f"Training {'OK' if success else 'FAILED'}",
            artifacts=[Path(output_dir) / "best.pt"],
        )


if __name__ == "__main__":
    from ..common import ProjectSpec
    spec = ProjectSpec()
    agent = TrainerAgent(spec)
    task = AgentTask(
        task_id="train_001",
        role=AgentRole.TRAINER,
        description="Train 1B",
        inputs={"config_path": "configs/v0_1b.yaml", "max_steps": 100}
    )
    # Не запускаем реально в тесте
    print("Trainer agent ready")
