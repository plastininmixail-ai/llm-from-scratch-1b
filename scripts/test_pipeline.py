"""Тест полного pipeline: Architect → Data Engineer → Trainer."""
import sys
sys.path.insert(0, ".")

from agents.common import ProjectSpec, AgentTask, AgentRole
from agents.orchestrator import Orchestrator

spec = ProjectSpec(max_steps=10)  # 10 шагов для теста
orch = Orchestrator(spec)
task = AgentTask(
    task_id="pipeline_test",
    role=AgentRole.ORCHESTRATOR,
    description="Test full pipeline",
    inputs={"auto_train": True, "max_steps": 10},
)
result = orch.execute(task)
print(f"\nResult: {result.message}")
print(f"Metrics: {result.metrics}")
