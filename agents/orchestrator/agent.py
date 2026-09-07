"""Orchestrator: координирует работу всех агентов."""
from pathlib import Path

from ..common import (
    BaseAgent, AgentRole, AgentTask, AgentResult, ProjectSpec, TaskStatus,
)
from ..architect import ArchitectAgent
from ..data_engineer import DataEngineerAgent
from ..trainer import TrainerAgent


class Orchestrator(BaseAgent):
    """Координатор всего pipeline."""

    def _define_role(self) -> AgentRole:
        return AgentRole.ORCHESTRATOR

    def execute(self, task: AgentTask) -> AgentResult:
        """Запустить полный pipeline."""
        self.log("=" * 60)
        self.log("ORCHESTRATOR: starting pipeline for Stage 0")
        self.log("=" * 60)

        results = {}

        # Step 1: Architect
        self.log("\n[1/3] Architect Agent...")
        arch_agent = ArchitectAgent(self.spec)
        arch_task = AgentTask(
            task_id="arch_stage0",
            role=AgentRole.ARCHITECT,
            description="Design 1B architecture",
        )
        arch_result = arch_agent.execute(arch_task)
        results["architect"] = arch_result

        if not arch_result.success:
            return AgentResult(
                task_id=task.task_id, role=self.role,
                success=False, message="Architect failed",
            )

        # Step 2: Data Engineer
        self.log("\n[2/3] Data Engineer Agent...")
        data_agent = DataEngineerAgent(self.spec)
        data_task = AgentTask(
            task_id="data_stage0",
            role=AgentRole.DATA_ENGINEER,
            description="Prepare training data",
            inputs=task.inputs.get("data_inputs", {
                "sources": ["sft_combined", "wiki_summaries_full", "openstax_textbooks", "chat_v23"]
            })
        )
        data_result = data_agent.execute(data_task)
        results["data"] = data_result

        # Step 3: Trainer (опционально)
        if task.inputs.get("auto_train", False):
            self.log("\n[3/3] Trainer Agent...")
            train_agent = TrainerAgent(self.spec)
            train_task = AgentTask(
                task_id="train_stage0",
                role=AgentRole.TRAINER,
                description="Train model",
                inputs={
                    "config_path": str(arch_result.artifacts[0]) if arch_result.artifacts else "configs/v0_1b.yaml",
                    "max_steps": task.inputs.get("max_steps", 100),
                }
            )
            train_result = train_agent.execute(train_task)
            results["trainer"] = train_result

        self.log("\n" + "=" * 60)
        self.log("ORCHESTRATOR: pipeline complete")
        self.log("=" * 60)

        return AgentResult(
            task_id=task.task_id,
            role=self.role,
            success=True,
            data=results,
            message=f"Pipeline complete: {len(results)} stages",
            metrics={
                "architect_success": results["architect"].success,
                "data_success": results["data"].success,
            },
        )


if __name__ == "__main__":
    from ..common import ProjectSpec
    spec = ProjectSpec()
    orch = Orchestrator(spec)
    task = AgentTask(
        task_id="pipeline_stage0",
        role=AgentRole.ORCHESTRATOR,
        description="Stage 0 pipeline",
        inputs={"auto_train": False}
    )
    result = orch.execute(task)
    print(f"\nFinal: {result.message}")
    print(f"Metrics: {result.metrics}")
