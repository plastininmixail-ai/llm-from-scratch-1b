"""Data Engineer Agent: скачивает и готовит данные."""
import json
from pathlib import Path

from ..common import (
    BaseAgent, AgentRole, AgentTask, AgentResult, ProjectSpec,
)


class DataEngineerAgent(BaseAgent):
    """Готовит обучающие данные: скачивает, чистит, токенизирует."""

    def _define_role(self) -> AgentRole:
        return AgentRole.DATA_ENGINEER

    def execute(self, task: AgentTask) -> AgentResult:
        """Подготовить данные согласно inputs."""
        sources = task.inputs.get("sources", ["wikipedia", "openstax", "sft_combined"])
        output_dir = Path(task.inputs.get("output_dir", "data/binary/v0"))
        max_bytes = task.inputs.get("max_bytes", 5 * 1024**3)  # 5 GB default

        self.log(f"Preparing data from sources: {sources}")
        self.log(f"Output dir: {output_dir}, max: {max_bytes/1024**3:.1f} GB")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Список уже доступных данных
        available = []
        for source in sources:
            src_path = Path(f"data/{source}.jsonl")
            if src_path.exists():
                size = src_path.stat().st_size
                available.append({"name": source, "path": str(src_path), "size_bytes": size})
                self.log(f"  ✓ Found {source}: {size/1024**2:.1f} MB")
            else:
                self.log(f"  ✗ Missing {source}")

        # Tokenize (TODO: подключить BPE)
        # Пока просто копируем информацию
        manifest = {
            "version": "v0",
            "total_sources": len(available),
            "sources": available,
            "output_dir": str(output_dir),
            "status": "ready_for_tokenization",
            "next_step": "tokenization",
        }

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        self.log(f"Manifest saved to {manifest_path}")
        self.send_message(
            AgentRole.TRAINER,
            "Data prepared",
            {"manifest": str(manifest_path), "sources": available}
        )

        return AgentResult(
            task_id=task.task_id,
            role=self.role,
            success=True,
            data=manifest,
            message=f"Found {len(available)} data sources",
            artifacts=[manifest_path],
            metrics={"sources_count": len(available)},
        )


if __name__ == "__main__":
    from ..common import ProjectSpec
    spec = ProjectSpec()
    agent = DataEngineerAgent(spec)
    task = AgentTask(
        task_id="data_001",
        role=AgentRole.DATA_ENGINEER,
        description="Prepare data",
        inputs={"sources": ["sft_combined", "wiki_summaries_full", "openstax_textbooks"]}
    )
    result = agent.execute(task)
    print(f"\nResult: {result.message}")
