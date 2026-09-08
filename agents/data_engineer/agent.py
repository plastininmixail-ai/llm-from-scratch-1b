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
        # Поиск файлов по разным путям (sources/, raw/, корневая)
        source_paths = {
            "wiki_summaries_full": ["data/wiki_summaries_full.jsonl"],
            "openstax_textbooks": ["data/openstax_textbooks.jsonl"],
            "openstax_pdfs": ["data/sources/openstax_pdfs/*.pdf", "data/sources/openstax_pdfs/*.json"],
            "fineweb_edu": ["data/sources/fineweb_edu/**/*.parquet"],
            "fastai": ["data/sources/fastai/**/*.txt", "data/sources/fastai/**/*.md"],
            "wiki_summaries": ["data/wiki_summaries.jsonl"],
            "sft_combined": ["data/sft_combined.jsonl"],
            "sft_v2": ["data/sft_v2.jsonl"],
            "sft_fastai": ["data/sft_fastai.jsonl"],
            "sft_sources_combined": ["data/sft_sources_combined.jsonl"],
            "chat_v23": ["data/chat_v23.jsonl"],
            "wiki_ru": ["data/sources/wiki_ru/*.jsonl"],
            "wiki_multi": ["data/sources/wiki_multi/*.jsonl"],
            "github_code": ["data/sources/github_code/*.txt"],
            "sep": ["data/sources/sep/*.html"],
            "iep": ["data/sources/iep/*.html"],
            "arxiv": ["data/sources/arxiv/*"],
        }

        for source in sources:
            found = False
            patterns = source_paths.get(source, [f"data/{source}.jsonl", f"data/{source}/*.jsonl"])

            from glob import glob
            for pattern in patterns:
                matches = glob(pattern, recursive=True)
                if matches:
                    total_size = sum(Path(m).stat().st_size for m in matches if Path(m).is_file())
                    if total_size > 0:
                        available.append({
                            "name": source,
                            "files": len(matches),
                            "size_bytes": total_size,
                            "size_mb": round(total_size/1024**2, 1),
                        })
                        self.log(f"  ✓ Found {source}: {len(matches)} files, {total_size/1024**2:.1f} MB")
                        found = True
                        break

            if not found:
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
