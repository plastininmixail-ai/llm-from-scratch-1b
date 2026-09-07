"""Common types and message protocol for agents."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class AgentRole(str, Enum):
    ARCHITECT = "architect"
    DATA_ENGINEER = "data_engineer"
    TRAINER = "trainer"
    TESTER = "tester"
    CURATOR = "curator"
    ORCHESTRATOR = "orchestrator"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    """Задача для агента."""
    task_id: str
    role: AgentRole
    description: str
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgentResult:
    """Результат работы агента."""
    task_id: str
    role: AgentRole
    success: bool
    data: Any = None
    message: str = ""
    artifacts: list[Path] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass
class ProjectSpec:
    """Спецификация проекта (общая для всех агентов)."""
    name: str = "self-improving-llm"
    target_params: int = 1_000_000_000  # 1B
    vocab_size: int = 32000
    seq_len: int = 1024
    target_tokens: int = 30_000_000_000  # 30B
    max_steps: int = 10000

    # Архитектура (Architect Agent заполняет)
    # Для 1B с tied weights (d=2048): 20 layers
    d_model: int = 2048
    n_heads: int = 32
    n_layers: int = 20

    # Paths
    project_root: Path = field(default_factory=lambda: Path("/c/Users/mixai/Desktop/llm-from-scratch"))


# Message protocol для межагентной коммуникации
@dataclass
class Message:
    """Сообщение между агентами."""
    from_agent: AgentRole
    to_agent: AgentRole
    subject: str
    body: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
