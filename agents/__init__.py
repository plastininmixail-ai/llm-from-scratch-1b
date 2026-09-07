"""Agents package — multi-agent orchestration."""
from .common import (
    AgentRole, TaskStatus, AgentTask, AgentResult, ProjectSpec, Message,
    BaseAgent,
)
from .architect import ArchitectAgent
from .data_engineer import DataEngineerAgent
from .trainer import TrainerAgent
from .orchestrator import Orchestrator

__all__ = [
    "AgentRole", "TaskStatus", "AgentTask", "AgentResult",
    "ProjectSpec", "Message", "BaseAgent",
    "ArchitectAgent", "DataEngineerAgent", "TrainerAgent", "Orchestrator",
]
