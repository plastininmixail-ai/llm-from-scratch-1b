"""Common agent infrastructure."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .protocol import (
    AgentRole, AgentTask, AgentResult, ProjectSpec, Message, TaskStatus
)


class BaseAgent(ABC):
    """Базовый класс для всех агентов."""

    def __init__(self, spec: ProjectSpec, log_dir: Path = None):
        self.spec = spec
        self.role: AgentRole = self._define_role()
        self.log_dir = log_dir or Path("logs") / self.role.value
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "agent.log"

    @abstractmethod
    def _define_role(self) -> AgentRole:
        pass

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """Выполнить задачу и вернуть результат."""
        pass

    def log(self, message: str):
        """Логировать сообщение."""
        import datetime
        ts = datetime.datetime.now().isoformat()
        line = f"[{ts}] {message}\n"
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line)
        print(line.rstrip())

    def send_message(self, to_agent: AgentRole, subject: str, body):
        """Отправить сообщение другому агенту."""
        msg = Message(
            from_agent=self.role,
            to_agent=to_agent,
            subject=subject,
            body=body,
        )
        # Сохраняем в общий mailbox
        mailbox = Path("logs") / "mailbox.jsonl"
        mailbox.parent.mkdir(parents=True, exist_ok=True)
        import json
        with mailbox.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "from": msg.from_agent.value,
                "to": msg.to_agent.value,
                "subject": msg.subject,
                "body": str(msg.body),
                "timestamp": msg.timestamp,
            }) + "\n")
        self.log(f"Sent to {to_agent.value}: {subject}")
