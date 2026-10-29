"""Common interfaces for SALAZAR task agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskResult:
    """Standard output returned by an agent."""

    agent_name: str
    task_type: str
    content: str
    success: bool = True
    sources: tuple[str, ...] = ()


class Agent(ABC):
    """Base contract for a specialized SALAZAR agent."""

    name: str
    task_type: str

    def __init__(self, name: str, task_type: str):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string.")

        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("Agent task_type must be a non-empty string.")

        self.name = name.strip()
        self.task_type = task_type.strip()

    @abstractmethod
    def execute(self, task: str) -> TaskResult:
        """Execute a task and return a structured result."""
        raise NotImplementedError

    def validate_task(self, task: str) -> str:
        """Validate and normalize task input."""
        if not isinstance(task, str):
            raise TypeError("task must be a string.")

        task = task.strip()
        if not task:
            raise ValueError("task must not be empty.")

        return task
