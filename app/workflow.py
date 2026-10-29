"""Sequential workflow orchestration for SALAZAR."""

from dataclasses import dataclass
from typing import Callable

from app.agents import Agent, TaskResult


@dataclass(frozen=True)
class WorkflowResult:
    """Outcome of a sequential workflow."""

    success: bool
    results: tuple[TaskResult, ...]
    failed_agent: str | None = None
    error: str | None = None


TaskBuilder = Callable[[str, TaskResult, Agent], str]


class WorkflowEngine:
    """Execute agents sequentially, optionally customizing each next task."""

    def run(
        self,
        agents: list[Agent],
        initial_task: str,
        task_builder: TaskBuilder | None = None,
    ) -> WorkflowResult:
        if not isinstance(initial_task, str):
            raise TypeError("initial_task must be a string.")

        current_task = initial_task.strip()
        if not current_task:
            raise ValueError("initial_task must not be empty.")

        if not agents:
            raise ValueError("A workflow requires at least one agent.")

        if task_builder is not None and not callable(task_builder):
            raise TypeError("task_builder must be callable or None.")

        original_task = current_task
        results = []

        for index, agent in enumerate(agents):
            if not isinstance(agent, Agent):
                raise TypeError("Every workflow step must be an Agent.")

            if index > 0 and task_builder is not None:
                try:
                    current_task = task_builder(
                        original_task,
                        results[-1],
                        agent,
                    )
                except Exception as exc:
                    return WorkflowResult(
                        success=False,
                        results=tuple(results),
                        failed_agent=agent.name,
                        error=f"Task builder failed: {type(exc).__name__}: {exc}",
                    )

                if not isinstance(current_task, str) or not current_task.strip():
                    return WorkflowResult(
                        success=False,
                        results=tuple(results),
                        failed_agent=agent.name,
                        error="Task builder must return a non-empty string.",
                    )

            try:
                result = agent.execute(current_task)
            except Exception as exc:
                return WorkflowResult(
                    success=False,
                    results=tuple(results),
                    failed_agent=agent.name,
                    error=f"{type(exc).__name__}: {exc}",
                )

            if not isinstance(result, TaskResult):
                return WorkflowResult(
                    success=False,
                    results=tuple(results),
                    failed_agent=agent.name,
                    error="Agent did not return a TaskResult.",
                )

            if result.agent_name != agent.name:
                return WorkflowResult(
                    success=False,
                    results=tuple(results),
                    failed_agent=agent.name,
                    error="Result agent_name does not match executing agent.",
                )

            if not result.success:
                return WorkflowResult(
                    success=False,
                    results=tuple(results) + (result,),
                    failed_agent=agent.name,
                    error="Agent reported task failure.",
                )

            results.append(result)
            current_task = result.content

        return WorkflowResult(
            success=True,
            results=tuple(results),
        )
