"""Adapter for SALAZAR's existing document-grounded RAG CLI."""

from collections.abc import Callable, Mapping, Sequence
from app.agents import Agent, TaskResult


class RAGAgent(Agent):
    """Use an injected RAG handler and preserve its source metadata."""

    def __init__(
        self,
        handler: Callable[[str], Mapping[str, object]],
    ):
        if not callable(handler):
            raise TypeError("handler must be callable.")

        super().__init__(name="Document RAG Agent", task_type="rag")
        self.handler = handler

    def execute(self, task: str) -> TaskResult:
        task = self.validate_task(task)
        response = self.handler(task)

        if not isinstance(response, Mapping):
            raise TypeError("RAG handler must return a mapping.")

        answer = response.get("answer")
        sources = response.get("sources", ())

        if not isinstance(answer, str):
            raise TypeError("RAG handler answer must be a string.")

        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            raise TypeError("RAG handler sources must be a sequence of strings.")

        if any(not isinstance(source, str) for source in sources):
            raise TypeError("Every RAG source must be a string.")

        answer = answer.strip()
        if not answer:
            return TaskResult(
                agent_name=self.name,
                task_type=self.task_type,
                content="The RAG pipeline returned no answer.",
                success=False,
                sources=tuple(sources),
            )

        return TaskResult(
            agent_name=self.name,
            task_type=self.task_type,
            content=answer,
            success=True,
            sources=tuple(sources),
        )
