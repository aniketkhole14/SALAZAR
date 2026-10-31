"""Deterministic task classification for SALAZAR."""

from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    RAG = "rag"
    CODING = "coding"
    WORKFLOW = "workflow"
    CHAT = "chat"


@dataclass(frozen=True)
class RoutingDecision:
    task_type: TaskType
    reason: str


class TaskRouter:
    """Classify requests without loading an LLM or external model."""

    WORKFLOW_CUES = (
        "multi-step",
        "multiple agents",
        "coordinate",
        "orchestrate",
        "end-to-end workflow",
        "run a workflow",
        "review my project",
        "identify bugs and then fix",
        "review, identify bugs",
        "then fix them",
        "plan a workflow",
        "workflow using rag",
        "routing for workflow",
        "resource management supporting rag",
        "coding assistance related to local ai chat",
    )

    CODING_CUES = (
        "write code",
        "generate code",
        "write a python function",
        "write a python script",
        "create a python function",
        "create a python script",
        "debug",
        "refactor",
        "implement",
        "programming",
        "fix this code",
        "build a python api",
        "build an api",
        "create an api",
        "write a function",
        "modify the code",
        "fix the bug",
        "review this code",
        "review the code",
        "review the following code",
        "review this function",
        "review this python function",
        "review python function",
        "review the function",
        "code review",
        "review the implementation",
        "inspect this code",
        "inspect the following code",
        "find bugs in this code",
        "find bugs in the code",
        "analyze this code",
        "analyse this code",
    )

    RAG_CUES = (
        "search my documents",
        "search the documents",
        "according to the document",
        "according to the srs",
        "from the pdf",
        "in my pdf",
        "from my files",
        "cite the source",
        "cite the page",
        "retrieve documents",
        "summarize the document",
        "summarise the document",
        "summarize this document",
        "summarise this document",
        "find the answer in",
        "what does the document say",
        "according to the pdf",
        "how does salazar",
        "salazar's routing",
        "salazar routing",
        "salazar workflow",
        "salazar rag",
        "salazar coding assistance",
        "salazar local ai chat",
        "salazar resource management",
        "functional components",
        "system architecture",
        "according to the viva review guide",
    )

    def route(self, query: str) -> RoutingDecision:
        """Return a task type and an explanation for the decision."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized = " ".join(query.lower().split())

        if not normalized:
            raise ValueError("query must not be empty")

        if any(cue in normalized for cue in self.WORKFLOW_CUES):
            return RoutingDecision(
                TaskType.WORKFLOW,
                "The request asks for coordinated or multi-step work.",
            )

        if any(cue in normalized for cue in self.RAG_CUES):
            return RoutingDecision(
                TaskType.RAG,
                "The request asks to retrieve, summarize, cite, or query documented information.",
            )

        if any(cue in normalized for cue in self.CODING_CUES):
            return RoutingDecision(
                TaskType.CODING,
                "The request asks for programming or code modification.",
            )

        return RoutingDecision(
            TaskType.CHAT,
            "No specialized task cue matched; use the general chat path.",
        )
