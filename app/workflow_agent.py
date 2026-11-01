"""Deterministic workflow orchestration for SALAZAR."""

import re

from app.agents import Agent, TaskResult
from app.agent_registry import AgentRegistry
from app.task_router import TaskType
from app.workflow import WorkflowEngine


class WorkflowAgent(Agent):
    """Coordinate registered agents using predefined workflow plans."""

    def __init__(self, registry: AgentRegistry):
        if not isinstance(registry, AgentRegistry):
            raise TypeError("registry must be an AgentRegistry.")

        super().__init__(
            name="Workflow Agent",
            task_type=TaskType.WORKFLOW.value,
        )
        self.registry = registry
        self.engine = WorkflowEngine()

    def _select_plan(self, task: str):
        normalized = " ".join(task.lower().split())

        if (
            "review my code" in normalized
            or "review the code" in normalized
            or "review code" in normalized
            or "identify bugs and then fix" in normalized
            or "review my project" in normalized
        ):
            return [TaskType.CODING, TaskType.CHAT]

        if (
            "research my documents" in normalized
            or "research the documents" in normalized
            or "search my documents" in normalized
            or "retrieve documents" in normalized
            or "according to the document" in normalized
            or "summarize the document" in normalized
            or "summarise the document" in normalized
        ):
            return [TaskType.RAG, TaskType.CHAT]

        return None

    @staticmethod
    def _has_code_input(task: str) -> bool:
        """Return whether the request contains a fenced code block."""
        return "```" in task

    @staticmethod
    def _strip_fenced_code(content: str) -> str:
        """Remove fenced code blocks while preserving surrounding prose."""
        return re.sub(
            r"```[\s\S]*?```",
            "[code omitted]",
            content,
        ).strip()

    @staticmethod
    def _coding_handoff(content: str) -> str:
        """Convert a code review into a compact, structured explanation brief."""
        content = WorkflowAgent._strip_fenced_code(content)

        def extract(label_pattern: str) -> str:
            match = re.search(
                rf"(?ims)^\s*(?:\d+\.\s*)?(?:{label_pattern})\s*:\s*"
                rf"(.*?)(?=^\s*(?:\d+\.\s*)?"
                rf"(?:Finding|Why it matters|Proposed fix|Corrected code)\s*:"
                rf"|\Z)",
                content,
            )
            if not match:
                return ""
            value = " ".join(match.group(1).split())
            value = re.sub(r"\[code omitted\]", "", value).strip()
            return value[:500]

        finding = extract("Finding")
        why = extract("Why it matters")
        fix = extract("Proposed fix")

        if not finding:
            lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith("[code omitted]")
            ]
            finding = lines[0][:500] if lines else "No clear finding was returned."

        return (
            f"Finding: {finding}\n"
            f"Why it matters: {why or 'The review did not provide a separate explanation.'}\n"
            f"Recommended fix: {fix or 'The review did not provide a clear fix.'}"
        )

    @staticmethod
    def _has_research_target(task: str) -> bool:
        """Reject generic research requests with no question or target."""
        normalized = " ".join(task.lower().split())

        generic_phrases = {
            "research the documents and then summarize the findings",
            "research the documents and then summarise the findings",
            "search my documents and summarize the findings",
            "search my documents and summarise the findings",
            "retrieve documents and summarize the findings",
            "retrieve documents and summarise the findings",
        }
        if normalized in generic_phrases:
            return False

        target_cues = (
            "about ",
            "for ",
            "whether ",
            "what ",
            "how ",
            "why ",
            "which ",
            "where ",
            "when ",
            "according to ",
            ".pdf",
            ".docx",
            ".txt",
            ".md",
        )
        return any(cue in normalized for cue in target_cues)

    def execute(self, task: str) -> TaskResult:
        task = self.validate_task(task)
        plan = self._select_plan(task)

        if plan is None:
            return TaskResult(
                agent_name=self.name,
                task_type=self.task_type,
                content=(
                    "Unsupported workflow. Supported workflows are "
                    "coding followed by explanation, or document retrieval "
                    "followed by summarization."
                ),
                success=False,
            )

        if plan[0] == TaskType.CODING and not self._has_code_input(task):
            return TaskResult(
                agent_name=self.name,
                task_type=self.task_type,
                content=(
                    "Cannot review code because no code was provided. "
                    "Include a code snippet in a fenced Markdown block, "
                    "or provide a workflow that references an accessible file."
                ),
                success=False,
            )

        if plan[0] == TaskType.RAG and not self._has_research_target(task):
            return TaskResult(
                agent_name=self.name,
                task_type=self.task_type,
                content=(
                    "Cannot start document research: specify a question, "
                    "topic, or document target."
                ),
                success=False,
            )

        try:
            agents = [self.registry.get(task_type) for task_type in plan]
        except LookupError as exc:
            return TaskResult(
                agent_name=self.name,
                task_type=self.task_type,
                content=f"Cannot start workflow: {exc}",
                success=False,
            )

        if plan[0] == TaskType.CODING:
            initial_task = (
                "Review the following user request and supplied code. "
                "Identify bugs, correctness issues, and a proposed fix. "
                "Do not claim to have executed the code.\n\n"
                f"Original user request and input:\n{task}"
            )
        else:
            initial_task = (
                "Retrieve source-grounded information for the following "
                "question or document target. Preserve citations and source "
                "details where available.\n\n"
                f"Original user request:\n{task}"
            )

        def build_next_task(original_task, previous_result, next_agent):
            if plan[0] == TaskType.CODING:
                instruction = (
                    "Explain the coding agent's findings, but act as the explainer "
                    "rather than a second code reviewer. Explain the main finding "
                    "and why the proposed fix helps. "
                    "Do not repeat the full code review, reproduce code blocks, "
                    "or suggest additional alternatives unless necessary. "
                    "Do not claim that code was executed or that a fix was applied. "
                    "Keep the explanation concise and distinct from the review."
                )
            else:
                instruction = (
                    "Produce one concise, coherent synthesis answering the "
                    "user's original request. Combine related findings, remove "
                    "duplicate points, and organize the answer logically. "
                    "Preserve relevant citations and source details. Distinguish "
                    "missing information from established findings. Use only "
                    "the supplied evidence. Do not repeat the retrieved report "
                    "or mention the intermediate agent."
                )

            previous_content = previous_result.content
            if plan[0] == TaskType.CODING:
                previous_content = self._coding_handoff(previous_content)

            user_request = original_task
            if plan[0] == TaskType.CODING:
                user_request = self._strip_fenced_code(original_task)

            return (
                f"User request: {user_request}\n\n"
                f"Task: {instruction}\n\n"
                f"Previous agent output:\n{previous_content}"
            )

        outcome = self.engine.run(
            agents,
            initial_task,
            task_builder=build_next_task,
        )

        if not outcome.success:
            return TaskResult(
                agent_name=self.name,
                task_type=self.task_type,
                content=(
                    f"Workflow failed at {outcome.failed_agent}: "
                    f"{outcome.error}"
                ),
                success=False,
            )

        if plan[0] == TaskType.CODING:
            # The final chat agent synthesizes the code review.
            # Avoid displaying the full review a second time.
            final_result = outcome.results[-1]
            final_content = final_result.content
        else:
            # Return only the Chat Agent's consolidated synthesis.
            final_content = outcome.results[-1].content

        return TaskResult(
            agent_name=self.name,
            task_type=self.task_type,
            content=final_content,
            success=True,
            sources=tuple(
                source
                for result in outcome.results
                for source in result.sources
            ),
        )
