"""Local-chat agent backed by SALAZAR's GGUF inference runner."""

from app.agents import Agent, TaskResult


class ChatAgent(Agent):
    """Handle general conversation and workflow follow-up using a local model."""

    def __init__(self, runner):
        if not callable(getattr(runner, "generate", None)):
            raise TypeError("runner must provide a callable generate(prompt) method.")

        super().__init__(name="Local Chat Agent", task_type="chat")
        self.runner = runner

    @staticmethod
    def _is_workflow_follow_up(task: str) -> bool:
        legacy_format = (
            "Original user request:" in task
            and "Instruction:" in task
            and "Previous agent output:" in task
        )
        compact_format = (
            "User request:" in task
            and "Task:" in task
            and "Evidence:" in task
        )
        return legacy_format or compact_format

    def execute(self, task: str) -> TaskResult:
        task = self.validate_task(task)

        if self._is_workflow_follow_up(task):
            prompt = f"""You are SALAZAR, an offline assistant.

Complete the requested follow-up using only the evidence supplied.
For document summaries, do not invent component responsibilities.
Preserve citations and source references. If evidence is insufficient, say what is unknown.
Provide a concise synthesis. Write only the final answer. Do not repeat the instructions or evidence.

{task}

Final answer:
"""
        else:
            prompt = f"""You are SALAZAR, a helpful offline AI assistant.

Answer the user's request directly and clearly.
Do not claim to have searched documents, accessed the internet, or executed code unless that actually happened.

User request:
{task}

Assistant response:
"""

        answer = self.runner.generate(prompt)

        if not answer.strip():
            return TaskResult(
                agent_name=self.name,
                task_type=self.task_type,
                content="The local model returned an empty response.",
                success=False,
            )

        return TaskResult(
            agent_name=self.name,
            task_type=self.task_type,
            content=answer.strip(),
            success=True,
        )
