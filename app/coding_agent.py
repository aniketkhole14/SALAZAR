"""Local code-generation and code-explanation agent for SALAZAR."""

from app.agents import Agent, TaskResult


class CodingAgent(Agent):
    """Handle coding requests using SALAZAR's local language model.

    Generated code is returned as text. This agent never executes it.
    """

    def __init__(self, runner):
        if not callable(getattr(runner, "generate", None)):
            raise TypeError(
                "runner must provide a callable generate(prompt) method."
            )

        super().__init__(
            name="Local Coding Agent",
            task_type="coding",
        )
        self.runner = runner

    def execute(self, task: str) -> TaskResult:
        task = self.validate_task(task)

        prompt = f"""You are SALAZAR's local Coding Agent, operating offline.

First identify what the user is asking you to do. If the request includes code
and asks for a review, perform an actual code review. Never answer only "No."
or give a one-word response to a valid coding request.

For a code review, use these sections:
1. Finding: identify concrete bugs, risks, and relevant edge cases. Do not
   declare code correct without considering its inputs and failure conditions.
2. Why it matters: explain the behavior and when a failure can occur.
3. Proposed fix: provide a minimal, appropriate correction and explain any
   important trade-off. Do not change the function's return type unexpectedly.
4. Corrected code: include a code block when a code change is warranted.

Check edge cases relevant to the code, such as division by zero for division
functions, empty inputs for collection operations, and invalid input types
when they are plausible. If no obvious bug is found, state which cases were
considered and distinguish that from proving the code is entirely correct.

For other programming requests, follow the user's requested task instead.
Prefer readable, maintainable solutions and state assumptions when necessary.
Never claim that code was tested or executed unless it actually was.
Do not execute generated code.

User's coding request:
{task}

Coding Agent response:
"""

        answer = self.runner.generate(prompt)

        if not isinstance(answer, str) or not answer.strip():
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
