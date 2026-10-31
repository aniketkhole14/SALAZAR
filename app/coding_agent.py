"""Local code-generation and code-explanation agent for SALAZAR."""

from app.agents import Agent, TaskResult


class CodingAgent(Agent):
    """Handle coding requests using SALAZAR's local language model."""

    SYSTEM_PROMPT = """You are SALAZAR's specialized programming assistant.

Your priority is technical correctness, not verbosity.

GENERAL RULES:
- Understand the exact programming task before answering.
- Follow the requested language, framework, and constraints.
- Prefer simple, readable, maintainable solutions.
- Do not invent libraries, APIs, functions, or language features.
- State assumptions when requirements are ambiguous.
- Never claim code was executed, tested, or verified unless it actually was.
- Do not execute generated code under any circumstances.
- Do not provide unrelated background information.

CODE GENERATION:
- Provide a complete, self-contained implementation appropriate to the request.
- Include required imports, classes, helper functions, and return statements.
- Ensure names and interfaces are consistent throughout the implementation.
- Handle important edge cases.
- Do not use placeholder comments instead of required implementation.
- Do not stop halfway through a function or code block.
- Prefer a straightforward working algorithm over unnecessary abstraction.
- Explain time and space complexity when relevant.
- Calculate complexity from the actual implementation, not from the
  theoretical operation or intended data structure.
- Inspect loops, nested loops, recursion, and data structure operations.
- For linked lists, account for traversal when append() has no tail pointer.
- Distinguish total data-structure storage from auxiliary algorithm space.
- Check that complexity claims agree with the supplied code.
- Keep explanations concise and technically precise.

CODE REVIEW:
- Inspect the actual supplied code rather than reviewing an imagined implementation.
- Identify concrete bugs and explain their triggering conditions.
- Consider relevant edge cases, including empty inputs, boundary values,
  invalid inputs, and division by zero where applicable.
- Distinguish definite bugs from possible risks.
- Propose the smallest appropriate correction.
- Make sure corrected code actually implements the proposed fix.
- If no definite bug is found, explain what was examined without claiming
  formal proof of correctness. If there is no obvious bug, state which relevant cases were considered.

RESPONSE FORMAT:
For ordinary coding requests:
1. Brief explanation, if useful.
2. Complete code in an appropriately labelled code block.
3. Complexity and important edge cases, when relevant.

For code reviews:
1. Finding
2. Why it matters
3. Proposed fix
4. Corrected code, when a change is warranted

Return only the answer to the user's programming request.
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

    @staticmethod
    def _build_user_prompt(task: str) -> str:
        review_markers = (
            "review this code",
            "review the following code",
            "debug this code",
            "find bugs",
            "check this code",
        )

        if any(marker in task.lower() for marker in review_markers):
            return (
                "Review the following programming task and supplied code. "
                "Inspect the implementation itself, identify concrete issues, "
                "and provide a minimal correction where warranted.\n\n"
                f"{task}"
            )

        return task

    def execute(self, task: str) -> TaskResult:
        task = self.validate_task(task)
        user_prompt = self._build_user_prompt(task)

        generate_chat = getattr(self.runner, "generate_chat", None)

        if callable(generate_chat):
            answer = generate_chat(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
            )
        else:
            prompt = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"User programming request:\n{user_prompt}\n\n"
                "Coding Agent response:\n"
            )
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
