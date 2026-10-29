"""Unit tests for SALAZAR's workflow orchestration agent."""

import unittest

from app.agents import Agent, TaskResult
from app.agent_registry import AgentRegistry
from app.task_router import TaskType
from app.workflow_agent import WorkflowAgent


class FakeAgent(Agent):
    def __init__(self, name, task_type, response, success=True):
        super().__init__(name, task_type)
        self.response = response
        self.success = success
        self.received_tasks = []

    def execute(self, task):
        task = self.validate_task(task)
        self.received_tasks.append(task)
        return TaskResult(
            agent_name=self.name,
            task_type=self.task_type,
            content=self.response,
            success=self.success,
        )


def make_registry():
    registry = AgentRegistry()
    registry.register(
        TaskType.CODING,
        FakeAgent(
            "Local Coding Agent",
            "coding",
            "Finding: division by zero can raise ZeroDivisionError.\n"
            "Why it matters: a zero divisor causes a runtime error.\n"
            "Proposed fix: check whether b is zero.\n"
            "```python\ndef divide(a, b):\n    return a / b\n```",
        ),
    )
    registry.register(
        TaskType.RAG,
        FakeAgent("Document RAG Agent", "rag", "Retrieved facts"),
    )
    registry.register(
        TaskType.CHAT,
        FakeAgent("Local Chat Agent", "chat", "Final explanation"),
    )
    return registry


class TestWorkflowAgent(unittest.TestCase):
    def test_agent_identity(self):
        agent = WorkflowAgent(make_registry())
        self.assertEqual(agent.name, "Workflow Agent")
        self.assertEqual(agent.task_type, "workflow")

    def test_invalid_registry_rejected(self):
        with self.assertRaises(TypeError):
            WorkflowAgent(object())

    def test_empty_task_rejected(self):
        with self.assertRaises(ValueError):
            WorkflowAgent(make_registry()).execute(" ")

    def test_non_string_task_rejected(self):
        with self.assertRaises(TypeError):
            WorkflowAgent(make_registry()).execute(None)

    def test_coding_handoff_removes_fenced_code_but_preserves_findings(self):
        review = (
            "Finding: division by zero can raise ZeroDivisionError.\n"
            "Why it matters: a zero divisor causes a runtime error.\n"
            "Proposed fix: check whether b is zero.\n"
            "```python\ndef divide(a, b):\n    return a / b\n```"
        )

        handoff = WorkflowAgent._coding_handoff(review)

        self.assertIn("Finding: division by zero", handoff)
        self.assertIn("Recommended fix: check whether b is zero", handoff)
        self.assertNotIn("[code omitted]", handoff)
        self.assertNotIn("def divide(a, b):", handoff)
        self.assertNotIn("return a / b", handoff)


    def test_coding_then_chat_workflow(self):
        registry = make_registry()
        agent = WorkflowAgent(registry)

        result = agent.execute(
            "Review my code and then explain the solution:\n```python\ndef add(a, b):\n    return a + b\n```"
        )

        self.assertTrue(result.success)
        self.assertEqual(result.task_type, "workflow")
        self.assertEqual(result.content, "Final explanation")
        self.assertNotIn("Local Coding Agent", result.content)
        self.assertNotIn("Local Chat Agent", result.content)
        chat_task = registry.get(TaskType.CHAT).received_tasks[0]
        self.assertIn("Original user request", chat_task)
        self.assertIn("Explain the coding agent's findings", chat_task)
        self.assertIn("rather than a second code reviewer", chat_task)
        self.assertIn("Do not claim that code was executed or that a fix was applied", chat_task)
        self.assertIn("Previous agent output:", chat_task)
        self.assertIn("Finding: division by zero", chat_task)
        self.assertIn("Why it matters: a zero divisor causes a runtime error", chat_task)
        self.assertIn("Recommended fix: check whether b is zero", chat_task)
        self.assertNotIn("def divide(a, b):", chat_task)
        self.assertNotIn("def add(a, b):", chat_task)

    def test_rag_then_chat_workflow(self):
        registry = make_registry()
        agent = WorkflowAgent(registry)

        result = agent.execute(
            "Research my documents for the system architecture and then summarize the findings"
        )

        self.assertTrue(result.success)
        self.assertIn("Document RAG Agent", result.content)
        self.assertIn("Local Chat Agent", result.content)
        chat_task = registry.get(TaskType.CHAT).received_tasks[0]
        self.assertIn("Original user request", chat_task)
        self.assertIn("summarize the retrieved information", chat_task.lower())
        self.assertIn("Previous agent output:", chat_task)
        self.assertIn("Retrieved facts", chat_task)
        self.assertIn("system architecture", chat_task)

    def test_workflow_stops_after_failed_agent(self):
        registry = AgentRegistry()
        registry.register(
            TaskType.CODING,
            FakeAgent(
                "Local Coding Agent", "coding", "Could not generate code",
                success=False,
            ),
        )
        registry.register(
            TaskType.CHAT,
            FakeAgent("Local Chat Agent", "chat", "Should not run"),
        )
        agent = WorkflowAgent(registry)

        result = agent.execute("Review my code and then explain the solution:\n```python\ndef add(a, b):\n    return a + b\n```")

        self.assertFalse(result.success)
        self.assertIn("Local Coding Agent", result.content)
        self.assertEqual(registry.get(TaskType.CHAT).received_tasks, [])

    def test_unsupported_workflow_is_reported(self):
        agent = WorkflowAgent(make_registry())

        result = agent.execute("Coordinate all available SALAZAR components")

        self.assertFalse(result.success)
        self.assertIn("supported workflow", result.content.lower())

    def test_code_review_without_code_fails_before_agents_run(self):
        registry = make_registry()
        agent = WorkflowAgent(registry)

        result = agent.execute(
            "Review my code and then explain the solution"
        )

        self.assertFalse(result.success)
        self.assertIn("code", result.content.lower())
        self.assertEqual(registry.get(TaskType.CODING).received_tasks, [])
        self.assertEqual(registry.get(TaskType.CHAT).received_tasks, [])

    def test_document_workflow_requires_a_retrieval_target(self):
        registry = make_registry()
        agent = WorkflowAgent(registry)

        result = agent.execute(
            "Research the documents and then summarize the findings"
        )

        self.assertFalse(result.success)
        self.assertIn("document", result.content.lower())
        self.assertEqual(registry.get(TaskType.RAG).received_tasks, [])
        self.assertEqual(registry.get(TaskType.CHAT).received_tasks, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
