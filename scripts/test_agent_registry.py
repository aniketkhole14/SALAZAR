"""Unit tests for SALAZAR's AgentRegistry."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents import Agent, TaskResult
from app.agent_registry import AgentRegistry
from app.task_router import RoutingDecision, TaskRouter, TaskType


class MockAgent(Agent):
    def __init__(self, name="MockCodingAgent", task_type="coding"):
        super().__init__(name, task_type)

    def execute(self, task: str) -> TaskResult:
        task = self.validate_task(task)
        return TaskResult(
            agent_name=self.name,
            task_type=self.task_type,
            content=f"Processed: {task}",
        )


class BadResultAgent(MockAgent):
    def execute(self, task: str) -> TaskResult:
        return "not a TaskResult"


class TestAgentRegistry(unittest.TestCase):
    def setUp(self):
        self.router = TaskRouter()
        self.registry = AgentRegistry()
        self.agent = MockAgent()

    def test_successful_dispatch(self):
        self.registry.register(TaskType.CODING, self.agent)
        query = "Can you help me build a Python API?"
        decision = self.router.route(query)

        result = self.registry.dispatch(decision, query)

        self.assertEqual(result.agent_name, self.agent.name)
        self.assertEqual(result.task_type, TaskType.CODING.value)
        self.assertIn("Python API", result.content)
        self.assertTrue(result.success)

    def test_missing_agent_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            self.registry.get(TaskType.CODING)

    def test_duplicate_registration_rejected(self):
        self.registry.register(TaskType.CODING, self.agent)

        with self.assertRaises(ValueError):
            self.registry.register(TaskType.CODING, MockAgent())

    def test_mismatched_agent_type_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.register(
                TaskType.RAG,
                self.agent,
            )

    def test_non_agent_registration_rejected(self):
        with self.assertRaises(TypeError):
            self.registry.register(TaskType.CODING, object())

    def test_invalid_routing_decision_rejected(self):
        with self.assertRaises(TypeError):
            self.registry.dispatch("coding", "Build an API")

    def test_invalid_agent_result_rejected(self):
        self.registry.register(
            TaskType.CODING,
            BadResultAgent(),
        )
        decision = RoutingDecision(
            TaskType.CODING,
            "Test invalid agent output.",
        )

        with self.assertRaises(TypeError):
            self.registry.dispatch(decision, "Build an API")

    def test_wrong_result_agent_name_rejected(self):
        class WrongNameAgent(MockAgent):
            def execute(self, task: str) -> TaskResult:
                return TaskResult(
                    agent_name="DifferentAgent",
                    task_type=self.task_type,
                    content="Output",
                )

        self.registry.register(
            TaskType.CODING,
            WrongNameAgent(),
        )
        decision = RoutingDecision(
            TaskType.CODING,
            "Test mismatched agent name.",
        )

        with self.assertRaises(ValueError):
            self.registry.dispatch(decision, "Build an API")


if __name__ == "__main__":
    unittest.main(verbosity=2)
