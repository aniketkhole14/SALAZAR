"""Unit tests for SALAZAR's sequential workflow engine."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents import Agent, TaskResult
from app.workflow import WorkflowEngine


class MockAgent(Agent):
    def __init__(self, name, task_type, prefix):
        super().__init__(name, task_type)
        self.prefix = prefix
        self.received_task = None

    def execute(self, task: str) -> TaskResult:
        task = self.validate_task(task)
        self.received_task = task
        return TaskResult(
            agent_name=self.name,
            task_type=self.task_type,
            content=f"{self.prefix}({task})",
        )


class FailingAgent(MockAgent):
    def execute(self, task: str) -> TaskResult:
        raise RuntimeError("Simulated agent failure")


class UnsuccessfulAgent(MockAgent):
    def execute(self, task: str) -> TaskResult:
        return TaskResult(
            agent_name=self.name,
            task_type=self.task_type,
            content="Task failed",
            success=False,
        )


class TestWorkflowEngine(unittest.TestCase):
    def setUp(self):
        self.engine = WorkflowEngine()

    def test_three_agent_workflow(self):
        planner = MockAgent("Planner", "workflow", "PLAN")
        coder = MockAgent("Coder", "coding", "CODE")
        reviewer = MockAgent("Reviewer", "review", "REVIEW")

        result = self.engine.run(
            [planner, coder, reviewer],
            "Build a Python API",
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.results), 3)
        self.assertEqual(
            coder.received_task,
            "PLAN(Build a Python API)",
        )
        self.assertEqual(
            reviewer.received_task,
            "CODE(PLAN(Build a Python API))",
        )

    def test_single_agent_workflow(self):
        agent = MockAgent("Planner", "workflow", "PLAN")
        result = self.engine.run([agent], "Build an API")

        self.assertTrue(result.success)
        self.assertEqual(len(result.results), 1)

    def test_empty_task_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.run([MockAgent("A", "chat", "A")], "  ")

    def test_non_string_task_rejected(self):
        with self.assertRaises(TypeError):
            self.engine.run([MockAgent("A", "chat", "A")], None)

    def test_empty_workflow_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.run([], "Build an API")

    def test_invalid_agent_rejected(self):
        with self.assertRaises(TypeError):
            self.engine.run([object()], "Build an API")

    def test_agent_exception_stops_workflow(self):
        first = MockAgent("First", "chat", "FIRST")
        failing = FailingAgent("Failing", "coding", "FAIL")
        last = MockAgent("Last", "review", "LAST")

        result = self.engine.run(
            [first, failing, last],
            "Build an API",
        )

        self.assertFalse(result.success)
        self.assertEqual(result.failed_agent, "Failing")
        self.assertEqual(len(result.results), 1)
        self.assertIn("Simulated agent failure", result.error)

    def test_unsuccessful_result_stops_workflow(self):
        first = UnsuccessfulAgent("First", "chat", "FIRST")
        last = MockAgent("Last", "review", "LAST")

        result = self.engine.run(
            [first, last],
            "Build an API",
        )

        self.assertFalse(result.success)
        self.assertEqual(result.failed_agent, "First")
        self.assertEqual(len(result.results), 1)
        self.assertEqual(last.received_task, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
