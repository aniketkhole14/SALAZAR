"""Tests for SALAZAR's router-to-registry dispatcher."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent_registry import AgentRegistry
from app.agents import Agent, TaskResult
from app.dispatcher import TaskDispatcher
from app.task_router import TaskRouter, TaskType


class MockAgent(Agent):
    def __init__(self, name, task_type):
        super().__init__(name, task_type)
        self.calls = []

    def execute(self, task):
        task = self.validate_task(task)
        self.calls.append(task)
        return TaskResult(
            agent_name=self.name,
            task_type=self.task_type,
            content=f"{self.task_type} handled: {task}",
        )


class TestTaskDispatcher(unittest.TestCase):
    def setUp(self):
        self.router = TaskRouter()
        self.registry = AgentRegistry()
        self.rag_agent = MockAgent("RAG Agent", "rag")
        self.coding_agent = MockAgent("Coding Agent", "coding")
        self.workflow_agent = MockAgent("Workflow Agent", "workflow")
        self.chat_agent = MockAgent("Chat Agent", "chat")

        self.registry.register(TaskType.RAG, self.rag_agent)
        self.registry.register(TaskType.CODING, self.coding_agent)
        self.registry.register(TaskType.WORKFLOW, self.workflow_agent)
        self.registry.register(TaskType.CHAT, self.chat_agent)

        self.dispatcher = TaskDispatcher(self.router, self.registry)

    def test_document_question_dispatches_to_rag(self):
        decision, result = self.dispatcher.dispatch(
            "Find the answer in my PDF and cite the page"
        )
        self.assertEqual(decision.task_type, TaskType.RAG)
        self.assertEqual(result.agent_name, "RAG Agent")
        self.assertEqual(len(self.rag_agent.calls), 1)

    def test_code_request_dispatches_to_coding(self):
        decision, result = self.dispatcher.dispatch("Debug this code")
        self.assertEqual(decision.task_type, TaskType.CODING)
        self.assertEqual(result.agent_name, "Coding Agent")

    def test_workflow_request_dispatches_to_workflow(self):
        decision, result = self.dispatcher.dispatch(
            "Coordinate the routing, workflow, and resource management components"
        )
        self.assertEqual(decision.task_type, TaskType.WORKFLOW)
        self.assertEqual(result.agent_name, "Workflow Agent")

    def test_general_question_dispatches_to_chat(self):
        decision, result = self.dispatcher.dispatch(
            "Please explain what a Python decorator is"
        )
        self.assertEqual(decision.task_type, TaskType.CHAT)
        self.assertEqual(result.agent_name, "Chat Agent")

    def test_trimmed_query_is_dispatched(self):
        _, result = self.dispatcher.dispatch("  Debug this code  ")
        self.assertEqual(self.coding_agent.calls, ["Debug this code"])
        self.assertTrue(result.success)

    def test_empty_query_rejected(self):
        for query in ("", "   "):
            with self.subTest(query=repr(query)):
                with self.assertRaises(ValueError):
                    self.dispatcher.dispatch(query)

    def test_non_string_query_rejected(self):
        with self.assertRaises(TypeError):
            self.dispatcher.dispatch(None)

    def test_missing_agent_propagates_lookup_error(self):
        registry = AgentRegistry()
        dispatcher = TaskDispatcher(self.router, registry)
        with self.assertRaises(LookupError):
            dispatcher.dispatch("Debug this code")


if __name__ == "__main__":
    unittest.main(verbosity=2)
