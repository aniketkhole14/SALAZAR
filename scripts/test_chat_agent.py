"""Unit tests for SALAZAR's local ChatAgent."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents import TaskResult
from app.chat_agent import ChatAgent


class MockRunner:
    def __init__(self, response="Hello from the local model."):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


class TestChatAgent(unittest.TestCase):
    def test_returns_structured_result(self):
        runner = MockRunner()
        agent = ChatAgent(runner)

        result = agent.execute("Hello SALAZAR")

        self.assertIsInstance(result, TaskResult)
        self.assertEqual(result.agent_name, "Local Chat Agent")
        self.assertEqual(result.task_type, "chat")
        self.assertEqual(result.content, "Hello from the local model.")
        self.assertTrue(result.success)

    def test_prompt_contains_user_request(self):
        runner = MockRunner()
        agent = ChatAgent(runner)

        agent.execute("Explain local inference")

        self.assertIn("Explain local inference", runner.prompts[0])
        self.assertIn("offline AI assistant", runner.prompts[0])

    def test_empty_model_response_is_reported(self):
        agent = ChatAgent(MockRunner("   "))

        result = agent.execute("Hello")

        self.assertFalse(result.success)
        self.assertIn("empty response", result.content)

    def test_workflow_follow_up_requires_grounded_concise_summary(self):
        runner = MockRunner("The workflow coordinates specialized agents.")
        agent = ChatAgent(runner)

        task = (
            "Original user request:\\n"
            "Search my documents for the SALAZAR system architecture.\\n\\n"
            "Instruction:\\n"
            "Summarize the retrieved information.\\n\\n"
            "Previous agent output:\\n"
            "The workflow coordinates specialized agents. "
            "(Source: architecture.pdf, page 4)"
        )

        result = agent.execute(task)
        prompt = runner.prompts[0].lower()

        self.assertTrue(result.success)
        self.assertIn("do not invent component responsibilities", prompt)
        self.assertIn("source references", prompt)
        self.assertIn("concise synthesis", prompt)

    def test_regular_chat_does_not_use_workflow_prompt(self):
        runner = MockRunner()
        agent = ChatAgent(runner)

        agent.execute("Explain local inference")

        prompt = runner.prompts[0].lower()
        self.assertIn("offline ai assistant", prompt)
        self.assertNotIn("previous agent output as evidence", prompt)

    def test_empty_task_rejected(self):
        agent = ChatAgent(MockRunner())

        with self.assertRaises(ValueError):
            agent.execute("   ")

    def test_non_string_task_rejected(self):
        agent = ChatAgent(MockRunner())

        with self.assertRaises(TypeError):
            agent.execute(None)

    def test_invalid_runner_rejected(self):
        with self.assertRaises(TypeError):
            ChatAgent(object())


if __name__ == "__main__":
    unittest.main(verbosity=2)
