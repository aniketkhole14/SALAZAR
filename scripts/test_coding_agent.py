"""Unit tests for SALAZAR's local Coding Agent."""

import unittest

from app.agents import TaskResult
from app.coding_agent import CodingAgent


class FakeRunner:
    def __init__(self, response="print('Hello, SALAZAR!')"):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


class TestCodingAgent(unittest.TestCase):
    def test_agent_identity(self):
        agent = CodingAgent(FakeRunner())
        self.assertEqual(agent.name, "Local Coding Agent")
        self.assertEqual(agent.task_type, "coding")

    def test_invalid_runner_rejected(self):
        with self.assertRaises(TypeError):
            CodingAgent(object())

    def test_empty_task_rejected(self):
        agent = CodingAgent(FakeRunner())
        with self.assertRaises(ValueError):
            agent.execute("   ")

    def test_non_string_task_rejected(self):
        agent = CodingAgent(FakeRunner())
        with self.assertRaises(TypeError):
            agent.execute(None)

    def test_prompt_contains_user_request(self):
        runner = FakeRunner()
        agent = CodingAgent(runner)
        request = "Write a Python function to reverse a string."

        agent.execute(request)

        self.assertIn(request, runner.prompts[0])
        self.assertIn("Do not execute generated code", runner.prompts[0])

    def test_review_prompt_requires_edge_case_analysis(self):
        runner = FakeRunner()
        agent = CodingAgent(runner)

        agent.execute(
            "Review this code: ```python\\ndef divide(a, b):\\n    return a / b\\n```"
        )

        prompt = runner.prompts[0].lower()
        self.assertIn("edge case", prompt)
        self.assertIn("division by zero", prompt)
        self.assertIn("no obvious bug", prompt)

    def test_returns_structured_result(self):
        runner = FakeRunner("def reverse_text(text):\n    return text[::-1]")
        agent = CodingAgent(runner)

        result = agent.execute("Write a string reversal function.")

        self.assertIsInstance(result, TaskResult)
        self.assertEqual(result.agent_name, "Local Coding Agent")
        self.assertEqual(result.task_type, "coding")
        self.assertTrue(result.success)
        self.assertIn("reverse_text", result.content)

    def test_empty_model_response_is_unsuccessful(self):
        agent = CodingAgent(FakeRunner("   "))

        result = agent.execute("Write a Python function.")

        self.assertFalse(result.success)
        self.assertIn("empty response", result.content.lower())

    def test_model_output_is_not_executed(self):
        runner = FakeRunner("__import__('os').system('touch /tmp/salazar_coding_agent_test')")
        agent = CodingAgent(runner)

        result = agent.execute("Generate a Python example.")

        self.assertTrue(result.success)
        self.assertFalse(
            __import__("pathlib").Path(
                "/tmp/salazar_coding_agent_test"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
