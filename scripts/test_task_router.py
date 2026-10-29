"""Unit tests for SALAZAR's deterministic TaskRouter."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.task_router import TaskRouter, TaskType


class TestTaskRouter(unittest.TestCase):
    def setUp(self):
        self.router = TaskRouter()

    def assert_route(self, query, expected_type):
        decision = self.router.route(query)
        self.assertEqual(decision.task_type, expected_type)
        self.assertTrue(decision.reason.strip())

    def test_coding_request(self):
        self.assert_route("Debug this code", TaskType.CODING)

    def test_write_python_function_routes_to_coding(self):
        self.assert_route(
            "Write a Python function that checks whether a string is a palindrome. Include two examples.",
            TaskType.CODING,
        )

    def test_document_retrieval_request(self):
        self.assert_route(
            "Search my documents for the answer", TaskType.RAG
        )

    def test_workflow_request(self):
        self.assert_route(
            "Run a multi-step workflow", TaskType.WORKFLOW
        )

    def test_document_search_inside_explicit_workflow_routes_to_workflow(self):
        self.assert_route(
            "Run a multi-step workflow: search my documents for the "
            "SALAZAR system architecture and summarize the findings.",
            TaskType.WORKFLOW,
        )

    def test_document_question_without_workflow_remains_rag(self):
        self.assert_route(
            "Search my documents for the SALAZAR system architecture "
            "and summarize the findings.",
            TaskType.RAG,
        )

    def test_general_chat_request(self):
        self.assert_route("Hello SALAZAR", TaskType.CHAT)

    def test_case_insensitivity(self):
        self.assert_route("DEBUG THIS CODE", TaskType.CODING)

    def test_extra_whitespace(self):
        self.assert_route(
            "  Run   a multi-step   workflow  ", TaskType.WORKFLOW
        )

    def test_build_python_api(self):
        self.assert_route(
            "Can you help me build a Python API?", TaskType.CODING
        )

    def test_pdf_question_with_citation(self):
        self.assert_route(
            "Find the answer in my PDF and cite the page", TaskType.RAG
        )

    def test_multi_agent_planning(self):
        self.assert_route(
            "Plan a multi-step task using multiple agents",
            TaskType.WORKFLOW,
        )

    def test_conceptual_question_remains_chat(self):
        self.assert_route(
            "Please explain what a Python decorator is", TaskType.CHAT
        )

    def test_fix_code(self):
        self.assert_route(
            "Fix this code and explain the bug", TaskType.CODING
        )

    def test_document_summary(self):
        self.assert_route("Summarize the document", TaskType.RAG)

    def test_greeting(self):
        self.assert_route("Hi, how are you?", TaskType.CHAT)

    def test_review_then_fix_workflow(self):
        self.assert_route(
            "Review my project, identify bugs, and then fix them",
            TaskType.WORKFLOW,
        )

    def test_coordinate_salazar_components_workflow(self):
        self.assert_route(
            "Coordinate the routing, workflow, and resource management components",
            TaskType.WORKFLOW,
        )

    def test_question_about_documented_workflow_is_rag(self):
        self.assert_route(
            "How does SALAZAR's workflow component work according to the SRS?",
            TaskType.RAG,
        )

    def test_workflow_using_rag(self):
        self.assert_route("Workflow using RAG", TaskType.WORKFLOW)

    def test_routing_for_workflow(self):
        self.assert_route("Routing for Workflow", TaskType.WORKFLOW)

    def test_empty_query_rejected(self):
        for query in ("", "   ", "\n\t"):
            with self.subTest(query=repr(query)):
                with self.assertRaises(ValueError):
                    self.router.route(query)

    def test_non_string_query_rejected(self):
        for query in (None, 123, ["debug"]):
            with self.subTest(query=repr(query)):
                with self.assertRaises(TypeError):
                    self.router.route(query)


if __name__ == "__main__":
    unittest.main(verbosity=2)
