"""Unit tests for SALAZAR's RAG agent adapter."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents import TaskResult
from app.rag_agent import RAGAgent


class TestRAGAgent(unittest.TestCase):
    def test_returns_answer_and_sources(self):
        agent = RAGAgent(lambda task: {
            "answer": f"Answer to: {task}",
            "sources": ["guide.pdf, page 4", "SRS.docx, page 2"],
        })

        result = agent.execute("Explain routing")

        self.assertIsInstance(result, TaskResult)
        self.assertEqual(result.agent_name, "Document RAG Agent")
        self.assertEqual(result.task_type, "rag")
        self.assertEqual(result.content, "Answer to: Explain routing")
        self.assertEqual(
            result.sources,
            ("guide.pdf, page 4", "SRS.docx, page 2"),
        )
        self.assertTrue(result.success)

    def test_empty_answer_is_unsuccessful(self):
        agent = RAGAgent(lambda task: {"answer": "  ", "sources": []})

        result = agent.execute("Explain routing")

        self.assertFalse(result.success)
        self.assertIn("no answer", result.content.lower())

    def test_invalid_handler_rejected(self):
        with self.assertRaises(TypeError):
            RAGAgent(object())

    def test_invalid_response_rejected(self):
        agent = RAGAgent(lambda task: "not a mapping")

        with self.assertRaises(TypeError):
            agent.execute("Explain routing")

    def test_missing_answer_rejected(self):
        agent = RAGAgent(lambda task: {"sources": []})

        with self.assertRaises(TypeError):
            agent.execute("Explain routing")

    def test_invalid_sources_rejected(self):
        agent = RAGAgent(lambda task: {
            "answer": "Answer",
            "sources": "not a source list",
        })

        with self.assertRaises(TypeError):
            agent.execute("Explain routing")

    def test_empty_task_rejected(self):
        agent = RAGAgent(lambda task: {"answer": "Answer", "sources": []})

        with self.assertRaises(ValueError):
            agent.execute(" ")

    def test_handler_receives_validated_task(self):
        received = []
        agent = RAGAgent(
            lambda task: received.append(task) or {
                "answer": "Answer",
                "sources": [],
            }
        )

        agent.execute("  Explain routing  ")

        self.assertEqual(received, ["Explain routing"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
