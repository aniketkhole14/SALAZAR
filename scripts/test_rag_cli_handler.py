"""Unit tests for the existing RAG CLI subprocess adapter."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag_cli_handler import run_rag_cli


class FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestRAGCLIHandler(unittest.TestCase):
    def test_parses_answer_and_sources(self):
        output = """[Routing preview] rag

--- SALAZAR Answer ---

SALAZAR routing uses task classification. (Source: guide.pdf, page 4)

--- Retrieved sources ---
- guide.pdf, page 4 (reranker score: -1.2000)
- SRS.docx, page 2 (reranker score: -2.3000)
"""
        with patch(
            "app.rag_cli_handler.subprocess.run",
            return_value=FakeCompleted(stdout=output),
        ):
            result = run_rag_cli("Explain routing")

        self.assertEqual(
            result["answer"],
            "SALAZAR routing uses task classification. (Source: guide.pdf, page 4)",
        )
        self.assertEqual(
            result["sources"],
            ["guide.pdf, page 4", "SRS.docx, page 2"],
        )

    def test_deduplicates_sources(self):
        output = """--- SALAZAR Answer ---

Answer

--- Retrieved sources ---
- guide.pdf, page 4 (reranker score: -1.0)
- guide.pdf, page 4 (reranker score: -2.0)
"""
        with patch(
            "app.rag_cli_handler.subprocess.run",
            return_value=FakeCompleted(stdout=output),
        ):
            result = run_rag_cli("Question")

        self.assertEqual(result["sources"], ["guide.pdf, page 4"])

    def test_nonzero_exit_raises(self):
        with patch(
            "app.rag_cli_handler.subprocess.run",
            return_value=FakeCompleted(
                stderr="model failed",
                returncode=1,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                run_rag_cli("Question")

    def test_missing_answer_section_raises(self):
        with patch(
            "app.rag_cli_handler.subprocess.run",
            return_value=FakeCompleted(stdout="No answer section here"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not locate"):
                run_rag_cli("Question")

    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            run_rag_cli("  ")

    def test_non_string_query_rejected(self):
        with self.assertRaises(TypeError):
            run_rag_cli(None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
