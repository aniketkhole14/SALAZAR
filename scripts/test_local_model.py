"""Unit tests for SALAZAR's local inference runner."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.local_model import LocalModelRunner


class TestLocalModelRunner(unittest.TestCase):
    def setUp(self):
        self.runner = LocalModelRunner(
            executable=Path("/tmp/fake-llama"),
            model_path=Path("/tmp/fake-model.gguf"),
            working_directory=Path("/tmp"),
            context_size=4096,
            max_new_tokens=768,
            threads=4,
        )

    def test_invalid_generation_configuration_rejected(self):
        with self.assertRaises(ValueError):
            LocalModelRunner(
                executable=Path("/tmp/llama"),
                model_path=Path("/tmp/model.gguf"),
                working_directory=Path("/tmp"),
                context_size=0,
            )

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            self.runner.generate("   ")

    def test_non_string_prompt_rejected(self):
        with self.assertRaises(TypeError):
            self.runner.generate(None)

    @patch("app.local_model.subprocess.run")
    @patch("app.local_model.Path.exists", return_value=True)
    def test_generate_uses_configured_limits(self, mock_exists, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Generated answer",
            stderr="",
        )

        result = self.runner.generate("Test prompt")

        self.assertEqual(result, "Generated answer")

        command = mock_run.call_args.args[0]
        self.assertIn("4096", command)
        self.assertIn("768", command)
        self.assertIn("4", command)

    @patch("app.local_model.subprocess.run")
    @patch("app.local_model.Path.exists", return_value=True)
    def test_nonzero_exit_code_raises_error(self, mock_exists, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Model failure",
        )

        with self.assertRaises(RuntimeError):
            self.runner.generate("Test prompt")



    @patch("app.local_model.subprocess.run")
    @patch("app.local_model.Path.exists", return_value=True)
    def test_generate_chat_separates_system_and_user(
        self, mock_exists, mock_run
    ):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Complete answer", stderr=""
        )

        result = self.runner.generate_chat(
            system_prompt="You are SALAZAR.",
            user_prompt="Explain linked lists.",
        )

        self.assertEqual(result, "Complete answer")
        command = mock_run.call_args.args[0]

        self.assertEqual(
            command[command.index("--system-prompt") + 1],
            "You are SALAZAR.",
        )
        self.assertEqual(
            command[command.index("--prompt") + 1],
            "Explain linked lists.",
        )

    @patch("app.local_model.subprocess.run")
    @patch("app.local_model.Path.exists", return_value=True)
    def test_generate_chat_uses_sampling_settings(
        self, mock_exists, mock_run
    ):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Answer", stderr=""
        )

        self.runner.generate_chat(
            "System instructions", "User request", temperature=0.2
        )

        command = mock_run.call_args.args[0]
        self.assertIn("0.2", command)
        self.assertIn("0.9", command)
        self.assertIn("1.05", command)

    def test_generate_chat_rejects_empty_system_prompt(self):
        with self.assertRaises(ValueError):
            self.runner.generate_chat("", "User request")

    def test_generate_chat_rejects_empty_user_prompt(self):
        with self.assertRaises(ValueError):
            self.runner.generate_chat("System instructions", " ")

    def test_generate_chat_rejects_invalid_temperature(self):
        with self.assertRaises(ValueError):
            self.runner.generate_chat(
                "System instructions", "User request", temperature=3.0
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
