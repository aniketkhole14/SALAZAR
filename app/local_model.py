"""Reusable local GGUF inference through llama.cpp's llama-cli."""

from pathlib import Path
import subprocess
import tempfile


class LocalModelRunner:
    """Run a local GGUF model without requiring network access."""

    def __init__(
        self,
        executable: Path,
        model_path: Path,
        working_directory: Path,
        context_size: int = 4096,
        max_new_tokens: int = 384,
        threads: int = 4,
    ):
        self.executable = Path(executable)
        self.model_path = Path(model_path)
        self.working_directory = Path(working_directory)
        self.context_size = context_size
        self.max_new_tokens = max_new_tokens
        self.threads = threads

        if context_size < 1 or max_new_tokens < 1 or threads < 1:
            raise ValueError("Context size, output tokens, and threads must be positive.")

    def generate(self, prompt: str) -> str:
        """Generate a response from the configured local model."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

        for resource in (
            self.executable,
            self.model_path,
            self.working_directory,
        ):
            if not resource.exists():
                raise FileNotFoundError(f"Required resource missing: {resource}")

        with tempfile.TemporaryDirectory(prefix="salazar-") as temp_dir:
            temp_path = Path(temp_dir)
            prompt_path = temp_path / "prompt.txt"
            output_path = temp_path / "output.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            output_path.write_text("", encoding="utf-8")

            command = [
                str(self.executable),
                "-m", str(self.model_path),
                "-c", str(self.context_size),
                "-n", str(self.max_new_tokens),
                "-t", str(self.threads),
                "--simple-io",
                "--single-turn",
                "--no-display-prompt",
                "--log-disable",
                "--file", str(prompt_path),
                "--output-file", str(output_path),
            ]

            completed = subprocess.run(
                command,
                cwd=self.working_directory,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                check=False,
            )

            if completed.returncode != 0:
                raise RuntimeError(
                    "llama-cli failed with exit code "
                    f"{completed.returncode}: {completed.stderr[-6000:]}"
                )

            output = output_path.read_text(encoding="utf-8").strip()
            if not output:
                output = completed.stdout.strip()

            if "ANSWER:" in output:
                output = output.rsplit("ANSWER:", 1)[-1].strip()

            # Remove llama-cli diagnostic suffixes.
            output = output.split("[ Prompt:", 1)[0].strip()

            # If llama-cli echoed the conversation, retain only the final
            # assistant message. These are actual newline characters.
            if "\nAssistant:" in output:
                output = output.rsplit("\nAssistant:", 1)[-1].strip()

            return output
