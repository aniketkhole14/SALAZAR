"""Reusable local GGUF inference through llama.cpp."""

from pathlib import Path
import os
import subprocess


class LocalModelRunner:
    """Run a local GGUF model without requiring network access."""

    def __init__(
        self,
        executable: Path,
        model_path: Path,
        working_directory: Path,
        context_size: int = 4096,
        max_new_tokens: int = 768,
        threads: int = 4,
        temperature: float = 0.35,
        top_p: float = 0.9,
        repeat_penalty: float = 1.05,
    ):
        self.executable = Path(executable)
        self.model_path = Path(model_path)
        self.working_directory = Path(working_directory)
        self.context_size = context_size
        self.max_new_tokens = max_new_tokens
        self.threads = threads
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

        if context_size < 1 or max_new_tokens < 1 or threads < 1:
            raise ValueError(
                "Context size, output tokens, and threads must be positive."
            )

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")

        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1.")

        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be positive.")

    def generate(self, prompt: str) -> str:
        """Generate a response from a plain-text prompt."""
        return self._generate(prompt)

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        """Generate a response using separate system and user messages."""
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string.")

        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")

        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")

        return self._generate(
            user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    def _generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Run inference with shared validation and generation settings."""
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

        effective_temperature = (
            self.temperature if temperature is None else temperature
        )

        command = [
            str(self.executable),
            "-m", str(self.model_path),
            "-c", str(self.context_size),
            "-n", str(self.max_new_tokens),
            "-t", str(self.threads),
            "--temp", str(effective_temperature),
            "--top-p", str(self.top_p),
            "--repeat-penalty", str(self.repeat_penalty),
            "--simple-io",
            "--single-turn",
            "--no-display-prompt",
            "--no-perf",
        ]

        if system_prompt is not None:
            command.extend(["--system-prompt", system_prompt])

        command.extend(["--prompt", prompt])

        process_env = os.environ.copy()

        # Load llama.cpp shared libraries from the portable runtime directory.
        if os.name == "posix":
            runtime_dir = str(self.executable.resolve().parent)
            existing = process_env.get("LD_LIBRARY_PATH", "")
            process_env["LD_LIBRARY_PATH"] = (
                runtime_dir + (os.pathsep + existing if existing else "")
            )

        completed = subprocess.run(
            command,
            cwd=self.working_directory,
            env=process_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "llama.cpp failed with exit code "
                f"{completed.returncode}: {completed.stderr[-6000:]}"
            )

        output = completed.stdout.strip()

        if "ANSWER:" in output:
            output = output.rsplit("ANSWER:", 1)[-1].strip()

        output = output.split("[ Prompt:", 1)[0].strip()
        output = output.replace("[end of text]", "").strip()

        if "\nAssistant:" in output:
            output = output.rsplit("\nAssistant:", 1)[-1].strip()

        return output
