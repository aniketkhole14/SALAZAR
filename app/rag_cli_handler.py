"""Subprocess adapter for SALAZAR's existing document-grounded RAG CLI."""

import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("SALAZAR_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAG_SCRIPT = ROOT / "scripts" / "ask_salazar.py"
RAG_EXECUTABLE = os.environ.get("SALAZAR_RAG_EXECUTABLE")


def run_rag_cli(query: str) -> dict[str, object]:
    """Run the existing RAG CLI and return its answer and source citations."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")

    query = query.strip()
    if not query:
        raise ValueError("query must not be empty.")

    if RAG_EXECUTABLE:
        executable = Path(RAG_EXECUTABLE).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"Packaged RAG executable not found: {executable}")
        command = [str(executable), query]
    else:
        if getattr(sys, "frozen", False):
            raise RuntimeError(
                "Packaged SALAZAR backend requires SALAZAR_RAG_EXECUTABLE."
            )
        if not RAG_SCRIPT.is_file():
            raise FileNotFoundError(f"RAG CLI not found: {RAG_SCRIPT}")
        command = [sys.executable, str(RAG_SCRIPT), query]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
        check=False,
    )

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Existing RAG CLI failed with exit code {completed.returncode}: "
            f"{detail[-4000:]}"
        )

    output = completed.stdout
    answer_match = re.search(
        r"--- SALAZAR Answer ---\s*(.*?)"
        r"(?=\s*--- Retrieved sources ---|\Z)",
        output,
        flags=re.DOTALL,
    )

    if not answer_match:
        raise RuntimeError(
            "Could not locate the answer section in RAG CLI output."
        )

    answer = answer_match.group(1).strip()
    source_section = output.split("--- Retrieved sources ---", 1)

    sources: list[str] = []
    if len(source_section) == 2:
        for line in source_section[1].splitlines():
            line = line.strip()
            if line.startswith("- "):
                # Remove the reranker score but preserve document and page.
                source = re.sub(r"\s+\(reranker score:.*\)$", "", line[2:])
                if source and source not in sources:
                    sources.append(source)

    if not answer:
        raise RuntimeError("The existing RAG CLI returned an empty answer.")

    return {
        "answer": answer,
        "sources": sources,
    }
