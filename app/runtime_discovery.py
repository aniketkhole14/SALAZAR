"""Discover a compatible llama.cpp executable for SALAZAR."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def discover_llama_executable() -> Path:
    """Return the first available llama.cpp executable.

    Bundled project runtimes take precedence over development fallbacks.
    """
    names = (
        "llama-completion",
        "llama-cli",
    )

    candidates = []

    # Portable, project-relative runtime locations.
    for name in names:
        candidates.extend(
            (
                PROJECT_ROOT / "runtime" / "bin" / name,
                PROJECT_ROOT / "runtime" / "llama.cpp" / "bin" / name,
                PROJECT_ROOT / "runtime" / "llama.cpp" / "build" / "bin" / name,
            )
        )

    # Development-only fallbacks; these are not portable installation paths.
    home = Path.home()
    candidates.extend(
        (
            home / "salazar-build" / "salazar-runtime-build" / "bin" / "llama-completion",
            home / "salazar-build" / "llama.cpp" / "bin" / "llama-cli",
        )
    )

    # Optional explicit override for development and testing.
    override = os.environ.get("SALAZAR_LLAMA_EXECUTABLE")
    if override:
        candidates.insert(0, Path(override).expanduser())

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "No executable llama.cpp runtime was found. "
        "Place a compatible executable in SALAZAR's runtime/bin directory "
        "or set SALAZAR_LLAMA_EXECUTABLE.\nSearched:\n" + searched
    )
