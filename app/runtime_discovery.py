"""Discover a compatible llama.cpp executable for SALAZAR."""

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("SALAZAR_ROOT", Path(__file__).resolve().parents[1])
).resolve()


def discover_llama_executable() -> Path:
    """Return the first available llama.cpp executable.

    Bundled project runtimes take precedence over development fallbacks.
    """
    suffix = ".exe" if os.name == "nt" else ""

    names = (
        f"llama-completion{suffix}",
        f"llama-cli{suffix}",
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

    # Development-only fallbacks.
    home = Path.home()
    candidates.extend(
        (
            home / "salazar-build" / "salazar-runtime-build" / "bin" / names[0],
            home / "salazar-build" / "llama.cpp" / "bin" / names[1],
        )
    )

    # Optional explicit override for development and testing.
    override = os.environ.get("SALAZAR_LLAMA_EXECUTABLE")
    if override:
        candidates.insert(0, Path(override).expanduser())

    for candidate in candidates:
        if not candidate.is_file():
            continue

        if os.name == "nt" or os.access(candidate, os.X_OK):
            return candidate.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "No executable llama.cpp runtime was found. "
        "Place a compatible executable in SALAZAR's runtime/bin directory "
        "or set SALAZAR_LLAMA_EXECUTABLE.\nSearched:\n" + searched
    )
