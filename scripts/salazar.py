"""Dispatcher-backed CLI for SALAZAR."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bootstrap import build_dispatcher


def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Usage: python scripts/salazar.py "your request"')
        raise SystemExit(1)

    dispatcher = build_dispatcher()
    decision = dispatcher.router.route(query)
    print(f"[SALAZAR] Route: {decision.task_type.value}")
    print(f"[SALAZAR] Reason: {decision.reason}")

    try:
        decision, result = dispatcher.dispatch(query)
    except LookupError:
        print(
            f"[SALAZAR] The {decision.task_type.value} agent is not yet "
            "registered for execution."
        )
        print(
            "[SALAZAR] Existing RAG CLI remains available at "
            "scripts/ask_salazar.py."
        )
        raise SystemExit(2)

    print(f"[SALAZAR] Agent: {result.agent_name}")
    if not result.success:
        print(result.content)
        raise SystemExit(1)

    print("\n--- SALAZAR Answer ---\n")
    print(result.content)

    if result.sources:
        print("\n--- Retrieved sources ---")
        for source in result.sources:
            print(f"- {source}")


if __name__ == "__main__":
    main()
