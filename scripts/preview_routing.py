"""Preview SALAZAR task-routing decisions without executing agents."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.task_router import TaskRouter


def main():
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        print('Usage: python scripts/preview_routing.py "your question"')
        raise SystemExit(1)

    decision = TaskRouter().route(query)

    print("Query:", query)
    print("Proposed task:", decision.task_type.value)
    print("Reason:", decision.reason)
    print("Execution: preview only; no agent or model was invoked.")


if __name__ == "__main__":
    main()
