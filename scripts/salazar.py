"""Dispatcher-backed CLI for SALAZAR."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_registry import AgentRegistry
from app.chat_agent import ChatAgent
from app.coding_agent import CodingAgent
from app.dispatcher import TaskDispatcher
from app.local_model import LocalModelRunner
from app.rag_agent import RAGAgent
from app.rag_cli_handler import run_rag_cli
from app.task_router import TaskRouter, TaskType
from app.workflow_agent import WorkflowAgent


def build_dispatcher():
    """Register only agents that are ready for real execution."""
    runner = LocalModelRunner(
        executable=Path.home() / "salazar-build/llama.cpp/bin/llama-cli",
        model_path=ROOT / "models/llm/smollm2-1.7b/smollm2-1.7b-instruct-q4_k_m.gguf",
        working_directory=ROOT,
        context_size=1024,
        max_new_tokens=384,
        threads=4,
    )

    registry = AgentRegistry()
    registry.register(TaskType.CHAT, ChatAgent(runner))
    registry.register(TaskType.CODING, CodingAgent(runner))
    registry.register(TaskType.RAG, RAGAgent(run_rag_cli))
    registry.register(TaskType.WORKFLOW, WorkflowAgent(registry))
    return TaskDispatcher(TaskRouter(), registry)


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
        print("[SALAZAR] Existing RAG CLI remains available at scripts/ask_salazar.py.")
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
