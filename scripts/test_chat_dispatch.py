"""Integration test for SALAZAR's real local chat dispatch path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent_registry import AgentRegistry
from app.chat_agent import ChatAgent
from app.dispatcher import TaskDispatcher
from app.local_model import LocalModelRunner
from app.runtime_discovery import discover_llama_executable
from app.task_router import TaskRouter, TaskType


runner = LocalModelRunner(
    executable=discover_llama_executable(),
    model_path=ROOT / "models/llm/smollm2-1.7b/smollm2-1.7b-instruct-q4_k_m.gguf",
    working_directory=ROOT,
    context_size=1024,
    max_new_tokens=96,
    threads=4,
)

registry = AgentRegistry()
registry.register(TaskType.CHAT, ChatAgent(runner))

dispatcher = TaskDispatcher(TaskRouter(), registry)

query = "Explain what a local language model is in one sentence."
decision, result = dispatcher.dispatch(query)

print("\n=== SALAZAR Local Chat Integration Test ===")
print(f"Query: {query}")
print(f"Route: {decision.task_type.value}")
print(f"Reason: {decision.reason}")
print(f"Agent: {result.agent_name}")
print(f"Success: {result.success}")
print(f"\nResponse:\n{result.content}")

if decision.task_type != TaskType.CHAT:
    raise SystemExit("FAIL: Query was not routed to CHAT.")

if not result.success or not result.content.strip():
    raise SystemExit("FAIL: The local chat agent did not return a usable response.")

print("\nPASS: Dispatcher-to-local-model execution completed.")
