"""Shared initialization for SALAZAR interfaces."""

import os
from pathlib import Path

from app.agent_registry import AgentRegistry
from app.chat_agent import ChatAgent
from app.coding_agent import CodingAgent
from app.dispatcher import TaskDispatcher
from app.local_model import LocalModelRunner
from app.runtime_discovery import discover_llama_executable
from app.rag_agent import RAGAgent
from app.rag_cli_handler import run_rag_cli
from app.task_router import TaskRouter, TaskType
from app.workflow_agent import WorkflowAgent


ROOT = Path(os.environ.get("SALAZAR_ROOT", Path(__file__).resolve().parents[1])).resolve()


def build_dispatcher() -> TaskDispatcher:
    """Build the dispatcher and register executable agents."""
    runner = LocalModelRunner(
        executable=discover_llama_executable(),
        model_path=ROOT
        / "models/llm/smollm2-1.7b/smollm2-1.7b-instruct-q4_k_m.gguf",
        working_directory=ROOT,
        context_size=4096,
        max_new_tokens=768,
        temperature=0.35,
        top_p=0.9,
        repeat_penalty=1.05,
        threads=4,
    )

    registry = AgentRegistry()
    registry.register(TaskType.CHAT, ChatAgent(runner))
    registry.register(TaskType.CODING, CodingAgent(runner))
    registry.register(TaskType.RAG, RAGAgent(run_rag_cli))
    registry.register(TaskType.WORKFLOW, WorkflowAgent(registry))

    return TaskDispatcher(TaskRouter(), registry)
