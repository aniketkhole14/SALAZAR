import subprocess
import sys

TESTS = [
    (
        "CHAT",
        "What does SALAZAR's Local AI Chat component do?",
        [
            "streams tokens",
            "Markdown/code blocks",
            "message/agent/model history",
        ],
    ),
    (
        "ROUTING",
        "How does SALAZAR's intelligent model routing work?",
        [
            "classifies task",
            "manual override",
            "page 4",
        ],
    ),
    (
        "WORKFLOW",
        "How does SALAZ execute multi-agent workflows, and what happens to intermediate outputs?",
        [
            "sequential or concurrent execution",
            "Planner, Coder, Reviewer",
            "intermediate outputs across stages",
        ],
    ),
    (
        "RAG",
        "How does SALAZ process documents and use RAG?",
        [
            "chunks text",
            "generates embeddings",
            "similarity search",
        ],
    ),
    (
        "CODING",
        "What does SALAZ's Coding Assistance component do?",
        [
            "code generation",
            "debugging",
            "directory-restricted file access",
        ],
    ),
    (
        "RESOURCE",
        "What does SALAZ's resource and lifecycle management component do?",
        [
            "CPU/RAM/VRAM",
            "Load",
            "prevents memory overflow",
        ],
    ),

    (
        "COMPONENT INTERACTION",
        "Trace how a user request is routed to an agent, executed through a workflow using a local model, and managed by the ResourceManager. Explain the documented relationships.",
        [
            "Workflow:",
            "Intelligent Model Routing:",
            "Resource and Lifecycle Management:",
            "does not explicitly specify the exact integration mechanism",
        ],
    ),
]

passed = 0

for name, question, expected_phrases in TESTS:
    print(f"\n{'=' * 18} {name} {'=' * 18}")
    result = subprocess.run(
        [sys.executable, "scripts/ask_salazar.py", question],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout
    print(output)

    missing = [
        phrase for phrase in expected_phrases
        if phrase.casefold() not in output.casefold()
    ]

    if result.returncode == 0 and not missing:
        print(f"TEST RESULT: PASS — {name}")
        passed += 1
    else:
        print(f"TEST RESULT: FAIL — {name}")
        if result.returncode != 0:
            print(f"Exit code: {result.returncode}")
        if missing:
            print("Missing expected phrases:", ", ".join(missing))

print(f"\nSUMMARY: {passed}/{len(TESTS)} tests passed.")
sys.exit(0 if passed == len(TESTS) else 1)
