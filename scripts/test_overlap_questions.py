import subprocess
import sys

TESTS = [
    (
        "Workflow using RAG",
        "How does the Workflow component use RAG?",
        ("workflow", "rag"),
    ),
    (
        "Routing for Workflow",
        "How does Routing select agents for a Workflow?",
        ("routing", "workflow"),
    ),
    (
        "Resource Management supporting RAG",
        "How does Resource Management support RAG?",
        ("resource and lifecycle management", "rag"),
    ),
    (
        "Coding Assistance related to Local AI Chat",
        "How do Coding Assistance and Local AI Chat relate to each other?",
        ("coding assistance", "local ai chat"),
    ),
    (
        "Routing, Workflow, and Resource Management",
        "How do Routing, Workflow, and Resource Management relate to each other?",
        ("routing", "workflow", "resource and lifecycle management"),
    ),
]

GROUNDING_CAVEAT = (
    "does not explicitly specify"
)

failures = 0

for name, question, required_terms in TESTS:
    print(f"\n--- {name} ---")
    result = subprocess.run(
        [sys.executable, "scripts/ask_salazar.py", question],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout)

    output = result.stdout.casefold()
    missing = [
        term for term in required_terms
        if term not in output
    ]

    passed = (
        result.returncode == 0
        and not missing
        and GROUNDING_CAVEAT in output
    )

    if passed:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name}")
        if missing:
            print(f"  Missing component references: {missing}")
        if GROUNDING_CAVEAT not in output:
            print("  Missing explicit source-grounding caveat.")
        failures += 1

print(f"\nOverlap tests: {len(TESTS) - failures}/{len(TESTS)} passed.")
sys.exit(1 if failures else 0)
