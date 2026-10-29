from pathlib import Path
from sentence_transformers import CrossEncoder

root = Path(__file__).resolve().parents[1]
model_path = root / "models" / "reranker" / "minilm"

print("Loading reranker on CPU...")
model = CrossEncoder(str(model_path), device="cpu")

query = "What are the functional components of SALAZAR?"

passages = [
    "SALAZAR supports local AI chat, intelligent model routing, multi-agent workflows, document processing and RAG, coding assistance, and resource and lifecycle management.",
    "UML composition describes a strong ownership relationship between objects such as users, conversations and messages.",
    "The system monitors CPU and RAM, manages model states, and prevents memory overflow.",
    "The frontend communicates with the backend through a REST or WebSocket API.",
]

scores = model.predict([(query, passage) for passage in passages])
results = sorted(
    zip(scores, passages),
    key=lambda item: float(item[0]),
    reverse=True,
)

print("\nQuery:", query)
print("\nRanked passages:")

for rank, (score, passage) in enumerate(results, start=1):
    print(f"\n{rank}. Score: {float(score):.4f}")
    print(f"   {passage}")

print("\nDevice: CPU")
print("Reranker test completed.")
