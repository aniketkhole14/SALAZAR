from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

root = Path(__file__).resolve().parents[1]
model_path = root / "models/embeddings/bge-small-en-v1.5"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = SentenceTransformer(
    str(model_path),
    device="cpu",
    local_files_only=True,
)

sentences = [
    "Virtual memory provides programs with an abstract memory space.",
    "Paging maps virtual memory pages to physical memory frames.",
    "A TCP handshake establishes a connection between network hosts.",
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
    convert_to_tensor=True,
)

print("Device:", embeddings.device)
print("Embedding shape:", tuple(embeddings.shape))
print("Memory-related similarity:", (embeddings[0] @ embeddings[1]).item())
print("Unrelated similarity:", (embeddings[0] @ embeddings[2]).item())
