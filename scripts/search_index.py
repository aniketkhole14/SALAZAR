from pathlib import Path
import sqlite3
import sys

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL_DIR = ROOT / "models/embeddings/bge-small-en-v1.5"
RERANKER_MODEL_DIR = ROOT / "models/reranker/minilm"
VECTOR_PATH = ROOT / "data/index/embeddings.npy"
DB_PATH = ROOT / "data/database/salazar.db"

RETRIEVE_K = 10
FINAL_K = 5


def main():
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        print('Usage: python scripts/search_index.py "your question"')
        raise SystemExit(1)

    for path in (
        EMBEDDING_MODEL_DIR,
        RERANKER_MODEL_DIR,
        VECTOR_PATH,
        DB_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required SALAZAR resource missing: {path}")

    vectors = np.load(VECTOR_PATH, allow_pickle=False)

    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute("""
            SELECT vector_index, document, page_number, text
            FROM chunks
            ORDER BY vector_index
        """).fetchall()

    if len(rows) != len(vectors):
        raise RuntimeError(
            f"Index mismatch: {len(rows)} chunks but {len(vectors)} vectors"
        )

    if not rows:
        print("The document index is empty.")
        return

    print("Loading embedding model on CPU...")
    embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_DIR),
        device="cpu",
        local_files_only=True,
    )

    query_vector = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    if vectors.ndim != 2 or vectors.shape[1] != query_vector.shape[0]:
        raise RuntimeError(
            f"Embedding dimension mismatch: vectors={vectors.shape}, "
            f"query={query_vector.shape}"
        )

    semantic_scores = vectors @ query_vector
    candidate_count = min(RETRIEVE_K, len(rows))
    candidate_indices = np.argsort(semantic_scores)[::-1][:candidate_count]

    print("Loading reranker on CPU...")
    reranker = CrossEncoder(
        str(RERANKER_MODEL_DIR),
        device="cpu",
        local_files_only=True,
    )

    pairs = [
        (query, rows[int(idx)][3])
        for idx in candidate_indices
    ]
    rerank_scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(candidate_indices, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:FINAL_K]

    print(f"\nQuery: {query}")
    print(
        f"Retrieved {candidate_count} candidates; "
        f"showing top {len(ranked_results)} after reranking.\n"
    )

    for rank, (idx, rerank_score) in enumerate(ranked_results, start=1):
        _, document, page, text = rows[int(idx)]
        print(f"{rank}. Reranker score: {float(rerank_score):.4f}")
        print(f"   Semantic score: {float(semantic_scores[idx]):.4f}")
        print(f"   Source: {document}, page {page}")
        print(f"   Passage: {text}\n")


if __name__ == "__main__":
    main()
