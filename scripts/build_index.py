from pathlib import Path
import hashlib
import sqlite3
import sys

import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data/documents"
MODEL_DIR = ROOT / "models/embeddings/bge-small-en-v1.5"
INDEX_DIR = ROOT / "data/index"
DB_PATH = ROOT / "data/database/salazar.db"
VECTOR_PATH = INDEX_DIR / "embeddings.npy"

CHUNK_SIZE = 180
OVERLAP = 30
BATCH_SIZE = 16
MIN_PAGE_WORDS = 20
MIN_MEANINGFUL_WORDS = 10


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if piece:
            yield " ".join(piece)


def count_meaningful_words(text):
    return sum(
        any(char.isalnum() for char in word)
        for word in text.split()
    )


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_DIR.exists():
        sys.exit(f"Embedding model not found: {MODEL_DIR}")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"No PDF files found in {PDF_DIR}")

    records = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if not text or len(text.split()) < MIN_PAGE_WORDS:
                    print(f"  Skipping short/empty page {page_number}")
                    continue

                for chunk_index, chunk in enumerate(chunk_text(text)):
                    if count_meaningful_words(chunk) < MIN_MEANINGFUL_WORDS:
                        print(
                            f"  Skipping noise chunk on page {page_number}, "
                            f"chunk {chunk_index}"
                        )
                        continue

                    chunk_id = hashlib.sha256(
                        f"{pdf_path.name}|{page_number}|{chunk_index}|{chunk}".encode()
                    ).hexdigest()

                    records.append({
                        "chunk_id": chunk_id,
                        "document": pdf_path.name,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "text": chunk,
                    })

    if not records:
        sys.exit("No meaningful text found in the PDFs.")

    print(f"\nChunks prepared: {len(records)}")
    print("Loading embedding model on CPU...")

    model = SentenceTransformer(
        str(MODEL_DIR),
        device="cpu",
        local_files_only=True,
    )

    vectors = []

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        encoded = model.encode(
            [item["text"] for item in batch],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vectors.append(encoded.astype(np.float32))
        print(f"Embedded {min(start + BATCH_SIZE, len(records))}/{len(records)}")

    matrix = np.concatenate(vectors, axis=0)

    # Save vectors in the same order as the records.
    np.save(VECTOR_PATH, matrix)

    # Rebuild the SQLite metadata index in matching vector order.
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                vector_index INTEGER NOT NULL,
                text TEXT NOT NULL
            )
        """)

        connection.execute("DELETE FROM chunks")

        connection.executemany("""
            INSERT INTO chunks (
                chunk_id, document, page_number,
                chunk_index, vector_index, text
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, [
            (
                item["chunk_id"],
                item["document"],
                item["page"],
                item["chunk_index"],
                index,
                item["text"],
            )
            for index, item in enumerate(records)
        ])

    print("\nIndex build complete.")
    print(f"Documents: {len(pdf_files)}")
    print(f"Chunks: {len(records)}")
    print(f"Vector matrix: {matrix.shape}")
    print(f"Vectors saved: {VECTOR_PATH}")
    print(f"Metadata saved: {DB_PATH}")


if __name__ == "__main__":
    main()
