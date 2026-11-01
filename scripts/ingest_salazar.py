"""Safe, versioned PDF ingestion for SALAZAR."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid

# Limit CPU thread pools before importing the ML runtime.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import pymupdf
import torch
from sentence_transformers import SentenceTransformer

# Keep CPU inference predictable on machines with limited memory.
torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


ROOT = Path(os.environ.get("SALAZAR_ROOT", Path(__file__).resolve().parents[1])).resolve()
PDF_DIR = ROOT / "data/documents"
MODEL_DIR = ROOT / "models/embeddings/bge-small-en-v1.5"
INDEX_DIR = ROOT / "data/index"
DB_PATH = ROOT / "data/database/salazar.db"
VECTOR_PATH = INDEX_DIR / "embeddings.npy"
MANIFEST_PATH = INDEX_DIR / "active.json"
GENERATION_DIR = INDEX_DIR / "generations"

CHUNK_SIZE = 180
OVERLAP = 30
BATCH_SIZE = 4
MIN_PAGE_WORDS = 20
MIN_MEANINGFUL_WORDS = 10


def chunk_text(text):
    words = text.split()
    step = CHUNK_SIZE - OVERLAP
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_SIZE]
        if piece:
            yield " ".join(piece)


def meaningful_words(text):
    return sum(any(char.isalnum() for char in word) for word in text.split())


def active_paths():
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        generation = GENERATION_DIR / manifest["generation"]
        return generation / "salazar.db", generation / "embeddings.npy"

    return DB_PATH, VECTOR_PATH


def extract_pdf(path):
    records = []

    with pymupdf.open(path) as document:
        if document.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text or len(text.split()) < MIN_PAGE_WORDS:
                continue

            for chunk_index, chunk in enumerate(chunk_text(text)):
                if meaningful_words(chunk) < MIN_MEANINGFUL_WORDS:
                    continue

                chunk_id = hashlib.sha256(
                    f"{path.name}|{page_number}|{chunk_index}|{chunk}".encode()
                ).hexdigest()

                records.append({
                    "chunk_id": chunk_id,
                    "document": path.name,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk,
                })

    if not records:
        raise ValueError("No meaningful text was found in this PDF.")

    return records


def embed_records(records):
    model = SentenceTransformer(
        str(MODEL_DIR),
        device="cpu",
        local_files_only=True,
    )

    batches = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        encoded = model.encode(
            [item["text"] for item in batch],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        batches.append(encoded.astype(np.float32))
        print(f"Embedded {min(start + BATCH_SIZE, len(records))}/{len(records)}")

    return np.concatenate(batches, axis=0)


def read_existing_index():
    db_path, vector_path = active_paths()

    if not db_path.exists() or not vector_path.exists():
        return [], np.empty((0, 384), dtype=np.float32)

    vectors = np.load(vector_path, allow_pickle=False)

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("""
            SELECT chunk_id, document, page_number, chunk_index,
                   vector_index, text
            FROM chunks
            ORDER BY vector_index
        """).fetchall()

    if len(rows) != len(vectors):
        raise RuntimeError("Existing index is inconsistent; refusing to modify it.")

    if vectors.ndim != 2 or vectors.shape[1] != 384:
        raise RuntimeError("Existing embedding dimensions are incompatible.")

    records = []
    for row in rows:
        records.append({
            "chunk_id": row[0],
            "document": row[1],
            "page": row[2],
            "chunk_index": row[3],
            "text": row[5],
        })

    return records, vectors


def commit_generation(records, vectors):
    if len(records) != len(vectors):
        raise RuntimeError("Cannot commit mismatched records and vectors.")

    GENERATION_DIR.mkdir(parents=True, exist_ok=True)
    generation_id = uuid.uuid4().hex
    staging = GENERATION_DIR / f".staging-{generation_id}"
    final = GENERATION_DIR / generation_id
    staging.mkdir()

    try:
        source_db, _ = active_paths()

        if source_db.exists():
            with sqlite3.connect(source_db) as source:
                with sqlite3.connect(staging / "salazar.db") as target:
                    source.backup(target)
        else:
            with sqlite3.connect(staging / "salazar.db") as connection:
                connection.execute("""
                    CREATE TABLE chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document TEXT NOT NULL,
                        page_number INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        vector_index INTEGER NOT NULL,
                        text TEXT NOT NULL
                    )
                """)

        with sqlite3.connect(staging / "salazar.db") as connection:
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

            count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count != len(vectors):
                raise RuntimeError("Staged database validation failed.")

        vector_file = staging / "embeddings.npy"
        with vector_file.open("wb") as handle:
            np.save(handle, vectors.astype(np.float32), allow_pickle=False)

        check = np.load(vector_file, allow_pickle=False)
        if check.shape != vectors.shape:
            raise RuntimeError("Staged vector validation failed.")

        staging.rename(final)

        manifest_tmp = INDEX_DIR / f".active-{generation_id}.tmp"
        manifest_tmp.write_text(json.dumps({
            "generation": generation_id,
            "chunks": len(records),
            "dimensions": 384,
        }))
        manifest_tmp.replace(MANIFEST_PATH)

        print(f"Active index generation: {generation_id}")
        print(f"Indexed chunks: {len(records)}")

    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def ingest(path):
    path = Path(path).resolve()

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF documents are supported.")

    if not path.is_file():
        raise FileNotFoundError(path)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Embedding model missing: {MODEL_DIR}")

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    destination = PDF_DIR / path.name
    transaction_id = uuid.uuid4().hex
    staging_dir = PDF_DIR / f".ingest-{transaction_id}"
    staged_pdf = staging_dir / path.name
    backup_pdf = staging_dir / "previous.pdf"

    staging_dir.mkdir(parents=True)

    try:
        shutil.copy2(path, staged_pdf)

        print(f"Extracting: {destination.name}")
        new_records = extract_pdf(staged_pdf)
        new_vectors = embed_records(new_records)

        old_records, old_vectors = read_existing_index()

        kept = [
            (record, old_vectors[index])
            for index, record in enumerate(old_records)
            if record["document"] != destination.name
        ]

        records = [item[0] for item in kept] + new_records
        vectors = (
            np.stack([item[1] for item in kept]).astype(np.float32)
            if kept else np.empty((0, 384), dtype=np.float32)
        )
        vectors = np.concatenate([vectors, new_vectors], axis=0)

        # Preserve the old PDF until the new index is committed.
        if destination.exists():
            destination.replace(backup_pdf)

        try:
            staged_pdf.replace(destination)
            commit_generation(records, vectors)
        except Exception:
            destination.unlink(missing_ok=True)
            if backup_pdf.exists():
                backup_pdf.replace(destination)
            raise

        print(f"Document indexed: {destination.name}")
        print(f"New chunks: {len(new_records)}")

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def delete_document(filename):
    if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
        raise ValueError("Invalid document filename.")

    old_records, old_vectors = read_existing_index()
    kept = [
        (record, old_vectors[index])
        for index, record in enumerate(old_records)
        if record["document"] != filename
    ]

    if len(kept) == len(old_records):
        raise ValueError("Document is not present in the index.")

    records = [item[0] for item in kept]
    vectors = (
        np.stack([item[1] for item in kept]).astype(np.float32)
        if kept else np.empty((0, 384), dtype=np.float32)
    )

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    document_path = PDF_DIR / filename
    backup_dir = PDF_DIR / f".delete-{uuid.uuid4().hex}"
    backup_dir.mkdir()
    backup_path = backup_dir / filename

    try:
        if document_path.exists():
            document_path.replace(backup_path)

        try:
            commit_generation(records, vectors)
        except Exception:
            if backup_path.exists():
                backup_path.replace(document_path)
            raise

        print(f"Document deleted: {filename}")

    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="SALAZAR document ingestion")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", metavar="PDF")
    group.add_argument("--delete", metavar="FILENAME")
    args = parser.parse_args()

    if args.ingest:
        ingest(args.ingest)
    else:
        delete_document(args.delete)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
