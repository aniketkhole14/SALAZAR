"""Local REST API for SALAZAR."""

from contextlib import asynccontextmanager
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.bootstrap import build_dispatcher
from app.conversation_store import ConversationStore


ROOT = Path(os.environ.get("SALAZAR_ROOT", Path(__file__).resolve().parents[1])).resolve()
PDF_DIR = ROOT / "data" / "documents"
UPLOAD_DIR = ROOT / "data" / "uploads"
INDEX_DIR = ROOT / "data" / "index"
MANIFEST_PATH = INDEX_DIR / "active.json"
LEGACY_DB = ROOT / "data" / "database" / "salazar.db"

RAG_EXECUTABLE = os.environ.get("SALAZAR_RAG_EXECUTABLE")
RAG_SCRIPT = ROOT / "scripts" / "ask_salazar.py"

MAX_UPLOAD_SIZE = 50 * 1024 * 1024

dispatcher = None
conversation_store = ConversationStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dispatcher
    dispatcher = build_dispatcher()
    yield
    dispatcher = None


app = FastAPI(
    title="SALAZAR API",
    description="Local-first API for the SALAZAR offline AI workstation.",
    version="0.3.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    success: bool
    conversation_id: str
    agent: str
    task_type: str
    reason: str
    response: str
    sources: list[str]


def run_ingestion_operation(*arguments: str) -> str:
    """Run document ingestion using the packaged RAG runtime or dev script."""

    if RAG_EXECUTABLE:
        executable = Path(RAG_EXECUTABLE).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(
                f"Packaged RAG executable not found: {executable}"
            )
        command = [str(executable), *arguments]
    else:
        if getattr(sys, "frozen", False):
            raise RuntimeError(
                "Packaged backend requires SALAZAR_RAG_EXECUTABLE."
            )
        if not RAG_SCRIPT.is_file():
            raise FileNotFoundError(f"RAG script not found: {RAG_SCRIPT}")
        command = [sys.executable, str(RAG_SCRIPT), *arguments]

    environment = os.environ.copy()
    environment["SALAZAR_ROOT"] = str(ROOT)

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
        check=False,
    )

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(detail[-4000:] or "Document operation failed.")

    return result.stdout.strip()


def get_active_database() -> Path:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        generation = manifest.get("generation", "")
        if not generation or Path(generation).name != generation:
            raise RuntimeError("Invalid active index manifest.")

        database = INDEX_DIR / "generations" / generation / "salazar.db"
        if not database.is_file():
            raise RuntimeError("Active index database is missing.")
        return database

    return LEGACY_DB


def get_document_chunk_counts() -> dict[str, int]:
    database = get_active_database()

    if not database.is_file():
        return {}

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT document, COUNT(*) FROM chunks GROUP BY document"
        ).fetchall()

    return {name: count for name, count in rows}


def validate_filename(filename: str) -> str:
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
        or filename.startswith(".")
        or not filename.lower().endswith(".pdf")
    ):
        raise HTTPException(status_code=400, detail="Invalid PDF filename.")

    return filename


@app.get("/health")
async def health():
    return {
        "status": "online",
        "application": "SALAZAR",
        "backend": "FastAPI",
    }


@app.get("/api/conversations")
async def list_conversations():
    return {"conversations": conversation_store.list()}


@app.post("/api/conversations")
async def create_conversation():
    conversation_id = conversation_store.create()
    return {"conversation_id": conversation_id}


@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    messages = conversation_store.get_messages(conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation_id": conversation_id, "messages": messages}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if not conversation_store.delete(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"success": True}


@app.get("/api/documents")
async def list_documents():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    try:
        chunk_counts = await run_in_threadpool(get_document_chunk_counts)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read document index: {exc}",
        ) from exc

    documents = []
    for path in sorted(PDF_DIR.glob("*.pdf"), key=lambda item: item.name.lower()):
        if path.name.startswith("."):
            continue

        documents.append({
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "chunks": chunk_counts.get(path.name, 0),
            "indexed": chunk_counts.get(path.name, 0) > 0,
        })

    return {"documents": documents}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    filename = validate_filename(file.filename or "")

    if file.content_type not in {
        "application/pdf",
        "application/octet-stream",
        "application/x-pdf",
    }:
        raise HTTPException(
            status_code=415,
            detail="Only PDF uploads are supported.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = UPLOAD_DIR / uuid.uuid4().hex
    staging_dir.mkdir(parents=True)
    staged_path = staging_dir / filename
    total_size = 0

    try:
        with staged_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 50 MB upload limit.",
                    )

                destination.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

        with staged_path.open("rb") as uploaded:
            if uploaded.read(5) != b"%PDF-":
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded file is not a valid PDF.",
                )

        output = await run_in_threadpool(
            run_ingestion_operation,
            "--ingest",
            str(staged_path),
        )

        try:
            chunks = get_document_chunk_counts().get(filename, 0)
        except Exception:
            chunks = 0

        return {
            "success": True,
            "filename": filename,
            "size_bytes": total_size,
            "chunks": chunks,
            "message": "PDF successfully indexed.",
            "details": output,
        }

    except HTTPException:
        raise
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="PDF ingestion timed out.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF ingestion failed: {exc}",
        ) from exc
    finally:
        import shutil
        shutil.rmtree(staging_dir, ignore_errors=True)
        await file.close()


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    filename = validate_filename(filename)

    try:
        output = await run_in_threadpool(
            run_ingestion_operation,
            "--delete",
            filename,
        )
        return {
            "success": True,
            "filename": filename,
            "message": "Document and its indexed chunks were deleted.",
            "details": output,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="Document deletion timed out.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {exc}",
        ) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    if dispatcher is None:
        raise HTTPException(status_code=503, detail="SALAZAR dispatcher is not ready.")

    conversation_id = request.conversation_id
    if conversation_id:
        if conversation_store.get_messages(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation_id = conversation_store.create()

    conversation_store.add_message(conversation_id, "user", message)

    try:
        decision, result = await run_in_threadpool(
            dispatcher.dispatch,
            message,
        )

        sources = list(result.sources)
        conversation_store.add_message(
            conversation_id,
            "assistant",
            result.content,
            agent=result.agent_name,
            task_type=result.task_type,
            sources=sources,
        )

        return ChatResponse(
            success=result.success,
            conversation_id=conversation_id,
            agent=result.agent_name,
            task_type=result.task_type,
            reason=decision.reason,
            response=result.content,
            sources=sources,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SALAZAR request failed: {exc}",
        ) from exc
