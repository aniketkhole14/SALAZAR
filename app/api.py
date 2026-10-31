"""Local REST API for SALAZAR."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.bootstrap import build_dispatcher


dispatcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dispatcher
    dispatcher = build_dispatcher()
    yield
    dispatcher = None


app = FastAPI(
    title="SALAZAR API",
    description="Local-first API for the SALAZAR offline AI workstation.",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)


class ChatResponse(BaseModel):
    success: bool
    agent: str
    task_type: str
    reason: str
    response: str
    sources: list[str]


@app.get("/health")
async def health():
    return {
        "status": "online",
        "application": "SALAZAR",
        "backend": "FastAPI",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    if dispatcher is None:
        raise HTTPException(status_code=503, detail="SALAZAR dispatcher is not ready.")

    try:
        decision, result = await run_in_threadpool(
            dispatcher.dispatch,
            message,
        )

        return ChatResponse(
            success=result.success,
            agent=result.agent_name,
            task_type=result.task_type,
            reason=decision.reason,
            response=result.content,
            sources=list(result.sources),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SALAZAR request failed: {exc}",
        ) from exc
