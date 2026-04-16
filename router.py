"""FastAPI router for the insurance copilot."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from db_utils import get_customer
from models import (
    ChatRequest,
    ChatResponse,
    CustomerRecord,
    ResetResponse,
    UploadResponse,
)
from service import ingest_policy, reset_all, run_chat, stream_chat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insurance-copilot", tags=["insurance-copilot"])


@router.get("/health")
async def health():
    return {"status": "healthy", "service": "insurance-copilot-langgraph"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Single-shot chat. Returns the final routed answer."""
    try:
        result = await run_chat(
            message=request.message,
            thread_id=request.thread_id,
            customer_id=request.customer_id,
        )
        return ChatResponse(**result)
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Server-Sent Events stream of the supervisor + sub-agent trace."""

    async def event_source():
        async for chunk in stream_chat(
            message=request.message,
            thread_id=request.thread_id,
            customer_id=request.customer_id,
        ):
            yield chunk

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/upload-policy", response_model=UploadResponse)
async def upload_policy(file: UploadFile = File(...), source_name: str | None = Form(None)):
    """Upload a policy PDF; it gets chunked and indexed into ChromaDB."""
    allowed = {".pdf", ".txt", ".md"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(allowed)}")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename or "policy.pdf")
    try:
        with open(temp_path, "wb") as fh:
            fh.write(await file.read())
        chunks = ingest_policy(temp_path, source_name=source_name or file.filename)
        return UploadResponse(status="indexed", chunks_indexed=chunks, filename=file.filename or "policy.pdf")
    except Exception as exc:
        logger.exception("upload-policy failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/customers/{customer_id}", response_model=CustomerRecord)
async def get_customer_record(customer_id: str):
    record = get_customer(customer_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return CustomerRecord(**record)


@router.post("/reset", response_model=ResetResponse)
async def reset():
    """Drop the ChromaDB index, re-seed SQLite, and clear graph memory."""
    result = reset_all()
    return ResetResponse(status=result["status"], cleared=result["cleared"])
