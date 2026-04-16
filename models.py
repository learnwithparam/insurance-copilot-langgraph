from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    """Incoming chat request to the insurance copilot."""
    message: str = Field(..., description="User message to the copilot")
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation thread id for multi-turn memory. If omitted a new one is created.",
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Optional customer id used by billing / claims agents to look up records.",
    )


class ChatResponse(BaseModel):
    """Non-streaming chat response shape."""
    thread_id: str
    route: str
    answer: str
    citations: List[str] = []


class UploadResponse(BaseModel):
    """Response after indexing policy PDFs into the RAG store."""
    status: str
    chunks_indexed: int
    filename: str


class CustomerRecord(BaseModel):
    """Customer + billing snapshot pulled from the SQL database."""
    customer_id: str
    name: str
    email: str
    plan: str
    premium_monthly: float
    balance_due: float
    claims: List[Dict[str, Any]] = []


class ResetResponse(BaseModel):
    status: str
    cleared: List[str]
