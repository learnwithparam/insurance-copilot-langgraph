"""ChromaDB-backed RAG helpers for policy documents.

The policy_agent queries `search_policies(query)` to ground its answers in
indexed PDF chunks. Indexing happens via `index_policy_pdf(path)` which is
called by the `/insurance-copilot/upload-policy` endpoint.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "insurance_policies")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


_client: Optional[chromadb.ClientAPI] = None
_embed_model = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def _get_collection():
    return _get_client().get_or_create_collection(name=COLLECTION_NAME)


def _get_embedder():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _embed(texts: List[str]) -> List[List[float]]:
    return _get_embedder().encode(texts, convert_to_tensor=False).tolist()


def _load_pdf_markdown(path: str) -> str:
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(path)
    except Exception as exc:  # pragma: no cover - exercised at runtime
        logger.warning("pymupdf4llm failed (%s). Falling back to plain read.", exc)
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", errors="ignore")


def _split_text(text: str) -> List[str]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def index_policy_pdf(path: str, source_name: Optional[str] = None) -> int:
    """Index a policy PDF into Chroma. Returns the number of chunks written."""
    text = _load_pdf_markdown(path)
    chunks = _split_text(text)
    if not chunks:
        return 0

    source = source_name or os.path.basename(path)
    ids = [f"{source}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
    metas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    embeddings = _embed(chunks)

    collection = _get_collection()
    collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
    logger.info("Indexed %d chunks from %s", len(chunks), source)
    return len(chunks)


def search_policies(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Semantic search over indexed policy chunks."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    embedded = _embed([query])
    results = collection.query(query_embeddings=embedded, n_results=k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)
    out: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, distances):
        out.append({"content": doc, "source": (meta or {}).get("source", "policy"), "distance": dist})
    return out


def reset_index() -> None:
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # collection may not exist yet
        pass
