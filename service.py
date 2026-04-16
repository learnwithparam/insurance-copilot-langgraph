"""Service layer between FastAPI routes and the LangGraph workflow."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from agent_graph import get_graph
from rag_utils import index_policy_pdf, reset_index
from db_utils import reset_database

logger = logging.getLogger(__name__)


def _new_thread_id() -> str:
    return f"thread-{uuid.uuid4().hex[:10]}"


def _thread_config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


async def run_chat(
    message: str,
    thread_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the graph once and return the final state as a dict."""
    thread_id = thread_id or _new_thread_id()
    graph = get_graph()

    # Preserve history across turns by re-reading the checkpoint's messages.
    snapshot = graph.get_state(_thread_config(thread_id))
    history = (snapshot.values or {}).get("messages", []) if snapshot else []
    history = history + [{"role": "user", "content": message}]

    initial: Dict[str, Any] = {
        "messages": history,
        "user_input": message,
        "customer_id": customer_id,
    }

    final_state: Dict[str, Any] = {}
    async for event in graph.astream(initial, config=_thread_config(thread_id)):
        for _node, node_state in event.items():
            final_state.update(node_state or {})

    answer = final_state.get("answer", "")
    route = final_state.get("route", "policy")
    citations = final_state.get("citations", []) or []

    # Persist assistant turn into messages for the next call.
    history.append({"role": "assistant", "content": answer})
    graph.update_state(_thread_config(thread_id), {"messages": history})

    return {
        "thread_id": thread_id,
        "route": route,
        "answer": answer,
        "citations": citations,
    }


async def stream_chat(
    message: str,
    thread_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted events as the graph progresses."""
    thread_id = thread_id or _new_thread_id()
    graph = get_graph()

    snapshot = graph.get_state(_thread_config(thread_id))
    history = (snapshot.values or {}).get("messages", []) if snapshot else []
    history = history + [{"role": "user", "content": message}]

    initial: Dict[str, Any] = {
        "messages": history,
        "user_input": message,
        "customer_id": customer_id,
    }

    yield _sse({"event": "start", "thread_id": thread_id})

    final_state: Dict[str, Any] = {}
    try:
        async for event in graph.astream(initial, config=_thread_config(thread_id)):
            for node_name, node_state in event.items():
                final_state.update(node_state or {})
                yield _sse({
                    "event": "node",
                    "node": node_name,
                    "route": final_state.get("route"),
                    "partial_answer": (node_state or {}).get("answer"),
                })
    except Exception as exc:
        logger.exception("Graph stream failed")
        yield _sse({"event": "error", "error": str(exc)})
        return

    history.append({"role": "assistant", "content": final_state.get("answer", "")})
    graph.update_state(_thread_config(thread_id), {"messages": history})

    yield _sse({
        "event": "done",
        "thread_id": thread_id,
        "route": final_state.get("route"),
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []) or [],
    })


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def ingest_policy(path: str, source_name: Optional[str] = None) -> int:
    return index_policy_pdf(path, source_name=source_name)


def reset_all() -> Dict[str, Any]:
    reset_database()
    reset_index()
    return {"status": "ok", "cleared": ["sqlite", "chromadb", "graph_memory"]}
