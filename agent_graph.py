"""LangGraph StateGraph for the insurance copilot.

Shape of the graph:

    START -> supervisor -> {policy_agent, billing_agent, claims_agent, escalation_agent} -> END

The supervisor is a cheap LLM call that classifies the user's latest message
into one of four routes. The sub-agents each write their answer back into the
shared `ClaimState`, and the graph terminates after a single hop. Conversation
memory is provided by a `MemorySaver` checkpointer keyed on `thread_id`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from utils.llm_provider import get_llm_provider
from rag_utils import search_policies
from db_utils import get_customer, list_claims, open_claim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class ClaimState(TypedDict, total=False):
    """State shared across every node in the graph.

    The supervisor populates `route`; each sub-agent appends to `messages`
    and fills `answer` / `citations`. Downstream turns re-use `messages` so
    the graph has short-term conversation memory.
    """
    messages: List[Dict[str, str]]       # [{role, content}, ...]
    user_input: str                      # latest user message
    customer_id: Optional[str]           # optional binding for SQL agents
    route: str                           # one of: policy | billing | claims | escalation
    retrieved_docs: List[Dict[str, Any]] # policy RAG hits
    customer_record: Optional[Dict[str, Any]]
    claims_record: List[Dict[str, Any]]
    answer: str
    citations: List[str]


# ---------------------------------------------------------------------------
# Router helpers
# ---------------------------------------------------------------------------
_VALID_ROUTES = {"policy", "billing", "claims", "escalation"}

_KEYWORD_HINTS = {
    "policy": ["coverage", "covered", "deductible", "policy", "terms", "exclusion", "benefit"],
    "billing": ["bill", "premium", "invoice", "payment", "charge", "balance", "due", "autopay"],
    "claims": ["claim", "accident", "damage", "incident", "file a", "reimburse", "payout"],
    "escalation": ["human", "agent", "manager", "supervisor", "complaint", "angry", "lawyer", "lawsuit"],
}


def _keyword_route(text: str) -> Optional[str]:
    lowered = text.lower()
    scores = {route: sum(1 for kw in kws if kw in lowered) for route, kws in _KEYWORD_HINTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


SUPERVISOR_PROMPT = """You are the supervisor agent for an insurance copilot.
Classify the user's latest message into exactly one of these routes:

- policy: questions about coverage, terms, deductibles, exclusions (use the RAG KB)
- billing: questions about premiums, invoices, balances, payments (use the SQL DB)
- claims: filing, checking, or disputing a claim (may use both KB and SQL)
- escalation: user is angry, asks for a human, legal threats, or anything unsafe

Respond with a single JSON object: {{"route": "policy|billing|claims|escalation"}}.

Conversation so far:
{history}

Latest user message:
{user_input}
"""


POLICY_PROMPT = """You are the policy agent. Answer the user's question using ONLY the
retrieved policy excerpts below. If the excerpts do not answer the question, say so
and recommend escalation.

User question:
{user_input}

Retrieved policy excerpts:
{context}

Write a concise, friendly answer. Cite sources inline like [source: filename].
"""


BILLING_PROMPT = """You are the billing agent. Use the customer record below to answer.
If no customer_id was provided, ask the user to share it.

User question:
{user_input}

Customer record (JSON):
{record}

Answer clearly. Show premium, balance due, and next steps where relevant.
"""


CLAIMS_PROMPT = """You are the claims agent. Use BOTH the policy excerpts and the
customer's claim history to answer. You can acknowledge a new claim intent but do
not promise approval; new claims enter the system in 'review' status.

User question:
{user_input}

Customer claims (JSON):
{claims}

Relevant policy excerpts:
{context}

Answer professionally. If this is a new claim filing, summarize what was opened.
"""


ESCALATION_PROMPT = """You are the escalation agent. The user needs a human. Write a
short, empathetic message that:
1. Acknowledges their frustration or request.
2. Confirms a human specialist will follow up within one business day.
3. Provides a reference handoff id (make one up in the format ESC-XXXXXX).

User message:
{user_input}
"""


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------
def _format_history(messages: List[Dict[str, str]]) -> str:
    if not messages:
        return "(no prior turns)"
    lines = []
    for m in messages[-6:]:  # cap context
        role = m.get("role", "user")
        lines.append(f"{role}: {m.get('content','')}")
    return "\n".join(lines)


async def supervisor_node(state: ClaimState) -> Dict[str, Any]:
    user_input = state.get("user_input", "")
    history = _format_history(state.get("messages", []))

    # Cheap keyword gate — catches obvious escalations before spending an LLM call.
    kw = _keyword_route(user_input)
    if kw == "escalation":
        return {"route": "escalation"}

    provider = get_llm_provider()
    prompt = SUPERVISOR_PROMPT.format(history=history, user_input=user_input)
    try:
        raw = await provider.generate_text(prompt, max_tokens=60, temperature=0)
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            route = parsed.get("route", "").strip().lower()
            if route in _VALID_ROUTES:
                return {"route": route}
    except Exception as exc:
        logger.warning("Supervisor LLM failed (%s); falling back to keyword routing.", exc)

    return {"route": kw or "policy"}


async def policy_agent_node(state: ClaimState) -> Dict[str, Any]:
    query = state.get("user_input", "")
    hits = search_policies(query, k=4)
    if hits:
        context = "\n\n".join(
            f"[source: {h['source']}]\n{h['content']}" for h in hits
        )
    else:
        context = "(no policy documents indexed yet — ask the user to upload one)"

    provider = get_llm_provider()
    answer = await provider.generate_text(
        POLICY_PROMPT.format(user_input=query, context=context),
        max_tokens=500,
        temperature=0.2,
    )
    citations = sorted({h["source"] for h in hits})
    return {"retrieved_docs": hits, "answer": answer, "citations": citations}


async def billing_agent_node(state: ClaimState) -> Dict[str, Any]:
    customer_id = state.get("customer_id")
    record = get_customer(customer_id) if customer_id else None

    provider = get_llm_provider()
    answer = await provider.generate_text(
        BILLING_PROMPT.format(
            user_input=state.get("user_input", ""),
            record=json.dumps(record, default=str) if record else "(no customer_id provided)",
        ),
        max_tokens=400,
        temperature=0.2,
    )
    return {"customer_record": record, "answer": answer, "citations": []}


async def claims_agent_node(state: ClaimState) -> Dict[str, Any]:
    customer_id = state.get("customer_id")
    user_input = state.get("user_input", "")
    claims = list_claims(customer_id) if customer_id else []

    # If the user is clearly filing a new claim, open one in 'review' status.
    filing_intent = any(
        phrase in user_input.lower()
        for phrase in ("file a claim", "open a claim", "new claim", "submit a claim")
    )
    if filing_intent and customer_id:
        amount_match = re.search(r"\$?([0-9]+(?:\.[0-9]+)?)", user_input)
        amount = float(amount_match.group(1)) if amount_match else 0.0
        opened = open_claim(customer_id, claim_type="general", amount=amount)
        claims = [opened] + claims

    hits = search_policies(user_input, k=3)
    context = "\n\n".join(f"[source: {h['source']}] {h['content']}" for h in hits) or "(no policy matches)"

    provider = get_llm_provider()
    answer = await provider.generate_text(
        CLAIMS_PROMPT.format(
            user_input=user_input,
            claims=json.dumps(claims, default=str),
            context=context,
        ),
        max_tokens=500,
        temperature=0.2,
    )
    return {
        "claims_record": claims,
        "retrieved_docs": hits,
        "answer": answer,
        "citations": sorted({h["source"] for h in hits}),
    }


async def escalation_agent_node(state: ClaimState) -> Dict[str, Any]:
    provider = get_llm_provider()
    answer = await provider.generate_text(
        ESCALATION_PROMPT.format(user_input=state.get("user_input", "")),
        max_tokens=200,
        temperature=0.3,
    )
    return {"answer": answer, "citations": []}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def _route_selector(state: ClaimState) -> str:
    return state.get("route", "policy")


def build_graph():
    """Compile the LangGraph StateGraph with an in-memory checkpointer."""
    graph = StateGraph(ClaimState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("policy_agent", policy_agent_node)
    graph.add_node("billing_agent", billing_agent_node)
    graph.add_node("claims_agent", claims_agent_node)
    graph.add_node("escalation_agent", escalation_agent_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_selector,
        {
            "policy": "policy_agent",
            "billing": "billing_agent",
            "claims": "claims_agent",
            "escalation": "escalation_agent",
        },
    )
    for leaf in ("policy_agent", "billing_agent", "claims_agent", "escalation_agent"):
        graph.add_edge(leaf, END)

    return graph.compile(checkpointer=MemorySaver())


# Lazy singleton so `uv sync` + import doesn't require API keys.
_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
