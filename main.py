from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router
from db_utils import bootstrap_database

app = FastAPI(
    title="Agentic Workflows with LangGraph: Insurance Copilot",
    description=(
        "Build a supervisor-routed multi-agent insurance assistant with LangGraph. "
        "Policy, billing, claims, and escalation agents collaborate over a RAG "
        "knowledge base and a SQL customer database, streaming answers back "
        "through FastAPI."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def _startup() -> None:
    bootstrap_database()


@app.get("/")
async def root():
    return {"service": "insurance-copilot-langgraph", "docs": "/docs"}
