# Agentic Workflows with LangGraph: Insurance Copilot

![learnwithparam.com](https://www.learnwithparam.com/ai-bootcamp/opengraph-image)

Build a supervisor-routed multi-agent insurance assistant with LangGraph. Policy, billing, claims, and escalation agents collaborate over a RAG knowledge base and a SQL customer database, streaming answers back through FastAPI.

> Start learning at [learnwithparam.com](https://learnwithparam.com). Regional pricing available with discounts of up to 60%.

## What You'll Learn

- Supervisor agent routing in LangGraph and how to keep the routing cheap
- Shared state design across policy, billing, claims, and escalation subagents
- Mixing RAG retrieval with SQL tool calls inside a single graph
- Streaming multi-agent output to the browser over FastAPI SSE
- Safe human-escalation patterns when the copilot should step aside

## Tech Stack

- **FastAPI** - High-performance async Python web framework
- **LangGraph** - Stateful multi-agent orchestration with the supervisor pattern
- **LangChain** - Tool + splitter primitives reused by the agents
- **ChromaDB** - Embedded vector store for policy RAG
- **SQLite** - Customer and claims database used by billing and claims agents
- **Pydantic** - Strict request and response schemas
- **LLM Provider Pattern** - Swap Fireworks, OpenRouter, Gemini, or OpenAI from `.env`
- **Docker** - Containerized development

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (installed automatically by `make setup`)
- An API key from any supported LLM provider

### Quick Start

```bash
# One command to set up and run
make dev

# Or step by step:
make setup          # Create .env and install dependencies
# Edit .env with your API key
make run            # Start the FastAPI server
```

### With Docker

```bash
make build          # Build the Docker image
make up             # Start the container
make logs           # View logs
make down           # Stop the container
```

### API Documentation

Once running, open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI. The core endpoint is `POST /insurance-copilot/chat/stream`, which streams supervisor decisions and sub-agent answers over Server-Sent Events.

## Challenges

Work through these incrementally to build the full copilot:

1. **The Policy Knowledge Base** - Load policy PDFs into ChromaDB with chunked, layout-aware parsing so the policy agent can cite real sources.
2. **The Customer Ledger** - Build a SQLite seed of five customers and ten claims, plus query helpers the billing and claims agents call as tools.
3. **Shared Graph State** - Define a `ClaimState` TypedDict that every node reads from and writes to, keeping routing, retrieval, and answers cleanly separated.
4. **The Four Subagents** - Implement `policy_agent` (RAG), `billing_agent` (SQL), `claims_agent` (both RAG and SQL), and `escalation_agent` (human handoff).
5. **The Supervisor Router** - Add a supervisor node that classifies each message into one of the four routes, with a keyword fast-path for obvious escalations.
6. **The Streaming Endpoint** - Wire `/insurance-copilot/chat/stream` so the supervisor decision and sub-agent output flow to the client as SSE events.
7. **Multi-Turn Memory** - Use LangGraph's checkpointer plus a `thread_id` so the copilot remembers prior turns and can handle follow-ups naturally.

## Makefile Targets

```
make help           Show all available commands
make setup          Initial setup (create .env, install deps)
make dev            Setup and run (one command!)
make run            Start FastAPI server
make build          Build Docker image
make up             Start container
make down           Stop container
make logs           View container logs
make restart        Restart the container
make clean          Remove venv and cache
make clean-all      Clean everything including Docker volumes
```

## Learn more

- Start the course: [learnwithparam.com/courses/supervisor-multi-agent-langgraph](https://www.learnwithparam.com/courses/supervisor-multi-agent-langgraph)
- AI Bootcamp for Software Engineers: [learnwithparam.com/ai-bootcamp](https://www.learnwithparam.com/ai-bootcamp)
- All courses: [learnwithparam.com/courses](https://www.learnwithparam.com/courses)
