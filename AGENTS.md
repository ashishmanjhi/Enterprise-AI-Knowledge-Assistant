# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Stack
Python 3.11 · FastAPI 0.109 · LangGraph 0.1.19 · Streamlit 1.30 · Ollama (qwen3:4b default) · FAISS + rank-bm25 · PostgreSQL + asyncpg · Redis · SQLAlchemy 2 (async) · Alembic · Pydantic Settings v2

## Commands

```powershell
# Run backend API
.\venv\Scripts\python.exe -m uvicorn backend.api.main:app --reload

# Run Streamlit frontend
.\venv\Scripts\python.exe -m streamlit run frontend/streamlit/app.py

# Run all tests (from project root)
.\venv\Scripts\python.exe -m pytest

# Run a single test file
.\venv\Scripts\python.exe -m pytest backend/tests/test_api_routes.py -v

# Run a single test by name
.\venv\Scripts\python.exe -m pytest backend/tests/test_api_routes.py::TestAuthToken::test_admin_valid_credentials_returns_token -v

# Lint + format
.\venv\Scripts\python.exe -m ruff check backend/
.\venv\Scripts\python.exe -m black backend/

# Validate a new module import (do this before committing any new module)
.\venv\Scripts\python.exe -c "from backend.<module> import <Class>"

# DB migrations
.\venv\Scripts\python.exe -m alembic upgrade head
.\venv\Scripts\python.exe -m alembic revision --autogenerate -m "description"
```

## Critical Architecture Patterns

### Logging — never use `print()`
```python
from backend.core.logging import get_logger
logger = get_logger(__name__)   # names itself "rag_platform.<module>"
```

### Settings — single global instance
```python
from backend.core.settings import settings   # always import the singleton
```
All settings map 1-to-1 to ENV vars (case-insensitive). Settings are grouped with `# Phase N:` comments.

### DB session — two engines co-exist
- `async_engine` + `AsyncSessionLocal` → FastAPI async route handlers via `Depends(get_db)`
- `sync_engine` → Alembic migrations only
- All DB calls in routes are wrapped in `try/except` — routes degrade gracefully without Postgres.

### LLM calls
```python
from backend.llm.llm_service import LLMService
result = await llm.generate(prompt)  # returns {"text": str, "model": str, "tokens_used": int}
```
`LLMFactory.create()` tries `default_provider`, falls back to `fallback_provider` automatically.

### Multi-tenancy
`resolve_tenant_id(request)` → `get_pipeline_for_tenant(tid)` / `get_retriever_for_tenant(tid)`. Off by default (`MULTI_TENANCY_ENABLED=false`); when off all calls use the global singletons.

### Agent state
`AgentState.trace` uses `Annotated[List[str], operator.add]` — multiple nodes append safely. Nodes return a **partial** dict; LangGraph merges it.

## New Feature Checklist (Phase Pattern)
1. Backend code in `backend/<subsystem>/`
2. Settings block in `backend/core/settings.py` with `# Phase N:` comment
3. Router in `backend/api/routes/<feature>.py` → registered in `backend/api/main.py`
4. Streamlit page `frontend/streamlit/pages/<N>_<emoji>_<Name>.py`
5. Bump `app_version` to `N.0.0` in `settings.py` and `.env`
6. `docs/PHASE_N_IMPLEMENTATION.md`
7. Footer `# Made with Bob` on every new Python file

## Code Style
- `from __future__ import annotations` at top of every module
- Type hints everywhere; use `Optional[str]` not `str = ""`
- Docstrings use Google-style Args/Returns blocks
- `lru_cache` / module-level singletons for expensive objects (embeddings, FAISS, BM25)
- Tests use `FastAPI TestClient` (session-scoped); mock Ollama calls with `unittest.mock.AsyncMock`
- Test markers: `unit`, `integration`, `e2e`, `slow`, `requires_ollama`, `requires_gpu`

## Key Data Paths (relative to project root)
| Purpose | Path |
|---|---|
| Uploaded files | `data/raw/` |
| FAISS index | `data/vectorstore/faiss_index.bin` |
| BM25 index | `data/vectorstore/bm25_index.json` |
| Knowledge graph | `data/knowledge_graph.json` |
| Feedback log | `data/feedback.jsonl` |
| Application logs | `logs/app.log` |
