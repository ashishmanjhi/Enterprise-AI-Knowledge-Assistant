# Project Coding Rules (Agent Mode)

## Non-negotiable conventions

- **Every new Python file ends with `# Made with Bob`** (last line)
- **`from __future__ import annotations`** is the first import in every module
- **Never `print()`** — always `from backend.core.logging import get_logger; logger = get_logger(__name__)`
- **Never import `settings` class** — always import the singleton: `from backend.core.settings import settings`

## Singletons — use shared instances, not new ones

Route handlers must use shared/cached singletons, not construct fresh instances:

```python
from backend.retrievers.vector_store_manager import get_shared_vector_store
from backend.retrievers.bm25_manager import get_shared_bm25_retriever
# NOT: FAISSVectorStore()  or  BM25Retriever()
```

DB session is also lazy-imported inside helpers to avoid startup crash when Postgres is down.

## Adding a new route

1. `router = APIRouter(prefix="/api/v1/<feature>", tags=["<feature>"])`
2. Import and `app.include_router(router)` in `backend/api/main.py`
3. Add entry to the `"endpoints"` dict in the root `GET /` handler

## New LLM provider

Subclass `BaseLLMProvider` (implements `generate`, `stream`, `health_check`, `get_model_info`), register the class string key in `LLMFactory._providers`, add default config branch in `LLMFactory._get_default_config`.

## Agent / LangGraph nodes

Node function signature: `async def node_name(state: AgentState) -> AgentState`
Append observability entries to `state["trace"]` (list, operator.add merge).
Compile graph with `graph.compile()` — do not call `.run()` on the raw StateGraph.

## Settings for new features

Group under `# Phase N: <name>` comment. Use inline comments for non-obvious defaults. Bump `app_version` to `N.0.0` in both `settings.py` and `.env`.

## Test patterns

- Session-scoped `client` fixture from `conftest.py` — reuse, don't create new `TestClient`
- Use `monkeypatch.setattr(settings, "field", value)` for per-test overrides
- Mock Ollama: `with patch("backend.llm.llm_service.LLMService.generate", new_callable=AsyncMock)`

## Import validation (before every commit)

```powershell
.\venv\Scripts\python.exe -c "from backend.<new_module> import <Class>"
```
