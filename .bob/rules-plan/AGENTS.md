# Project Architecture Rules (Plan Mode)

## Architectural constraints

### Postgres is optional — filesystem is the source of truth for vectors
FAISS index + BM25 JSON are the live retrieval stores. Postgres is a metadata/audit layer. Every route that touches the DB wraps DB calls in `try/except` and continues via filesystem fallback. New features must follow this pattern.

### Two async models coexist
- FastAPI routes: `async def` + `await` (asyncpg)
- LangGraph nodes: `async def` (same event loop)
- Alembic / scripts: sync (psycopg2)  
Never mix asyncpg sessions into sync code or vice versa.

### LLM calls are always async and always go through `LLMService`
`LLMService` wraps `LLMFactory` which manages provider fallback. Direct provider instantiation bypasses fallback logic — don't do it in route handlers or nodes.

### Retrieval is tenant-scoped via `tenant_registry.py`
`get_pipeline_for_tenant` / `get_retriever_for_tenant` maintain in-process caches keyed by tenant slug. With `MULTI_TENANCY_ENABLED=false` these return the global singletons — zero overhead. New ingestion or retrieval code must go through these functions, not directly access `get_shared_vector_store`.

### LangGraph state merges partial dicts
Nodes return only changed keys. `AgentState.trace` is `Annotated[List[str], operator.add]` — intentionally append-only so parallel nodes don't clobber each other's trace entries.

### Streaming responses use SSE
The chat route yields `data: <json>\n\n` chunks via `StreamingResponse`. Streamlit pages consume them with `requests.get(..., stream=True)`. The final chunk from `generate_response_stream` is always `{"type": "sources", "sources": [...]}`.

### Knowledge graph is in-process NetworkX
`data/knowledge_graph.json` is loaded once per process. There is no graph database. KG queries are entirely in-memory graph traversals — O(nodes) — so keep `kg_max_subgraph_nodes` low for production.

### Rate limiting is per-endpoint, not per-user
`slowapi` limits are IP-based. There is no per-user quota mechanism. Auth and rate limiting are independent layers.

### No background worker / task queue
Long ingestion runs on a `BackgroundTasks` callback (FastAPI built-in). There is no Celery/RQ. For very large document batches this can block the next ingestion.
