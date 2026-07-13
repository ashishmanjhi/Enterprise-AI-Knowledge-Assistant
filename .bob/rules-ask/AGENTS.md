# Project Documentation Context (Ask Mode)

## Codebase layout — non-obvious facts

- `backend/retrievers/` contains **both** the FAISS vector store (`vector_store.py`) and BM25 (`bm25_retriever.py`); these are not in `backend/llm/` or `backend/ingestion/`
- `backend/retrievers/tenant_registry.py` is the only place that wires multi-tenancy to vector stores and pipelines — not in middleware
- `backend/ingestion/loaders/` has two PDF loaders: `pdf_loader.py` (PyPDF2) and `pdf_loader_v2.py` (pdfplumber + table extraction + OCR); the v2 loader is selected at runtime via `settings.pdf_use_enhanced_loader`
- The FAISS index and BM25 index are **file-based** (flat `.bin` / `.json`), not a database; Postgres only stores document metadata and chunk records
- `backend/db/session.py` exposes both `async_engine` (asyncpg, for routes) and `sync_engine` (psycopg2, for Alembic only) — they use different URL schemes

## Auth is opt-in and transparent

`JWTAuthMiddleware` is always registered but is a no-op when `AUTH_ENABLED=false`. `_EXEMPT_PATHS` (defined in `backend/api/middleware/auth.py`) always bypass auth even when enabled.

## Phases map to code

The codebase is organized in numbered phases (0–15 as of v15.0.0). Each phase has a `docs/PHASE_N_IMPLEMENTATION.md`. Phase N settings live in a comment block `# Phase N:` inside `backend/core/settings.py`.

## Streamlit pages call the backend over HTTP

All Streamlit pages use `requests` (sync) or `httpx` against `http://localhost:8000`. They import from `backend.core.settings` directly only to read constants like `api_port` — no shared state.

## Rate limiting

Implemented via `slowapi` decorators on individual route functions in each router file, not centrally in middleware. Limits are configurable per-endpoint via `settings.rate_limit_*` fields.

## Feedback storage

Feedback is appended to a flat JSON-Lines file (`data/feedback.jsonl`), not stored in Postgres.
