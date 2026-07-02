---
name: rag-platform-conventions
description: Use when working on the Enterprise Agentic RAG Platform project — automatically loads stack conventions, file patterns, phase rules, and coding standards so Bob follows the exact project style without being re-explained each session.
---

# Enterprise Agentic RAG Platform — Project Conventions

Activate this skill at the start of any coding session on this project. It encodes all patterns that must be followed exactly to stay consistent with the existing codebase.

---

## 1. Project Identity

- **Name**: Enterprise Agentic RAG Platform
- **Version**: tracked in `backend/core/settings.py` → `app_version` field AND `.env` → `APP_VERSION`
- **Branch**: `main` (all work committed directly)
- **Python interpreter**: `.\venv\Scripts\python.exe`
- **Platform**: Windows 10, PowerShell, CPU-only (32 GB RAM)

---

## 2. Stack at a Glance

| Layer | Technology |
|---|---|
| API | FastAPI 0.109, Uvicorn |
| Agent | LangGraph 0.1.19 (StateGraph) |
| Frontend | Streamlit 1.30 |
| LLM (local) | Ollama — qwen3:4b, gemma3:4b, phi4-mini, llava:7b |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim, sentence-transformers) |
| Vector store | FAISS IndexFlatL2 |
| Keyword search | rank-bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Graph | NetworkX MultiDiGraph (JSON persistence) |
| PDF extraction | pdfplumber (text + tables), llava:7b (charts), pytesseract (optional OCR) |
| Memory | Redis (in-process fallback) |
| Auth | PyJWT HS256 |
| Tracing | LangSmith + OpenTelemetry |
| Evaluation | RAGAS |
| Deploy | Podman + Caddy |

---

## 3. Settings Pattern

All settings live in `backend/core/settings.py` as typed `BaseSettings` fields:

```python
# Phase N: <short description>
field_name: type = default  # inline comment explaining the field
```

- Use `Optional[str]` for nullable strings, not bare `str = ""`
- Group all new fields under a `# Phase N: <name>` comment block
- Each new phase bumps `app_version` in `settings.py` AND `.env`
- Settings load from `.env` automatically via pydantic-settings

---

## 4. Logging Pattern

```python
from backend.core.logging import get_logger
logger = get_logger(__name__)
```

Never use `print()`. Always use `logger.info / warning / error / debug`.

---

## 5. Tracing Pattern

```python
from backend.core.tracing import trace_span, langsmith_callbacks
async with trace_span("operation.name"):
    ...
```

---

## 6. LLM Service Pattern

```python
from backend.llm.llm_service import LLMService
llm = LLMService()
result = await llm.generate(prompt, system_prompt=...)
# result = {"text": str, "model": str, "tokens_used": int}
```

---

## 7. Retriever Pattern

```python
from backend.retrievers.hybrid_retriever import HybridRetriever
retriever = HybridRetriever()
results = await retriever.retrieve(query, top_k=5, method="hybrid")
# results: list[HybridRetrievalResult]
# each has: .chunk_id, .content, .score, .filename, .page_number
```

---

## 8. Agent Pattern (LangGraph)

- Each agent is a compiled `StateGraph` stored in a class with `async run(state_dict) -> state_dict`
- Dependencies injected via `functools.partial` into node functions
- State is a `TypedDict`
- Agents live in `backend/agents/` (single-agent) or `backend/agents/multi/` (multi-agent)
- Node functions are pure async functions with signature `async def node_name(state: StateType) -> StateType`

---

## 9. FastAPI Router Pattern

Every new feature gets its own route file:

```
backend/api/routes/<feature>.py
```

Router must be:
1. Created with `router = APIRouter(prefix="/api/v1/<feature>", tags=["<feature>"])`
2. Imported in `backend/api/main.py`
3. Registered with `app.include_router(router, tags=[...])`
4. Added to the root endpoint dict in `app.get("/")`

---

## 10. Phase Rules

Every new phase MUST include all of:
- [ ] New backend code in the correct `backend/` subdirectory
- [ ] New settings block in `backend/core/settings.py` with `# Phase N:` comment
- [ ] New/updated FastAPI routes wired into `backend/api/main.py`
- [ ] New or updated Streamlit page in `frontend/streamlit/pages/`
- [ ] `docs/PHASE_N_IMPLEMENTATION.md` documentation file
- [ ] `README.md` updated — phase table row added, feature section added, version bumped
- [ ] Import validation: `.\venv\Scripts\python.exe -c "from backend.<new_module> import <Class>"`
- [ ] Git commit: `git commit -m "feat: Phase N - <descriptive title>"`
- [ ] `app_version` bumped to `N.0.0` in `settings.py`

---

## 11. Streamlit Page Naming

Pages follow the pattern: `<N>_<emoji>_<Name>.py`

Current pages:
```
1_📄_Documents.py
2_💬_Chat.py
3_📊_Evaluate.py
4_🤖_Agent.py
5_🌐_Multi_Agent.py
6_🕸️_Knowledge_Graph.py
```

New pages continue the numbering. API base URL is always `http://localhost:8000`.

---

## 12. Ingestion / Chunk Types

Every chunk in FAISS + BM25 carries `chunk_type` metadata:

| `chunk_type` | Prefix | Source |
|---|---|---|
| `"text"` | none | pdfplumber paragraph text |
| `"table"` | `[TABLE]` | pdfplumber tables → Markdown |
| `"chart"` | `[CHART]` | llava:7b image description |

---

## 13. Knowledge Graph

- Store: `NetworkX.MultiDiGraph`, persisted as JSON at `data/knowledge_graph.json`
- Entity types: `PERSON, ORG, PRODUCT, LOCATION, CONCEPT, TECHNICAL`
- Singleton: `from backend.knowledge_graph import get_graph_store`
- All KG routes under `/api/v1/kg/`

---

## 14. Git Conventions

- All work on `main` branch
- Commit messages: `feat: Phase N - <description>` for phases, `fix: <description>` for bug fixes
- Validate imports before every commit
- Footer on all new Python files: `# Made with Bob`

---

## 15. Completed Phases Reference

| Phase | Name | Key modules |
|---|---|---|
| 0–0.5 | Foundation | `backend/core/`, `backend/providers/`, `backend/llm/` |
| 1 | Basic RAG | `backend/ingestion/`, `backend/retrievers/`, `backend/api/routes/documents.py` |
| 2 | Hybrid Retrieval | `backend/retrievers/hybrid_retriever.py`, `bm25_retriever.py` |
| 3 | Query Understanding | `backend/query_understanding/` |
| 4 | Reranking | `backend/rerankers/cross_encoder.py` |
| 5 | Evaluation | `backend/evaluators/` |
| 6 | Memory | `backend/memory/` |
| 7 | Safety | `backend/guardrails/` |
| 8 | UX | SSE streaming in `backend/api/routes/chat.py` |
| 9 | Agentic RAG | `backend/agents/graph.py`, `nodes.py`, `state.py` |
| 10 | Production | `backend/api/middleware/auth.py`, `backend/core/tracing.py`, `backend/providers/` |
| 11 | Multi-Agent | `backend/agents/multi/orchestrator.py` + 5 sub-agents |
| 12 | Knowledge Graph | `backend/knowledge_graph/` |
| 13 | PDF Tables | `backend/ingestion/loaders/pdf_loader_v2.py`, `table_serializer.py` |
| 14 | Chart AI | `backend/ingestion/chart_describer.py` (llava:7b via Ollama) |

---

## 16. Before Starting Any Task

1. Read any files that are relevant — never speculate about code not yet opened
2. Check `git status` to see current state
3. Use `update_todo_list` to track multi-step work
4. Run import validation after any new module is created
5. Always commit at the end of a completed phase

# Made with Bob
