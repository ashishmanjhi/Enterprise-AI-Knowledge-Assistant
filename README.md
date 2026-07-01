# Enterprise Agentic RAG Platform

A production-ready Enterprise RAG (Retrieval-Augmented Generation) Platform with multi-provider LLM support, intelligent document processing, advanced hybrid retrieval strategies, conversational memory, enterprise-grade safety controls, and a full agentic LangGraph pipeline.

---

## Project Status

**Current Phase: Phase 13 Complete — Enhanced PDF Ingestion (Tables + OCR)**

| Phase | Name | Status |
|-------|------|--------|
| 0 | Core Foundation | ✅ Complete |
| 0.25 | Local AI Infrastructure | ✅ Complete |
| 0.5 | Provider Abstraction Layer | ✅ Complete |
| 1 | Basic RAG | ✅ Complete |
| 2 | Hybrid Retrieval | ✅ Complete |
| 3 | Query Understanding | ✅ Complete |
| 4 | Retrieval Optimization (Reranking) | ✅ Complete |
| 5 | Evaluation Framework (RAGAS) | ✅ Complete |
| 6 | Conversational Memory | ✅ Complete |
| 7 | Safety & Governance | ✅ Complete |
| 8 | User Experience | ✅ Complete |
| 9 | Agentic RAG (LangGraph) | ✅ Complete |
| 10 | Production Readiness | ✅ Complete |
| 11 | Multi-Agent Ecosystem | ✅ Complete |
| 12 | Knowledge Graph Enhancement | ✅ Complete |
| 13 | Enhanced PDF Ingestion (Tables + OCR) | ✅ Complete |

---

## Features

### Implemented (Phases 0–13)

#### Phase 0–0.5: Foundation
- **FastAPI Backend** with health checks and status monitoring
- **Streamlit Frontend** with intuitive multi-page UI
- **Provider Abstraction Layer** supporting multiple LLM providers
- **Ollama Integration** for local inference (qwen3:4b, gemma3:4b, phi4-mini)
- **Hugging Face Transformers** support
- **Cloud Provider Stubs** (OpenAI, Anthropic, Gemini, Azure) — activated in Phase 10
- **LLM Factory** with automatic fallback mechanism
- **Configuration Management** using Pydantic Settings
- **Structured Logging** with rotation
- **Podman Containers** for PostgreSQL and Redis
- **Health Monitoring** for all services

#### Phase 1: Basic RAG
- **Document Ingestion** — Upload and process PDF and DOCX files
- **Intelligent Chunking** — RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Semantic Embeddings** — BAAI/bge-small-en-v1.5 (384 dimensions)
- **Vector Search** — FAISS IndexFlatL2 for exact similarity search
- **RAG Chat** — Context-aware question answering with source attribution
- **Document Management** — Upload, list, search, delete operations

#### Phase 2: Hybrid Retrieval
- **BM25 Keyword Search** — Statistical keyword-based retrieval with TF-IDF scoring
- **Reciprocal Rank Fusion (RRF)** — Intelligent merging of semantic and keyword results
- **Hybrid Retriever** — Combines FAISS semantic search with BM25 keyword search
- **Parallel Execution** — Simultaneous FAISS and BM25 queries
- **Method Selection** — Choose between hybrid, semantic-only, or keyword-only retrieval
- **Dual Score Display** — Shows both semantic and keyword scores with rankings

#### Phase 3: Query Understanding
- **Query Reformulation** — LLM rewrites vague or ambiguous queries
- **Query Expansion** — Generates 3 alternative phrasings to increase recall
- **HyDE (Hypothetical Document Embeddings)** — Generates a hypothetical answer for better semantic search
- **Query Processor** — Orchestrates all three techniques with graceful degradation

#### Phase 4: Retrieval Optimization
- **Cross-Encoder Reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2` for second-pass scoring
- **Configurable Top-N** — Keep only the best N results after reranking
- **Lazy Model Loading** — Downloads ~22 MB on first use, cached thereafter
- **Per-Request Toggle** — Enable/disable reranking per chat request

#### Phase 5: Evaluation Framework (RAGAS)
- **Faithfulness** — Does the answer stick to the retrieved context?
- **Answer Relevancy** — Does the answer actually address the question?
- **Context Precision** — Were the right chunks retrieved?
- **Context Recall** — Was all needed information retrieved? (requires ground truth)
- **Factual Correctness** — Is the answer factually correct? (requires ground truth)
- **Evaluation UI** — Manual entry and JSON batch input via Streamlit page

#### Phase 6: Conversational Memory
- **Redis-Backed Session Store** — Per-conversation message history with 24 h TTL
- **Graceful Fallback** — In-process dict when Redis is unavailable
- **Auto-History Loading** — Chat route loads persisted history automatically
- **Conversation Summaries** — LLM compacts old history when it grows large
- **Memory API** — REST endpoints to fetch/delete conversation history

#### Phase 7: Safety & Governance
- **Prompt Injection Detection** — 14 regex patterns (jailbreaks, instruction overrides, delimiter injection)
- **PII Detection & Redaction** — SSN, credit cards, emails, phone numbers, API keys
- **Toxicity Detection** — Violence, self-harm, and harassment patterns
- **Hallucination Detection** — LLM-as-judge with heuristic token-overlap fallback
- **Guardrails Pipeline** — Pre-generation input checks + post-generation output checks
- **Configurable Blocking** — Per-check block-vs-warn settings via environment variables
- **Safety UI** — Guardrail status panel in Chat sidebar

#### Phase 8: User Experience
- **Streaming Responses** — Server-sent events (SSE) for token-by-token output in Chat UI
- **History Restore** — Conversation history automatically reloaded on page refresh
- **Safety Badges** — Guardrail warning indicators shown on each response
- **Query Metadata Panel** — Shows techniques applied (HyDE, expansion, reranking) per query
- **Filename Search Filter** — Filter documents by name in the Document Library tab

#### Phase 11: Multi-Agent Ecosystem
- **5 Specialised Sub-Agents** — Research, Retrieval, Knowledge, Evaluation, Governance — each a compiled LangGraph sub-graph
- **Multi-Agent Orchestrator** — classifies query intent, routes to the right agent pipeline, then chains all five agents in sequence
- **Research Agent** — decomposes complex questions into sub-questions, answers each independently, synthesises findings
- **Retrieval Agent** — expands queries, runs parallel multi-strategy retrieval, deduplicates and re-ranks
- **Knowledge Agent** — extracts named entities, runs targeted entity lookups, builds enriched knowledge context
- **Evaluation Agent** — heuristic + optional RAGAS quality scoring on every answer
- **Governance Agent** — final safety gate: guardrails + confidence + attribution checks
- **Multi-Agent UI** — `🌐 Multi-Agent` Streamlit page with per-agent cards, quality metrics, entity panel, governance badge

#### Phase 12: Knowledge Graph Enhancement
- **GraphStore** — NetworkX `MultiDiGraph` KG with entity nodes and relation edges, persisted as JSON (`data/knowledge_graph.json`)
- **Entity Extractor** — LLM-first NER with typed labels (PERSON, ORG, PRODUCT, LOCATION, CONCEPT, TECHNICAL) + regex fallback
- **Relation Mapper** — LLM extracts `subject | predicate | object` triples; heuristic regex fallback for `is / uses / has` patterns
- **Graph Retriever** — entity substring search + ego-graph neighbour expansion with hop-distance scoring
- **Hybrid Graph Retriever** — RRF fusion of FAISS+BM25 vector results and graph-aware results (configurable per-source weights)
- **Knowledge Agent Upgrade** — Phase 12 replaces the Phase 11 regex baseline with the full KG pipeline (EntityExtractor + RelationMapper + GraphRetriever)
- **KG REST API** — `POST /api/v1/kg/build`, `GET /kg/query`, `GET /kg/stats`, `GET /kg/entities`, `GET /kg/relations`, `DELETE /kg/clear`
- **Knowledge Graph UI** — `🕸️ Knowledge Graph` Streamlit page with Build, Query, Entities, and Relations tabs

#### Phase 13: Enhanced PDF Ingestion (Tables + OCR)
- **EnhancedPDFLoader** — pdfplumber-based PDF loader; extracts paragraph text AND structured tables per page
- **TableSerializer** — converts pdfplumber `list[list]` rows to Markdown (default) or CSV strings ready for chunking
- **OCRProcessor** — pytesseract wrapper for scanned / image-only pages; gracefully degrades when not installed
- **Table Chunks** — tables indexed as dedicated `[TABLE]` chunks with `chunk_type="table"` metadata for traceability
- **Graceful Fallback** — if pdfplumber not installed → PyPDF2; if pytesseract not installed → blank pages → empty string; never crashes
- **Pipeline Stats** — `GET /api/v1/documents/stats/overview` now returns `pdf_extraction` backend info
- **Documents UI Update** — Statistics tab shows extraction backend, table support, and OCR status; Upload Tips updated

#### Phase 10: Production Readiness
- **JWT Authentication** — `POST /auth/token` issues signed Bearer tokens; `JWTAuthMiddleware` enforces on all routes when `AUTH_ENABLED=true`
- **Cloud Provider Activation** — OpenAI, Anthropic, Gemini, and Azure OpenAI fully implemented (set the matching API key to activate)
- **LangSmith Tracing** — every `AgentGraph.run()` and its nodes are traced in LangSmith when `LANGSMITH_ENABLED=true`
- **OpenTelemetry Spans** — FastAPI HTTP spans + `llm.generate` / `agent.run` spans exported to any OTLP collector when `OTEL_ENABLED=true`
- **User Feedback Collection** — `POST /api/v1/feedback` persists thumbs-up/down ratings; 👍👎 buttons in Chat UI
- **Production Podman Compose** — Caddy TLS termination, internal-only DB network, resource limits, health checks

#### Phase 9: Agentic RAG (LangGraph)
- **LangGraph State Machine** — Full directed graph with conditional edges and loop guards
- **Automatic Routing** — LLM classifies each query and picks `hybrid`, `faiss`, or `bm25`
- **Query Rewriting** — Agent rewrites vague queries before retrieval (configurable max rewrites)
- **Document Grading** — LLM scores each retrieved chunk yes/no for relevance; irrelevant chunks are dropped before generation
- **Grounding Verification** — Post-generation check verifies the answer only uses information from retrieved context
- **Loop Recovery** — If grounding fails and rewrite budget remains, graph loops back for another attempt
- **Full Graph Trace** — Every node execution is logged in the response for observability
- **Agentic Chat UI** — Dedicated Streamlit page showing strategy chosen, rewrite count, and grounding verdict

---

## Prerequisites

### Required
- Python 3.9 or higher
- Ollama with at least one model (e.g. `qwen3:4b`)

### Optional
- Podman + podman-compose (for PostgreSQL and Redis)
- Redis (for persistent conversation memory)
- GPU (for faster embeddings and LLM inference)
- OpenAI / Anthropic / Gemini API keys (for cloud LLM — Phase 10)

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Enterprise-AI-Knowledge-Assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Install and Configure Ollama

```bash
# Download from https://ollama.ai
ollama serve
ollama pull qwen3:4b
```

### 6. Start the Application

**Terminal 1 — Backend:**
```bash
uvicorn backend.api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/streamlit/app.py
```

### 7. Access the Application

| Service | URL |
|---------|-----|
| Frontend UI | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |
| Agentic Chat | http://localhost:8501 → 🤖 Agent page |

---

## Project Structure

```
Enterprise-AI-Knowledge-Assistant/
├── backend/
│   ├── agents/                         # Phase 9: Agentic RAG (LangGraph)
│   │   ├── state.py                    # AgentState TypedDict
│   │   ├── nodes.py                    # Node functions (route, rewrite, retrieve, grade, generate, ground)
│   │   └── graph.py                    # Compiled LangGraph state machine
│   ├── api/
│   │   ├── main.py                     # FastAPI application entry point
│   │   ├── middleware/                 # Auth middleware (Phase 10)
│   │   ├── models/                     # Pydantic request/response schemas
│   │   └── routes/
│   │       ├── agent.py                # Agentic chat endpoint (Phase 9)
│   │       ├── chat.py                 # RAG chat (history + guardrails + streaming)
│   │       ├── documents.py            # Document upload/list/delete
│   │       ├── evaluate.py             # RAGAS evaluation (Phase 5)
│   │       ├── guardrails.py           # Safety check endpoints (Phase 7)
│   │       ├── memory.py               # Conversation history API (Phase 6)
│   │       ├── admin.py                # Admin utilities
│   │       ├── health.py               # Health checks
│   │       └── status.py               # Service status
│   ├── core/
│   │   ├── settings.py                 # Pydantic Settings (env-driven config)
│   │   ├── security.py                 # JWT helpers (Phase 10)
│   │   └── logging.py                  # Structured logging with rotation
│   ├── evaluators/                     # Phase 5: RAGAS evaluation engine
│   │   ├── metrics.py
│   │   └── ragas_evaluator.py
│   ├── guardrails/                     # Phase 7: Safety & governance
│   │   ├── detectors.py
│   │   ├── hallucination.py
│   │   └── pipeline.py
│   ├── ingestion/                      # Document loading and chunking
│   │   ├── loaders/
│   │   ├── chunking.py
│   │   ├── metadata.py
│   │   └── pipeline.py
│   ├── llm/
│   │   ├── embeddings.py               # BAAI/bge-small-en-v1.5
│   │   ├── llm_service.py              # Multi-provider LLM service
│   │   └── rag_chain.py                # Linear RAG orchestration
│   ├── memory/                         # Phase 6: Conversational memory
│   │   ├── session_memory.py
│   │   └── conversation_manager.py
│   ├── providers/                      # LLM provider implementations
│   │   ├── base.py
│   │   ├── ollama.py
│   │   ├── huggingface.py
│   │   ├── openai.py                   # Phase 10
│   │   ├── anthropic.py                # Phase 10
│   │   ├── gemini.py                   # Phase 10
│   │   ├── azure.py                    # Phase 10
│   │   └── factory.py
│   ├── query_understanding/            # Phase 3
│   │   ├── query_processor.py
│   │   ├── query_reformulator.py
│   │   ├── query_expander.py
│   │   └── hyde_generator.py
│   ├── rerankers/                      # Phase 4
│   │   └── cross_encoder.py
│   ├── retrievers/                     # Phases 1–2
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── fusion.py
│   │   └── hybrid_retriever.py
│   └── tests/
│       ├── test_chunking.py
│       ├── test_loaders.py
│       ├── test_query_understanding.py
│       ├── test_reranker.py
│       ├── test_memory.py
│       └── test_guardrails.py
├── frontend/
│   └── streamlit/
│       ├── app.py                      # Home / dashboard
│       └── pages/
│           ├── 1_📄_Documents.py       # Document management
│           ├── 2_💬_Chat.py            # RAG chat (streaming + memory + safety)
│           ├── 3_📊_Evaluate.py        # RAGAS evaluation UI
│           └── 4_🤖_Agent.py           # Agentic chat UI (Phase 9)
├── data/
│   ├── raw/                            # Uploaded documents
│   ├── processed/                      # Processed text
│   └── vectorstore/                    # FAISS and BM25 indices
├── docs/                               # Phase documentation
├── deploy/podman/                      # Container configuration
├── scripts/                            # Utility scripts
├── .env.example                        # Environment variable template
├── requirements.txt                    # Python dependencies
└── pytest.ini                         # Test configuration
```

---

## API Endpoints

### Document Operations
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload a PDF or DOCX file |
| `GET` | `/api/v1/documents` | List all documents |
| `GET` | `/api/v1/documents/{id}` | Get document details |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `GET` | `/api/v1/documents/stats/overview` | System statistics |

### Chat Operations
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/chat` | RAG chat (with memory + guardrails) |
| `POST` | `/api/v1/chat/stream` | Streaming chat (SSE) |
| `POST` | `/api/v1/chat/direct` | Direct LLM (no retrieval) |
| `GET` | `/api/v1/chat/health` | LLM service health |

### Agentic RAG (Phase 9)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/agent/chat` | Agentic chat via LangGraph pipeline |
| `GET` | `/api/v1/agent/health` | Agent subsystem health |

### Conversation Memory (Phase 6)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/memory/{id}` | Fetch conversation history |
| `GET` | `/api/v1/memory/{id}/info` | Conversation metadata |
| `DELETE` | `/api/v1/memory/{id}` | Delete conversation history |

### Evaluation (Phase 5)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/evaluate` | Run RAGAS evaluation |
| `GET` | `/api/v1/evaluate/metrics` | List available metrics |

### Safety & Guardrails (Phase 7)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/guardrails/check/input` | Run input safety checks |
| `POST` | `/api/v1/guardrails/check/output` | Run output safety checks |
| `GET` | `/api/v1/guardrails/status` | Enabled guardrail checks |

### Admin & System
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/admin/clear-vector-stores` | Clear all indices |
| `GET` | `/api/v1/admin/system-info` | System information |
| `GET` | `/health` | Basic health check |
| `GET` | `/healthz` | Kubernetes-style health |
| `GET` | `/readyz` | Readiness check |
| `GET` | `/api/v1/status` | Detailed service status |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

### Key Settings by Phase

#### Core (All Phases)
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_MODEL=qwen3:4b
DEFAULT_PROVIDER=ollama
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

#### Phase 2 — Hybrid Retrieval
```env
DEFAULT_RETRIEVAL_METHOD=hybrid   # hybrid | faiss | bm25
FAISS_WEIGHT=0.5
BM25_WEIGHT=0.5
RRF_K=60
```

#### Phase 3 — Query Understanding
```env
ENABLE_QUERY_REFORMULATION=true
ENABLE_QUERY_EXPANSION=true
ENABLE_HYDE=true
NUM_QUERY_EXPANSIONS=3
```

#### Phase 4 — Reranking
```env
ENABLE_RERANKING=false            # downloads model on first use
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=3
```

#### Phase 5 — Evaluation
```env
EVAL_JUDGE_MODEL=                 # blank = use OLLAMA_DEFAULT_MODEL
EVAL_MAX_SAMPLES=50
```

#### Phase 6 — Conversational Memory
```env
MEMORY_SESSION_TTL=86400          # 24 hours
MEMORY_MAX_HISTORY_MESSAGES=20
MEMORY_ENABLE_SUMMARISATION=true
MEMORY_SUMMARY_THRESHOLD=10
```

#### Phase 7 — Safety & Guardrails
```env
GUARDRAILS_ENABLE_INJECTION=true
GUARDRAILS_ENABLE_TOXICITY=true
GUARDRAILS_ENABLE_PII=true
GUARDRAILS_ENABLE_HALLUCINATION=true
GUARDRAILS_BLOCK_ON_INJECTION=true
GUARDRAILS_BLOCK_ON_TOXICITY=true
GUARDRAILS_BLOCK_ON_PII_INPUT=false
GUARDRAILS_BLOCK_ON_HALLUCINATION=false
```

#### Phase 9 — Agentic RAG
```env
AGENT_ENABLE_DOCUMENT_GRADING=true   # LLM grades each chunk for relevance
AGENT_ENABLE_GROUNDING_CHECK=true    # Post-generation grounding verification
AGENT_MAX_REWRITES=2                 # Max query rewrite iterations
```

#### Phase 10 — Production (Cloud Providers & Auth)
```env
# Cloud LLM providers (set to activate)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Authentication
AUTH_ENABLED=false                   # set true to require JWT on all routes
JWT_SECRET_KEY=change-me-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
```

---

## Testing

```bash
# Run all unit tests
pytest backend/tests/ -v

# Run specific phase tests
pytest backend/tests/test_memory.py -v       # Phase 6
pytest backend/tests/test_guardrails.py -v   # Phase 7

# Run with coverage
pytest backend/tests/ --cov=backend --cov-report=term-missing
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Chunking | 16 | Document chunking logic |
| Loaders | 12 | PDF and DOCX ingestion |
| Query Understanding | 22 | Reformulation, expansion, HyDE |
| Reranker | 12 | Cross-encoder reranking |
| Memory (Phase 6) | 30 | Session store + conversation manager |
| Guardrails (Phase 7) | 41 | Detectors, hallucination, pipeline |
| **Total** | **133+** | |

---

## Troubleshooting

### Backend Won't Start
```bash
# Check if port 8000 is in use (Windows)
netstat -an | findstr 8000

# Verify environment
python -c "from backend.api.main import app; print('OK')"
```

### Ollama Connection Failed
```bash
ollama list
ollama serve
ollama pull qwen3:4b
curl http://localhost:11434/api/tags
```

### Redis Unavailable
Redis is optional. When unavailable, conversational memory falls back to an in-process dictionary (data is lost on restart). Start Redis with:
```bash
podman-compose -f deploy/podman/podman-compose.yml up -d redis
```

### Chat Blocked by Guardrails (HTTP 400)
A `request_blocked_by_guardrails` error means the input triggered injection or toxicity detection. Check `block_reason` in the response body. To temporarily disable:
```env
GUARDRAILS_BLOCK_ON_INJECTION=false
GUARDRAILS_BLOCK_ON_TOXICITY=false
```

### Document Upload Fails
- Supported formats: PDF, DOCX only
- Max file size: 10 MB (configurable via `MAX_FILE_SIZE`)
- Check backend logs: `logs/app.log`

### Slow First Response
- First LLM request: Ollama loads the model (~10–30 s on CPU)
- First embedding: Model downloads on first run (~50 MB)
- Reranker first use: Downloads ~22 MB cross-encoder model
- Agent mode: Each query runs 3–6 LLM calls (routing, grading, generation, grounding)

### Agent Grounding Check Always Fails
Set `AGENT_ENABLE_GROUNDING_CHECK=false` to disable (e.g. when Ollama is slow and you want faster responses), or raise `AGENT_MAX_REWRITES=0` to skip the recovery loop.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/PHASE_9_IMPLEMENTATION.md](docs/PHASE_9_IMPLEMENTATION.md) | Phase 9 Agentic RAG architecture and implementation |
| [docs/PHASES_3_TO_7_SUMMARY.md](docs/PHASES_3_TO_7_SUMMARY.md) | Implementation summary for Phases 3–7 |
| [docs/PHASE_8_IMPLEMENTATION_PLAN.md](docs/PHASE_8_IMPLEMENTATION_PLAN.md) | Phase 8 UX plan |
| [docs/PHASE_2_IMPLEMENTATION_PLAN.md](docs/PHASE_2_IMPLEMENTATION_PLAN.md) | Phase 2 architecture |
| [docs/PHASE_1_DOCUMENTATION.md](docs/PHASE_1_DOCUMENTATION.md) | Phase 1 full documentation |
| [docs/phase-0-architecture.md](docs/phase-0-architecture.md) | System architecture overview |
| [docs/project-roadmap.md](docs/project-roadmap.md) | Full 12-phase roadmap |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute setup guide |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Overall implementation summary |

---

## Roadmap

- [x] **Phase 0**: Core Foundation
- [x] **Phase 0.25**: Local AI Infrastructure
- [x] **Phase 0.5**: Provider Abstraction Layer
- [x] **Phase 1**: Basic RAG
- [x] **Phase 2**: Hybrid Retrieval
- [x] **Phase 3**: Query Understanding
- [x] **Phase 4**: Retrieval Optimization
- [x] **Phase 5**: Evaluation Framework (RAGAS)
- [x] **Phase 6**: Conversational Memory
- [x] **Phase 7**: Safety & Governance
- [x] **Phase 8**: User Experience
- [x] **Phase 9**: Agentic RAG (LangGraph)
- [x] **Phase 10**: Production Readiness (JWT auth, cloud providers, LangSmith, OTel, feedback)
- [x] **Phase 11**: Multi-Agent Ecosystem (5 specialised agents + orchestrator)
- [ ] **Phase 12**: Knowledge Graph Enhancement

---

## Acknowledgments

- FastAPI for the excellent web framework
- Streamlit for rapid UI development
- Ollama for local LLM inference
- Hugging Face for transformer models
- LangChain / LangGraph for agentic RAG components
- FAISS for efficient vector search
- rank-bm25 for BM25 implementation
- RAGAS for RAG evaluation metrics

---

**Version**: 11.0.0
**Status**: Phase 11 Complete ✅
**Last Updated**: 2026-07-01
