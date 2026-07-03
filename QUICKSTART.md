# Quick Start Guide

Get the Enterprise Agentic RAG Platform (v15.0.0) running in 5 minutes.

## Prerequisites Check

Before starting, ensure you have:

- ✅ Python 3.11+ installed
- ✅ Ollama installed and running
- ✅ 16 GB RAM minimum (32 GB recommended for llava:7b + multi-agent)
- ✅ Windows 10/11, Linux, or macOS

---

## Step-by-Step Setup

### 1. Install Dependencies

```powershell
# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Linux/Mac
# source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

```powershell
# Copy environment template
copy .env.example .env

# Minimum edits for local development:
#   OLLAMA_HOST=http://localhost:11434
#   JWT_SECRET_KEY=<any-random-string>
# Everything else has safe defaults.
```

### 3. Start Services (Optional but Recommended)

```powershell
# Start PostgreSQL and Redis via Podman
cd deploy\podman
podman-compose up -d
cd ..\..

# Apply database schema
.\venv\Scripts\python.exe -m alembic upgrade head
```

> **Note:** Redis and Postgres are optional. Without them, conversation memory falls back to in-process storage and document metadata is served from the filesystem.

### 4. Verify Ollama Models

```powershell
# Check which models are installed
ollama list

# Pull required models if missing:
ollama pull qwen3:4b       # primary LLM (required)
ollama pull llava:7b       # vision model for chart descriptions (Phase 14, optional)

# Optional models:
ollama pull gemma3:4b      # secondary LLM for A/B testing
ollama pull phi4-mini      # lightweight/fast testing
```

### 5. Start Backend

```powershell
# From project root
uvicorn backend.api.main:app --reload
```

Open http://localhost:8000/docs to see the full API documentation.

### 6. Start Frontend

```powershell
# In a new terminal, from project root
streamlit run frontend/streamlit/app.py
```

Open http://localhost:8501 to access the UI.

---

## Verify Everything Works

### Import Check

```powershell
.\venv\Scripts\python.exe -c "from backend.api.main import app; print('OK')"
```

Expected: `INFO - Rate limiting enabled` + `OK`

### Backend Health

```powershell
curl http://localhost:8000/health
# Returns: {"status":"healthy","service":"rag-platform"}
```

### Service Status

```powershell
curl http://localhost:8000/api/v1/status
```

### Agent Health

```powershell
curl http://localhost:8000/api/v1/agent/health
# Returns: {"status":"healthy","graph":"compiled",...}
```

### Knowledge Graph

```powershell
curl http://localhost:8000/api/v1/kg/stats
# Returns: {"nodes":0,"edges":0,...}
```

### Frontend

1. Open http://localhost:8501
2. Sidebar should show "✅ Backend Connected"
3. Six pages available:
   - 📄 Documents
   - 💬 Chat
   - 📊 Evaluate
   - 🤖 Agent
   - 🌐 Multi Agent
   - 🕸️ Knowledge Graph

---

## First Use Walkthrough

### Step 1: Upload a Document

1. Go to **📄 Documents** page
2. Click **Browse files** and select a PDF or DOCX file
3. Click **🚀 Upload**
4. Wait for processing (~10–30 seconds)
5. Check **Statistics** tab — the **PDF Extraction Engine** card shows:
   - Backend: `pdfplumber ✅`
   - Table extraction: `✅ Enabled`
   - Chart / image AI: `✅ llava:7b` (if installed)

> **Phase 13 + 14:** Tables auto-extracted as `[TABLE]` chunks. Charts/diagrams described by llava:7b as `[CHART]` chunks. Both fully searchable.

### Step 2: Chat with RAG

1. Go to **💬 Chat** page
2. Enter a session ID (or leave default)
3. Type a question about your uploaded document
4. Enable **Hybrid retrieval**, **Query expansion**, and **HyDE** for best results
5. Streaming response shows tokens as they arrive

### Step 3: Try Agentic Mode

1. Go to **🤖 Agent** page
2. Ask a complex, multi-step question
3. Watch the LangGraph trace: `route → retrieve → grade → generate → ground`

### Step 4: Explore Multi-Agent

1. Go to **🌐 Multi Agent** page
2. Ask a research-style question (e.g. "Summarise the key findings and evaluate quality")
3. See 5 agents: Research → Retrieval → Knowledge → Evaluation → Governance

### Step 5: Build Knowledge Graph

1. Go to **🕸️ Knowledge Graph** page
2. Click **Build KG** to extract entities and relations from indexed documents
3. Browse entities and relations in the Explorer tabs

---

## Rate Limits (Phase 15)

All LLM-calling endpoints are rate-limited per IP by default:

| Endpoint                    | Default Limit |
| --------------------------- | ------------- |
| `POST /api/v1/chat`         | 20 / minute   |
| `POST /api/v1/chat/stream`  | 10 / minute   |
| `POST /api/v1/chat/direct`  | 30 / minute   |
| `POST /api/v1/agent/chat`   | 10 / minute   |
| `POST /api/v1/multi-agent/chat` | 5 / minute |

To disable rate limiting during development:
```env
RATE_LIMIT_ENABLED=false
```

---

## Common Issues

### Port Already in Use

```powershell
netstat -ano | findstr "8000 8501 5432 6379"
taskkill /PID <PID> /F
```

### Rate Limit Hit (HTTP 429)

```env
# Add to .env to disable:
RATE_LIMIT_ENABLED=false
```

### Ollama Not Responding

```powershell
ollama list
# Windows: restart from system tray or:
taskkill /IM "ollama.exe" /F
ollama serve
```

### Chat Blocked by Guardrails (HTTP 400)

Check `block_reason` in the response. To disable blocking:

```env
GUARDRAILS_BLOCK_ON_INJECTION=false
GUARDRAILS_BLOCK_ON_TOXICITY=false
```

### llava:7b Not Installed (Chart descriptions disabled)

```powershell
ollama pull llava:7b
# Or disable in .env:
# PDF_CHART_DESCRIPTION_ENABLED=false
```

### Agent / Multi-Agent Responses Are Slow

Each agent query runs 3–6 LLM calls on CPU-only Ollama:

```env
AGENT_ENABLE_DOCUMENT_GRADING=false
AGENT_ENABLE_GROUNDING_CHECK=false
AGENT_MAX_REWRITES=0
```

### Import Errors

```powershell
venv\Scripts\activate
pip install -r requirements.txt --force-reinstall
```

---

## Running Tests

```powershell
# Full test suite (203 tests)
.\venv\Scripts\pytest backend/tests/ -v

# Quick smoke tests only
.\venv\Scripts\pytest backend/tests/test_api_integration.py -v

# Route-level tests (covers rate limiting, auth, documents, chat, admin)
.\venv\Scripts\pytest backend/tests/test_api_routes.py -v
```

---

## Stopping Services

```powershell
# Stop backend: Ctrl+C in terminal
# Stop frontend: Ctrl+C in terminal

# Stop Podman services
cd deploy\podman
podman-compose down
```

---

## Documentation Reference

| Document | Description |
|---|---|
| [README.md](README.md) | Full feature reference and configuration guide |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Detailed troubleshooting |
| [WINDOWS_SETUP.md](WINDOWS_SETUP.md) | Windows-specific setup |
| [docs/PHASE_15_IMPLEMENTATION.md](docs/PHASE_15_IMPLEMENTATION.md) | Phase 15 — rate limiting, async embeddings, compaction |
| [docs/PHASE_14_IMPLEMENTATION.md](docs/PHASE_14_IMPLEMENTATION.md) | Chart AI guide |
| [docs/PHASE_13_IMPLEMENTATION.md](docs/PHASE_13_IMPLEMENTATION.md) | Table extraction guide |
| [docs/PHASE_9_IMPLEMENTATION.md](docs/PHASE_9_IMPLEMENTATION.md) | Agentic RAG guide |

---

**Estimated Setup Time**: 5–10 minutes  
**Version**: 15.0.0  
**Status**: Phases 0–15 Complete ✅

# Made with Bob
