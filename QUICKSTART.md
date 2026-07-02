# Quick Start Guide

Get the Enterprise Agentic RAG Platform running in 5 minutes!

## Prerequisites Check

Before starting, ensure you have:

- ✅ Python 3.12+ installed
- ✅ Ollama installed and running
- ✅ 16 GB RAM minimum (32 GB recommended for llava:7b)
- ✅ Windows 10/11, Linux, or macOS

## Step-by-Step Setup

### 1. Install Dependencies (2 minutes)

```powershell
# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Linux/Mac
# source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment (30 seconds)

```powershell
# Copy environment template
copy .env.example .env

# Default values work for local development
# Edit .env only if you need custom settings
```

### 3. Start Services (1 minute)

```powershell
# Start PostgreSQL and Redis (optional but recommended)
cd deploy\podman
podman-compose up -d

# Check status
podman-compose ps
```

> **Note:** Redis is optional. Without it, conversation memory falls back to in-process storage (no persistence across restarts). See `TROUBLESHOOTING.md` for alternatives.

### 4. Verify Ollama Models (30 seconds)

```powershell
# Check which models are installed
ollama list

# Pull required models if missing:
ollama pull qwen3:4b       # primary LLM (required)
ollama pull llava:7b       # vision model for chart descriptions (Phase 14)

# Optional models:
ollama pull gemma3:4b      # secondary LLM
ollama pull phi4-mini      # lightweight testing
```

### 5. Start Backend (30 seconds)

```powershell
# From project root
uvicorn backend.api.main:app --reload
```

Open http://localhost:8000/docs to see the full API documentation.

### 6. Start Frontend (30 seconds)

```powershell
# In a new terminal, from project root
streamlit run frontend/streamlit/app.py
```

Open http://localhost:8501 to access the UI.

---

## Verify Everything Works

### Check Backend Health

```powershell
curl http://localhost:8000/health
# Returns: {"status":"healthy","service":"rag-platform"}
```

### Check Service Status

```powershell
curl http://localhost:8000/api/v1/status
# Shows all services as "connected"
```

### Check Agent Health

```powershell
curl http://localhost:8000/api/v1/agent/health
# Returns: {"status":"healthy","graph":"compiled",...}
```

### Check Knowledge Graph

```powershell
curl http://localhost:8000/api/v1/kg/stats
# Returns: {"nodes":0,"edges":0,...}
```

### Check Frontend

1. Open http://localhost:8501
2. Sidebar should show "✅ Backend Connected"
3. Six pages available in the sidebar:
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
4. Wait for processing (~10–30 seconds depending on document size)
5. Check **Statistics** tab — the **PDF Extraction Engine** card shows:
   - Backend: `pdfplumber ✅`
   - Table extraction: `✅ Enabled`
   - Chart / image AI: `✅ llava:7b`

> **Phase 13 + 14 note**: Tables are automatically extracted as `[TABLE]` chunks. Charts and diagrams are described by llava:7b as `[CHART]` chunks. Both are fully searchable.

### Step 2: Chat with RAG

1. Go to **💬 Chat** page
2. Enter a session ID (or leave default)
3. Type a question about your uploaded document
4. Enable **Hybrid retrieval**, **Query expansion**, and **HyDE** for best results

### Step 3: Try Agentic Mode

1. Go to **🤖 Agent** page
2. Ask a complex, multi-step question
3. Watch the LangGraph trace: `route → retrieve → grade → generate → ground`

### Step 4: Explore Multi-Agent

1. Go to **🌐 Multi Agent** page
2. Ask a research-style question (e.g. "Summarise the key findings and evaluate quality")
3. See 5 agents working: Research → Retrieval → Knowledge → Evaluation → Governance

### Step 5: Build Knowledge Graph

1. Go to **🕸️ Knowledge Graph** page
2. Click **Build KG** with a query to extract entities and relations
3. Browse entities and relations in the Explorer tabs

---

## Common Issues

### Port Already in Use

```powershell
netstat -ano | findstr "8000 8501 5432 6379"
taskkill /PID <PID> /F
```

### Ollama Not Responding

```powershell
# Check Ollama is running
ollama list

# Restart Ollama service if needed
# Windows: restart from system tray or:
# taskkill /IM "ollama.exe" /F && ollama serve
```

### llava:7b Not Installed (Chart descriptions disabled)

```powershell
# Install llava for chart/image understanding
ollama pull llava:7b

# Or disable chart descriptions entirely in .env:
# PDF_CHART_DESCRIPTION_ENABLED=false
```

### Import Errors

```powershell
# Make sure virtual environment is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Agent / Multi-Agent Responses Are Slow

The agent runs 3–6 LLM calls per query on CPU-only Ollama. Speed up with:

```env
# Add to .env:
AGENT_ENABLE_DOCUMENT_GRADING=false
AGENT_ENABLE_GROUNDING_CHECK=false
AGENT_MAX_REWRITES=0
```

### Chart Descriptions Slow Ingestion

llava:7b takes ~5–15 s per image on CPU. For bulk ingestion:

```env
# Disable during batch upload, re-enable afterwards:
PDF_CHART_DESCRIPTION_ENABLED=false
```

### pdfplumber Table Extraction Issues

```env
# Fall back to PyPDF2 (no table support):
PDF_USE_ENHANCED_LOADER=false
```

---

## Next Steps

1. ✅ **Upload documents** via 📄 Documents — tables and charts auto-extracted
2. ✅ **Chat with RAG** via 💬 Chat — streaming, memory, safety guardrails
3. ✅ **Agentic mode** via 🤖 Agent — LangGraph adaptive retrieval
4. ✅ **Multi-agent** via 🌐 Multi Agent — 5-agent ensemble
5. ✅ **Knowledge Graph** via 🕸️ Knowledge Graph — entity/relation explorer
6. ✅ **Evaluate quality** via 📊 Evaluate — RAGAS metrics
7. ✅ Explore full API at http://localhost:8000/docs

## Stopping Services

```powershell
# Stop backend: Ctrl+C in terminal
# Stop frontend: Ctrl+C in terminal

# Stop Podman services
cd deploy\podman
podman-compose down
```

## Getting Help

- 📚 [Full Documentation](README.md)
- 🔧 [Troubleshooting Guide](TROUBLESHOOTING.md)
- 📖 [Phase 14 Chart AI Guide](docs/PHASE_14_IMPLEMENTATION.md)
- 📖 [Phase 13 Table Extraction Guide](docs/PHASE_13_IMPLEMENTATION.md)
- 📖 [Phase 12 Knowledge Graph Guide](docs/PHASE_12_IMPLEMENTATION.md)
- 🤖 [Phase 9 Agentic RAG Guide](docs/PHASE_9_IMPLEMENTATION.md)

---

**Estimated Setup Time**: 5–10 minutes
**Version**: 14.0.0
**Status**: Phases 0–14 Complete ✅

# Made with Bob
