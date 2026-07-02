# Troubleshooting Guide

## Podman Machine Issues on Windows

### Issue: "machine did not transition into running state"

This is a common issue with Podman on Windows. Here are solutions:

### Solution 1: Reset Podman Machine

```powershell
# Stop and remove existing machine
podman machine stop
podman machine rm podman-machine-default

# Create new machine with more resources
podman machine init --cpus 4 --memory 8192 --disk-size 50

# Start the machine
podman machine start
```

### Solution 2: Use Podman Desktop

1. Open **Podman Desktop** application
2. Go to **Settings** → **Resources**
3. Click **Start** on the Podman machine
4. Wait for it to fully start (may take 2-3 minutes)

### Solution 3: Use Docker Desktop Instead

If Podman continues to have issues, you can use Docker Desktop as an alternative:

#### Install Docker Desktop

1. Download from: https://www.docker.com/products/docker-desktop/
2. Install and start Docker Desktop
3. Ensure WSL 2 backend is enabled

#### Use Docker Compose Instead

```powershell
cd deploy\podman

# Use docker-compose instead of podman-compose
docker-compose up -d

# Check status
docker ps
```

The `podman-compose.yml` file is compatible with Docker Compose.

### Solution 4: Run Services Manually with Podman

If podman-compose doesn't work, run containers individually:

```powershell
# Create network
podman network create rag_network

# Start PostgreSQL
podman run -d `
  --name rag_postgres `
  --network rag_network `
  -e POSTGRES_DB=rag_platform `
  -e POSTGRES_USER=rag_user `
  -e POSTGRES_PASSWORD=changeme `
  -p 5432:5432 `
  docker.io/library/postgres:16

# Start Redis
podman run -d `
  --name rag_redis `
  --network rag_network `
  -p 6379:6379 `
  docker.io/library/redis:7-alpine

# Verify
podman ps
```

---

## Alternative: Skip Podman/Docker for Now

You can run the application without containers by installing PostgreSQL and Redis locally:

### Install PostgreSQL Locally

1. Download from: https://www.postgresql.org/download/windows/
2. Install with default settings
3. Create database:

```sql
CREATE DATABASE rag_platform;
CREATE USER rag_user WITH PASSWORD 'changeme';
GRANT ALL PRIVILEGES ON DATABASE rag_platform TO rag_user;
```

### Install Redis Locally

1. Download from: https://github.com/microsoftarchive/redis/releases
2. Install and start Redis service
3. Or use Redis in WSL:

```bash
wsl
sudo apt-get install redis-server
sudo service redis-server start
```

### Update .env File

```env
DB_HOST=localhost
DB_PORT=5432
REDIS_HOST=localhost
REDIS_PORT=6379
```

---

## Common Issues

### 1. Port Already in Use

**Check what's using the port:**

```powershell
netstat -ano | findstr "5432"
netstat -ano | findstr "6379"
netstat -ano | findstr "8000"
netstat -ano | findstr "8501"
```

**Kill the process:**

```powershell
taskkill /PID <PID> /F
```

### 2. WSL Issues

**Reset WSL:**

```powershell
wsl --shutdown
wsl --unregister podman-machine-default
```

**Update WSL:**

```powershell
wsl --update
```

### 3. Hyper-V Issues

Ensure Hyper-V is enabled:

```powershell
# Run as Administrator
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
```

### 4. Firewall Blocking

Add firewall rules:

```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "PostgreSQL" -Direction Inbound -LocalPort 5432 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Redis" -Direction Inbound -LocalPort 6379 -Protocol TCP -Action Allow
```

---

## Ollama Issues

### Ollama Not Responding

```powershell
# Check Ollama is running
ollama list

# Check which port Ollama is listening on
netstat -ano | findstr "11434"

# Restart Ollama (Windows — from PowerShell as admin)
taskkill /IM "ollama.exe" /F
Start-Process ollama -ArgumentList "serve"
```

### Ollama Model Not Pulling

```powershell
# Pull models individually
ollama pull qwen3:4b       # primary LLM (~2.5 GB)
ollama pull llava:7b       # vision model for charts (~4.7 GB)
ollama pull gemma3:4b      # optional secondary (~3.3 GB)
ollama pull phi4-mini      # optional lightweight (~2.5 GB)
```

### Ollama Timeout on CPU

For CPU-only inference, generation can be slow. Increase timeout:

```env
# In .env:
OLLAMA_TIMEOUT=600
OLLAMA_NUM_CTX=2048
```

---

## Phase 13: PDF Table Extraction Issues

### Tables Not Being Extracted

Check if pdfplumber is installed:

```powershell
.\venv\Scripts\python.exe -c "import pdfplumber; print('pdfplumber OK')"
```

If not installed:

```powershell
.\venv\Scripts\pip install "pdfplumber>=0.10.0"
```

### pdfplumber Extraction Fails on a Specific PDF

Some PDFs use non-standard encoding. Force fallback to PyPDF2:

```env
# In .env:
PDF_USE_ENHANCED_LOADER=false
```

### Tables Appear Garbled in Chunks

The table Markdown is too large for one chunk. Reduce the threshold:

```env
PDF_MAX_TABLE_CHUNK_CHARS=1000
```

Or switch to CSV format (more compact):

```env
PDF_TABLE_FORMAT=csv
```

---

## Phase 14: Chart / Image Description Issues

### Charts Not Being Described

Check if llava:7b is installed:

```powershell
ollama list
# Should show: llava:7b
```

If missing:

```powershell
ollama pull llava:7b
```

Check the setting is enabled:

```powershell
.\venv\Scripts\python.exe -c "
from backend.ingestion.chart_describer import get_chart_describer
cd = get_chart_describer()
print('enabled:', cd.enabled, '| model:', cd.model)
"
```

### Chart Descriptions Are Slow / Ingestion Takes Very Long

llava:7b takes 5–15 seconds per image on CPU. Options:

```env
# Reduce images processed per page (default 3):
PDF_CHART_MAX_PER_PAGE=1

# Increase minimum area to only describe large charts:
PDF_CHART_MIN_AREA_PTS=20000

# Disable entirely for bulk ingestion, re-enable afterwards:
PDF_CHART_DESCRIPTION_ENABLED=false
```

### Charts Are Tiny / Decorative Images Being Described

Increase the minimum area filter:

```env
# Default 5000 pts² (~71×71 pts); raise to ~100×100 pts:
PDF_CHART_MIN_AREA_PTS=10000
```

### llava Returns Empty or Nonsensical Descriptions

Try a lower DPI for less noise:

```env
PDF_CHART_RENDER_DPI=100
```

Or check the image coordinate flip is correct — test with a known PDF that has a visible chart.

---

## Phase 12: Knowledge Graph Issues

### KG Build Returns No Entities

Check that LLM extraction is enabled and the model is running:

```powershell
curl http://localhost:8000/api/v1/kg/stats
# Should return {"nodes": N, "edges": N}
```

If `nodes: 0` after building, check Ollama is responding:

```powershell
ollama list
curl http://localhost:11434/api/tags
```

### KG Persist File Missing

The KG is saved to `data/knowledge_graph.json`. Ensure the `data/` directory exists:

```powershell
New-Item -ItemType Directory -Force -Path data
```

---

## Import Errors

```powershell
# Make sure virtual environment is activated
venv\Scripts\activate

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Test key imports
.\venv\Scripts\python.exe -c "from backend.api.main import app; print('API OK')"
.\venv\Scripts\python.exe -c "from backend.ingestion.pipeline import IngestionPipeline; print('Pipeline OK')"
.\venv\Scripts\python.exe -c "from backend.knowledge_graph import GraphStore; print('KG OK')"
.\venv\Scripts\python.exe -c "from backend.ingestion.chart_describer import get_chart_describer; print('Chart OK')"
```

---

## Testing Without Database

You can test the backend without PostgreSQL/Redis:

```powershell
# The platform starts fine without databases
# Redis missing → memory falls back to in-process dict
# PostgreSQL missing → only status endpoint shows "disconnected"
uvicorn backend.api.main:app --reload
curl http://localhost:8000/health
```

---

## Recommended Approach for Windows

For the smoothest experience on Windows:

1. **Use Docker Desktop** instead of Podman (if Podman has issues)
2. **Or** install PostgreSQL and Redis locally
3. **Or** skip database setup — Redis and PostgreSQL are optional for core functionality
4. **Ensure Ollama is running** with `qwen3:4b` and `llava:7b` pulled

The core pipeline (document ingestion, RAG chat, agent, knowledge graph, chart descriptions) all work without databases.

---

## Quick Validation Checklist

Run this to verify the full stack is healthy:

```powershell
# 1. Check Python imports
.\venv\Scripts\python.exe -c "from backend.api.main import app; print('Backend OK')"

# 2. Check Ollama models
ollama list

# 3. Start backend
uvicorn backend.api.main:app --reload

# 4. Health check
curl http://localhost:8000/health

# 5. Check PDF extraction engine
curl http://localhost:8000/api/v1/documents/stats/overview
# Look for: "pdf_extraction": {"backend": "pdfplumber", "chart_support": true}

# 6. Check KG endpoints
curl http://localhost:8000/api/v1/kg/stats
```

---

## Getting Help

If issues persist:

1. Check backend logs in `logs/app.log`
2. Check Podman Desktop logs
3. Check Windows Event Viewer
4. Try Docker Desktop as alternative to Podman
5. Run without containers using local services

# Made with Bob
