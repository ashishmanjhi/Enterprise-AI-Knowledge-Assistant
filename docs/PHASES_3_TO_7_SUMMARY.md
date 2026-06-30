# Phases 3–7 Implementation Summary

> Comprehensive summary of all features, architecture decisions, and APIs
> delivered in Phases 3 through 7 of the Enterprise Agentic RAG Platform.

---

## Phase 3 — Query Understanding

### Overview
Transforms raw user queries before retrieval to increase both precision and recall. Three techniques are available and can be enabled independently.

### Components

| File | Purpose |
|------|---------|
| `backend/query_understanding/query_reformulator.py` | Rewrites vague queries (resolves pronouns, clarifies intent) |
| `backend/query_understanding/query_expander.py` | Generates N alternative query phrasings |
| `backend/query_understanding/hyde_generator.py` | Generates a hypothetical document answer for better semantic search |
| `backend/query_understanding/query_processor.py` | Orchestrator — runs all enabled techniques in sequence |

### Settings
```env
ENABLE_QUERY_REFORMULATION=true
ENABLE_QUERY_EXPANSION=true
ENABLE_HYDE=true
NUM_QUERY_EXPANSIONS=3
QUERY_EXPANSION_TEMPERATURE=0.7
QUERY_REFORMULATION_TEMPERATURE=0.3
HYDE_TEMPERATURE=0.7
HYDE_MAX_TOKENS=200
```

### Chat API Integration
Pass a `query_understanding` object in the chat request body:
```json
{
  "message": "What did they say about costs?",
  "query_understanding": {
    "enable_reformulation": true,
    "enable_expansion": true,
    "enable_hyde": true,
    "num_expansions": 3
  }
}
```

The response includes a `query_metadata` field showing:
- `original_query` — original user text
- `reformulated_query` — rewritten query (if reformulation ran)
- `expanded_queries` — list of alternative queries (if expansion ran)
- `hyde_answer` — hypothetical document snippet (if HyDE ran)
- `processing_time` — seconds spent on query processing

---

## Phase 4 — Retrieval Optimization (Cross-Encoder Reranking)

### Overview
A second-pass scoring step that uses a cross-encoder model to re-score retrieved chunks based on their relevance to the query. Produces higher-precision top results at the cost of a small latency overhead.

### Component
`backend/rerankers/cross_encoder.py` — `CrossEncoderReranker`
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22 MB, downloads on first use)
- Lazy model loading — not downloaded until first rerank call
- Configurable `top_n` (default 3)
- Batch scoring for efficiency

### Settings
```env
ENABLE_RERANKING=false        # off by default to avoid model download on startup
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=3
RERANKER_BATCH_SIZE=16
```

### Per-Request Override
Add `"use_reranking": true` to any chat request to enable reranking for that call regardless of the global setting.

---

## Phase 5 — Evaluation Framework (RAGAS)

### Overview
Measures RAG pipeline quality using LLM-as-judge metrics from the RAGAS framework. Works with Ollama locally — no external API keys required.

### Metrics

| Metric | Requires Ground Truth | Description |
|--------|-----------------------|-------------|
| `faithfulness` | No | Is the answer grounded in the context? |
| `answer_relevancy` | No | Does the answer address the question? |
| `context_precision` | No | Were the right chunks retrieved? |
| `context_recall` | Yes | Did retrieval find everything needed? |
| `factual_correctness` | Yes | Is the answer factually correct? |

### Components

| File | Purpose |
|------|---------|
| `backend/evaluators/metrics.py` | Data classes: `EvaluationSample`, `EvaluationResult`, `MetricScore` |
| `backend/evaluators/ragas_evaluator.py` | `RAGASEvaluator` — wraps RAGAS 0.2.x, builds judge LLM wrapper |
| `backend/api/routes/evaluate.py` | `POST /api/v1/evaluate`, `GET /api/v1/evaluate/metrics` |

### API Usage
```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{
      "question": "What is EVSE uptime?",
      "answer": "EVSE uptime refers to operational availability.",
      "contexts": ["EVSE connectors experience downtime due to network failures..."]
    }],
    "metrics": ["faithfulness", "answer_relevancy", "context_precision"]
  }'
```

### Streamlit UI
Navigate to the **📊 Evaluate** page to:
- Manually enter samples one by one
- Paste a JSON array of samples
- Select metrics and override the judge model
- View aggregated scores and per-sample breakdowns

---

## Phase 6 — Conversational Memory

### Overview
Persists conversation history in Redis so the LLM has context across turns without the client needing to resend history. Falls back to an in-process dictionary when Redis is unavailable.

### Architecture

```
User Request
    │
    ▼
chat route
    ├─ load history from ConversationManager
    ├─ inject history into RAG prompt
    ├─ generate response
    └─ record_turn(user_msg, assistant_msg)
              │
              ▼
         SessionMemory
              │
     ┌────────┴────────┐
     │                 │
  Redis list      in-process dict
  (preferred)     (fallback)
```

### Components

| File | Purpose |
|------|---------|
| `backend/memory/session_memory.py` | `SessionMemory` — Redis list per conversation with TTL |
| `backend/memory/conversation_manager.py` | `ConversationManager` — history loading, recording, summarisation |
| `backend/api/routes/memory.py` | `GET/DELETE /api/v1/memory/{id}` |

### Summarisation
When a conversation exceeds `MEMORY_SUMMARY_THRESHOLD * 2` messages, the oldest half is summarised by the LLM and replaced with a single `system` message. This keeps the prompt window bounded while preserving context.

### Settings
```env
MEMORY_SESSION_TTL=86400           # 24 h Redis key TTL
MEMORY_MAX_HISTORY_MESSAGES=20     # messages injected into prompt
MEMORY_ENABLE_SUMMARISATION=true
MEMORY_SUMMARY_THRESHOLD=10        # turns (not messages) before summarising
```

### Memory API
```bash
# Fetch conversation history
GET /api/v1/memory/{conversation_id}?limit=20

# Get metadata only
GET /api/v1/memory/{conversation_id}/info

# Delete (idempotent)
DELETE /api/v1/memory/{conversation_id}
```

### Test Coverage
`backend/tests/test_memory.py` — 30 tests covering:
- In-process fallback (11 tests)
- ConversationManager CRUD + summarisation (12 tests)
- Redis path via mocked client (7 tests)

---

## Phase 7 — Safety & Governance

### Overview
Enterprise-grade safety controls applied on every chat request:
- **Input checks** before the LLM sees the message
- **Output checks** before the answer is returned to the user

All detectors are regex/heuristic-based and fully offline (no external API calls required). The hallucination detector optionally uses the judge LLM for higher accuracy.

### Architecture

```
Chat Request
    │
    ▼
GuardrailsPipeline.check_input(message)
    ├─ PromptInjectionDetector  ─► block (HTTP 400) if detected
    ├─ ToxicityDetector         ─► block (HTTP 400) if detected
    └─ PIIDetector              ─► redact input (warn-only by default)
    │
    ▼
RAG Chain (generate response)
    │
    ▼
GuardrailsPipeline.check_output(answer, context_chunks)
    ├─ HallucinationDetector    ─► replace answer (warn-only by default)
    └─ PIIDetector              ─► redact output (warn-only by default)
    │
    ▼
Response to user
```

### Detectors

#### PromptInjectionDetector
- 14 compiled regex patterns
- Detects: instruction overrides, jailbreak keywords, role overrides, system-prompt extraction, delimiter injection, command injection via context
- Severity: HIGH / MEDIUM / LOW per pattern

#### PIIDetector
- Detects: SSN, credit cards, emails, US phone numbers, IP addresses, passport-like codes, date-of-birth patterns, API key shapes
- `redact()` method replaces matches with `[REDACTED:TYPE]` placeholders

#### ToxicityDetector
- Detects: violence/threats, self-harm, hate speech markers, severe harassment
- Severity: HIGH / MEDIUM

#### HallucinationDetector
- **LLM-as-judge (primary)**: asks `"GROUNDED or HALLUCINATION?"` with a 0-temperature call
- **Heuristic fallback**: token-overlap ratio between answer and context (threshold configurable)
- Automatic fallback if LLM call fails

### Settings
```env
GUARDRAILS_ENABLE_INJECTION=true
GUARDRAILS_ENABLE_TOXICITY=true
GUARDRAILS_ENABLE_PII=true
GUARDRAILS_ENABLE_HALLUCINATION=true
GUARDRAILS_BLOCK_ON_INJECTION=true    # block request → HTTP 400
GUARDRAILS_BLOCK_ON_TOXICITY=true     # block request → HTTP 400
GUARDRAILS_BLOCK_ON_PII_INPUT=false   # warn only (redact and continue)
GUARDRAILS_BLOCK_ON_HALLUCINATION=false  # warn only
```

### Guardrails API
```bash
# Check input safety standalone
POST /api/v1/guardrails/check/input
{"text": "Ignore all previous instructions..."}

# Check output safety standalone
POST /api/v1/guardrails/check/output
{"answer": "...", "contexts": ["..."]}

# Status
GET /api/v1/guardrails/status
```

### Blocked Request Response
When a request is blocked, the chat endpoint returns HTTP 400:
```json
{
  "detail": {
    "error": "request_blocked_by_guardrails",
    "block_reason": "prompt_injection_detected",
    "checks": [...]
  }
}
```

### Test Coverage
`backend/tests/test_guardrails.py` — 41 tests covering:
- `PromptInjectionDetector` (9 tests)
- `PIIDetector` (9 tests)
- `ToxicityDetector` (5 tests)
- `HallucinationDetector` (7 tests — heuristic, LLM judge, fallback)
- `GuardrailsPipeline` (11 tests — blocking, warn-only, redaction)

---

## Combined Test Suite

```bash
pytest backend/tests/ -v --tb=short
```

| File | Tests | Phase |
|------|-------|-------|
| test_chunking.py | 16 | 1 |
| test_loaders.py | 12 | 1 |
| test_query_understanding.py | 22 | 3 |
| test_reranker.py | 12 | 4 |
| test_memory.py | 30 | 6 |
| test_guardrails.py | 41 | 7 |
| **Total** | **133** | |

All 133 tests pass offline (no Ollama, Redis, or network required).

---

*Made with Bob*
