# Phase 11 — Multi-Agent Ecosystem: Implementation Summary

## Overview

Phase 11 replaces the single-graph approach of Phase 9 with a **coordinated ecosystem of five specialised sub-agents**, each compiled as its own LangGraph `StateGraph` and orchestrated by a top-level router graph.

Every query passes through:

```
classify_intent
      │
      ├─ research ──▶ ResearchAgent   (complex multi-doc questions)
      └─ retrieval ─▶ RetrievalAgent  (factual questions)
                          │
                   KnowledgeAgent     (entity extraction + lookup)
                          │
                   EvaluationAgent    (quality scoring)
                          │
                   GovernanceAgent    (safety + compliance gate)
                          │
                      generate        (final answer assembly)
                          │
                         END
```

---

## The Five Agents

### 1. Research Agent (`ResearchAgent`)

**Purpose:** Handle complex questions that require synthesis across multiple documents.

**Pipeline:** `decompose_query → retrieve_answer → synthesise`

| Node | What it does |
|------|-------------|
| `decompose_query` | LLM decomposes the question into ≤ `RESEARCH_AGENT_MAX_SUB_QUESTIONS` sub-questions |
| `retrieve_answer` | For each sub-question: hybrid retrieval + independent LLM answer |
| `synthesise` | Merges all findings into a single coherent paragraph |

**When triggered:** query classified as `research` intent by the orchestrator.

---

### 2. Retrieval Agent (`RetrievalAgent`)

**Purpose:** Maximise retrieval recall and precision for factual queries.

**Pipeline:** `select_strategy → expand_and_retrieve → rerank`

| Node | What it does |
|------|-------------|
| `select_strategy` | LLM picks `hybrid`, `faiss`, or `bm25` for this query |
| `expand_and_retrieve` | Generates N query expansions, runs parallel retrieval, deduplicates results |
| `rerank` | Optional cross-encoder reranking of the merged result set |

**When triggered:** query classified as `retrieval` or `general` intent.

---

### 3. Knowledge Agent (`KnowledgeAgent`)

**Purpose:** Enrich the pipeline with entity-level intelligence.

**Pipeline:** `extract_entities → lookup_context → build_knowledge_context`

| Node | What it does |
|------|-------------|
| `extract_entities` | LLM-based NER from query + top document chunks (regex fallback) |
| `lookup_context` | Targeted retrieval for each entity — adds new chunks not already retrieved |
| `build_knowledge_context` | Assembles a compact context string prepended to the final prompt |

**Always runs** (after Research or Retrieval).

---

### 4. Evaluation Agent (`EvaluationAgent`)

**Purpose:** Measure answer quality before governance and delivery.

**Pipeline:** `heuristic_eval → ragas_eval`

| Node | What it does |
|------|-------------|
| `heuristic_eval` | Computes three fast local metrics: faithfulness token overlap, answer length OK, has sources; derives composite score |
| `ragas_eval` | Optional full RAGAS (faithfulness + answer relevancy) when `EVAL_AGENT_USE_RAGAS=true` |

Scores stored in `state["eval_scores"]` for the Governance Agent and the UI.

---

### 5. Governance Agent (`GovernanceAgent`)

**Purpose:** Final safety and compliance gate.

**Pipeline:** `guardrails_check → confidence_and_attribution`

| Node | What it does |
|------|-------------|
| `guardrails_check` | Runs Phase 7 guardrails pipeline on the generated answer (PII, hallucination, toxicity) |
| `confidence_and_attribution` | Appends low-confidence disclaimer when composite score < threshold; notes missing sources |

If `guardrails_check` blocks the answer, the final answer is replaced with a safe fallback message.

---

## Shared State

All five agents read from and write into a single [`MultiAgentState`](../backend/agents/multi/state.py) TypedDict. Key sections:

```python
{
  # Core
  "query": str,
  "intent": "research" | "retrieval" | "general",

  # Research Agent outputs
  "research_plan":     List[str],
  "research_findings": List[{"question", "answer", "sources"}],
  "research_summary":  str,

  # Retrieval Agent outputs
  "retrieval_strategy": "hybrid" | "faiss" | "bm25",
  "retrieval_results":  List[{chunk_id, content, score, ...}],

  # Knowledge Agent outputs
  "entities":          List[{"text", "label"}],
  "knowledge_context": str,

  # Evaluation Agent outputs
  "eval_scores":  {"faithfulness_heuristic", "has_sources", "composite", ...},
  "eval_summary": str,

  # Governance Agent outputs
  "governance_passed": bool,
  "governance_issues": List[{"check", "severity", "detail"}],
  "final_answer":      str,

  # Orchestrator
  "active_agents":  List[str],
  "final_response": str,
  "trace":          List[str],   # append-only
}
```

---

## Files Created

### Backend

| File | Purpose |
|------|---------|
| `backend/agents/multi/state.py` | `MultiAgentState` TypedDict |
| `backend/agents/multi/research_agent.py` | Research Agent LangGraph sub-graph |
| `backend/agents/multi/retrieval_agent.py` | Retrieval Agent LangGraph sub-graph |
| `backend/agents/multi/evaluation_agent.py` | Evaluation Agent LangGraph sub-graph |
| `backend/agents/multi/governance_agent.py` | Governance Agent LangGraph sub-graph |
| `backend/agents/multi/knowledge_agent.py` | Knowledge Agent LangGraph sub-graph |
| `backend/agents/multi/orchestrator.py` | `MultiAgentOrchestrator` — top-level router |
| `backend/agents/multi/__init__.py` | Package exports |
| `backend/api/routes/multi_agent.py` | `POST /api/v1/multi-agent/chat`, `GET /api/v1/multi-agent/health` |

### Frontend

| File | Purpose |
|------|---------|
| `frontend/streamlit/pages/5_🌐_Multi_Agent.py` | Multi-Agent Chat UI with per-agent cards, quality scores, entity panel, governance status |

---

## API

### `POST /api/v1/multi-agent/chat`

**Request:**
```json
{
  "message": "Compare the key findings across all uploaded documents",
  "conversation_id": "ma_abc123",
  "top_k": 5,
  "temperature": 0.7,
  "max_tokens": 600,
  "conversation_history": []
}
```

**Response:**
```json
{
  "conversation_id": "ma_abc123",
  "answer": "Across all documents, the key findings are ...",
  "intent": "research",
  "active_agents": ["research", "knowledge", "evaluation", "governance"],
  "agent_results": [
    {"name": "research",   "ran": true,  "summary": "Synthesised 3 findings"},
    {"name": "retrieval",  "ran": false, "summary": "0 docs"},
    {"name": "knowledge",  "ran": true,  "summary": "7 entities"},
    {"name": "evaluation", "ran": true,  "summary": "composite=0.71 PASS"},
    {"name": "governance", "ran": true,  "summary": "passed"}
  ],
  "research_plan": ["What are the main topics?", "What are the recommendations?", ...],
  "research_findings": 3,
  "eval_scores": {"faithfulness_heuristic": 0.62, "has_sources": 1.0, "composite": 0.71},
  "governance_passed": true,
  "entities": ["Document A", "Policy 3.2", "Q4 Results"],
  "trace": ["orchestrator.classify: intent=research", "research.decompose: 3 sub-questions", ...],
  "processing_time": 45.2
}
```

### `GET /api/v1/multi-agent/health`

```json
{
  "status": "healthy",
  "orchestrator": "compiled",
  "agents": ["research", "retrieval", "knowledge", "evaluation", "governance"],
  "research_max_sub": 4,
  "retrieval_expansions": 3,
  "eval_pass_threshold": 0.4,
  "eval_use_ragas": false
}
```

---

## Configuration

```env
# Multi-Agent Ecosystem (Phase 11)
RESEARCH_AGENT_MAX_SUB_QUESTIONS=4    # max sub-questions research agent decomposes into
RETRIEVAL_AGENT_EXPANSIONS=3          # query expansions for parallel retrieval
EVAL_AGENT_PASS_THRESHOLD=0.4         # composite score below this triggers disclaimer
EVAL_AGENT_USE_RAGAS=false            # set true for full RAGAS scoring (slow on CPU)
KNOWLEDGE_AGENT_MAX_ENTITIES=5        # max entities to entity-lookup per query
```

---

## Performance Notes

On CPU-only Ollama, the multi-agent pipeline makes **10–25 LLM calls** per query:

| Config | Approx. LLM calls | Latency |
|--------|-------------------|---------|
| Research intent, full pipeline | 15–25 | 5–20 min |
| Retrieval intent, full pipeline | 8–12 | 2–8 min |
| RAGAS disabled (default) | −2 calls | faster |

For development, use:
```env
RESEARCH_AGENT_MAX_SUB_QUESTIONS=2
RETRIEVAL_AGENT_EXPANSIONS=1
EVAL_AGENT_USE_RAGAS=false
AGENT_ENABLE_DOCUMENT_GRADING=false
```

---

**Phase**: 11
**Status**: ✅ Complete
**Date**: 2026-07-01
