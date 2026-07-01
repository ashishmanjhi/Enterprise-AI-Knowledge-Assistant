# Phase 12 — Knowledge Graph Enhancement

## Overview

Phase 12 adds a fully-integrated **Knowledge Graph** layer to the Enterprise Agentic RAG Platform. Every query that flows through the multi-agent pipeline now benefits from structured entity–relation knowledge extracted from your ingested documents, fused with the existing FAISS + BM25 hybrid vector retrieval.

---

## Architecture

```
Ingested text
      │
      ▼
EntityExtractor ──────────────► GraphStore (NetworkX, JSON persist)
      │                                │
      ▼                                │
RelationMapper ─── triples ──────────►│
                                       │
Query ──► GraphRetriever ◄─────────────┘
      │          │
      │          ▼  (graph-aware results)
      │   HybridGraphRetriever ◄── HybridRetriever (FAISS+BM25)
      │          │
      │     RRF fusion
      │          │
      ▼          ▼
  KnowledgeAgent (multi-agent sub-graph)
```

---

## New Files

| File | Purpose |
|------|---------|
| `backend/knowledge_graph/graph_store.py` | NetworkX `MultiDiGraph` KG — entity nodes, relation edges, JSON persist/load |
| `backend/knowledge_graph/entity_extractor.py` | LLM + regex NER → typed entities (PERSON, ORG, PRODUCT, LOCATION, CONCEPT, TECHNICAL) |
| `backend/knowledge_graph/relation_mapper.py` | LLM relation triple extraction (`subject \| predicate \| object`) |
| `backend/knowledge_graph/graph_retriever.py` | Graph-aware retrieval: entity search + neighbour expansion |
| `backend/knowledge_graph/hybrid_graph_retriever.py` | RRF fusion of vector + graph results |
| `backend/knowledge_graph/__init__.py` | Package exports |
| `backend/api/routes/knowledge_graph.py` | REST API for KG build, query, stats, entities, relations, clear |
| `frontend/streamlit/pages/6_🕸️_Knowledge_Graph.py` | Graph Explorer UI |

---

## Modified Files

| File | Change |
|------|--------|
| `backend/core/settings.py` | Added 10 Phase 12 KG settings; bumped `app_version` to `12.0.0` |
| `backend/agents/multi/knowledge_agent.py` | Upgraded from regex baseline to full KG pipeline (EntityExtractor + RelationMapper + GraphRetriever) |
| `backend/agents/multi/orchestrator.py` | Pass shared `GraphStore` singleton to `KnowledgeAgent` |
| `backend/api/main.py` | Register `kg_router`; add `knowledge_graph` to root endpoint dict |

---

## Settings Reference (Phase 12)

| Setting | Default | Description |
|---------|---------|-------------|
| `KG_PERSIST_PATH` | `data/knowledge_graph.json` | JSON file for graph persistence |
| `KG_ENTITY_TYPES` | `["PERSON","ORG","PRODUCT","LOCATION","CONCEPT","TECHNICAL"]` | Allowed entity types |
| `KG_MAX_ENTITIES_PER_DOC` | `50` | Entity cap per ingestion call |
| `KG_MAX_RELATIONS_PER_DOC` | `30` | Relation triple cap per ingestion call |
| `KG_NEIGHBOUR_HOPS` | `2` | Graph expansion hops during retrieval |
| `KG_MAX_SUBGRAPH_NODES` | `50` | Max nodes returned from a subgraph query |
| `KG_HYBRID_VECTOR_WEIGHT` | `0.6` | RRF weight for vector results in hybrid fusion |
| `KG_HYBRID_GRAPH_WEIGHT` | `0.4` | RRF weight for graph results in hybrid fusion |
| `KG_LLM_EXTRACTION_ENABLED` | `true` | Use LLM for NER / relation extraction |
| `KG_EXTRACTION_MAX_TOKENS` | `300` | Token budget for extraction prompts |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/kg/build` | Extract entities + relations from text and update the graph |
| `GET` | `/api/v1/kg/query?q=...&top_k=10` | Search entities and return subgraph context |
| `GET` | `/api/v1/kg/stats` | Graph statistics (counts, entity types, persist path) |
| `GET` | `/api/v1/kg/entities?entity_type=ORG&limit=100` | List entities with optional type filter |
| `GET` | `/api/v1/kg/relations?entity_id=...` | Outgoing relation edges for an entity |
| `DELETE` | `/api/v1/kg/clear?persist=true` | Wipe the knowledge graph |

### Build Request

```json
{
  "text": "LangGraph is developed by LangChain for building multi-agent apps.",
  "source_doc": "langchain_docs.pdf",
  "extract_relations": true,
  "persist": true
}
```

### Build Response

```json
{
  "entities_added": 3,
  "relations_added": 1,
  "elapsed_seconds": 1.24,
  "graph_stats": {
    "num_entities": 42,
    "num_relations": 31,
    "entity_types": {"TECHNICAL": 18, "ORG": 12, "CONCEPT": 12},
    "persist_path": "data/knowledge_graph.json",
    "is_persistent": true
  }
}
```

---

## GraphStore Design

- **Backend**: `networkx.MultiDiGraph` — allows multiple edges between the same pair of nodes (different predicates).
- **Node ID**: SHA-1 of `entity_type::normalised_text` — stable across sessions, enables deduplication.
- **Persistence**: `nx.node_link_data` serialised to JSON; loaded automatically on startup if the file exists.
- **Singleton**: `get_graph_store()` returns a module-level singleton shared across the API routes and the multi-agent orchestrator.

---

## Entity Extraction Strategy

1. **LLM path** (`kg_llm_extraction_enabled=True`): Prompts the configured LLM with a structured `TYPE: entity text` format. Output is parsed line-by-line; unknown type labels are mapped to `CONCEPT`.
2. **Regex fallback**: Fires whenever the LLM is unavailable or returns no results.  Captures:
   - Capitalised proper-noun sequences (PERSON / ORG / LOCATION proxy)
   - Version-tagged technical terms (`Python 3.12`, `GPT-4`)
   - Uppercase acronyms
   - Quoted strings (potential product names / concepts)

---

## Relation Mapping Strategy

1. **LLM path**: Prompts for `subject | predicate | object` triples. Short, lowercase predicates are encouraged (`is a`, `uses`, `developed by`, etc.).
2. **Heuristic fallback**: Regex pattern matching for `"X is/uses/has Y"` constructions.
3. **Entity resolution**: Both subject and object texts are fuzzy-matched against existing graph nodes (substring search). Unmatched ends are auto-created as `CONCEPT` nodes.

---

## Graph Retrieval

`GraphRetriever.retrieve(query, top_k)`:

1. Substring-searches entity nodes for query terms.
2. For each direct hit, performs ego-graph expansion (`kg_neighbour_hops` radius).
3. Scores nodes using decaying RRF-style weights: direct hit = `1/(rank+1)`, neighbours = `base × 0.5^hop`.
4. Builds a natural-language context string per node:  entity name + type + sources + relation summary + neighbour list.

---

## Hybrid Graph Retrieval (RRF Fusion)

`HybridGraphRetriever.retrieve(query, top_k, method)`:

- Runs `HybridRetriever` (FAISS + BM25) and `GraphRetriever` in parallel via `asyncio.gather`.
- Fuses ranked lists with weighted RRF:
  - `score(d) = Σ weight_s / (rrf_k + rank_in_s(d))`
  - Vector weight: `kg_hybrid_vector_weight` (default 0.6)
  - Graph weight:  `kg_hybrid_graph_weight`  (default 0.4)
- Each result carries a `source` field (`"vector"` | `"graph"` | `"both"`) for observability.

---

## Knowledge Agent Upgrade (Phase 11 → 12)

The `KnowledgeAgent` LangGraph sub-graph now runs four nodes instead of two:

```
extract_entities  →  map_relations  →  lookup_context  →  build_knowledge_context
```

| Node | Phase 11 | Phase 12 |
|------|----------|----------|
| `extract_entities` | LLM prompt + regex fallback, label=`"ENTITY"` | `EntityExtractor` → typed entities, upserted to GraphStore |
| `map_relations` | ❌ not present | `RelationMapper` → triples stored as GraphStore edges |
| `lookup_context` | HybridRetriever only | Graph expansion (GraphRetriever) + HybridRetriever |
| `build_knowledge_context` | Entities + doc snippets | Entities + relation triples (from GraphStore) + doc snippets |

---

## Streamlit UI

The **🕸️ Knowledge Graph** page (page 6) provides four tabs:

| Tab | Description |
|-----|-------------|
| 🔨 Build Graph | Paste text, choose source label, extract entities & relations |
| 🔍 Query / Explore | Search entities, inspect subgraph context, relations, neighbours |
| 📋 Entities | Paginated entity browser with type filter |
| 🔗 Relations | Relation edge inspector for any entity ID |

The sidebar shows live graph statistics (entity/relation counts, type breakdown) with a one-click **Clear** action.

---

## Implementation Notes

- `networkx` is a pure-Python dependency — no external graph database required.
- The JSON persist file is read at `GraphStore.__init__()` and written on `save()` calls (triggered by `POST /kg/build` with `persist=True`, and by `DELETE /kg/clear`).
- All extractor/retriever classes are async-first; sync regex fallbacks are called inline (no blocking I/O).
- The `KnowledgeAgent` receives the same `GraphStore` singleton as the API routes via `get_graph_store()`.

---

**Phase**: 12  
**Version**: 12.0.0  
**Status**: ✅ Complete  
**Date**: 2026-07-01
