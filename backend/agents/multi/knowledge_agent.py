"""
Knowledge Agent (Phase 11) — Multi-Agent Ecosystem.

The Knowledge Agent enriches the pipeline with entity-level intelligence:

  1. extract_entities  — pulls named entities from the query and top documents
  2. lookup_context    — finds additional document chunks related to the
                         extracted entities (entity-driven retrieval boost)
  3. build_knowledge_context — assembles a compact context string for the
                         downstream generator

The agent operates on a best-effort basis: if entity extraction fails or
no additional chunks are found, it simply passes the state through unchanged.

Note: Phase 12 will replace the simple regex extraction here with a full
knowledge-graph store (NetworkX / Neo4j) and relation mapping.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Set

from langgraph.graph import END, StateGraph

from backend.agents.multi.state import MultiAgentState
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.core.tracing import trace_span

logger = get_logger(__name__)

# ── Simple regex NER (Phase 11 baseline — replaced by spaCy / LLM in Ph 12) ─

# Capitalised word sequences (2–4 words) that are likely named entities
_NE_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b")

_NER_PROMPT = """\
Extract the key named entities (people, organisations, products, locations, \
technical terms) from the text below.  Return one entity per line, nothing else.

Text: {text}

Entities:"""


# ── Node functions ────────────────────────────────────────────────────────

async def extract_entities(state: MultiAgentState, llm) -> Dict[str, Any]:
    """
    Extract named entities from the query and the top retrieved chunks.
    Uses LLM extraction when available, falls back to regex.
    """
    query = state["query"]
    docs  = state.get("retrieval_results", [])
    start = time.time()

    # Combine query + first 3 docs for extraction input
    text = query + "\n" + "\n".join(d.get("content", "")[:300] for d in docs[:3])

    entities: List[Dict[str, str]] = []
    try:
        result = await llm.generate(
            prompt=_NER_PROMPT.format(text=text[:800]),
            temperature=0.0,
            max_tokens=150,
        )
        raw_lines = [ln.strip() for ln in result.get("text", "").splitlines() if ln.strip()]
        entities  = [{"text": ln, "label": "ENTITY"} for ln in raw_lines[:20]]
    except Exception:
        # Regex fallback
        matches = _NE_RE.findall(text)
        seen: Set[str] = set()
        for m in matches:
            if m not in seen:
                seen.add(m)
                entities.append({"text": m, "label": "NE_REGEX"})
        entities = entities[:20]

    elapsed = time.time() - start
    logger.info(f"knowledge.extract_entities: {len(entities)} entities ({elapsed:.2f}s)")
    return {
        "entities":      entities,
        "active_agents": ["knowledge"],
        "trace": [f"knowledge.extract_entities: {len(entities)} entities"],
    }


async def lookup_context(state: MultiAgentState, retriever) -> Dict[str, Any]:
    """
    For each extracted entity, run a targeted retrieval query and collect
    any new chunks not already in retrieval_results.
    """
    entities = state.get("entities", [])
    existing = {d["chunk_id"] for d in state.get("retrieval_results", [])}
    start    = time.time()

    new_chunks: List[Dict[str, Any]] = []
    max_ent = settings.knowledge_agent_max_entities

    for ent in entities[:max_ent]:
        entity_text = ent["text"]
        try:
            results = await retriever.retrieve(
                query=entity_text, top_k=2, method="hybrid"
            )
            for doc in results:
                if doc.chunk_id not in existing:
                    existing.add(doc.chunk_id)
                    new_chunks.append({
                        "chunk_id":    doc.chunk_id,
                        "content":     doc.content,
                        "score":       doc.score,
                        "document_id": doc.document_id,
                        "filename":    doc.filename,
                        "page_number": doc.page_number,
                        "retrieval_method": "knowledge_lookup",
                        "entity":      entity_text,
                    })
        except Exception as exc:
            logger.warning(f"knowledge lookup failed for '{entity_text}': {exc}")

    elapsed = time.time() - start
    logger.info(f"knowledge.lookup_context: {len(new_chunks)} new chunks ({elapsed:.2f}s)")
    return {
        "retrieval_results": state.get("retrieval_results", []) + new_chunks,
        "trace": [f"knowledge.lookup_context: +{len(new_chunks)} entity chunks in {elapsed:.1f}s"],
    }


async def build_knowledge_context(state: MultiAgentState) -> Dict[str, Any]:
    """
    Build a compact knowledge context string from entity-retrieved chunks.
    This will be prepended to the final generation prompt.
    """
    entities = state.get("entities", [])
    docs     = [d for d in state.get("retrieval_results", []) if d.get("retrieval_method") == "knowledge_lookup"]

    if not entities and not docs:
        return {
            "knowledge_context": None,
            "trace": ["knowledge.build_context: nothing to add"],
        }

    entity_list = ", ".join(e["text"] for e in entities[:10])
    chunk_texts = "\n".join(
        f"• [{d['filename']}] {d['content'][:200]}"
        for d in docs[:5]
    )

    ctx = f"Key entities: {entity_list}"
    if chunk_texts:
        ctx += f"\n\nEntity-related context:\n{chunk_texts}"

    logger.info(f"knowledge.build_context: {len(ctx)} chars")
    return {
        "knowledge_context": ctx,
        "trace": [f"knowledge.build_context: entities={len(entities)} extra_chunks={len(docs)}"],
    }


# ── Graph builder ─────────────────────────────────────────────────────────

class KnowledgeAgent:
    """
    LangGraph sub-graph for the Knowledge Agent.

    Extracts entities → looks up additional context → builds knowledge context.
    """

    def __init__(self, llm, retriever) -> None:
        self.llm       = llm
        self.retriever = retriever
        self._graph    = self._build()
        logger.info("KnowledgeAgent compiled")

    def _build(self):
        import functools
        builder = StateGraph(MultiAgentState)

        builder.add_node("extract_entities",      functools.partial(extract_entities,     llm=self.llm))
        builder.add_node("lookup_context",         functools.partial(lookup_context,        retriever=self.retriever))
        builder.add_node("build_knowledge_context", build_knowledge_context)

        builder.set_entry_point("extract_entities")
        builder.add_edge("extract_entities",       "lookup_context")
        builder.add_edge("lookup_context",          "build_knowledge_context")
        builder.add_edge("build_knowledge_context", END)

        return builder.compile()

    async def run(self, state: Dict[str, Any]) -> MultiAgentState:
        async with trace_span("knowledge_agent.run"):
            return await self._graph.ainvoke(state)  # type: ignore[return-value]


# Made with Bob
