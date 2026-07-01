"""
Knowledge Agent (Phase 12) — Multi-Agent Ecosystem.

Phase 12 upgrade: replaces the Phase 11 regex/LLM-only extraction with a
full Knowledge Graph pipeline (NetworkX GraphStore + EntityExtractor +
RelationMapper).

The Knowledge Agent now:
  1. extract_entities     — LLM + regex NER via EntityExtractor; entities
                             are upserted into the shared GraphStore.
  2. map_relations         — extracts relation triples via RelationMapper and
                             wires them as edges in the GraphStore.
  3. lookup_context        — graph-aware retrieval for each entity: neighbour
                             expansion via GraphRetriever, plus entity-driven
                             HybridRetriever lookups for new doc chunks.
  4. build_knowledge_context — assembles enriched context string (entities +
                             relation triples + graph-retrieved passages).

All nodes are best-effort: failures are logged and the pipeline continues
with whatever partial results are available.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Dict, List, Set

from langgraph.graph import END, StateGraph

from backend.agents.multi.state import MultiAgentState
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.core.tracing import trace_span
from backend.knowledge_graph.graph_store import GraphStore
from backend.knowledge_graph.entity_extractor import EntityExtractor
from backend.knowledge_graph.relation_mapper import RelationMapper
from backend.knowledge_graph.graph_retriever import GraphRetriever

logger = get_logger(__name__)


# ── Node functions ─────────────────────────────────────────────────────────

async def extract_entities(
    state: MultiAgentState,
    llm,
    graph_store: GraphStore,
    entity_extractor: EntityExtractor,
) -> Dict[str, Any]:
    """
    Extract named entities from the query and top retrieved chunks.
    Entities are upserted into the shared GraphStore.
    """
    query = state["query"]
    docs  = state.get("retrieval_results", [])
    start = time.time()

    # Combine query + first 3 doc chunks for extraction
    text = query + "\n" + "\n".join(d.get("content", "")[:400] for d in docs[:3])

    # Determine source_doc label from first available doc
    source_doc = docs[0].get("filename", "") if docs else ""

    try:
        raw_entities = await entity_extractor.extract(
            text=text,
            source_doc=source_doc,
            graph_store=graph_store,
            max_entities=settings.knowledge_agent_max_entities * 2,
        )
        entities = [
            {"text": e["text"], "label": e["entity_type"], "id": e.get("id")}
            for e in raw_entities
        ]
    except Exception as exc:
        logger.warning(f"knowledge.extract_entities failed: {exc}")
        entities = []

    elapsed = time.time() - start
    logger.info(f"knowledge.extract_entities: {len(entities)} entities ({elapsed:.2f}s)")
    return {
        "entities":      entities,
        "active_agents": ["knowledge"],
        "trace": [f"knowledge.extract_entities: {len(entities)} entities ({elapsed:.1f}s)"],
    }


async def map_relations(
    state: MultiAgentState,
    llm,
    graph_store: GraphStore,
    relation_mapper: RelationMapper,
) -> Dict[str, Any]:
    """
    Extract relation triples from the top retrieved chunks and store as
    edges in the GraphStore.
    """
    docs  = state.get("retrieval_results", [])
    start = time.time()

    total_triples = 0
    for doc in docs[:3]:
        content    = doc.get("content", "")[:800]
        source_doc = doc.get("filename", "")
        if not content:
            continue
        try:
            triples = await relation_mapper.map(
                text=content,
                source_doc=source_doc,
                graph_store=graph_store,
            )
            total_triples += len(triples)
        except Exception as exc:
            logger.warning(f"knowledge.map_relations failed for doc '{source_doc}': {exc}")

    elapsed = time.time() - start
    logger.info(f"knowledge.map_relations: {total_triples} triples ({elapsed:.2f}s)")
    return {
        "trace": [f"knowledge.map_relations: {total_triples} triples ({elapsed:.1f}s)"],
    }


async def lookup_context(
    state: MultiAgentState,
    retriever,
    graph_store: GraphStore,
    graph_retriever: GraphRetriever,
) -> Dict[str, Any]:
    """
    For each extracted entity, run:
      1. Graph-aware retrieval (neighbour expansion in GraphStore)
      2. Targeted HybridRetriever lookups for additional doc chunks
    New chunks not already in retrieval_results are appended.
    """
    entities = state.get("entities", [])
    existing: Set[str] = {d["chunk_id"] for d in state.get("retrieval_results", [])}
    start = time.time()

    new_chunks: List[Dict[str, Any]] = []
    max_ent = settings.knowledge_agent_max_entities

    for ent in entities[:max_ent]:
        entity_text = ent["text"]
        entity_id   = ent.get("id")

        # --- Graph-aware expansion -----------------------------------------
        if entity_id:
            try:
                graph_results = await graph_retriever.retrieve(
                    query=entity_text,
                    top_k=3,
                    graph_store=graph_store,
                )
                for gr in graph_results:
                    cid = gr.chunk_id
                    if cid not in existing:
                        existing.add(cid)
                        new_chunks.append({
                            "chunk_id":         cid,
                            "content":          gr.context_text,
                            "score":            gr.score,
                            "document_id":      "",
                            "filename":         gr.filename,
                            "page_number":      None,
                            "retrieval_method": "graph_expansion",
                            "entity":           entity_text,
                        })
            except Exception as exc:
                logger.warning(f"graph expansion failed for '{entity_text}': {exc}")

        # --- Vector/hybrid lookup for doc chunks ----------------------------
        try:
            results = await retriever.retrieve(
                query=entity_text, top_k=2, method="hybrid"
            )
            for doc in results:
                if doc.chunk_id not in existing:
                    existing.add(doc.chunk_id)
                    new_chunks.append({
                        "chunk_id":         doc.chunk_id,
                        "content":          doc.content,
                        "score":            doc.score,
                        "document_id":      getattr(doc, "document_id", ""),
                        "filename":         doc.filename,
                        "page_number":      doc.page_number,
                        "retrieval_method": "knowledge_lookup",
                        "entity":           entity_text,
                    })
        except Exception as exc:
            logger.warning(f"knowledge hybrid lookup failed for '{entity_text}': {exc}")

    elapsed = time.time() - start
    logger.info(f"knowledge.lookup_context: {len(new_chunks)} new chunks ({elapsed:.2f}s)")
    return {
        "retrieval_results": state.get("retrieval_results", []) + new_chunks,
        "trace": [f"knowledge.lookup_context: +{len(new_chunks)} entity chunks ({elapsed:.1f}s)"],
    }


async def build_knowledge_context(
    state: MultiAgentState,
    graph_store: GraphStore,
) -> Dict[str, Any]:
    """
    Build an enriched knowledge context string from:
      - extracted entities (with types)
      - relation triples from the GraphStore
      - entity-driven doc chunk snippets
    """
    entities = state.get("entities", [])
    kg_docs  = [
        d for d in state.get("retrieval_results", [])
        if d.get("retrieval_method") in ("knowledge_lookup", "graph_expansion")
    ]

    if not entities and not kg_docs:
        return {
            "knowledge_context": None,
            "trace": ["knowledge.build_context: nothing to add"],
        }

    # Entity list with types
    entity_lines = [
        f"{e['text']} [{e.get('label', 'ENTITY')}]"
        for e in entities[:10]
    ]
    entity_summary = ", ".join(entity_lines)

    # Relation triples from graph for top entity
    relation_lines: List[str] = []
    for ent in entities[:3]:
        eid = ent.get("id")
        if eid:
            try:
                rels = graph_store.get_relations(eid)
                for r in rels[:3]:
                    relation_lines.append(
                        f"{ent['text']} —{r['predicate']}→ {r['object_text']}"
                    )
            except Exception:
                pass

    # Entity-retrieved doc snippets
    chunk_texts = "\n".join(
        f"• [{d.get('filename', '?')}] {d['content'][:200]}"
        for d in kg_docs[:5]
    )

    ctx_parts = [f"Key entities: {entity_summary}"]
    if relation_lines:
        ctx_parts.append("Graph relations:\n" + "\n".join(relation_lines))
    if chunk_texts:
        ctx_parts.append(f"Entity-related context:\n{chunk_texts}")

    ctx = "\n\n".join(ctx_parts)
    logger.info(f"knowledge.build_context: {len(ctx)} chars")
    return {
        "knowledge_context": ctx,
        "trace": [
            f"knowledge.build_context: entities={len(entities)} "
            f"relations={len(relation_lines)} "
            f"extra_chunks={len(kg_docs)}"
        ],
    }


# ── Graph builder ──────────────────────────────────────────────────────────

class KnowledgeAgent:
    """
    LangGraph sub-graph for the Knowledge Agent (Phase 12).

    Extracts entities → maps relations → looks up context → builds
    enriched knowledge context for the downstream generator.
    """

    def __init__(self, llm, retriever, graph_store: GraphStore) -> None:
        self.llm             = llm
        self.retriever       = retriever
        self.graph_store     = graph_store
        self.entity_extractor = EntityExtractor(llm=llm)
        self.relation_mapper  = RelationMapper(llm=llm)
        self.graph_retriever  = GraphRetriever(graph_store=graph_store)
        self._graph           = self._build()
        logger.info("KnowledgeAgent compiled (Phase 12 — full KG pipeline)")

    def _build(self):
        builder = StateGraph(MultiAgentState)

        builder.add_node(
            "extract_entities",
            functools.partial(
                extract_entities,
                llm=self.llm,
                graph_store=self.graph_store,
                entity_extractor=self.entity_extractor,
            ),
        )
        builder.add_node(
            "map_relations",
            functools.partial(
                map_relations,
                llm=self.llm,
                graph_store=self.graph_store,
                relation_mapper=self.relation_mapper,
            ),
        )
        builder.add_node(
            "lookup_context",
            functools.partial(
                lookup_context,
                retriever=self.retriever,
                graph_store=self.graph_store,
                graph_retriever=self.graph_retriever,
            ),
        )
        builder.add_node(
            "build_knowledge_context",
            functools.partial(
                build_knowledge_context,
                graph_store=self.graph_store,
            ),
        )

        builder.set_entry_point("extract_entities")
        builder.add_edge("extract_entities",        "map_relations")
        builder.add_edge("map_relations",            "lookup_context")
        builder.add_edge("lookup_context",           "build_knowledge_context")
        builder.add_edge("build_knowledge_context",  END)

        return builder.compile()

    async def run(self, state: Dict[str, Any]) -> MultiAgentState:
        async with trace_span("knowledge_agent.run"):
            return await self._graph.ainvoke(state)  # type: ignore[return-value]


# Made with Bob
