"""
Knowledge Graph API routes (Phase 12).

Endpoints
─────────
  POST   /api/v1/kg/build       — ingest text and build/update the graph
  GET    /api/v1/kg/query       — entity + subgraph query
  GET    /api/v1/kg/stats       — graph statistics
  GET    /api/v1/kg/entities    — list entities (optionally filtered by type)
  GET    /api/v1/kg/relations   — list relations for a given entity_id
  DELETE /api/v1/kg/clear       — wipe the entire knowledge graph
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.knowledge_graph.graph_store import GraphStore, get_graph_store
from backend.knowledge_graph.entity_extractor import EntityExtractor
from backend.knowledge_graph.relation_mapper import RelationMapper
from backend.knowledge_graph.graph_retriever import GraphRetriever
from backend.llm.llm_service import LLMService
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/kg", tags=["knowledge-graph"])


# ── Lazy singletons ────────────────────────────────────────────────────────

_llm: Optional[LLMService] = None
_entity_extractor: Optional[EntityExtractor] = None
_relation_mapper: Optional[RelationMapper] = None
_graph_retriever: Optional[GraphRetriever] = None


def _get_llm() -> LLMService:
    global _llm
    if _llm is None:
        _llm = LLMService()
    return _llm


def _get_entity_extractor() -> EntityExtractor:
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor(llm=_get_llm())
    return _entity_extractor


def _get_relation_mapper() -> RelationMapper:
    global _relation_mapper
    if _relation_mapper is None:
        _relation_mapper = RelationMapper(llm=_get_llm())
    return _relation_mapper


def _get_graph_retriever() -> GraphRetriever:
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever(graph_store=get_graph_store())
    return _graph_retriever


# ── Request / Response models ──────────────────────────────────────────────

class BuildRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to extract entities and relations from")
    source_doc: str = Field(default="", description="Document identifier / filename")
    extract_relations: bool = Field(default=True, description="Also extract relation triples")
    persist: bool = Field(default=True, description="Persist the graph to disk after build")


class BuildResponse(BaseModel):
    entities_added: int
    relations_added: int
    elapsed_seconds: float
    graph_stats: Dict[str, Any]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100)


class EntityResponse(BaseModel):
    id: str
    text: str
    entity_type: str
    source_docs: List[str]
    metadata: Dict[str, Any]


class RelationResponse(BaseModel):
    subject_id: str
    object_id: str
    object_text: str
    predicate: str
    confidence: float
    source_doc: str


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/build", response_model=BuildResponse, summary="Build / update the knowledge graph")
async def build_kg(req: BuildRequest) -> BuildResponse:
    """
    Extract entities and (optionally) relation triples from the provided text
    and add them to the knowledge graph.  Persists to disk when `persist=True`.
    """
    gs       = get_graph_store()
    extractor = _get_entity_extractor()
    mapper    = _get_relation_mapper()
    start    = time.time()

    # Entity extraction
    entities = await extractor.extract(
        text=req.text,
        source_doc=req.source_doc,
        graph_store=gs,
    )

    # Relation extraction
    relations: List[Dict[str, Any]] = []
    if req.extract_relations:
        relations = await mapper.map(
            text=req.text,
            source_doc=req.source_doc,
            graph_store=gs,
        )

    if req.persist:
        gs.save()

    elapsed = time.time() - start
    logger.info(
        f"KG build: +{len(entities)} entities, +{len(relations)} relations, "
        f"persist={req.persist}, elapsed={elapsed:.2f}s"
    )
    return BuildResponse(
        entities_added=len(entities),
        relations_added=len(relations),
        elapsed_seconds=round(elapsed, 3),
        graph_stats=gs.stats(),
    )


@router.get("/query", summary="Query the knowledge graph")
async def query_kg(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100),
) -> Dict[str, Any]:
    """
    Search for entities matching the query and return their subgraph context.
    """
    gs       = get_graph_store()
    retriever = _get_graph_retriever()
    start    = time.time()

    results = await retriever.retrieve(query=q, top_k=top_k, graph_store=gs)
    elapsed = time.time() - start

    return {
        "query":   q,
        "results": [r.to_dict() for r in results],
        "count":   len(results),
        "elapsed": round(elapsed, 3),
    }


@router.get("/stats", summary="Knowledge graph statistics")
async def kg_stats() -> Dict[str, Any]:
    """Return counts and metadata about the current knowledge graph."""
    return get_graph_store().stats()


@router.get("/entities", response_model=List[EntityResponse], summary="List entities")
async def list_entities(
    entity_type: Optional[str] = Query(
        default=None,
        description=f"Filter by type — one of: {', '.join(settings.kg_entity_types)}",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
) -> List[EntityResponse]:
    """
    Return entities stored in the knowledge graph, optionally filtered by type.
    """
    gs       = get_graph_store()
    entities = gs.get_entities(entity_type=entity_type)[:limit]
    return [
        EntityResponse(
            id=e["id"],
            text=e.get("text", ""),
            entity_type=e.get("entity_type", ""),
            source_docs=e.get("source_docs", []),
            metadata=e.get("metadata", {}),
        )
        for e in entities
    ]


@router.get("/relations", response_model=List[RelationResponse], summary="List relations for entity")
async def list_relations(
    entity_id: str = Query(..., description="Entity node ID to fetch relations for"),
) -> List[RelationResponse]:
    """Return all outgoing relation edges for the given entity node."""
    gs        = get_graph_store()
    entity    = gs.get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")
    relations = gs.get_relations(entity_id)
    return [
        RelationResponse(
            subject_id  = r["subject_id"],
            object_id   = r["object_id"],
            object_text = r["object_text"],
            predicate   = r["predicate"],
            confidence  = r["confidence"],
            source_doc  = r["source_doc"],
        )
        for r in relations
    ]


@router.delete("/clear", summary="Clear the knowledge graph")
async def clear_kg(persist: bool = Query(default=True)) -> Dict[str, Any]:
    """
    Remove all entities and relations from the knowledge graph.
    When `persist=True` (default) the empty graph is written to disk.
    """
    gs = get_graph_store()
    gs.clear()
    if persist:
        gs.save()
    logger.info(f"KG cleared, persist={persist}")
    return {"message": "Knowledge graph cleared", "persist": persist}


# Made with Bob
