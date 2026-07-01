"""
backend.knowledge_graph (Phase 12) — Knowledge Graph Enhancement.

Package exports:
  GraphStore             NetworkX KG with entity nodes + relation edges
  EntityExtractor        LLM + regex NER → typed entities
  RelationMapper         LLM relation triple extraction
  GraphRetriever         Neighbour-expansion graph retrieval
  HybridGraphRetriever   RRF fusion of vector + graph results
  get_graph_store        Module-level GraphStore singleton factory
"""

from backend.knowledge_graph.graph_store import GraphStore, get_graph_store
from backend.knowledge_graph.entity_extractor import EntityExtractor
from backend.knowledge_graph.relation_mapper import RelationMapper
from backend.knowledge_graph.graph_retriever import GraphRetriever, GraphRetrievalResult
from backend.knowledge_graph.hybrid_graph_retriever import (
    HybridGraphRetriever,
    HybridGraphResult,
)

__all__ = [
    "GraphStore",
    "get_graph_store",
    "EntityExtractor",
    "RelationMapper",
    "GraphRetriever",
    "GraphRetrievalResult",
    "HybridGraphRetriever",
    "HybridGraphResult",
]

# Made with Bob
