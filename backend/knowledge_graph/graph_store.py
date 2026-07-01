"""
GraphStore (Phase 12) — NetworkX-backed Knowledge Graph.

Provides a persistent, queryable knowledge graph storing:
  - Entity nodes  (id, text, entity_type, source_doc, metadata)
  - Relation edges (subject, predicate, object, confidence, source_doc)

The graph is persisted as a JSON file (path from settings.kg_persist_path)
and loaded back on initialisation, so it survives server restarts.

Public API
──────────
  add_entity(text, entity_type, source_doc, metadata) → node_id
  add_relation(subject_id, predicate, object_id, confidence, source_doc) → edge_key
  get_entity(node_id) → dict | None
  get_entities(entity_type) → list[dict]
  get_relations(subject_id) → list[dict]
  get_neighbours(node_id, hops) → list[dict]
  get_subgraph(node_ids) → dict
  search_entities(query) → list[dict]
  stats() → dict
  clear()
  save() / load()
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


def _entity_id(text: str, entity_type: str) -> str:
    """Stable, URL-safe node id derived from text + type."""
    raw = f"{entity_type}::{text.lower().strip()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class GraphStore:
    """
    NetworkX-backed Knowledge Graph with JSON persistence.

    Thread-safety: this class is NOT thread-safe by itself; callers that
    share an instance across coroutines must synchronise externally.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._path = Path(persist_path or settings.kg_persist_path)
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._load_if_exists()
        logger.info(f"GraphStore initialised — {self._path} "
                    f"({self._graph.number_of_nodes()} nodes, "
                    f"{self._graph.number_of_edges()} edges)")

    # ── Entity API ────────────────────────────────────────────────────────

    def add_entity(
        self,
        text: str,
        entity_type: str,
        source_doc: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add or update an entity node.  Returns the node id.

        If a node with the same (text, entity_type) already exists, its
        source_docs list is extended (no duplicate) and the node is returned.
        """
        node_id = _entity_id(text, entity_type)
        if self._graph.has_node(node_id):
            # Merge source_docs list
            existing = self._graph.nodes[node_id]
            if source_doc and source_doc not in existing.get("source_docs", []):
                existing["source_docs"].append(source_doc)
            existing["updated_at"] = time.time()
        else:
            self._graph.add_node(
                node_id,
                text=text,
                entity_type=entity_type,
                source_docs=[source_doc] if source_doc else [],
                metadata=metadata or {},
                created_at=time.time(),
                updated_at=time.time(),
            )
        return node_id

    def get_entity(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return node attribute dict or None if not found."""
        if not self._graph.has_node(node_id):
            return None
        return {"id": node_id, **self._graph.nodes[node_id]}

    def get_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all entities, optionally filtered by type."""
        results = []
        for node_id, data in self._graph.nodes(data=True):
            if entity_type is None or data.get("entity_type") == entity_type:
                results.append({"id": node_id, **data})
        return results

    def search_entities(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Simple substring search over entity text (case-insensitive).
        Returns up to top_k matches ordered by text length (shorter = more specific).
        """
        q = query.lower().strip()
        matches = [
            {"id": nid, **data}
            for nid, data in self._graph.nodes(data=True)
            if q in data.get("text", "").lower()
        ]
        matches.sort(key=lambda x: len(x.get("text", "")))
        return matches[:top_k]

    # ── Relation API ──────────────────────────────────────────────────────

    def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float = 1.0,
        source_doc: str = "",
    ) -> Optional[int]:
        """
        Add a directed relation edge (subject --predicate--> object).
        Returns the NetworkX edge key, or None if either node is missing.
        """
        if not self._graph.has_node(subject_id) or not self._graph.has_node(object_id):
            logger.warning(
                f"add_relation: missing node(s) subject={subject_id} object={object_id}"
            )
            return None
        key = self._graph.add_edge(
            subject_id,
            object_id,
            predicate=predicate,
            confidence=confidence,
            source_doc=source_doc,
            created_at=time.time(),
        )
        return key

    def get_relations(self, subject_id: str) -> List[Dict[str, Any]]:
        """Return all outgoing relations from subject node."""
        if not self._graph.has_node(subject_id):
            return []
        relations = []
        for _, obj_id, key, data in self._graph.out_edges(subject_id, data=True, keys=True):
            obj_data = self._graph.nodes.get(obj_id, {})
            relations.append({
                "subject_id": subject_id,
                "object_id":  obj_id,
                "object_text": obj_data.get("text", obj_id),
                "predicate":  data.get("predicate", ""),
                "confidence": data.get("confidence", 1.0),
                "source_doc": data.get("source_doc", ""),
                "key":        key,
            })
        return relations

    # ── Graph Traversal ───────────────────────────────────────────────────

    def get_neighbours(
        self, node_id: str, hops: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Return all nodes reachable within `hops` steps from node_id
        (in both directions — predecessor and successor edges).
        """
        if not self._graph.has_node(node_id):
            return []
        undirected = self._graph.to_undirected(as_view=True)
        try:
            ego = nx.ego_graph(undirected, node_id, radius=hops)
        except nx.NetworkXError:
            return []
        nodes = [
            {"id": nid, **self._graph.nodes[nid]}
            for nid in ego.nodes()
            if nid != node_id and self._graph.has_node(nid)
        ]
        return nodes

    def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Return a JSON-serialisable subgraph dict for a list of node ids.

        Returns {"nodes": [...], "edges": [...]}
        """
        valid = [n for n in node_ids if self._graph.has_node(n)]
        sub = self._graph.subgraph(valid)
        nodes = [{"id": nid, **data} for nid, data in sub.nodes(data=True)]
        edges = [
            {
                "source": u,
                "target": v,
                "predicate": data.get("predicate", ""),
                "confidence": data.get("confidence", 1.0),
            }
            for u, v, data in sub.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the graph to the configured JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        logger.info(f"GraphStore saved → {self._path} "
                    f"({self._graph.number_of_nodes()}n / {self._graph.number_of_edges()}e)")

    def load(self) -> None:
        """Load the graph from the configured JSON file (overwrites in-memory state)."""
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._graph = nx.node_link_graph(data, directed=True, multigraph=True)
                logger.info(f"GraphStore loaded ← {self._path}")
            except Exception as exc:
                logger.warning(f"GraphStore: could not load {self._path}: {exc} — starting fresh")
                self._graph = nx.MultiDiGraph()

    # ── Utility ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all nodes and edges (in-memory only — call save() to persist)."""
        self._graph.clear()
        logger.info("GraphStore cleared")

    def stats(self) -> Dict[str, Any]:
        """Return a summary of the current graph state."""
        entity_types: Dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            et = data.get("entity_type", "UNKNOWN")
            entity_types[et] = entity_types.get(et, 0) + 1
        return {
            "num_entities": self._graph.number_of_nodes(),
            "num_relations": self._graph.number_of_edges(),
            "entity_types": entity_types,
            "persist_path": str(self._path),
            "is_persistent": self._path.exists(),
        }


# Module-level singleton (lazy — created on first import of this module)
_graph_store: Optional[GraphStore] = None


def get_graph_store() -> GraphStore:
    """Return (or create) the module-level GraphStore singleton."""
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
    return _graph_store


# Made with Bob
