"""
GraphRetriever (Phase 12) — Graph-aware retrieval.

Given a natural-language query, the GraphRetriever:
  1. Searches the GraphStore for entities matching query terms.
  2. Expands each match by traversing neighbour nodes (configurable hops).
  3. Collects relation context (triples involving the matched entities).
  4. Returns a ranked list of GraphRetrievalResult objects.

Public API
──────────
  async retrieve(query, top_k, graph_store) → list[GraphRetrievalResult]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class GraphRetrievalResult:
    """
    A single result from the GraphRetriever.

    Each result represents an entity node (and its surrounding graph context)
    that is relevant to the query.
    """

    def __init__(
        self,
        entity_id: str,
        entity_text: str,
        entity_type: str,
        score: float,
        source_docs: List[str],
        neighbours: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        context_text: str,
    ) -> None:
        self.entity_id   = entity_id
        self.entity_text = entity_text
        self.entity_type = entity_type
        self.score       = score
        self.source_docs = source_docs
        self.neighbours  = neighbours
        self.relations   = relations
        self.context_text = context_text

        # Convenience aliases used by HybridGraphRetriever and the KG route
        self.chunk_id  = entity_id
        self.content   = context_text
        self.filename  = source_docs[0] if source_docs else ""
        self.page_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id":    self.entity_id,
            "entity_text":  self.entity_text,
            "entity_type":  self.entity_type,
            "score":        self.score,
            "source_docs":  self.source_docs,
            "neighbours":   self.neighbours,
            "relations":    self.relations,
            "context_text": self.context_text,
        }


class GraphRetriever:
    """
    Retrieves graph context for a query from a GraphStore.

    Matching is performed by substring-searching entity texts.  Each match
    is expanded by traversing neighbour nodes up to `settings.kg_neighbour_hops`
    hops.  The final score is a combination of:
      - direct match bonus (1.0)
      - neighbour proximity decay (1 / (hop_distance + 1))
    """

    def __init__(self, graph_store=None) -> None:
        """
        Args:
            graph_store: GraphStore instance.  If None, retrieve() returns [].
        """
        self._graph_store = graph_store
        logger.info("GraphRetriever initialised")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        graph_store=None,
    ) -> List[GraphRetrievalResult]:
        """
        Retrieve graph-context results for a query.

        Args:
            query:       Natural-language query string.
            top_k:       Maximum results to return.
            graph_store: Optional override for the instance graph_store.

        Returns:
            List of GraphRetrievalResult objects, ranked by score descending.
        """
        gs = graph_store or self._graph_store
        if gs is None:
            logger.warning("GraphRetriever: no graph_store — returning empty results")
            return []

        # 1. Find directly matching entities
        direct_hits = gs.search_entities(query, top_k=top_k * 2)
        if not direct_hits:
            return []

        hops  = settings.kg_neighbour_hops
        max_n = settings.kg_max_subgraph_nodes

        # 2. Expand each direct hit with neighbours
        seen_ids: set = set()
        scored: Dict[str, float] = {}
        entity_map: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(direct_hits[:top_k]):
            node_id = hit["id"]
            # Direct match score: decays with rank
            base_score = 1.0 / (rank + 1)
            scored[node_id] = scored.get(node_id, 0.0) + base_score
            entity_map[node_id] = hit
            seen_ids.add(node_id)

            neighbours = gs.get_neighbours(node_id, hops=hops)
            for hop_i, nbr in enumerate(neighbours[: max_n]):
                nbr_id = nbr["id"]
                nbr_score = base_score * (0.5 ** (hop_i + 1))
                scored[nbr_id] = scored.get(nbr_id, 0.0) + nbr_score
                entity_map[nbr_id] = nbr
                seen_ids.add(nbr_id)

        # 3. Build results for top-scoring nodes
        ranked_ids = sorted(scored, key=lambda k: scored[k], reverse=True)[:top_k]
        results: List[GraphRetrievalResult] = []

        for node_id in ranked_ids:
            node_data  = entity_map.get(node_id, {})
            relations  = gs.get_relations(node_id)
            neighbours = gs.get_neighbours(node_id, hops=1)

            context = self._build_context(node_data, relations, neighbours)

            results.append(GraphRetrievalResult(
                entity_id   = node_id,
                entity_text = node_data.get("text", node_id),
                entity_type = node_data.get("entity_type", "UNKNOWN"),
                score       = scored[node_id],
                source_docs = node_data.get("source_docs", []),
                neighbours  = neighbours,
                relations   = relations,
                context_text = context,
            ))

        logger.debug(f"GraphRetriever: {len(results)} results for query='{query[:60]}'")
        return results

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(
        node_data: Dict[str, Any],
        relations: List[Dict[str, Any]],
        neighbours: List[Dict[str, Any]],
    ) -> str:
        """
        Build a compact natural-language context string for an entity node.
        """
        entity_text = node_data.get("text", "")
        entity_type = node_data.get("entity_type", "")
        source_docs = node_data.get("source_docs", [])

        lines: List[str] = []
        lines.append(f"Entity: {entity_text} [{entity_type}]")

        if source_docs:
            lines.append(f"Sources: {', '.join(source_docs[:3])}")

        if relations:
            rel_strs = [
                f"{entity_text} {r['predicate']} {r['object_text']}"
                for r in relations[:5]
            ]
            lines.append("Relations: " + "; ".join(rel_strs))

        if neighbours:
            nbr_strs = [
                f"{n.get('text', n['id'])} ({n.get('entity_type', '')})"
                for n in neighbours[:5]
            ]
            lines.append("Neighbours: " + ", ".join(nbr_strs))

        return "\n".join(lines)


# Made with Bob
