"""
HybridGraphRetriever (Phase 12) — RRF fusion of vector + graph results.

Combines:
  - Vector retrieval (FAISS + BM25 hybrid via HybridRetriever)
  - Graph-aware retrieval (entity neighbourhood expansion via GraphRetriever)

Results are fused using Reciprocal Rank Fusion (RRF) with configurable
per-source weights (settings.kg_hybrid_vector_weight / kg_hybrid_graph_weight).

Public API
──────────
  async retrieve(query, top_k, method) → list[HybridGraphResult]
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


# ── Result type ────────────────────────────────────────────────────────────

class HybridGraphResult:
    """
    Fused retrieval result combining vector and graph signals.
    """

    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        source: str,           # "vector" | "graph" | "both"
        filename: str = "",
        page_number: Optional[int] = None,
        entity_type: Optional[str] = None,
        vector_rank: Optional[int] = None,
        graph_rank: Optional[int]  = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.chunk_id    = chunk_id
        self.content     = content
        self.score       = score
        self.source      = source
        self.filename    = filename
        self.page_number = page_number
        self.entity_type = entity_type
        self.vector_rank = vector_rank
        self.graph_rank  = graph_rank
        self.metadata    = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id":    self.chunk_id,
            "content":     self.content,
            "score":       self.score,
            "source":      self.source,
            "filename":    self.filename,
            "page_number": self.page_number,
            "entity_type": self.entity_type,
            "vector_rank": self.vector_rank,
            "graph_rank":  self.graph_rank,
            "metadata":    self.metadata,
        }


# ── HybridGraphRetriever ───────────────────────────────────────────────────

class HybridGraphRetriever:
    """
    Fuses vector retrieval and graph retrieval using weighted Reciprocal Rank Fusion.

    Instantiate with a HybridRetriever (vector) and a GraphRetriever (graph).
    Both are optional; if one is absent only the other is used.
    """

    def __init__(
        self,
        vector_retriever=None,
        graph_retriever=None,
    ) -> None:
        """
        Args:
            vector_retriever: HybridRetriever instance (Phase 2).
            graph_retriever:  GraphRetriever instance (Phase 12).
        """
        self._vector = vector_retriever
        self._graph  = graph_retriever
        logger.info(
            f"HybridGraphRetriever initialised — "
            f"vector={'enabled' if vector_retriever else 'disabled'}, "
            f"graph={'enabled' if graph_retriever else 'disabled'}"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        method: str = "hybrid",
    ) -> List[HybridGraphResult]:
        """
        Run both retrievers in parallel and fuse results with RRF.

        Args:
            query:   Natural-language query.
            top_k:   Maximum results to return.
            method:  Retrieval method hint passed to the vector retriever.

        Returns:
            List of HybridGraphResult ranked by fused score.
        """
        fetch_k = top_k * 2  # fetch more then cut to top_k after fusion

        vector_task = self._run_vector(query, fetch_k, method)
        graph_task  = self._run_graph(query, fetch_k)

        vector_results, graph_results = await asyncio.gather(
            vector_task, graph_task, return_exceptions=False
        )

        fused = self._rrf_fuse(vector_results, graph_results)
        return fused[:top_k]

    # ── Internal ──────────────────────────────────────────────────────────

    async def _run_vector(
        self, query: str, top_k: int, method: str
    ) -> List[Any]:
        if self._vector is None:
            return []
        try:
            return await self._vector.retrieve(query=query, top_k=top_k, method=method)
        except Exception as exc:
            logger.warning(f"HybridGraphRetriever vector error: {exc}")
            return []

    async def _run_graph(self, query: str, top_k: int) -> List[Any]:
        if self._graph is None:
            return []
        try:
            return await self._graph.retrieve(query=query, top_k=top_k)
        except Exception as exc:
            logger.warning(f"HybridGraphRetriever graph error: {exc}")
            return []

    def _rrf_fuse(
        self,
        vector_results: List[Any],
        graph_results: List[Any],
    ) -> List[HybridGraphResult]:
        """
        Reciprocal Rank Fusion with per-source weights.

        rrf_score(d) = Σ weight_s / (k + rank_in_s(d))
        where k = settings.rrf_k
        """
        k          = settings.rrf_k
        v_weight   = settings.kg_hybrid_vector_weight
        g_weight   = settings.kg_hybrid_graph_weight

        scores: Dict[str, float] = {}
        result_map: Dict[str, HybridGraphResult] = {}

        # Vector pass
        for rank, r in enumerate(vector_results):
            cid = getattr(r, "chunk_id", str(rank))
            inc = v_weight / (k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + inc

            if cid not in result_map:
                result_map[cid] = HybridGraphResult(
                    chunk_id    = cid,
                    content     = getattr(r, "content", ""),
                    score       = 0.0,
                    source      = "vector",
                    filename    = getattr(r, "filename", ""),
                    page_number = getattr(r, "page_number", None),
                    vector_rank = rank + 1,
                )
            else:
                result_map[cid].vector_rank = rank + 1
                result_map[cid].source = "both"

        # Graph pass
        for rank, r in enumerate(graph_results):
            cid = getattr(r, "entity_id", getattr(r, "chunk_id", str(rank)))
            inc = g_weight / (k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + inc

            if cid not in result_map:
                result_map[cid] = HybridGraphResult(
                    chunk_id    = cid,
                    content     = getattr(r, "context_text", getattr(r, "content", "")),
                    score       = 0.0,
                    source      = "graph",
                    filename    = getattr(r, "filename", ""),
                    entity_type = getattr(r, "entity_type", None),
                    graph_rank  = rank + 1,
                )
            else:
                result_map[cid].graph_rank = rank + 1
                result_map[cid].source = "both"

        # Assign fused scores
        for cid, r in result_map.items():
            r.score = scores.get(cid, 0.0)

        # Sort by fused score descending
        ranked = sorted(result_map.values(), key=lambda x: x.score, reverse=True)
        logger.debug(
            f"HybridGraphRetriever fused {len(vector_results)}v + "
            f"{len(graph_results)}g → {len(ranked)} results"
        )
        return ranked


# Made with Bob
