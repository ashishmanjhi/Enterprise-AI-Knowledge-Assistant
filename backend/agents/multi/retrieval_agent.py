"""
Retrieval Agent (Phase 11) — Multi-Agent Ecosystem.

The Retrieval Agent is a specialist in choosing and executing the best
retrieval strategy for a given query.  It:

  1. Analyses the query to select the optimal method (hybrid / faiss / bm25)
  2. Expands the query with synonyms / paraphrases for better recall
  3. Runs parallel multi-strategy retrieval and fuses the results
  4. Re-ranks the combined result set using the cross-encoder

The retrieved results are stored in ``state["retrieval_results"]`` for
downstream agents (Evaluation, Governance, Research).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from backend.agents.multi.state import MultiAgentState
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.core.tracing import trace_span

logger = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────

_STRATEGY_PROMPT = """\
You are a retrieval strategy selector.
Given the query below, choose the best primary retrieval method.

- hybrid  : use for most questions — combines semantic + keyword search
- faiss   : prefer when the question is conceptual or needs paraphrasing
- bm25    : prefer when the question contains exact technical terms or codes

Respond with exactly one word: hybrid, faiss, or bm25.

Query: {query}
Strategy:"""

_EXPAND_PROMPT = """\
Generate {n} short alternative phrasings of the query below that capture \
the same meaning but use different words.  Return one per line, no numbering.

Query: {query}
Alternatives:"""


# ── Node functions ────────────────────────────────────────────────────────

async def select_strategy(state: MultiAgentState, llm) -> Dict[str, Any]:
    """Ask the LLM to pick the best retrieval strategy."""
    query = state["query"]
    try:
        result = await llm.generate(
            prompt=_STRATEGY_PROMPT.format(query=query),
            temperature=0.0,
            max_tokens=5,
        )
        strategy = result.get("text", "").strip().lower()
        if strategy not in ("hybrid", "faiss", "bm25"):
            strategy = settings.default_retrieval_method
    except Exception:
        strategy = settings.default_retrieval_method

    return {
        "retrieval_strategy": strategy,
        "active_agents": ["retrieval"],
        "trace": [f"retrieval.select_strategy: {strategy}"],
    }


async def expand_and_retrieve(state: MultiAgentState, llm, retriever) -> Dict[str, Any]:
    """
    Expand the query into N variants, retrieve for each, then fuse & deduplicate.
    """
    query    = state["query"]
    strategy = state.get("retrieval_strategy", settings.default_retrieval_method)
    top_k    = state.get("top_k", settings.top_k_retrieval)
    n_expand = settings.retrieval_agent_expansions
    start    = time.time()

    # Generate expansions
    expansions = [query]
    try:
        result = await llm.generate(
            prompt=_EXPAND_PROMPT.format(query=query, n=n_expand),
            temperature=0.5,
            max_tokens=150,
        )
        lines = [ln.strip() for ln in result.get("text", "").splitlines() if ln.strip()]
        expansions.extend(lines[:n_expand])
    except Exception as exc:
        logger.warning(f"retrieval.expand failed ({exc}); using original query only")

    # Retrieve in parallel for all expansions
    async def _retrieve_one(q: str):
        try:
            return await retriever.retrieve(query=q, top_k=top_k, method=strategy)
        except Exception as exc:
            logger.warning(f"retrieval failed for '{q[:40]}': {exc}")
            return []

    all_lists = await asyncio.gather(*[_retrieve_one(q) for q in expansions])

    # Deduplicate by chunk_id, keep highest score
    seen: Dict[str, Any] = {}
    for result_list in all_lists:
        for doc in result_list:
            cid = doc.chunk_id
            if cid not in seen or doc.score > seen[cid].score:
                seen[cid] = doc

    # Sort by score and cap
    merged = sorted(seen.values(), key=lambda d: d.score, reverse=True)[:top_k * 2]

    # Serialize to plain dicts
    docs = [
        {
            "chunk_id":         d.chunk_id,
            "content":          d.content,
            "score":            d.score,
            "document_id":      d.document_id,
            "filename":         d.filename,
            "page_number":      d.page_number,
            "retrieval_method": getattr(d, "retrieval_method", strategy),
            "faiss_score":      getattr(d, "faiss_score", None),
            "bm25_score":       getattr(d, "bm25_score", None),
        }
        for d in merged
    ]

    elapsed = time.time() - start
    logger.info(
        f"retrieval.expand_and_retrieve: {len(expansions)} expansions, "
        f"{len(docs)} unique docs in {elapsed:.2f}s"
    )
    return {
        "retrieval_results": docs,
        "trace": [
            f"retrieval.expand_and_retrieve: expansions={len(expansions)} "
            f"docs={len(docs)} strategy={strategy} time={elapsed:.1f}s"
        ],
    }


async def rerank_results(state: MultiAgentState, reranker) -> Dict[str, Any]:
    """Optional cross-encoder reranking of the merged result set."""
    if reranker is None:
        return {"trace": ["retrieval.rerank: skipped (no reranker)"]}

    query   = state["query"]
    docs    = state.get("retrieval_results", [])
    if not docs:
        return {"trace": ["retrieval.rerank: skipped (no docs)"]}

    start = time.time()
    try:
        # Convert plain dicts back to lightweight objects the reranker accepts
        class _Doc:
            def __init__(self, d):
                self.__dict__.update(d)

        reranked = reranker.rerank(query=query, results=[_Doc(d) for d in docs])
        # Serialize back to dicts
        reranked_dicts = []
        for d in reranked:
            entry = d.__dict__.copy() if hasattr(d, "__dict__") else dict(d)
            reranked_dicts.append(entry)
    except Exception as exc:
        logger.warning(f"retrieval.rerank failed ({exc}); keeping original order")
        reranked_dicts = docs

    elapsed = time.time() - start
    logger.info(f"retrieval.rerank: {len(reranked_dicts)} docs kept ({elapsed:.2f}s)")
    return {
        "retrieval_results": reranked_dicts,
        "trace": [f"retrieval.rerank: {len(reranked_dicts)} docs in {elapsed:.1f}s"],
    }


# ── Graph builder ─────────────────────────────────────────────────────────

class RetrievalAgent:
    """
    LangGraph sub-graph for the Retrieval Agent.

    Parameters
    ----------
    llm       : LLMService
    retriever : HybridRetriever
    reranker  : CrossEncoderReranker | None
    """

    def __init__(self, llm, retriever, reranker=None) -> None:
        self.llm       = llm
        self.retriever = retriever
        self.reranker  = reranker
        self._graph    = self._build()
        logger.info("RetrievalAgent compiled")

    def _build(self):
        import functools
        builder = StateGraph(MultiAgentState)

        builder.add_node("select_strategy",    functools.partial(select_strategy,     llm=self.llm))
        builder.add_node("expand_and_retrieve", functools.partial(expand_and_retrieve, llm=self.llm, retriever=self.retriever))
        builder.add_node("rerank",              functools.partial(rerank_results,      reranker=self.reranker))

        builder.set_entry_point("select_strategy")
        builder.add_edge("select_strategy",     "expand_and_retrieve")
        builder.add_edge("expand_and_retrieve", "rerank")
        builder.add_edge("rerank",              END)

        return builder.compile()

    async def run(self, state: Dict[str, Any]) -> MultiAgentState:
        async with trace_span("retrieval_agent.run", {"query": state.get("query", "")[:80]}):
            return await self._graph.ainvoke(state)  # type: ignore[return-value]


# Made with Bob
