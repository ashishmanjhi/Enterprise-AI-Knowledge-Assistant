"""
Agentic RAG Graph (Phase 9) — LangGraph state machine.

Flow
────
                         ┌─────────────────────────┐
                         │         START            │
                         └────────────┬────────────┘
                                      ▼
                               route_query
                                      │
                         ┌────────────▼────────────┐
                         │   should_rewrite?        │
                         │   rewrite_count < max    │
                         └──── yes ──┬──── no ──────┘
                                     ▼               │
                              rewrite_query          │
                                     │               │
                                     └──────┬────────┘
                                            ▼
                                         retrieve
                                            │
                                     grade_documents
                                            │
                                         generate
                                            │
                                     check_grounding
                                            │
                         ┌──────────────────▼─────────────────┐
                         │   grounded or max rewrites reached? │
                         └── no (re-route) ──┬──── yes ────────┘
                                             ▼
                                            END

Usage
─────
    from backend.agents.graph import AgentGraph

    graph = AgentGraph(llm=llm, retriever=retriever)
    result = await graph.run(query="What is X?")
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from backend.agents.state import AgentState
from backend.agents import nodes
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.core.tracing import trace_span, langsmith_callbacks

logger = get_logger(__name__)


# ── Conditional edge helpers ──────────────────────────────────────────────

def _decide_rewrite(state: AgentState) -> str:
    """
    After route_query: decide whether to rewrite the query first.

    If the query hasn't been rewritten yet AND we haven't hit the max
    rewrite limit, go to rewrite_query.  Otherwise go straight to retrieve.
    """
    count = state.get("rewrite_count", 0)
    already_rewritten = bool(state.get("rewritten_query"))
    if not already_rewritten and count < settings.agent_max_rewrites:
        return "rewrite_query"
    return "retrieve"


def _decide_after_grounding(state: AgentState) -> str:
    """
    After check_grounding: if the answer is grounded (or we've run out of
    rewrite budget) finish; otherwise loop back to route_query for another
    attempt.
    """
    grounded = state.get("is_grounded", True)
    rewrites  = state.get("rewrite_count", 0)
    if grounded or rewrites >= settings.agent_max_rewrites:
        return END
    return "route_query"


# ── Graph builder ─────────────────────────────────────────────────────────

class AgentGraph:
    """
    Compiled LangGraph for agentic RAG.

    Parameters
    ----------
    llm :
        Any object with an async ``generate(prompt, temperature, max_tokens)``
        method that returns ``{"text": str, "model": str, "tokens_used": int}``.
        Typically an ``LLMService`` instance.
    retriever :
        An object with an async ``retrieve(query, top_k, method)`` method that
        returns a list of ``HybridRetrievalResult``-like objects.
        Typically a ``HybridRetriever`` instance.
    """

    def __init__(self, llm, retriever) -> None:
        self.llm       = llm
        self.retriever = retriever
        self._graph    = self._build()
        logger.info("AgentGraph compiled successfully")

    # ── Internal build ────────────────────────────────────────────────────

    def _build(self):
        """Construct and compile the StateGraph."""
        builder = StateGraph(AgentState)

        # Bind dependencies into each node function
        _route    = functools.partial(nodes.route_query,    llm=self.llm)
        _rewrite  = functools.partial(nodes.rewrite_query,  llm=self.llm)
        _retrieve = functools.partial(nodes.retrieve,       retriever=self.retriever)
        _grade    = functools.partial(nodes.grade_documents, llm=self.llm)
        _generate = functools.partial(nodes.generate,        llm=self.llm)
        _ground   = functools.partial(nodes.check_grounding, llm=self.llm)

        # Register nodes
        builder.add_node("route_query",      _route)
        builder.add_node("rewrite_query",    _rewrite)
        builder.add_node("retrieve",         _retrieve)
        builder.add_node("grade_documents",  _grade)
        builder.add_node("generate",         _generate)
        builder.add_node("check_grounding",  _ground)

        # Entry point
        builder.set_entry_point("route_query")

        # Conditional: after routing, optionally rewrite the query
        builder.add_conditional_edges(
            "route_query",
            _decide_rewrite,
            {"rewrite_query": "rewrite_query", "retrieve": "retrieve"},
        )

        # After rewriting, always retrieve
        builder.add_edge("rewrite_query",   "retrieve")
        builder.add_edge("retrieve",        "grade_documents")
        builder.add_edge("grade_documents", "generate")
        builder.add_edge("generate",        "check_grounding")

        # Conditional: after grounding check, either finish or loop
        builder.add_conditional_edges(
            "check_grounding",
            _decide_after_grounding,
            {"route_query": "route_query", END: END},
        )

        return builder.compile()

    # ── Public API ────────────────────────────────────────────────────────

    async def run(self, initial_state: Dict[str, Any]) -> AgentState:
        """
        Execute the graph and return the final state.

        Parameters
        ----------
        initial_state :
            Must contain at least ``{"query": "<user question>"}``.
            Optional keys: ``top_k``, ``temperature``, ``max_tokens``,
            ``retrieval_method``, ``use_reranking``, ``conversation_history``.
        """
        query = initial_state.get("query", "")
        logger.info(f"AgentGraph.run: query='{query[:80]}'")
        async with trace_span("agent.run", {"query": query[:120]}):
            config: Dict[str, Any] = {}
            cbs = langsmith_callbacks()
            if cbs:
                config["callbacks"] = cbs
            final_state: AgentState = await self._graph.ainvoke(  # type: ignore[assignment]
                initial_state, config=config or None
            )
        return final_state


# Made with Bob
