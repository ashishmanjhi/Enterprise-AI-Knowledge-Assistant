"""
Multi-Agent Orchestrator (Phase 11) — top-level LangGraph router.

Pipeline
────────

    classify_intent
          │
    ┌─────┴──────────────────────────────────────────────────┐
    │  research (complex multi-doc)   │  retrieval (factual) │
    └─────────────┬───────────────────┴──────────┬───────────┘
                  ▼                               ▼
           research_agent                  retrieval_agent
                  │                               │
                  └──────────────┬────────────────┘
                                 ▼
                         knowledge_agent        (entity enrichment)
                                 │
                         evaluation_agent       (quality scoring)
                                 │
                         governance_agent       (safety & compliance)
                                 │
                             generate           (final answer)
                                 │
                                END

The orchestrator decides which primary pipeline to use based on the query
intent, then always passes results through knowledge → evaluation → governance
before the final generation step.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from backend.agents.multi.state import MultiAgentState
from backend.agents.multi.research_agent import ResearchAgent
from backend.agents.multi.retrieval_agent import RetrievalAgent
from backend.agents.multi.evaluation_agent import EvaluationAgent
from backend.agents.multi.governance_agent import GovernanceAgent
from backend.agents.multi.knowledge_agent import KnowledgeAgent
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.core.tracing import trace_span, langsmith_callbacks
from backend.knowledge_graph.graph_store import get_graph_store

logger = get_logger(__name__)

# ── Intent classification ─────────────────────────────────────────────────

_INTENT_PROMPT = """\
You are a query classifier for a multi-agent document QA system.

Classify the question into one of these intents:
- research   : complex question requiring multi-step investigation or synthesis across documents
- retrieval  : factual question answerable by retrieving a specific passage
- general    : conversational or simple question that doesn't need document lookup

Respond with exactly one word: research, retrieval, or general.

Question: {query}
Intent:"""

_GENERATE_PROMPT = """\
You are a helpful AI assistant.  Using the context below, provide a concise \
and accurate answer.  Cite sources by filename and page where available.

{knowledge_section}
Retrieved context:
{context}

Conversation history:
{history}

Question: {query}

Answer:"""


# ── Orchestrator nodes ────────────────────────────────────────────────────

async def classify_intent(state: MultiAgentState, llm) -> Dict[str, Any]:
    """Use the LLM to classify the query intent."""
    query = state["query"]
    try:
        result = await llm.generate(
            prompt=_INTENT_PROMPT.format(query=query),
            temperature=0.0,
            max_tokens=5,
        )
        intent = result.get("text", "").strip().lower()
        if intent not in ("research", "retrieval", "general"):
            intent = "retrieval"
    except Exception:
        intent = "retrieval"

    logger.info(f"orchestrator.classify_intent: intent={intent}")
    return {
        "intent":        intent,
        "active_agents": [],
        "trace": [f"orchestrator.classify: intent={intent}"],
    }


async def run_research_agent(state: MultiAgentState, agent: ResearchAgent) -> Dict[str, Any]:
    result = await agent.run(dict(state))
    return {k: v for k, v in result.items() if k not in ("query",)}


async def run_retrieval_agent(state: MultiAgentState, agent: RetrievalAgent) -> Dict[str, Any]:
    result = await agent.run(dict(state))
    return {k: v for k, v in result.items() if k not in ("query",)}


async def run_knowledge_agent(state: MultiAgentState, agent: KnowledgeAgent) -> Dict[str, Any]:
    result = await agent.run(dict(state))
    return {k: v for k, v in result.items() if k not in ("query",)}


async def run_evaluation_agent(state: MultiAgentState, agent: EvaluationAgent) -> Dict[str, Any]:
    result = await agent.run(dict(state))
    return {k: v for k, v in result.items() if k not in ("query",)}


async def run_governance_agent(state: MultiAgentState, agent: GovernanceAgent) -> Dict[str, Any]:
    result = await agent.run(dict(state))
    return {k: v for k, v in result.items() if k not in ("query",)}


async def generate_final_answer(state: MultiAgentState, llm) -> Dict[str, Any]:
    """
    Assemble all agent outputs and call the LLM for the final answer.
    If governance already set final_answer (e.g. blocked), skip generation.
    """
    # Governance may have already produced the final answer
    if state.get("governance_passed") is False:
        return {
            "final_response": state.get("final_answer", ""),
            "trace": ["orchestrator.generate: skipped (governance blocked)"],
        }

    # Use research summary if available, otherwise build from retrieval results
    existing = state.get("research_summary") or state.get("final_answer") or ""
    if existing:
        return {
            "final_response": existing,
            "trace": ["orchestrator.generate: using agent-generated summary"],
        }

    # Fallback — generate directly from retrieved docs
    docs    = state.get("retrieval_results", [])
    query   = state["query"]
    history = state.get("conversation_history", [])
    start   = time.time()

    context = "\n\n".join(
        f"[{i+1}] {d.get('filename','?')}: {d.get('content','')[:400]}"
        for i, d in enumerate(docs[:5])
    ) or "No context available."

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m.get('content','')}"
        for m in history[-6:]
    ) if history else "None"

    knowledge_section = ""
    kc = state.get("knowledge_context")
    if kc:
        knowledge_section = f"Knowledge context:\n{kc}\n\n"

    try:
        result = await llm.generate(
            prompt=_GENERATE_PROMPT.format(
                query=query,
                context=context,
                history=history_text,
                knowledge_section=knowledge_section,
            ),
            temperature=state.get("temperature", 0.7),
            max_tokens=state.get("max_tokens", 500),
        )
        answer = result.get("text", "").strip()
        meta_model  = result.get("model", "unknown")
        meta_tokens = result.get("tokens_used", 0)
    except Exception as exc:
        logger.error(f"orchestrator.generate_final_answer failed: {exc}")
        answer      = "I'm sorry, I was unable to generate a response at this time."
        meta_model  = "unknown"
        meta_tokens = 0

    elapsed = time.time() - start
    return {
        "final_response": answer,
        "metadata": {
            **(state.get("metadata") or {}),
            "model":        meta_model,
            "tokens_used":  meta_tokens,
            "gen_time":     round(elapsed, 3),
        },
        "trace": [f"orchestrator.generate: {len(answer)} chars in {elapsed:.1f}s"],
    }


# ── Conditional routing ───────────────────────────────────────────────────

def _route_by_intent(state: MultiAgentState) -> str:
    intent = state.get("intent", "retrieval")
    if intent == "research":
        return "research"
    return "retrieval"   # "general" falls through to retrieval (lighter path)


# ── Orchestrator graph ────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    Top-level LangGraph orchestrator that wires all five sub-agents.

    Parameters
    ----------
    llm       : LLMService
    retriever : HybridRetriever
    reranker  : CrossEncoderReranker | None  (optional)
    """

    def __init__(self, llm, retriever, reranker=None) -> None:
        self.llm        = llm
        self.retriever  = retriever
        self.reranker   = reranker

        # Instantiate sub-agents
        self._research   = ResearchAgent(llm, retriever)
        self._retrieval  = RetrievalAgent(llm, retriever, reranker)
        self._evaluation = EvaluationAgent(llm)
        self._governance = GovernanceAgent()
        self._knowledge  = KnowledgeAgent(llm, retriever, graph_store=get_graph_store())

        self._graph = self._build()
        logger.info("MultiAgentOrchestrator compiled")

    def _build(self):
        builder = StateGraph(MultiAgentState)

        # ── Nodes ────────────────────────────────────────────────────────
        builder.add_node("classify",    functools.partial(classify_intent,        llm=self.llm))
        builder.add_node("research",    functools.partial(run_research_agent,     agent=self._research))
        builder.add_node("retrieval",   functools.partial(run_retrieval_agent,    agent=self._retrieval))
        builder.add_node("knowledge",   functools.partial(run_knowledge_agent,    agent=self._knowledge))
        builder.add_node("evaluation",  functools.partial(run_evaluation_agent,   agent=self._evaluation))
        builder.add_node("governance",  functools.partial(run_governance_agent,   agent=self._governance))
        builder.add_node("generate",    functools.partial(generate_final_answer,  llm=self.llm))

        # ── Edges ────────────────────────────────────────────────────────
        builder.set_entry_point("classify")

        # Intent-based fork: research vs retrieval
        builder.add_conditional_edges(
            "classify",
            _route_by_intent,
            {"research": "research", "retrieval": "retrieval"},
        )

        # Both paths converge on knowledge → evaluation → governance → generate
        builder.add_edge("research",   "knowledge")
        builder.add_edge("retrieval",  "knowledge")
        builder.add_edge("knowledge",  "evaluation")
        builder.add_edge("evaluation", "governance")
        builder.add_edge("governance", "generate")
        builder.add_edge("generate",   END)

        return builder.compile()

    async def run(self, initial_state: Dict[str, Any]) -> MultiAgentState:
        """
        Run the full multi-agent pipeline.

        Parameters
        ----------
        initial_state :
            Must contain at least ``{"query": "<user question>"}``.
        """
        query = initial_state.get("query", "")
        logger.info(f"MultiAgentOrchestrator.run: query='{query[:80]}'")

        async with trace_span("multi_agent.run", {"query": query[:120]}):
            config: Dict[str, Any] = {}
            cbs = langsmith_callbacks()
            if cbs:
                config["callbacks"] = cbs
            state: MultiAgentState = await self._graph.ainvoke(  # type: ignore[assignment]
                initial_state, config=config or None
            )
        return state


# Made with Bob
