"""
Evaluation Agent (Phase 11) — Multi-Agent Ecosystem.

The Evaluation Agent runs automatic quality checks on the answer that was
produced by the Retrieval / Research agents and scores it on three fast
heuristic metrics before optionally calling the RAGAS evaluator for full
LLM-as-judge scoring.

Metrics (always computed — no external calls):
  faithfulness_heuristic  — token overlap between answer and context chunks
  answer_length_ok        — answer is between 20 and 2000 characters
  has_sources             — at least one source chunk was retrieved

Optional (when eval_agent_use_ragas=true AND ragas is installed):
  ragas_faithfulness, ragas_answer_relevancy
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from backend.agents.multi.state import MultiAgentState
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.core.tracing import trace_span

logger = get_logger(__name__)


# ── Heuristic scorers ─────────────────────────────────────────────────────

def _token_overlap(answer: str, contexts: List[str]) -> float:
    """Normalised token overlap between answer and concatenated context."""
    if not answer or not contexts:
        return 0.0
    answer_tokens  = set(re.findall(r"\w+", answer.lower()))
    context_tokens = set(re.findall(r"\w+", " ".join(contexts).lower()))
    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens | context_tokens)


# ── Node functions ────────────────────────────────────────────────────────

async def heuristic_eval(state: MultiAgentState) -> Dict[str, Any]:
    """Run fast, local heuristic quality checks — no LLM calls."""
    answer  = state.get("final_response") or state.get("research_summary", "")
    docs    = state.get("retrieval_results", []) or state.get("research_findings", [])
    start   = time.time()

    contexts = [d.get("content", "") or d.get("answer", "") for d in docs]

    scores: Dict[str, float] = {
        "faithfulness_heuristic": round(_token_overlap(answer, contexts), 3),
        "answer_length_ok":       float(20 <= len(answer) <= 2000),
        "has_sources":            float(len(docs) > 0),
    }

    # Composite quality score (simple average)
    composite = round(sum(scores.values()) / len(scores), 3)
    scores["composite"] = composite

    passed = composite >= settings.eval_agent_pass_threshold

    summary = (
        f"Quality composite={composite:.2f} "
        f"({'PASS' if passed else 'WARN'}) | "
        + " | ".join(f"{k}={v:.2f}" for k, v in scores.items() if k != "composite")
    )

    elapsed = time.time() - start
    logger.info(f"evaluation.heuristic_eval: {summary} ({elapsed:.3f}s)")
    return {
        "eval_scores":  scores,
        "eval_summary": summary,
        "active_agents": ["evaluation"],
        "trace": [f"evaluation.heuristic: {summary}"],
    }


async def ragas_eval(state: MultiAgentState, llm) -> Dict[str, Any]:
    """
    Optional RAGAS evaluation (LLM-as-judge).
    Skipped when eval_agent_use_ragas=False or no documents were retrieved.
    """
    if not settings.eval_agent_use_ragas:
        return {"trace": ["evaluation.ragas: skipped (disabled)"]}

    answer = state.get("final_response") or state.get("research_summary", "")
    docs   = state.get("retrieval_results", [])
    query  = state["query"]

    if not docs or not answer:
        return {"trace": ["evaluation.ragas: skipped (no docs/answer)"]}

    start = time.time()
    try:
        from backend.evaluators.ragas_evaluator import RAGASEvaluator
        from backend.evaluators.metrics import EvaluationSample, MetricName

        evaluator = RAGASEvaluator()
        sample = EvaluationSample(
            question=query,
            answer=answer,
            contexts=[d.get("content", "")[:800] for d in docs[:5]],
        )
        result = await evaluator.evaluate(
            samples=[sample],
            metrics=[MetricName.FAITHFULNESS, MetricName.ANSWER_RELEVANCY],
        )
        scores = {r.metric: r.score for r in (result.scores or [])}
        existing = state.get("eval_scores", {})
        merged   = {**existing, **scores}

        elapsed = time.time() - start
        logger.info(f"evaluation.ragas: {scores} ({elapsed:.2f}s)")
        return {
            "eval_scores": merged,
            "trace": [f"evaluation.ragas: {scores} in {elapsed:.1f}s"],
        }
    except Exception as exc:
        logger.warning(f"evaluation.ragas_eval failed ({exc}); skipping RAGAS")
        return {"trace": [f"evaluation.ragas: failed ({exc})"]}


# ── Graph builder ─────────────────────────────────────────────────────────

class EvaluationAgent:
    """
    LangGraph sub-graph for the Evaluation Agent.

    Runs heuristic scoring first, then optionally full RAGAS scoring.
    """

    def __init__(self, llm) -> None:
        self.llm    = llm
        self._graph = self._build()
        logger.info("EvaluationAgent compiled")

    def _build(self):
        import functools
        builder = StateGraph(MultiAgentState)

        builder.add_node("heuristic_eval", heuristic_eval)
        builder.add_node("ragas_eval",     functools.partial(ragas_eval, llm=self.llm))

        builder.set_entry_point("heuristic_eval")
        builder.add_edge("heuristic_eval", "ragas_eval")
        builder.add_edge("ragas_eval",     END)

        return builder.compile()

    async def run(self, state: Dict[str, Any]) -> MultiAgentState:
        async with trace_span("evaluation_agent.run"):
            return await self._graph.ainvoke(state)  # type: ignore[return-value]


# Made with Bob
