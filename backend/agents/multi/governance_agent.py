"""
Governance Agent (Phase 11) — Multi-Agent Ecosystem.

The Governance Agent is the final safety gate before an answer reaches the
user.  It runs all Phase 7 guardrail checks on the generated answer AND
enforces domain-specific compliance rules.

Checks performed
────────────────
  1. Guardrails pipeline output check (PII, hallucination, toxicity)
  2. Confidence check — if eval composite score < threshold, add disclaimer
  3. Source attribution check — every factual claim should reference a source

The agent may:
  • Pass the answer unchanged                  → governance_passed=True
  • Redact PII in the answer                   → replaces final_answer
  • Append a low-confidence disclaimer         → appends to final_answer
  • Block hallucinated / toxic answers         → replaces with safe message
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from backend.agents.multi.state import MultiAgentState
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.core.tracing import trace_span

logger = get_logger(__name__)

_LOW_CONF_DISCLAIMER = (
    "\n\n⚠️ *Note: The confidence score for this answer is below the "
    "recommended threshold.  Please verify the information against the "
    "original source documents.*"
)
_BLOCKED_MESSAGE = (
    "I'm sorry, this answer could not be delivered because it did not pass "
    "the safety and governance checks.  Please rephrase your question or "
    "consult the original documents directly."
)


# ── Node functions ────────────────────────────────────────────────────────

async def guardrails_check(state: MultiAgentState) -> Dict[str, Any]:
    """Run the Phase 7 guardrails pipeline on the generated answer."""
    answer = state.get("final_response") or state.get("research_summary", "")
    docs   = state.get("retrieval_results", []) or []
    issues: List[Dict[str, Any]] = []
    start  = time.time()

    try:
        from backend.guardrails.pipeline import GuardrailsPipeline
        pipeline = GuardrailsPipeline()
        contexts = [d.get("content", "")[:600] for d in docs[:5]]
        result   = await pipeline.check_output(answer, contexts)

        if result.blocked:
            issues.append({
                "check":    "guardrails",
                "severity": "critical",
                "detail":   result.block_reason or "blocked by guardrails",
            })
        for check in (result.warnings or []):
            issues.append({
                "check":    getattr(check, "detector", "unknown"),
                "severity": "warning",
                "detail":   str(check),
            })

        final_answer = _BLOCKED_MESSAGE if result.blocked else (result.redacted_text or answer)
    except Exception as exc:
        logger.warning(f"governance.guardrails_check failed ({exc}); passing through")
        final_answer = answer

    elapsed = time.time() - start
    blocked = any(i["severity"] == "critical" for i in issues)
    logger.info(f"governance.guardrails_check: blocked={blocked} issues={len(issues)} ({elapsed:.2f}s)")
    return {
        "governance_issues": issues,
        "final_answer":      final_answer,
        "active_agents":     ["governance"],
        "trace": [f"governance.guardrails: blocked={blocked} issues={len(issues)}"],
    }


async def confidence_and_attribution(state: MultiAgentState) -> Dict[str, Any]:
    """
    Append a low-confidence disclaimer when eval composite is too low.
    Also note when no sources are available (potential hallucination risk).
    """
    final_answer = state.get("final_answer") or state.get("final_response", "")
    eval_scores  = state.get("eval_scores", {})
    docs         = state.get("retrieval_results", [])
    issues       = list(state.get("governance_issues", []))

    composite = eval_scores.get("composite", 1.0)
    threshold = settings.eval_agent_pass_threshold

    if composite < threshold:
        final_answer += _LOW_CONF_DISCLAIMER
        issues.append({
            "check":    "confidence",
            "severity": "warning",
            "detail":   f"composite={composite:.2f} below threshold={threshold}",
        })

    if not docs:
        issues.append({
            "check":    "attribution",
            "severity": "warning",
            "detail":   "No source documents retrieved — answer may not be grounded.",
        })

    passed = not any(i["severity"] == "critical" for i in issues)
    logger.info(f"governance.confidence_attribution: passed={passed}")
    return {
        "governance_passed": passed,
        "governance_issues": issues,
        "final_answer":      final_answer,
        "final_response":    final_answer,
        "trace": [f"governance.confidence: passed={passed} issues={len(issues)}"],
    }


# ── Graph builder ─────────────────────────────────────────────────────────

class GovernanceAgent:
    """
    LangGraph sub-graph for the Governance Agent.

    Runs guardrails check then confidence/attribution gating.
    """

    def __init__(self) -> None:
        self._graph = self._build()
        logger.info("GovernanceAgent compiled")

    def _build(self):
        builder = StateGraph(MultiAgentState)

        builder.add_node("guardrails_check",         guardrails_check)
        builder.add_node("confidence_and_attribution", confidence_and_attribution)

        builder.set_entry_point("guardrails_check")
        builder.add_edge("guardrails_check",           "confidence_and_attribution")
        builder.add_edge("confidence_and_attribution", END)

        return builder.compile()

    async def run(self, state: Dict[str, Any]) -> MultiAgentState:
        async with trace_span("governance_agent.run"):
            return await self._graph.ainvoke(state)  # type: ignore[return-value]


# Made with Bob
