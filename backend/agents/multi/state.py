"""
Multi-Agent State (Phase 11) — shared TypedDict for the orchestrated pipeline.

All five sub-agents read from and write into this single state object.
Each agent appends its own section so results are never clobbered.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
import operator


class MultiAgentState(TypedDict, total=False):
    """
    Shared state for the Multi-Agent Ecosystem (Phase 11).

    Fields
    ──────
    query               Original user question (never mutated).
    intent              Classified intent: research | retrieval | evaluation | governance | general
    conversation_history  Prior turns injected into prompts.
    temperature         LLM temperature (propagated to all sub-agents).
    max_tokens          Token budget per LLM call.
    top_k               Documents to retrieve per sub-agent retrieval call.

    -- Research Agent --
    research_plan       Numbered list of sub-questions the research agent generated.
    research_findings   List of {"question": str, "answer": str, "sources": [...]} dicts.
    research_summary    Synthesised paragraph from all findings.

    -- Retrieval Agent --
    retrieval_strategy  Strategy chosen by the retrieval agent: hybrid | faiss | bm25.
    retrieval_results   Raw retrieved chunks (same schema as Phase 9 documents list).

    -- Evaluation Agent --
    eval_scores         Dict of metric_name → score (faithfulness, relevancy, etc.).
    eval_summary        Human-readable summary of the evaluation.

    -- Governance Agent --
    governance_passed   True if all governance checks cleared.
    governance_issues   List of {"check": str, "severity": str, "detail": str} dicts.
    final_answer        Possibly redacted / replaced answer after governance.

    -- Knowledge Agent --
    entities            Extracted named entities: [{"text": str, "label": str}].
    knowledge_context   Additional context assembled from entity lookups.

    -- Orchestrator --
    active_agents       Ordered list of agent names that ran (for trace / UI).
    final_response      The answer delivered back to the user.
    trace               Append-only execution log (one entry per node).
    metadata            Arbitrary key-value pairs (timing, model, tokens, etc.).
    """
    # Core
    query:                str
    intent:               Optional[str]
    conversation_history: List[Dict[str, str]]
    temperature:          float
    max_tokens:           int
    top_k:                int

    # Research Agent
    research_plan:        List[str]
    research_findings:    List[Dict[str, Any]]
    research_summary:     Optional[str]

    # Retrieval Agent
    retrieval_strategy:   Optional[str]
    retrieval_results:    List[Dict[str, Any]]

    # Evaluation Agent
    eval_scores:          Dict[str, float]
    eval_summary:         Optional[str]

    # Governance Agent
    governance_passed:    Optional[bool]
    governance_issues:    List[Dict[str, Any]]
    final_answer:         Optional[str]

    # Knowledge Agent
    entities:             List[Dict[str, str]]
    knowledge_context:    Optional[str]

    # Orchestrator
    active_agents:        List[str]
    final_response:       Optional[str]
    trace:                Annotated[List[str], operator.add]
    metadata:             Dict[str, Any]


# Made with Bob
