"""
Multi-Agent chat route (Phase 11).

POST /api/v1/multi-agent/chat   — run full 5-agent pipeline
GET  /api/v1/multi-agent/health — orchestrator health check
"""

from __future__ import annotations

import uuid, time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.agents.multi.orchestrator import MultiAgentOrchestrator
from backend.llm.llm_service import LLMService, get_llm_service
from backend.retrievers.hybrid_retriever import HybridRetriever
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.api.middleware.rate_limit import limiter

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/multi-agent", tags=["multi-agent"])

# ── Lazy singleton ────────────────────────────────────────────────────────
_orchestrator: Optional[MultiAgentOrchestrator] = None


def _get_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        llm       = get_llm_service()
        retriever = HybridRetriever()
        _orchestrator = MultiAgentOrchestrator(llm=llm, retriever=retriever)
        logger.info("MultiAgentOrchestrator singleton created")
    return _orchestrator


# ── Request / Response models ─────────────────────────────────────────────

class MultiAgentRequest(BaseModel):
    message:              str                            = Field(..., description="User question")
    conversation_id:      Optional[str]                  = Field(None)
    top_k:                int                            = Field(5, ge=1, le=20)
    temperature:          float                          = Field(0.7, ge=0.0, le=2.0)
    max_tokens:           int                            = Field(500, ge=1, le=4096)
    conversation_history: Optional[List[Dict[str, str]]] = Field(None)


class AgentResult(BaseModel):
    """Per-agent result section."""
    name:    str
    ran:     bool
    summary: Optional[str] = None


class MultiAgentResponse(BaseModel):
    conversation_id:   str
    answer:            str
    intent:            Optional[str]            = None
    active_agents:     List[str]                = Field(default_factory=list)
    agent_results:     List[AgentResult]        = Field(default_factory=list)
    research_plan:     Optional[List[str]]      = None
    research_findings: Optional[int]            = None   # count
    retrieval_docs:    Optional[int]            = None   # count
    eval_scores:       Optional[Dict[str, float]] = None
    governance_passed: Optional[bool]           = None
    governance_issues: Optional[int]            = None   # count
    entities:          Optional[List[str]]      = None
    trace:             List[str]                = Field(default_factory=list)
    model:             str                      = "unknown"
    tokens_used:       int                      = 0
    processing_time:   float                    = 0.0
    timestamp:         datetime                 = Field(default_factory=datetime.utcnow)


# ── Endpoint ──────────────────────────────────────────────────────────────

@limiter.limit(settings.rate_limit_multi_agent)
@router.post("/chat", response_model=MultiAgentResponse)
async def multi_agent_chat(request: Request, body: MultiAgentRequest) -> MultiAgentResponse:
    """
    Run the full multi-agent pipeline:
    classify → (research | retrieval) → knowledge → evaluation → governance → generate
    """
    conversation_id = body.conversation_id or f"ma_{uuid.uuid4().hex[:12]}"
    start = time.time()

    try:
        orch = _get_orchestrator()

        initial_state: Dict[str, Any] = {
            "query":                body.message,
            "top_k":               body.top_k,
            "temperature":         body.temperature,
            "max_tokens":          body.max_tokens,
            "conversation_history": body.conversation_history or [],
            "trace":               [],
        }

        state = await orch.run(initial_state)

        elapsed   = round(time.time() - start, 3)
        meta      = state.get("metadata") or {}
        agents    = state.get("active_agents", [])
        entities  = [e["text"] for e in state.get("entities", [])]

        # Build per-agent result summaries
        agent_results = [
            AgentResult(name="research",   ran="research"   in agents, summary=state.get("research_summary")),
            AgentResult(name="retrieval",  ran="retrieval"  in agents, summary=f"{len(state.get('retrieval_results',[]))} docs"),
            AgentResult(name="knowledge",  ran="knowledge"  in agents, summary=f"{len(state.get('entities',[]))} entities"),
            AgentResult(name="evaluation", ran="evaluation" in agents, summary=state.get("eval_summary")),
            AgentResult(name="governance", ran="governance" in agents,
                        summary="passed" if state.get("governance_passed") else "issues found"),
        ]

        logger.info(
            f"multi_agent_chat: conv={conversation_id} intent={state.get('intent')} "
            f"agents={agents} time={elapsed}s"
        )

        return MultiAgentResponse(
            conversation_id   = conversation_id,
            answer            = state.get("final_response") or "",
            intent            = state.get("intent"),
            active_agents     = agents,
            agent_results     = agent_results,
            research_plan     = state.get("research_plan"),
            research_findings = len(state.get("research_findings", [])) or None,
            retrieval_docs    = len(state.get("retrieval_results", [])) or None,
            eval_scores       = state.get("eval_scores") or None,
            governance_passed = state.get("governance_passed"),
            governance_issues = len(state.get("governance_issues", [])) or None,
            entities          = entities[:10] if entities else None,
            trace             = state.get("trace", []),
            model             = meta.get("model", "unknown"),
            tokens_used       = meta.get("tokens_used", 0),
            processing_time   = elapsed,
        )

    except Exception as exc:
        logger.error(f"multi_agent_chat failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Multi-agent pipeline failed: {exc}")


@router.get("/health")
async def multi_agent_health() -> Dict[str, Any]:
    """Health check for the multi-agent orchestrator."""
    try:
        orch = _get_orchestrator()
        return {
            "status":                  "healthy",
            "orchestrator":            "compiled",
            "agents":                  ["research", "retrieval", "knowledge", "evaluation", "governance"],
            "research_max_sub":        settings.research_agent_max_sub_questions,
            "retrieval_expansions":    settings.retrieval_agent_expansions,
            "eval_pass_threshold":     settings.eval_agent_pass_threshold,
            "eval_use_ragas":          settings.eval_agent_use_ragas,
            "knowledge_max_entities":  settings.knowledge_agent_max_entities,
        }
    except Exception as exc:
        logger.error(f"multi_agent_health failed: {exc}")
        return {"status": "unhealthy", "error": str(exc)}


# Made with Bob
