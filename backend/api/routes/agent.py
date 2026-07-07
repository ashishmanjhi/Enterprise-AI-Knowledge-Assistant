"""
Agentic RAG chat route (Phase 9).

Exposes a single POST endpoint that runs the full LangGraph agentic pipeline:
    route_query → (rewrite_query) → retrieve → grade_documents
                → generate → check_grounding

The endpoint accepts the same parameters as /api/v1/chat but routes through
the graph instead of the linear RAGChain, giving it adaptive retrieval strategy
selection, query rewriting, document grading, and grounding verification.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agents.graph import AgentGraph
from backend.llm.llm_service import LLMService, get_llm_service
from backend.retrievers.hybrid_retriever import HybridRetriever
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.api.middleware.rate_limit import limiter
from backend.api.middleware.tenant import resolve_tenant_id
from backend.retrievers.tenant_registry import get_retriever_for_tenant

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

# ── Per-tenant graph instances (lazy-initialised on first request) ────────
_graph_cache: dict = {}


def _get_graph(tenant_id: str = "default") -> AgentGraph:
    """Return an AgentGraph wired to the tenant's retriever (created once per tenant)."""
    if tenant_id not in _graph_cache:
        llm       = get_llm_service()
        retriever = get_retriever_for_tenant(tenant_id)
        _graph_cache[tenant_id] = AgentGraph(llm=llm, retriever=retriever)
        logger.info(f"AgentGraph created for tenant='{tenant_id}'")
    return _graph_cache[tenant_id]


# ── Request / Response models ─────────────────────────────────────────────

class AgentChatRequest(BaseModel):
    """Request body for the agentic chat endpoint."""
    message: str = Field(..., description="User question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    top_k: int = Field(5, ge=1, le=20, description="Documents to retrieve")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(500, ge=1, le=4096)
    retrieval_method: Optional[str] = Field(
        None,
        description="Override retrieval method: hybrid | faiss | bm25. "
                    "When None the agent router decides automatically.",
    )
    use_reranking: Optional[bool] = Field(None, description="Override reranking toggle")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Prior messages [{role, content}, ...]"
    )


class SourceRef(BaseModel):
    """A single retrieved source chunk."""
    chunk_id: str
    document_id: str
    filename: str
    content: str
    score: float
    page_number: Optional[int] = None
    retrieval_method: Optional[str] = None
    faiss_score: Optional[float] = None
    bm25_score: Optional[float] = None


class AgentChatResponse(BaseModel):
    """Response from the agentic chat endpoint."""
    conversation_id: str
    answer: str
    sources: List[SourceRef]
    retrieval_method: Optional[str] = None
    is_grounded: Optional[bool] = None
    rewrite_count: int = 0
    rewritten_query: Optional[str] = None
    trace: List[str] = Field(default_factory=list)
    model: str = "unknown"
    tokens_used: int = 0
    processing_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Endpoint ──────────────────────────────────────────────────────────────

@limiter.limit(settings.rate_limit_agent)
@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: Request, body: AgentChatRequest) -> AgentChatResponse:
    """
    Agentic RAG chat.

    Runs the full LangGraph pipeline — routing, optional query rewriting,
    retrieval, document grading, generation, and grounding verification.
    Returns the final answer with full trace metadata.
    """
    conversation_id = body.conversation_id or f"agent_{uuid.uuid4().hex[:12]}"

    try:
        start = time.time()
        tenant_id = resolve_tenant_id(request)
        graph = _get_graph(tenant_id)

        # Build initial state
        initial_state: Dict[str, Any] = {
            "query":                body.message,
            "top_k":               body.top_k,
            "temperature":         body.temperature,
            "max_tokens":          body.max_tokens,
            "use_reranking":       body.use_reranking if body.use_reranking is not None
                                   else settings.enable_reranking,
            "conversation_history": body.conversation_history or [],
            "trace":               [],
        }

        # Honour an explicit retrieval method override if provided
        if body.retrieval_method:
            initial_state["retrieval_method"] = body.retrieval_method

        final_state = await graph.run(initial_state)

        # Extract outputs
        answer   = final_state.get("generation") or ""
        docs     = final_state.get("documents",   [])
        metadata = final_state.get("metadata",    {})
        trace    = final_state.get("trace",        [])

        # Format sources
        sources = [
            SourceRef(
                chunk_id=d.get("chunk_id",      "unknown"),
                document_id=d.get("document_id","unknown"),
                filename=d.get("filename",       "unknown"),
                content=(d.get("content", "")[:200] + "..."
                         if len(d.get("content", "")) > 200
                         else d.get("content", "")),
                score=round(d.get("score",       0.0), 4),
                page_number=d.get("page_number"),
                retrieval_method=d.get("retrieval_method"),
                faiss_score=d.get("faiss_score"),
                bm25_score=d.get("bm25_score"),
            )
            for d in docs
        ]

        elapsed = round(time.time() - start, 3)
        logger.info(
            f"agent_chat: conv={conversation_id} "
            f"rewrite_count={final_state.get('rewrite_count', 0)} "
            f"is_grounded={final_state.get('is_grounded')} "
            f"sources={len(sources)} time={elapsed}s"
        )

        return AgentChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            sources=sources,
            retrieval_method=final_state.get("retrieval_method"),
            is_grounded=final_state.get("is_grounded"),
            rewrite_count=final_state.get("rewrite_count", 0),
            rewritten_query=final_state.get("rewritten_query"),
            trace=trace,
            model=metadata.get("model", "unknown"),
            tokens_used=metadata.get("tokens_used", 0),
            processing_time=elapsed,
        )

    except Exception as exc:
        logger.error(f"agent_chat failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent chat failed: {exc}")


# ── Streaming endpoint ────────────────────────────────────────────────────

@limiter.limit(settings.rate_limit_agent_stream)
@router.post("/chat/stream")
async def agent_chat_stream(request: Request, body: AgentChatRequest) -> StreamingResponse:
    """
    Agentic RAG chat — SSE streaming variant.

    Streams three event types as newline-delimited JSON over
    ``text/event-stream``:

    ``{"type": "node", "node": "<name>", "data": {...}}``
        One event per LangGraph node completion carrying the partial
        state update produced by that node.

    ``{"type": "token", "content": "word "}``
        Word-level events for the generation node's answer text,
        enabling a live typing effect in the UI.

    ``{"type": "done", "data": {...AgentChatResponse fields...}}``
        Terminal event with the fully assembled response payload.
        The client should parse this to persist sources, trace, and
        metadata in session state.
    """
    conversation_id = body.conversation_id or f"agent_{uuid.uuid4().hex[:12]}"
    start = time.time()
    tenant_id = resolve_tenant_id(request)

    initial_state: dict = {
        "query":                body.message,
        "top_k":               body.top_k,
        "temperature":         body.temperature,
        "max_tokens":          body.max_tokens,
        "use_reranking":       body.use_reranking if body.use_reranking is not None
                               else settings.enable_reranking,
        "conversation_history": body.conversation_history or [],
        "trace":               [],
    }
    if body.retrieval_method:
        initial_state["retrieval_method"] = body.retrieval_method

    async def _generate():
        try:
            graph = _get_graph(tenant_id)
            async for event in graph.stream_run(initial_state):
                if event["type"] == "done":
                    # Build the full response payload for the terminal event
                    fs = event["data"]
                    docs     = fs.get("documents", [])
                    metadata = fs.get("metadata", {})
                    elapsed  = round(time.time() - start, 3)

                    sources = [
                        {
                            "chunk_id":         d.get("chunk_id",       "unknown"),
                            "document_id":      d.get("document_id",    "unknown"),
                            "filename":         d.get("filename",        "unknown"),
                            "content":          (d.get("content", "")[:200] + "..."
                                                 if len(d.get("content", "")) > 200
                                                 else d.get("content", "")),
                            "score":            round(d.get("score", 0.0), 4),
                            "page_number":      d.get("page_number"),
                            "retrieval_method": d.get("retrieval_method"),
                            "faiss_score":      d.get("faiss_score"),
                            "bm25_score":       d.get("bm25_score"),
                        }
                        for d in docs
                    ]

                    done_payload = {
                        "conversation_id":  conversation_id,
                        "answer":           fs.get("generation") or "",
                        "sources":          sources,
                        "retrieval_method": fs.get("retrieval_method"),
                        "is_grounded":      fs.get("is_grounded"),
                        "rewrite_count":    fs.get("rewrite_count", 0),
                        "rewritten_query":  fs.get("rewritten_query"),
                        "trace":            fs.get("trace", []),
                        "model":            metadata.get("model", "unknown"),
                        "tokens_used":      metadata.get("tokens_used", 0),
                        "processing_time":  elapsed,
                    }
                    yield f"data: {json.dumps({'type': 'done', 'data': done_payload})}\n\n"
                    logger.info(
                        f"agent_chat_stream: conv={conversation_id} "
                        f"rewrite_count={fs.get('rewrite_count', 0)} "
                        f"is_grounded={fs.get('is_grounded')} "
                        f"sources={len(sources)} time={elapsed}s"
                    )
                else:
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.error(f"agent_chat_stream failed: {exc}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.get("/health")
async def agent_health() -> Dict[str, Any]:
    """Health check for the agent subsystem."""
    try:
        graph = _get_graph(settings.default_tenant_id)
        return {
            "status": "healthy",
            "graph": "compiled",
            "max_rewrites": settings.agent_max_rewrites,
            "document_grading": settings.agent_enable_document_grading,
            "grounding_check": settings.agent_enable_grounding_check,
        }
    except Exception as exc:
        logger.error(f"agent_health failed: {exc}")
        return {"status": "unhealthy", "error": str(exc)}


# Made with Bob
