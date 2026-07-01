"""
User Feedback collection — Phase 10 Production Readiness.

POST /api/v1/feedback       — submit a thumbs-up / thumbs-down rating
GET  /api/v1/feedback       — list recent feedback entries (admin use)
GET  /api/v1/feedback/stats — aggregate thumbs-up / thumbs-down counts

Feedback is persisted as append-only JSON Lines (one JSON object per line)
in the path configured by ``settings.feedback_store_path``.  This is
intentionally simple — swap for a PostgreSQL table when needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])

_store = Path(settings.feedback_store_path)


# ── Pydantic models ───────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """A single user feedback submission."""
    conversation_id:  str                       = Field(..., description="Conversation or session ID")
    message:          str                       = Field(..., description="The user message that was answered")
    answer:           str                       = Field(..., description="The assistant answer being rated")
    rating:           Literal["up", "down"]     = Field(..., description="Thumbs-up or thumbs-down")
    comment:          Optional[str]             = Field(None, description="Optional free-text comment")
    pipeline:         Optional[str]             = Field(None, description="'rag' or 'agent' — which pipeline produced the answer")
    retrieval_method: Optional[str]             = Field(None, description="hybrid | faiss | bm25")
    sources:          Optional[List[str]]       = Field(None, description="Source chunk IDs cited in the answer")


class FeedbackResponse(BaseModel):
    id:      str
    status:  str = "recorded"


class FeedbackEntry(FeedbackRequest):
    id:         str
    timestamp:  float


class FeedbackStats(BaseModel):
    total:     int
    thumbs_up: int
    thumbs_down: int
    up_pct:    float


# ── Helpers ───────────────────────────────────────────────────────────────

def _write(entry: dict) -> None:
    """Append one JSON object to the feedback store."""
    _store.parent.mkdir(parents=True, exist_ok=True)
    with _store.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _read_all() -> List[dict]:
    """Read every feedback entry from the store."""
    if not _store.exists():
        return []
    entries = []
    with _store.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit a thumbs-up or thumbs-down rating for an assistant answer.

    Persisted immediately to ``settings.feedback_store_path`` (JSON Lines).
    """
    import uuid
    entry_id = f"fb_{uuid.uuid4().hex[:12]}"
    entry = {
        "id":               entry_id,
        "timestamp":        time.time(),
        **request.model_dump(),
    }
    try:
        _write(entry)
    except Exception as exc:
        logger.error(f"Failed to write feedback: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {exc}")

    logger.info(
        f"Feedback recorded: id={entry_id} rating={request.rating} "
        f"conv={request.conversation_id}"
    )
    return FeedbackResponse(id=entry_id)


@router.get("", response_model=List[FeedbackEntry])
async def list_feedback(
    limit: int = Query(50, ge=1, le=500, description="Max entries to return (most recent first)"),
    rating: Optional[Literal["up", "down"]] = Query(None, description="Filter by rating"),
) -> List[FeedbackEntry]:
    """Return recent feedback entries (most recent first)."""
    try:
        entries = _read_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if rating:
        entries = [e for e in entries if e.get("rating") == rating]

    # most recent first
    entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    return [FeedbackEntry(**e) for e in entries[:limit]]


@router.get("/stats", response_model=FeedbackStats)
async def feedback_stats() -> FeedbackStats:
    """Return aggregate thumbs-up / thumbs-down counts."""
    try:
        entries = _read_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    total   = len(entries)
    up      = sum(1 for e in entries if e.get("rating") == "up")
    down    = total - up
    up_pct  = round(up / total * 100, 1) if total else 0.0

    return FeedbackStats(total=total, thumbs_up=up, thumbs_down=down, up_pct=up_pct)


# Made with Bob
