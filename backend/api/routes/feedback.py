"""
User Feedback collection — Phase 10 / Phase 15 Analytics.

POST /api/v1/feedback              — submit a thumbs-up / thumbs-down rating
GET  /api/v1/feedback              — list recent feedback entries (admin use)
GET  /api/v1/feedback/stats        — aggregate thumbs-up / thumbs-down counts
GET  /api/v1/feedback/analytics    — multi-dimension breakdown (pipeline, method, daily trend)
GET  /api/v1/feedback/trend        — day-by-day thumbs counts for charting
GET  /api/v1/feedback/export       — download full JSONL file

Feedback is persisted as append-only JSON Lines (one JSON object per line)
in the path configured by ``settings.feedback_store_path``.  This is
intentionally simple — swap for a PostgreSQL table when needed.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
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
    total:       int
    thumbs_up:   int
    thumbs_down: int
    up_pct:      float


class DailyBucket(BaseModel):
    """Counts for a single UTC day (YYYY-MM-DD)."""
    date:        str
    thumbs_up:   int
    thumbs_down: int
    total:       int


class PipelineBreakdown(BaseModel):
    """Per-pipeline satisfaction breakdown."""
    pipeline:    str
    thumbs_up:   int
    thumbs_down: int
    total:       int
    up_pct:      float


class MethodBreakdown(BaseModel):
    """Per-retrieval-method breakdown."""
    method:      str
    thumbs_up:   int
    thumbs_down: int
    total:       int
    up_pct:      float


class FeedbackAnalytics(BaseModel):
    """Full analytics summary across all dimensions."""
    overall:              FeedbackStats
    by_pipeline:          List[PipelineBreakdown]
    by_retrieval_method:  List[MethodBreakdown]
    daily_trend:          List[DailyBucket]
    # date range of the data returned
    from_date:            Optional[str] = None
    to_date:              Optional[str] = None
    entries_analysed:     int = 0


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


# ── Analytics helpers ─────────────────────────────────────────────────────

def _ts_to_date(ts: float) -> str:
    """Convert a Unix timestamp to a UTC date string YYYY-MM-DD."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _build_analytics(entries: List[dict]) -> FeedbackAnalytics:
    """Compute all analytics dimensions from a list of raw JSONL dicts."""
    # ── overall ──────────────────────────────────────────────────────────
    total = len(entries)
    up    = sum(1 for e in entries if e.get("rating") == "up")
    down  = total - up
    overall = FeedbackStats(
        total=total,
        thumbs_up=up,
        thumbs_down=down,
        up_pct=round(up / total * 100, 1) if total else 0.0,
    )

    # ── per-pipeline ─────────────────────────────────────────────────────
    pipeline_up:    Dict[str, int] = defaultdict(int)
    pipeline_down:  Dict[str, int] = defaultdict(int)
    for e in entries:
        key = e.get("pipeline") or "unknown"
        if e.get("rating") == "up":
            pipeline_up[key]   += 1
        else:
            pipeline_down[key] += 1

    by_pipeline = [
        PipelineBreakdown(
            pipeline=k,
            thumbs_up=pipeline_up[k],
            thumbs_down=pipeline_down[k],
            total=pipeline_up[k] + pipeline_down[k],
            up_pct=round(
                pipeline_up[k] / (pipeline_up[k] + pipeline_down[k]) * 100, 1
            ) if (pipeline_up[k] + pipeline_down[k]) else 0.0,
        )
        for k in sorted(set(list(pipeline_up) + list(pipeline_down)))
    ]

    # ── per-retrieval-method ──────────────────────────────────────────────
    method_up:   Dict[str, int] = defaultdict(int)
    method_down: Dict[str, int] = defaultdict(int)
    for e in entries:
        key = e.get("retrieval_method") or "unknown"
        if e.get("rating") == "up":
            method_up[key]   += 1
        else:
            method_down[key] += 1

    by_method = [
        MethodBreakdown(
            method=k,
            thumbs_up=method_up[k],
            thumbs_down=method_down[k],
            total=method_up[k] + method_down[k],
            up_pct=round(
                method_up[k] / (method_up[k] + method_down[k]) * 100, 1
            ) if (method_up[k] + method_down[k]) else 0.0,
        )
        for k in sorted(set(list(method_up) + list(method_down)))
    ]

    # ── daily trend ───────────────────────────────────────────────────────
    day_up:   Dict[str, int] = defaultdict(int)
    day_down: Dict[str, int] = defaultdict(int)
    for e in entries:
        ts  = e.get("timestamp", 0.0)
        day = _ts_to_date(ts) if ts else "unknown"
        if e.get("rating") == "up":
            day_up[day]   += 1
        else:
            day_down[day] += 1

    all_days = sorted(set(list(day_up) + list(day_down)))
    daily_trend = [
        DailyBucket(
            date=d,
            thumbs_up=day_up[d],
            thumbs_down=day_down[d],
            total=day_up[d] + day_down[d],
        )
        for d in all_days
    ]

    from_date = all_days[0]  if all_days else None
    to_date   = all_days[-1] if all_days else None

    return FeedbackAnalytics(
        overall=overall,
        by_pipeline=by_pipeline,
        by_retrieval_method=by_method,
        daily_trend=daily_trend,
        from_date=from_date,
        to_date=to_date,
        entries_analysed=total,
    )


# ── Analytics endpoints ───────────────────────────────────────────────────

@router.get("/analytics", response_model=FeedbackAnalytics)
async def feedback_analytics(
    pipeline: Optional[str] = Query(
        None, description="Filter to a specific pipeline (rag | agent | multi_agent)"
    ),
    since: Optional[str] = Query(
        None, description="ISO date lower bound inclusive, e.g. 2024-01-01"
    ),
    until: Optional[str] = Query(
        None, description="ISO date upper bound inclusive, e.g. 2024-12-31"
    ),
) -> FeedbackAnalytics:
    """
    Multi-dimension analytics over all collected feedback.

    Returns overall satisfaction, per-pipeline breakdown, per-retrieval-method
    breakdown, and a daily time-series trend.  All dimensions respect the
    optional ``pipeline``, ``since``, and ``until`` filters.
    """
    try:
        entries = _read_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # ── apply filters ─────────────────────────────────────────────────────
    if pipeline:
        entries = [e for e in entries if (e.get("pipeline") or "unknown") == pipeline]

    if since:
        try:
            since_ts = datetime.strptime(since, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).timestamp()
        except ValueError:
            raise HTTPException(
                status_code=422, detail="'since' must be YYYY-MM-DD"
            )
        entries = [e for e in entries if e.get("timestamp", 0) >= since_ts]

    if until:
        try:
            # inclusive: include all of the "until" day
            until_ts = datetime.strptime(until, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).timestamp() + 86399
        except ValueError:
            raise HTTPException(
                status_code=422, detail="'until' must be YYYY-MM-DD"
            )
        entries = [e for e in entries if e.get("timestamp", 0) <= until_ts]

    return _build_analytics(entries)


@router.get("/trend", response_model=List[DailyBucket])
async def feedback_trend(
    days: int = Query(30, ge=1, le=365, description="Number of most recent days to return"),
) -> List[DailyBucket]:
    """
    Day-by-day thumbs counts for the last *N* days.

    Suitable for feeding directly into a line/bar chart in the UI.
    Days with zero feedback are **omitted** (sparse representation).
    """
    try:
        entries = _read_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    cutoff_ts = time.time() - days * 86400
    entries   = [e for e in entries if e.get("timestamp", 0) >= cutoff_ts]

    analytics = _build_analytics(entries)
    return analytics.daily_trend


@router.get("/export")
async def feedback_export() -> PlainTextResponse:
    """
    Download the raw feedback store as a JSONL file.

    Returns the entire ``settings.feedback_store_path`` contents with
    ``Content-Disposition: attachment`` so browsers save it directly.
    The response MIME type is ``application/x-ndjson`` (newline-delimited JSON).
    """
    if not _store.exists():
        # Return an empty JSONL rather than 404 — consistent with /stats behaviour
        return PlainTextResponse(
            content="",
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="feedback.jsonl"'},
        )

    try:
        content = _store.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error(f"Failed to read feedback store: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not read store: {exc}")

    return PlainTextResponse(
        content=content,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="feedback.jsonl"'},
    )


# Made with Bob
