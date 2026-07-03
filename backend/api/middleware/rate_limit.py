"""
Rate limiting middleware — shared Limiter instance (Phase 15).

A single ``Limiter`` object is created here and imported by every route
module that needs to apply per-IP limits on LLM-calling endpoints.
``SlowAPIMiddleware`` and the 429 exception handler are wired in
``backend.api.main`` via ``create_app()``.

Key function: ``_get_real_ip``
  Reads ``X-Forwarded-For`` first (set by Caddy / reverse-proxy), then
  falls back to the direct ``client.host``.  This avoids every request
  appearing to come from the same proxy IP.
"""

from __future__ import annotations

from starlette.requests import Request
from slowapi import Limiter

from backend.core.logging import get_logger

logger = get_logger(__name__)


def _get_real_ip(request: Request) -> str:
    """Return the originating client IP, honouring X-Forwarded-For."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # The header may contain a comma-separated chain; the first is the client.
        ip = forwarded.split(",")[0].strip()
        return ip
    if request.client:
        return request.client.host
    return "unknown"


# Singleton Limiter — imported by route modules.
limiter = Limiter(key_func=_get_real_ip)

# Made with Bob
