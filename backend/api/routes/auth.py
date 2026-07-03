"""
Authentication route — Phase 10 Production Readiness.

Exposes:
    POST /auth/token           — issue a JWT for username + password
    POST /auth/token/refresh   — refresh an existing (non-expired) token

This is a minimal implementation suitable for development and internal
tooling.  For production, integrate with your identity provider (OAuth2,
LDAP, SAML) instead of storing credentials here.

Demo credentials (development only):
    username: admin   password: changeme
    username: user    password: changeme

Set AUTH_ENABLED=true in .env to enforce JWT on all API routes.
"""

from __future__ import annotations

import hmac
from datetime import timedelta
from typing import Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.core.settings import settings
from backend.core.security import create_access_token
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Demo credential store — credentials sourced from settings (.env) ─────
# Never commit real passwords. Set AUTH_ADMIN_PASSWORD / AUTH_USER_PASSWORD
# in your .env file before any non-local deployment.

def _get_demo_users() -> Dict[str, str]:
    """Return credential map sourced from settings (loaded from .env)."""
    return {
        "admin": settings.auth_admin_password,
        "user":  settings.auth_user_password,
    }


# ── Request / Response models ─────────────────────────────────────────────

class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshRequest(BaseModel):
    token: str


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/token", response_model=TokenResponse, summary="Obtain a JWT access token")
async def get_token(request: TokenRequest) -> TokenResponse:
    """
    Exchange username + password for a signed JWT access token.

    The token must be sent as ``Authorization: Bearer <token>`` on all
    protected routes when ``AUTH_ENABLED=true``.
    """
    demo_users = _get_demo_users()
    stored_password = demo_users.get(request.username, "")
    # Use constant-time comparison to prevent timing-based username enumeration
    if not hmac.compare_digest(stored_password, request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        subject=request.username,
        extra_claims={"role": "admin" if request.username == "admin" else "user"},
    )
    logger.info(f"Token issued for user={request.username}")
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
    )


@router.post("/token/refresh", response_model=TokenResponse, summary="Refresh a JWT token")
async def refresh_token(request: RefreshRequest) -> TokenResponse:
    """
    Issue a new token from an existing valid (non-expired) token.
    The original token is decoded to extract the subject.
    """
    from backend.core.security import decode_access_token
    try:
        payload = decode_access_token(request.token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token invalid or expired: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    new_token = create_access_token(
        subject=payload["sub"],
        extra_claims={k: v for k, v in payload.items() if k not in ("sub", "exp", "iat")},
    )
    logger.info(f"Token refreshed for user={payload.get('sub')}")
    return TokenResponse(
        access_token=new_token,
        expires_in=settings.jwt_expire_minutes * 60,
    )


@router.get("/status", summary="Auth configuration status")
async def auth_status():
    """Return whether authentication is currently enforced."""
    return {
        "auth_enabled":     settings.auth_enabled,
        "jwt_algorithm":    settings.jwt_algorithm,
        "expire_minutes":   settings.jwt_expire_minutes,
    }


# Made with Bob
