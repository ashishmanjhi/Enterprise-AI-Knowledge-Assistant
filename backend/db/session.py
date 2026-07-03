"""
Database session factory for the Enterprise Agentic RAG Platform.

Usage (async routes)
--------------------
    from backend.db.session import get_db

    @router.get("/items")
    async def list_items(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Item))
        return result.scalars().all()

Usage (sync / scripts)
----------------------
    from backend.db.session import sync_engine
    from backend.db.models import Base
    Base.metadata.create_all(sync_engine)  # only for quick dev — use Alembic in prod
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


# ── Connection URLs ───────────────────────────────────────────────────────

def _sync_url() -> str:
    """Standard psycopg2 URL for Alembic migrations (sync)."""
    return (
        f"postgresql+psycopg2://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )


def _async_url() -> str:
    """asyncpg URL for FastAPI async routes."""
    return (
        f"postgresql+asyncpg://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )


# ── Engines ───────────────────────────────────────────────────────────────

# Sync engine — used by Alembic and any synchronous scripts / tests
sync_engine = create_engine(
    _sync_url(),
    pool_pre_ping=True,    # detect stale connections before using them
    echo=settings.debug,   # log SQL in debug mode
)

# Async engine — used by FastAPI route handlers
async_engine = create_async_engine(
    _async_url(),
    pool_pre_ping=True,
    echo=settings.debug,
)

# Session factories
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── FastAPI dependency ────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session for use as a FastAPI dependency.

    Automatically commits on success and rolls back on exception.

    Example
    -------
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Made with Bob
