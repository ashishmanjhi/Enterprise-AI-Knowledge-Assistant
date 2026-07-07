"""
Tenant-aware retriever and pipeline registry (Phase 15 — Multi-Tenancy).

When ``settings.multi_tenancy_enabled`` is True every tenant gets its own
FAISS index and BM25 index, stored under::

    data/vectorstore/<tenant_id>/faiss_index.bin
    data/vectorstore/<tenant_id>/metadata.json
    data/vectorstore/<tenant_id>/bm25_index.json

When the flag is False (the default) all calls fall back to the global
singleton instances — behaviour is identical to the pre-multi-tenancy code.

Public API
----------
``get_pipeline_for_tenant(tenant_id) -> IngestionPipeline``
    Returns an IngestionPipeline whose vector_store and bm25_retriever are
    scoped to *tenant_id*.

``get_retriever_for_tenant(tenant_id) -> HybridRetriever``
    Returns a HybridRetriever whose FAISS and BM25 instances are scoped to
    *tenant_id*.

Both functions cache instances in an in-process dict so the per-tenant index
is only loaded from disk once per process lifetime.  Call
``evict_tenant(tenant_id)`` to force a reload (e.g. after an index rebuild).
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Dict, Optional

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

# ── Thread safety ─────────────────────────────────────────────────────────

_lock = threading.Lock()

# ── Per-tenant caches ─────────────────────────────────────────────────────

_pipeline_cache:  Dict[str, "IngestionPipeline"]  = {}  # noqa: F821
_retriever_cache: Dict[str, "HybridRetriever"]    = {}  # noqa: F821

# ── Slug validation ───────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def _validate_tenant_id(tenant_id: str) -> str:
    """
    Sanitise and validate a tenant_id so it is safe to use as a directory name.

    Raises ``ValueError`` for slugs that contain path-traversal characters or
    exceed 64 characters.
    """
    if not _SLUG_RE.match(tenant_id):
        raise ValueError(
            f"Invalid tenant_id {tenant_id!r}. "
            "Must be 1–64 characters: letters, digits, hyphens, underscores."
        )
    return tenant_id


# ── Path helpers ──────────────────────────────────────────────────────────

def _tenant_dir(tenant_id: str) -> Path:
    base = Path(settings.vectorstore_dir)
    return base / tenant_id


def _faiss_path(tenant_id: str) -> Path:
    return _tenant_dir(tenant_id) / "faiss_index.bin"


def _metadata_path(tenant_id: str) -> Path:
    return _tenant_dir(tenant_id) / "metadata.json"


def _bm25_path(tenant_id: str) -> Path:
    return _tenant_dir(tenant_id) / "bm25_index.json"


# ── Factories ─────────────────────────────────────────────────────────────

def get_pipeline_for_tenant(tenant_id: str) -> "IngestionPipeline":
    """
    Return an IngestionPipeline scoped to *tenant_id*.

    When multi-tenancy is disabled, always returns the process-global
    singleton pipeline (identical to the original behaviour).
    """
    from backend.ingestion.pipeline import IngestionPipeline
    from backend.retrievers.vector_store_manager import get_shared_vector_store
    from backend.retrievers.bm25_manager import get_shared_bm25_retriever

    if not settings.multi_tenancy_enabled:
        return IngestionPipeline(
            vector_store=get_shared_vector_store(),
            bm25_retriever=get_shared_bm25_retriever(),
        )

    tid = _validate_tenant_id(tenant_id)
    with _lock:
        if tid not in _pipeline_cache:
            _pipeline_cache[tid] = _make_pipeline(tid)
        return _pipeline_cache[tid]


def get_retriever_for_tenant(tenant_id: str) -> "HybridRetriever":
    """
    Return a HybridRetriever scoped to *tenant_id*.

    When multi-tenancy is disabled, returns the process-global singleton
    retriever (identical to the original behaviour).
    """
    from backend.retrievers.hybrid_retriever import HybridRetriever
    from backend.retrievers.vector_store_manager import get_shared_vector_store
    from backend.retrievers.bm25_manager import get_shared_bm25_retriever
    from backend.retrievers.retriever import DocumentRetriever

    if not settings.multi_tenancy_enabled:
        return HybridRetriever(
            faiss_retriever=DocumentRetriever(vector_store=get_shared_vector_store()),
            bm25_retriever=get_shared_bm25_retriever(),
        )

    tid = _validate_tenant_id(tenant_id)
    with _lock:
        if tid not in _retriever_cache:
            _retriever_cache[tid] = _make_retriever(tid)
        return _retriever_cache[tid]


def evict_tenant(tenant_id: str) -> None:
    """Remove cached instances for *tenant_id* so they are rebuilt on next access."""
    tid = _validate_tenant_id(tenant_id)
    with _lock:
        _pipeline_cache.pop(tid, None)
        _retriever_cache.pop(tid, None)
    logger.info(f"Evicted tenant cache for '{tid}'")


def list_tenants() -> list:
    """Return the slugs of all tenants that currently have a data directory."""
    base = Path(settings.vectorstore_dir)
    if not base.exists():
        return []
    return [d.name for d in base.iterdir() if d.is_dir() and _SLUG_RE.match(d.name)]


# ── Internal builders ─────────────────────────────────────────────────────

def _make_pipeline(tenant_id: str) -> "IngestionPipeline":
    from backend.ingestion.pipeline import IngestionPipeline
    from backend.retrievers.vector_store import FAISSVectorStore
    from backend.retrievers.bm25_retriever import BM25Retriever

    _tenant_dir(tenant_id).mkdir(parents=True, exist_ok=True)

    vs = FAISSVectorStore(
        index_path=_faiss_path(tenant_id),
        metadata_path=_metadata_path(tenant_id),
    )
    bm25 = BM25Retriever(index_path=str(_bm25_path(tenant_id)))

    # Load existing indexes if present
    if _faiss_path(tenant_id).exists():
        try:
            vs.load()
        except Exception as exc:
            logger.warning(f"Could not load FAISS index for tenant '{tenant_id}': {exc}")
    bm25.load(str(_bm25_path(tenant_id)))

    logger.info(
        f"Created IngestionPipeline for tenant '{tenant_id}' "
        f"(FAISS vectors: {vs.index.ntotal}, BM25 docs: {len(bm25.documents)})"
    )
    return IngestionPipeline(vector_store=vs, bm25_retriever=bm25)


def _make_retriever(tenant_id: str) -> "HybridRetriever":
    from backend.retrievers.hybrid_retriever import HybridRetriever
    from backend.retrievers.retriever import DocumentRetriever
    from backend.retrievers.vector_store import FAISSVectorStore
    from backend.retrievers.bm25_retriever import BM25Retriever

    _tenant_dir(tenant_id).mkdir(parents=True, exist_ok=True)

    vs = FAISSVectorStore(
        index_path=_faiss_path(tenant_id),
        metadata_path=_metadata_path(tenant_id),
    )
    bm25 = BM25Retriever(index_path=str(_bm25_path(tenant_id)))

    if _faiss_path(tenant_id).exists():
        try:
            vs.load()
        except Exception as exc:
            logger.warning(f"Could not load FAISS index for tenant '{tenant_id}': {exc}")
    bm25.load(str(_bm25_path(tenant_id)))

    logger.info(
        f"Created HybridRetriever for tenant '{tenant_id}' "
        f"(FAISS vectors: {vs.index.ntotal}, BM25 docs: {len(bm25.documents)})"
    )
    return HybridRetriever(
        faiss_retriever=DocumentRetriever(vector_store=vs),
        bm25_retriever=bm25,
    )


# Made with Bob
