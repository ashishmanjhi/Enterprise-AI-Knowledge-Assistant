"""
Tenant resolution helper (Phase 15 — Multi-Tenancy).

``resolve_tenant_id(request)`` reads the configured header
(``settings.tenant_id_header``, default ``X-Tenant-ID``) from the incoming
Starlette request.  When multi-tenancy is disabled **or** the header is absent
it returns ``settings.default_tenant_id`` ("default") — giving backward-
compatible single-tenant behaviour with zero changes required in route code.

Usage in route handlers::

    from backend.api.middleware.tenant import resolve_tenant_id

    @router.post("/upload")
    async def upload(request: Request, ...):
        tenant_id = resolve_tenant_id(request)
        pipeline  = get_pipeline_for_tenant(tenant_id)
        ...
"""

from __future__ import annotations

from starlette.requests import Request

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


def resolve_tenant_id(request: Request) -> str:
    """
    Return the tenant slug for this request.

    Resolution order:
    1. When ``settings.multi_tenancy_enabled`` is False → ``settings.default_tenant_id``
    2. Value of the ``X-Tenant-ID`` header (or whatever ``settings.tenant_id_header`` names)
    3. ``settings.default_tenant_id`` when the header is absent or empty

    The returned value is safe to use as a dict key and as a directory name
    component — the tenant registry validates it further when constructing paths.
    """
    if not settings.multi_tenancy_enabled:
        return settings.default_tenant_id

    header_value = request.headers.get(settings.tenant_id_header, "").strip()
    if header_value:
        return header_value

    logger.debug(
        f"Header '{settings.tenant_id_header}' absent — "
        f"using default tenant '{settings.default_tenant_id}'"
    )
    return settings.default_tenant_id


# Made with Bob
