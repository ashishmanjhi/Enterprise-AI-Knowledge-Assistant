"""
LangSmith + OpenTelemetry observability — Phase 10 Production Readiness.

This module is the single place where both tracing backends are initialised.
Everything is opt-in through settings:

    LANGSMITH_ENABLED=true
    LANGSMITH_API_KEY=ls__...
    LANGSMITH_PROJECT=my-project

    OTEL_ENABLED=true
    OTEL_ENDPOINT=http://localhost:4317          # gRPC collector
    OTEL_SERVICE_NAME=enterprise-rag-platform

When disabled (the default) both initialise as transparent no-ops so the
rest of the codebase never needs ``if tracing_enabled`` guards.

Public API
──────────
    setup_tracing()          → call once at app startup (main.py)
    get_tracer()             → returns an OTel Tracer (or NoopTracer)
    trace_span(name, attrs)  → async context manager for manual spans
    langsmith_callbacks()    → returns LangSmith callback list (or [])
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────
_tracer = None          # OTel Tracer
_ls_client = None       # LangSmith Client


# ── LangSmith setup ───────────────────────────────────────────────────────

def _setup_langsmith() -> None:
    global _ls_client
    if not settings.langsmith_enabled:
        return
    try:
        import os
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY",    settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT",    settings.langsmith_project)
        # Validate the client is reachable
        from langsmith import Client
        _ls_client = Client()
        logger.info(
            f"LangSmith tracing enabled — project='{settings.langsmith_project}'"
        )
    except ImportError:
        logger.warning(
            "langsmith package not installed — tracing disabled. "
            "Run: pip install langsmith"
        )
    except Exception as exc:
        logger.warning(f"LangSmith init failed (continuing without tracing): {exc}")


def langsmith_callbacks() -> List[Any]:
    """Return a list of LangSmith callbacks for LangChain/LangGraph runs."""
    if _ls_client is None:
        return []
    try:
        from langsmith.run_helpers import LangSmithTracer
        return [LangSmithTracer(project_name=settings.langsmith_project, client=_ls_client)]
    except Exception:
        return []


# ── OpenTelemetry setup ───────────────────────────────────────────────────

def _setup_otel() -> None:
    global _tracer
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)

        # OTLP gRPC exporter (collector must be running)
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(
                f"OpenTelemetry tracing enabled — endpoint='{settings.otel_endpoint}'"
            )
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp-proto-grpc not installed; "
                "spans will not be exported. "
                "Run: pip install opentelemetry-exporter-otlp-proto-grpc"
            )

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(settings.otel_service_name)

    except ImportError:
        logger.warning(
            "opentelemetry-sdk not installed — OTel disabled. "
            "Run: pip install opentelemetry-sdk opentelemetry-api"
        )
    except Exception as exc:
        logger.warning(f"OpenTelemetry init failed (continuing without spans): {exc}")


def get_tracer():
    """
    Return the active OTel Tracer.
    Falls back to a no-op tracer when OTel is disabled or not installed.
    """
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        return trace.get_tracer("noop")
    except ImportError:
        return _NoopTracer()


@contextlib.asynccontextmanager
async def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Async context manager that wraps a block of code in an OTel span.

    Usage::

        async with trace_span("llm.generate", {"model": "qwen3:4b"}):
            result = await llm.generate(prompt)
    """
    tracer = get_tracer()
    if isinstance(tracer, _NoopTracer):
        yield
        return
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v))
        yield span


# ── FastAPI OpenTelemetry middleware ──────────────────────────────────────

def add_otel_middleware(app) -> None:
    """
    Instrument a FastAPI app with OpenTelemetry HTTP tracing.
    Call this from ``create_app()`` when ``settings.otel_enabled`` is True.
    """
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-fastapi not installed. "
            "Run: pip install opentelemetry-instrumentation-fastapi"
        )
    except Exception as exc:
        logger.warning(f"FastAPI OTel instrumentation failed: {exc}")


# ── Public init ───────────────────────────────────────────────────────────

def setup_tracing() -> None:
    """Initialise all enabled tracing backends. Call once at app startup."""
    _setup_langsmith()
    _setup_otel()


# ── No-op tracer fallback ─────────────────────────────────────────────────

class _NoopTracer:
    """Silent stand-in used when OTel is not installed."""

    @contextlib.contextmanager
    def start_as_current_span(self, name: str, **_):  # type: ignore[override]
        yield self

    def set_attribute(self, *_):
        pass


# Made with Bob
