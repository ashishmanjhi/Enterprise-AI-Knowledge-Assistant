"""
FastAPI application entry point.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from backend.core.settings import settings
from backend.core.logging import logger
from backend.core.tracing import setup_tracing, add_otel_middleware
from backend.api.routes import health, status, documents, chat, admin, evaluate, memory, guardrails, agent, auth, feedback, multi_agent
from backend.api.routes.knowledge_graph import router as kg_router
from backend.api.middleware.auth import JWTAuthMiddleware
from backend.api.middleware.rate_limit import limiter


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        description="Enterprise Agentic RAG Platform with multi-provider LLM support"
    )

    # Phase 15: Rate limiting — attach limiter state and register 429 handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    if settings.rate_limit_enabled:
        app.add_middleware(SlowAPIMiddleware)
        logger.info("Rate limiting enabled")
    else:
        logger.info("Rate limiting disabled (RATE_LIMIT_ENABLED=false)")

    # JWT auth middleware (Phase 10) — transparent pass-through when auth_enabled=False
    app.add_middleware(JWTAuthMiddleware)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(status.router, prefix=settings.api_prefix, tags=["status"])
    
    # Phase 1: RAG endpoints
    app.include_router(documents.router, tags=["documents"])
    app.include_router(chat.router, tags=["chat"])
    
    # Admin endpoints
    app.include_router(admin.router, tags=["admin"])

    # Phase 5: Evaluation endpoints
    app.include_router(evaluate.router, tags=["evaluation"])

    # Phase 6: Memory endpoints
    app.include_router(memory.router, tags=["memory"])

    # Phase 7: Guardrails endpoints
    app.include_router(guardrails.router, tags=["guardrails"])

    # Phase 9: Agentic RAG endpoints
    app.include_router(agent.router, tags=["agent"])

    # Phase 10: Auth endpoints (always registered; enforced only when auth_enabled=True)
    app.include_router(auth.router, tags=["auth"])

    # Phase 10: Feedback collection
    app.include_router(feedback.router, tags=["feedback"])

    # Phase 11: Multi-Agent Ecosystem
    app.include_router(multi_agent.router, tags=["multi-agent"])

    # Phase 12: Knowledge Graph Enhancement
    app.include_router(kg_router, tags=["knowledge-graph"])
    
    # Phase 10: OpenTelemetry HTTP instrumentation (no-op when otel_enabled=False)
    add_otel_middleware(app)

    @app.on_event("startup")
    async def startup_event():
        """Application startup event handler."""
        setup_tracing()  # Phase 10: initialise LangSmith + OTel

        # F-04: Warn (dev) or hard-fail (prod) when the JWT secret is the insecure default
        _INSECURE_SECRET = "change-me-in-production"
        if settings.jwt_secret_key == _INSECURE_SECRET:
            if settings.is_production:
                raise RuntimeError(
                    "JWT_SECRET_KEY must be changed from the default value before running "
                    "in production. Set a random 32+ character value in your .env file."
                )
            else:
                logger.warning(
                    "SECURITY WARNING: JWT_SECRET_KEY is still the default insecure value. "
                    "Set JWT_SECRET_KEY in .env before any non-local deployment."
                )

        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"API listening on {settings.api_host}:{settings.api_port}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event handler."""
        logger.info("Shutting down application")
    
    @app.get("/")
    async def root():
        """Root endpoint with basic application information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "environment": settings.environment,
            "features": {
                "document_upload": True,
                "rag_chat": True,
                "supported_formats": ["pdf", "docx"]
            },
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "documents": "/api/v1/documents",
                "chat": "/api/v1/chat",
                "admin": "/api/v1/admin",
                "evaluate": "/api/v1/evaluate",
                "memory":      "/api/v1/memory",
                "guardrails":  "/api/v1/guardrails",
                "agent":       "/api/v1/agent",
                "auth":        "/auth/token",
                "feedback":    "/api/v1/feedback",
                "multi_agent":      "/api/v1/multi-agent",
                "knowledge_graph":  "/api/v1/kg",
            }
        }
    
    return app


# Create application instance
app = create_app()

# Made with Bob
