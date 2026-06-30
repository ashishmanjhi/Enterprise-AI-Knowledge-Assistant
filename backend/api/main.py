"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.settings import settings
from backend.core.logging import logger
from backend.api.routes import health, status, documents, chat, admin


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
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup event handler."""
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
                "admin": "/api/v1/admin"
            }
        }
    
    return app


# Create application instance
app = create_app()

# Made with Bob
