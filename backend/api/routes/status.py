"""
Status check endpoints with service connectivity validation.
"""

from fastapi import APIRouter
from backend.core.settings import settings
from backend.core.logging import get_logger
import psycopg2
import redis
import requests

logger = get_logger(__name__)
router = APIRouter()


@router.get("/status")
async def get_status():
    """
    Comprehensive status check for all services.
    
    Returns:
        dict: Status of application and all connected services
    """
    logger.info("Status check requested")
    
    from backend.retrievers.tenant_registry import list_tenants

    status = {
        "application": "running",
        "version": settings.app_version,
        "environment": settings.environment,
        "multi_tenancy_enabled": settings.multi_tenancy_enabled,
        "tenants": list_tenants() if settings.multi_tenancy_enabled else [],
        "services": {}
    }
    
    # Check PostgreSQL
    try:
        conn = psycopg2.connect(settings.database_url, connect_timeout=5)
        conn.close()
        status["services"]["postgresql"] = "connected"
        logger.debug("PostgreSQL connection successful")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        status["services"]["postgresql"] = "disconnected"
    
    # Check Redis
    try:
        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=5)
        r.ping()
        status["services"]["redis"] = "connected"
        logger.debug("Redis connection successful")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        status["services"]["redis"] = "disconnected"
    
    # Check Ollama
    try:
        response = requests.get(
            f"{settings.ollama_host}/api/tags",
            timeout=5
        )
        if response.status_code == 200:
            status["services"]["ollama"] = "connected"
            # Get available models
            models = response.json().get("models", [])
            status["services"]["ollama_models"] = [m.get("name") for m in models]
            logger.debug(f"Ollama connection successful, {len(models)} models available")
        else:
            status["services"]["ollama"] = "disconnected"
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        status["services"]["ollama"] = "disconnected"
    
    return status


@router.get("/status/providers")
async def get_providers_status():
    """
    Get status of all LLM providers.
    
    Returns:
        dict: Status of each provider
    """
    from backend.providers.factory import LLMFactory
    
    providers = LLMFactory.list_providers()
    status = {
        "available_providers": providers,
        "default_provider": settings.default_provider,
        "fallback_provider": settings.fallback_provider
    }
    
    return status

# Made with Bob
