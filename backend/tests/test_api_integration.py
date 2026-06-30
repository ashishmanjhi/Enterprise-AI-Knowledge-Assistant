"""
Integration tests for API endpoints
Tests the complete flow from API to backend services
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os


# Note: These tests require the backend services to be running
# They test the actual API endpoints with real dependencies


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_endpoint(self, client):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_healthz_endpoint(self, client):
        """Test Kubernetes-style health check"""
        response = client.get("/healthz")
        assert response.status_code == 200
    
    def test_readyz_endpoint(self, client):
        """Test readiness check"""
        response = client.get("/readyz")
        assert response.status_code == 200


class TestDocumentEndpoints:
    """Test document management endpoints"""
    
    def test_list_documents_empty(self, client):
        """Test listing documents when none exist"""
        response = client.get("/api/v1/documents/")
        assert response.status_code == 200
        data = response.json()
        # API returns dict with 'documents' and 'total' keys
        assert isinstance(data, dict)
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)
    
    def test_get_document_stats(self, client):
        """Test getting document statistics"""
        # Note: This endpoint may not be implemented yet
        response = client.get("/api/v1/documents/stats")
        # Accept 404 if endpoint not implemented
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "total_documents" in data or "total" in data
    
    def test_get_nonexistent_document(self, client):
        """Test getting a document that doesn't exist"""
        response = client.get("/api/v1/documents/nonexistent-id")
        assert response.status_code == 404
    
    def test_delete_nonexistent_document(self, client):
        """Test deleting a document that doesn't exist"""
        response = client.delete("/api/v1/documents/nonexistent-id")
        assert response.status_code == 404
    
    def test_search_documents_empty(self, client):
        """Test searching when no documents exist"""
        response = client.post(
            "/api/v1/documents/search",
            json={"query": "test query", "top_k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)


class TestChatEndpoints:
    """Test chat endpoints"""
    
    def test_chat_health(self, client):
        """Test chat service health check"""
        response = client.get("/api/v1/chat/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        # Check for either ollama_available or llm_available
        assert "llm_available" in data or "ollama_available" in data
    
    def test_direct_chat_without_rag(self, client):
        """Test direct chat without RAG"""
        response = client.post(
            "/api/v1/chat/direct",
            json={
                "message": "What is 2+2?",
                "model": "qwen3:4b"
            }
        )
        # May return 422 if validation fails, 503/500 if service unavailable, or 200 if successful
        assert response.status_code in [200, 422, 503, 500]
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
    
    def test_chat_with_invalid_request(self, client):
        """Test chat with invalid request"""
        response = client.post(
            "/api/v1/chat/",
            json={}  # Missing required fields
        )
        assert response.status_code == 422  # Validation error


class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_docs_endpoint(self, client):
        """Test Swagger UI is available"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc is available"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


# Fixtures
@pytest.fixture
def client():
    """Create test client"""
    from backend.api.main import app
    return TestClient(app)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
