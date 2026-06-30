"""
Singleton vector store manager to ensure all components share the same vector store instance.
"""

from typing import Optional
from pathlib import Path
from backend.retrievers.vector_store import FAISSVectorStore
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """
    Singleton manager for vector store to ensure all components use the same instance.
    """
    
    _instance: Optional['VectorStoreManager'] = None
    _vector_store: Optional[FAISSVectorStore] = None
    
    def __new__(cls):
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_vector_store(self) -> FAISSVectorStore:
        """
        Get the shared vector store instance.
        
        Returns:
            Shared FAISSVectorStore instance
        """
        if self._vector_store is None:
            self._vector_store = FAISSVectorStore()
            
            # Load existing index if it exists
            index_path = Path(settings.faiss_index_path)
            if index_path.exists():
                try:
                    self._vector_store.load()
                    logger.info(f"Loaded existing FAISS index from {index_path}")
                    logger.info(f"Index contains {self._vector_store.index.ntotal} vectors")
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
            else:
                logger.info("No existing FAISS index found - starting with empty store")
        
        return self._vector_store
    
    def reload_index(self) -> bool:
        """
        Reload the vector store index from disk.
        
        Useful after documents are added to ensure the latest index is loaded.
        
        Returns:
            True if reload successful, False otherwise
        """
        if self._vector_store is None:
            return False
        
        index_path = Path(settings.faiss_index_path)
        if not index_path.exists():
            logger.warning(f"Index file not found: {index_path}")
            return False
        
        try:
            self._vector_store.load()
            logger.info(f"Reloaded FAISS index from {index_path}")
            logger.info(f"Index now contains {self._vector_store.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to reload FAISS index: {e}")
            return False
    
    def reset(self):
        """Reset the vector store instance (mainly for testing)."""
        self._vector_store = None
        logger.info("Vector store instance reset")


# Global function to get the shared vector store
def get_shared_vector_store() -> FAISSVectorStore:
    """
    Get the shared vector store instance.
    
    Returns:
        Shared FAISSVectorStore instance
    """
    manager = VectorStoreManager()
    return manager.get_vector_store()


def reload_vector_store() -> bool:
    """
    Reload the vector store from disk.
    
    Returns:
        True if reload successful, False otherwise
    """
    manager = VectorStoreManager()
    return manager.reload_index()


def reset_shared_vector_store() -> None:
    """
    Reset the shared vector store instance.
    Useful for clearing the index or testing.
    """
    manager = VectorStoreManager()
    manager.reset()


# Made with Bob