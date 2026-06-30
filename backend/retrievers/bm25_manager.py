"""
Shared BM25 retriever manager.
Ensures a single BM25 retriever instance is used across the application.
"""

from typing import Optional
from backend.retrievers.bm25_retriever import BM25Retriever
from backend.core.logging import get_logger

logger = get_logger(__name__)

# Global shared BM25 retriever instance
_shared_bm25_retriever: Optional[BM25Retriever] = None


def get_shared_bm25_retriever() -> BM25Retriever:
    """
    Get the shared BM25 retriever instance.
    Creates one if it doesn't exist.
    
    Returns:
        Shared BM25Retriever instance
    """
    global _shared_bm25_retriever
    
    if _shared_bm25_retriever is None:
        logger.info("Creating shared BM25 retriever instance")
        _shared_bm25_retriever = BM25Retriever()
        
        # Try to load existing index
        try:
            _shared_bm25_retriever.load()
            logger.info(f"Loaded existing BM25 index with {len(_shared_bm25_retriever.documents)} documents")
        except FileNotFoundError:
            logger.info("No existing BM25 index found")
    
    return _shared_bm25_retriever


def reset_shared_bm25_retriever() -> None:
    """
    Reset the shared BM25 retriever instance.
    Useful for testing or when clearing the index.
    """
    global _shared_bm25_retriever
    _shared_bm25_retriever = None
    logger.info("Reset shared BM25 retriever instance")


# Made with Bob