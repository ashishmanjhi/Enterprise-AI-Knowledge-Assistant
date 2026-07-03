"""
Embedding service using sentence-transformers.
Generates vector embeddings for text chunks using BAAI/bge-small-en-v1.5.
"""

import asyncio
from functools import partial
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.core.settings import settings
from backend.core.logging import get_logger
import torch

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Uses sentence-transformers with BAAI/bge-small-en-v1.5 model
    for high-quality embeddings optimized for retrieval.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None
    ):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing EmbeddingService with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Model loaded successfully. Embedding dimension: {self.dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings
            
        Returns:
            NumPy array of embeddings with shape (n_texts, dimension)
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return np.array([])
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")

            # Run CPU-bound encode() in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                partial(
                    self.model.encode,
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                ),
            )

            logger.info(
                f"Generated embeddings with shape: {embeddings.shape}"
            )

            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def embed_query(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            text: Query text to embed
            normalize: Whether to normalize embedding
            
        Returns:
            NumPy array of embedding with shape (dimension,)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for query embedding")
            return np.zeros(self.dimension)
        
        try:
            # Run CPU-bound encode() in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                partial(
                    self.model.encode,
                    [text],
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                ),
            )
            return result[0]
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings in batches.
        
        Useful for processing large numbers of texts with memory constraints.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size (uses default if None)
            
        Returns:
            List of embedding arrays
        """
        batch_size = batch_size or self.batch_size
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.embed_documents(
                batch,
                show_progress=False
            )
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_seq_length": self.model.max_seq_length,
        }
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine', 'dot', 'euclidean')
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            return float(
                np.dot(embedding1, embedding2) /
                (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            )
        elif metric == "dot":
            # Dot product
            return float(np.dot(embedding1, embedding2))
        elif metric == "euclidean":
            # Negative Euclidean distance (higher is more similar)
            return -float(np.linalg.norm(embedding1 - embedding2))
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Calculate similarity between query and multiple documents.
        
        Args:
            query_embedding: Query embedding (dimension,)
            document_embeddings: Document embeddings (n_docs, dimension)
            metric: Similarity metric
            
        Returns:
            Array of similarity scores (n_docs,)
        """
        if metric == "cosine":
            # Cosine similarity (assuming normalized embeddings)
            return np.dot(document_embeddings, query_embedding)
        elif metric == "dot":
            # Dot product
            return np.dot(document_embeddings, query_embedding)
        elif metric == "euclidean":
            # Negative Euclidean distance
            return -np.linalg.norm(
                document_embeddings - query_embedding,
                axis=1
            )
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings.
    
    Useful for avoiding re-computation of embeddings for frequently
    accessed texts.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, text: str) -> Union[np.ndarray, None]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to look up
            
        Returns:
            Cached embedding or None if not found
        """
        if text in self.cache:
            self.access_count[text] = self.access_count.get(text, 0) + 1
            return self.cache[text]
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """
        Add embedding to cache.
        
        Args:
            text: Text key
            embedding: Embedding to cache
        """
        # Evict least accessed item if cache is full
        if len(self.cache) >= self.max_size:
            least_accessed = min(
                self.access_count.items(),
                key=lambda x: x[1]
            )[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
        
        self.cache[text] = embedding
        self.access_count[text] = 0
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


# Global embedding service instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create global embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    
    return _embedding_service


# Made with Bob