"""
BM25 Retriever for keyword-based document search.
Implements the BM25 algorithm for efficient keyword matching.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import time
from rank_bm25 import BM25Okapi
from backend.core.logging import get_logger

logger = get_logger(__name__)


class BM25Document:
    """Container for a document in BM25 index."""
    
    def __init__(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        tokens: List[str]
    ):
        """
        Initialize BM25 document.
        
        Args:
            doc_id: Unique document identifier
            content: Original document content
            metadata: Document metadata
            tokens: Tokenized content
        """
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata
        self.tokens = tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "tokens": self.tokens
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BM25Document':
        """Create from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            metadata=data["metadata"],
            tokens=data["tokens"]
        )


class BM25RetrievalResult:
    """Container for a BM25 retrieval result."""
    
    def __init__(
        self,
        doc_id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
        rank: int
    ):
        """
        Initialize BM25 retrieval result.
        
        Args:
            doc_id: Document identifier
            content: Document content
            score: BM25 score
            metadata: Document metadata
            rank: Rank in results (1-based)
        """
        self.doc_id = doc_id
        self.content = content
        self.score = score
        self.metadata = metadata
        self.rank = rank
        
        # Extract common metadata
        self.chunk_id = metadata.get("chunk_id", doc_id)
        self.document_id = metadata.get("document_id", "unknown")
        self.filename = metadata.get("filename", "unknown")
        self.page_number = metadata.get("page_number")
        self.chunk_index = metadata.get("chunk_index", 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "retrieval_method": "bm25"
        }


class BM25Retriever:
    """
    BM25-based keyword retriever.
    
    Implements BM25 algorithm for fast keyword-based document retrieval.
    Complements semantic search with exact term matching.
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            index_path: Path to save/load BM25 index
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        from backend.core.settings import settings
        
        self.index_path = Path(index_path) if index_path else Path(settings.bm25_index_path)
        self.k1 = k1
        self.b = b
        
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[BM25Document] = []
        self.doc_id_to_index: Dict[str, int] = {}
        
        logger.info(f"Initialized BM25Retriever with k1={k1}, b={b}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Simple tokenization: lowercase + split on whitespace.
        Can be enhanced with stemming, stopword removal, etc.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization
        tokens = text.lower().split()
        
        # Remove very short tokens (< 2 chars)
        tokens = [t for t in tokens if len(t) >= 2]
        
        return tokens
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Add documents to BM25 index.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
            
        Returns:
            Number of documents added
        """
        start_time = time.time()
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Generate doc_id if not provided
            doc_id = metadata.get("chunk_id") or metadata.get("document_id") or f"doc_{len(self.documents)}"
            
            # Tokenize content
            tokens = self._tokenize(content)
            
            # Create BM25 document
            bm25_doc = BM25Document(
                doc_id=doc_id,
                content=content,
                metadata=metadata,
                tokens=tokens
            )
            
            # Add to index
            self.doc_id_to_index[doc_id] = len(self.documents)
            self.documents.append(bm25_doc)
        
        # Build BM25 index
        if self.documents:
            corpus = [doc.tokens for doc in self.documents]
            self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        
        elapsed = time.time() - start_time
        logger.info(f"Added {len(documents)} documents to BM25 index in {elapsed:.3f}s")
        
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[BM25RetrievalResult]:
        """
        Search documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum BM25 score threshold
            
        Returns:
            List of BM25 retrieval results
        """
        if not self.bm25 or not self.documents:
            logger.warning("BM25 index is empty")
            return []
        
        start_time = time.time()
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            logger.warning("Query tokenization resulted in empty tokens")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top K indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Create results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            
            # Apply minimum score threshold
            if score < min_score:
                continue
            
            doc = self.documents[idx]
            
            result = BM25RetrievalResult(
                doc_id=doc.doc_id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
                rank=rank
            )
            
            results.append(result)
        
        elapsed = time.time() - start_time
        logger.info(f"BM25 search returned {len(results)} results in {elapsed:.3f}s")
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save BM25 index to disk.
        
        Args:
            path: Path to save index (uses default if not provided)
        """
        save_path = Path(path) if path else self.index_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare data for serialization
            data = {
                "documents": [doc.to_dict() for doc in self.documents],
                "doc_id_to_index": self.doc_id_to_index,
                "k1": self.k1,
                "b": self.b,
                "version": "1.0"
            }
            
            # Save using pickle
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved BM25 index to {save_path}")
            logger.info(f"Index contains {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            raise
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load BM25 index from disk.
        
        Args:
            path: Path to load index from (uses default if not provided)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = Path(path) if path else self.index_path
        
        if not load_path.exists():
            logger.warning(f"BM25 index file not found: {load_path}")
            return False
        
        try:
            # Load data
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            # Restore documents
            self.documents = [
                BM25Document.from_dict(doc_data)
                for doc_data in data["documents"]
            ]
            
            self.doc_id_to_index = data["doc_id_to_index"]
            self.k1 = data.get("k1", self.k1)
            self.b = data.get("b", self.b)
            
            # Rebuild BM25 index
            if self.documents:
                corpus = [doc.tokens for doc in self.documents]
                self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
            
            logger.info(f"Loaded BM25 index from {load_path}")
            logger.info(f"Index contains {len(self.documents)} documents")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
    
    def clear(self) -> None:
        """Clear the BM25 index."""
        self.bm25 = None
        self.documents = []
        self.doc_id_to_index = {}
        logger.info("Cleared BM25 index")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get BM25 index statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.documents),
            "k1": self.k1,
            "b": self.b,
            "index_built": self.bm25 is not None,
            "avg_doc_length": sum(len(doc.tokens) for doc in self.documents) / len(self.documents) if self.documents else 0
        }


# Made with Bob