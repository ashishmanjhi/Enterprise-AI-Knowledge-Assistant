"""
Document retrieval service for RAG.
Handles similarity search and result ranking.
"""

from typing import List, Dict, Any, Optional
import time
from backend.llm.embeddings import EmbeddingService
from backend.retrievers.vector_store import FAISSVectorStore
from backend.retrievers.vector_store_manager import get_shared_vector_store
from backend.core.logging import get_logger

logger = get_logger(__name__)


class RetrievalResult:
    """Container for a single retrieval result."""
    
    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any]
    ):
        """
        Initialize retrieval result.
        
        Args:
            chunk_id: Unique chunk identifier
            content: Chunk content
            score: Similarity score
            metadata: Chunk metadata
        """
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.metadata = metadata
        
        # Extract common metadata fields
        self.document_id = metadata.get("document_id", "unknown")
        self.filename = metadata.get("filename", "unknown")
        self.page_number = metadata.get("page_number")
        self.chunk_index = metadata.get("chunk_index", 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "content": self.content,
            "score": self.score,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        return (
            f"RetrievalResult(chunk_id={self.chunk_id}, "
            f"document_id={self.document_id}, score={self.score:.3f})"
        )


class DocumentRetriever:
    """
    Service for retrieving relevant documents.
    
    Provides similarity search with filtering and ranking capabilities.
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[FAISSVectorStore] = None
    ):
        """
        Initialize retriever.
        
        Args:
            embedding_service: Service for generating query embeddings
            vector_store: Vector store for similarity search (uses shared instance if not provided)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or get_shared_vector_store()
        
        logger.info("Initialized DocumentRetriever")
        logger.info(f"Vector store contains {self.vector_store.index.ntotal} vectors")
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            document_ids: Filter by document IDs
            file_types: Filter by file types
            min_score: Minimum similarity score threshold
            
        Returns:
            List of retrieval results
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding,
                k=top_k * 2  # Get more results for filtering
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for metadata, distance in results:
                # Convert distance to similarity score (lower distance = higher similarity)
                score = 1.0 / (1.0 + distance)
                
                # Apply filters
                if document_ids and metadata.get("document_id") not in document_ids:
                    continue
                
                if file_types and metadata.get("file_type") not in file_types:
                    continue
                
                if score < min_score:
                    continue
                
                # Get content and chunk_id from metadata
                content = metadata.get("content", "")
                chunk_id = metadata.get("chunk_id", f"chunk_{metadata.get('chunk_index', 0)}")
                
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    content=content,
                    score=score,
                    metadata=metadata
                )
                
                retrieval_results.append(result)
                
                # Stop if we have enough results
                if len(retrieval_results) >= top_k:
                    break
            
            retrieval_time = time.time() - start_time
            
            logger.info(
                f"Retrieved {len(retrieval_results)} documents for query "
                f"in {retrieval_time:.3f}s"
            )
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    async def retrieve_by_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks from a specific document.
        
        Args:
            query: Search query
            document_id: Document ID to search within
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        return await self.retrieve(
            query=query,
            top_k=top_k,
            document_ids=[document_id]
        )
    
    async def retrieve_similar_chunks(
        self,
        chunk_id: str,
        top_k: int = 5,
        exclude_same_document: bool = False
    ) -> List[RetrievalResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: Reference chunk ID
            top_k: Number of results to return
            exclude_same_document: Whether to exclude chunks from same document
            
        Returns:
            List of similar chunks
        """
        try:
            # Get the chunk's embedding from vector store
            chunk_data = self.vector_store.get_by_id(chunk_id)
            
            if not chunk_data:
                logger.warning(f"Chunk not found: {chunk_id}")
                return []
            
            embedding = chunk_data["embedding"]
            metadata = chunk_data["metadata"]
            
            # Search for similar chunks
            results = self.vector_store.search(
                embedding,
                k=top_k + 1  # +1 to account for the chunk itself
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            source_doc_id = metadata.get("document_id")
            
            for result_metadata, distance in results:
                # Convert distance to similarity score
                score = 1.0 / (1.0 + distance)
                
                # Get chunk_id from metadata
                result_chunk_id = result_metadata.get("chunk_id", f"chunk_{result_metadata.get('chunk_index', 0)}")
                
                # Skip the source chunk itself
                if result_chunk_id == chunk_id:
                    continue
                
                # Skip same document if requested
                if exclude_same_document:
                    if result_metadata.get("document_id") == source_doc_id:
                        continue
                
                content = result_metadata.get("content", "")
                
                result = RetrievalResult(
                    chunk_id=result_chunk_id,
                    content=content,
                    score=score,
                    metadata=result_metadata
                )
                
                retrieval_results.append(result)
                
                if len(retrieval_results) >= top_k:
                    break
            
            logger.info(f"Found {len(retrieval_results)} similar chunks")
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Similar chunk retrieval failed: {e}")
            raise
    
    def format_context(
        self,
        results: List[RetrievalResult],
        max_length: Optional[int] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieval results into context string for LLM.
        
        Args:
            results: List of retrieval results
            max_length: Maximum context length in characters
            include_metadata: Whether to include source metadata
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            # Format chunk with metadata
            if include_metadata:
                chunk_text = (
                    f"[Source {i}: {result.filename}"
                )
                
                if result.page_number:
                    chunk_text += f", Page {result.page_number}"
                
                chunk_text += f"]\n{result.content}\n"
            else:
                chunk_text = f"{result.content}\n\n"
            
            # Check length limit
            if max_length:
                if current_length + len(chunk_text) > max_length:
                    # Try to fit partial content
                    remaining = max_length - current_length
                    if remaining > 100:  # Only add if meaningful
                        chunk_text = chunk_text[:remaining] + "...\n"
                        context_parts.append(chunk_text)
                    break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "vector_store": self.vector_store.get_stats(),
            "embedding_service": self.embedding_service.get_model_info()
        }


# Made with Bob