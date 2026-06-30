"""
Hybrid Retriever combining FAISS (semantic) and BM25 (keyword) search.
Uses Reciprocal Rank Fusion to merge results from both retrievers.
"""

from typing import List, Dict, Any, Optional, Literal
import asyncio
import time
from backend.retrievers.retriever import DocumentRetriever, RetrievalResult
from backend.retrievers.bm25_retriever import BM25Retriever, BM25RetrievalResult
from backend.retrievers.bm25_manager import get_shared_bm25_retriever
from backend.retrievers.fusion import ReciprocalRankFusion
from backend.core.logging import get_logger

logger = get_logger(__name__)


class HybridRetrievalResult:
    """
    Enhanced retrieval result with hybrid search metadata.
    """
    
    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
        retrieval_method: str,
        faiss_score: Optional[float] = None,
        bm25_score: Optional[float] = None,
        faiss_rank: Optional[int] = None,
        bm25_rank: Optional[int] = None
    ):
        """
        Initialize hybrid retrieval result.
        
        Args:
            chunk_id: Chunk identifier
            content: Chunk content
            score: Final hybrid score (RRF score)
            metadata: Chunk metadata
            retrieval_method: Method used (hybrid/faiss/bm25)
            faiss_score: FAISS similarity score
            bm25_score: BM25 relevance score
            faiss_rank: Rank in FAISS results
            bm25_rank: Rank in BM25 results
        """
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.metadata = metadata
        self.retrieval_method = retrieval_method
        self.faiss_score = faiss_score
        self.bm25_score = bm25_score
        self.faiss_rank = faiss_rank
        self.bm25_rank = bm25_rank
        
        # Extract common metadata
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
            "metadata": self.metadata,
            "retrieval_method": self.retrieval_method,
            "faiss_score": self.faiss_score,
            "bm25_score": self.bm25_score,
            "faiss_rank": self.faiss_rank,
            "bm25_rank": self.bm25_rank
        }


class HybridRetriever:
    """
    Hybrid retriever combining semantic (FAISS) and keyword (BM25) search.
    
    Uses Reciprocal Rank Fusion to intelligently merge results from both
    retrieval methods, providing better recall and precision.
    """
    
    def __init__(
        self,
        faiss_retriever: Optional[DocumentRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        faiss_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            faiss_retriever: FAISS-based semantic retriever
            bm25_retriever: BM25-based keyword retriever (uses shared instance if not provided)
            fusion: RRF fusion algorithm
            faiss_weight: Weight for FAISS results (0-1)
            bm25_weight: Weight for BM25 results (0-1)
            rrf_k: RRF constant
        """
        from backend.core.settings import settings
        
        self.faiss_retriever = faiss_retriever or DocumentRetriever()
        self.bm25_retriever = bm25_retriever or get_shared_bm25_retriever()
        
        self.fusion = fusion or ReciprocalRankFusion(k=rrf_k)
        
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        
        logger.info(
            f"Initialized HybridRetriever with "
            f"faiss_weight={faiss_weight}, bm25_weight={bm25_weight}, rrf_k={rrf_k}, "
            f"BM25 docs={len(self.bm25_retriever.documents)}"
        )
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: Literal["hybrid", "faiss", "bm25"] = "hybrid",
        document_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[HybridRetrievalResult]:
        """
        Retrieve documents using specified method.
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: Retrieval method (hybrid/faiss/bm25)
            document_ids: Filter by document IDs
            min_score: Minimum score threshold
            
        Returns:
            List of hybrid retrieval results
        """
        start_time = time.time()
        
        if method == "faiss":
            results = await self._retrieve_faiss_only(
                query, top_k, document_ids, min_score
            )
        elif method == "bm25":
            results = await self._retrieve_bm25_only(
                query, top_k, min_score
            )
        else:  # hybrid
            results = await self._retrieve_hybrid(
                query, top_k, document_ids, min_score
            )
        
        elapsed = time.time() - start_time
        logger.info(
            f"Hybrid retrieval ({method}) returned {len(results)} results "
            f"in {elapsed:.3f}s"
        )
        
        return results
    
    async def _retrieve_faiss_only(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
        min_score: float
    ) -> List[HybridRetrievalResult]:
        """Retrieve using FAISS only."""
        faiss_results = await self.faiss_retriever.retrieve(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
            min_score=min_score
        )
        
        # Convert to hybrid results
        hybrid_results = []
        for result in faiss_results:
            hybrid_result = HybridRetrievalResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                retrieval_method="faiss",
                faiss_score=result.score,
                faiss_rank=len(hybrid_results) + 1
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    async def _retrieve_bm25_only(
        self,
        query: str,
        top_k: int,
        min_score: float
    ) -> List[HybridRetrievalResult]:
        """Retrieve using BM25 only."""
        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=top_k,
            min_score=min_score
        )
        
        # Convert to hybrid results
        hybrid_results = []
        for result in bm25_results:
            hybrid_result = HybridRetrievalResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                retrieval_method="bm25",
                bm25_score=result.score,
                bm25_rank=result.rank
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    async def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
        min_score: float
    ) -> List[HybridRetrievalResult]:
        """Retrieve using hybrid approach (FAISS + BM25 + RRF)."""
        # Retrieve from both sources in parallel
        # Get more results than needed for better fusion
        retrieval_k = top_k * 2
        
        faiss_task = self.faiss_retriever.retrieve(
            query=query,
            top_k=retrieval_k,
            document_ids=document_ids,
            min_score=0.0  # Apply threshold after fusion
        )
        
        # BM25 search is synchronous, wrap in async
        async def bm25_search():
            return self.bm25_retriever.search(
                query=query,
                top_k=retrieval_k,
                min_score=0.0
            )
        
        # Run both retrievals in parallel
        faiss_results, bm25_results = await asyncio.gather(
            faiss_task,
            bm25_search()
        )
        
        logger.info(
            f"Retrieved {len(faiss_results)} FAISS results, "
            f"{len(bm25_results)} BM25 results"
        )
        
        # Convert to dictionaries for fusion
        faiss_dicts = [
            {
                **result.to_dict(),
                "retrieval_method": "faiss",
                "original_score": result.score
            }
            for result in faiss_results
        ]
        
        bm25_dicts = [
            {
                **result.to_dict(),
                "retrieval_method": "bm25",
                "original_score": result.score
            }
            for result in bm25_results
        ]
        
        # Fuse results using weighted RRF
        fused_results = self.fusion.fuse_with_weights(
            results_lists=[faiss_dicts, bm25_dicts],
            weights=[self.faiss_weight, self.bm25_weight],
            top_k=top_k,
            doc_id_key="chunk_id"
        )
        
        # Convert to hybrid results with enhanced metadata
        hybrid_results = []
        
        # Debug: Log chunk IDs for matching
        faiss_chunk_ids = {f.chunk_id for f in faiss_results}
        bm25_chunk_ids = {b.doc_id for b in bm25_results}
        logger.debug(f"FAISS chunk IDs sample: {list(faiss_chunk_ids)[:3]}")
        logger.debug(f"BM25 doc IDs sample: {list(bm25_chunk_ids)[:3]}")
        
        for fused in fused_results:
            # Extract scores and ranks from fusion metadata
            source_retrievers = fused.get("source_retrievers", [])
            source_ranks = fused.get("source_ranks", [])
            
            faiss_score = None
            bm25_score = None
            faiss_rank = None
            bm25_rank = None
            
            # Map scores and ranks
            for retriever, rank in zip(source_retrievers, source_ranks):
                if retriever == "faiss":
                    faiss_rank = rank
                    # Find original FAISS score
                    for f_result in faiss_results:
                        if f_result.chunk_id == fused["chunk_id"]:
                            faiss_score = f_result.score
                            break
                elif retriever == "bm25":
                    bm25_rank = rank
                    # Find original BM25 score
                    for b_result in bm25_results:
                        # BM25 results use doc_id which should match chunk_id
                        if b_result.doc_id == fused["chunk_id"] or b_result.chunk_id == fused["chunk_id"]:
                            bm25_score = b_result.score
                            break
            
            # Apply minimum score threshold
            if fused["score"] < min_score:
                continue
            
            hybrid_result = HybridRetrievalResult(
                chunk_id=fused["chunk_id"],
                content=fused["content"],
                score=fused["score"],  # RRF score
                metadata=fused["metadata"],
                retrieval_method="hybrid",
                faiss_score=faiss_score,
                bm25_score=bm25_score,
                faiss_rank=faiss_rank,
                bm25_rank=bm25_rank
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get hybrid retriever statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "faiss_stats": self.faiss_retriever.get_stats(),
            "bm25_stats": self.bm25_retriever.get_stats(),
            "faiss_weight": self.faiss_weight,
            "bm25_weight": self.bm25_weight,
            "rrf_k": self.fusion.k
        }
    
    def format_context(
        self,
        results: List[HybridRetrievalResult],
        max_length: Optional[int] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieval results into context string for LLM.
        
        Args:
            results: List of hybrid retrieval results
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
                chunk_text = f"[Source {i}: {result.filename}"
                
                if result.page_number:
                    chunk_text += f", Page {result.page_number}"
                
                # Add retrieval method info
                if result.retrieval_method == "hybrid":
                    methods = []
                    if result.faiss_rank:
                        methods.append(f"Semantic#{result.faiss_rank}")
                    if result.bm25_rank:
                        methods.append(f"Keyword#{result.bm25_rank}")
                    chunk_text += f" ({', '.join(methods)})"
                elif result.retrieval_method == "faiss":
                    chunk_text += " (Semantic)"
                elif result.retrieval_method == "bm25":
                    chunk_text += " (Keyword)"
                
                chunk_text += f"]\n{result.content}\n"
            else:
                chunk_text = f"{result.content}\n\n"
            
            # Check length limit
            if max_length:
                if current_length + len(chunk_text) > max_length:
                    remaining = max_length - current_length
                    if remaining > 100:
                        chunk_text = chunk_text[:remaining] + "...\n"
                        context_parts.append(chunk_text)
                    break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)


# Made with Bob