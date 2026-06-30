"""
Reciprocal Rank Fusion (RRF) for merging results from multiple retrievers.
Combines rankings from different retrieval methods into a unified ranking.
"""

from typing import List, Dict, Any, Set
from collections import defaultdict
from backend.core.logging import get_logger

logger = get_logger(__name__)


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion algorithm for combining retrieval results.
    
    RRF is a simple yet effective method for merging ranked lists from
    multiple retrieval systems. It assigns scores based on the reciprocal
    of the rank, making it robust to differences in score scales.
    
    Formula: RRF_score(doc) = Σ 1 / (k + rank_i(doc))
    where:
        - k is a constant (typically 60)
        - rank_i(doc) is the rank of doc in retriever i
        - Σ is the sum across all retrievers
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.
        
        Args:
            k: RRF constant (default 60, as per original paper)
               Higher k gives more weight to lower-ranked documents
        """
        self.k = k
        logger.info(f"Initialized ReciprocalRankFusion with k={k}")
    
    def fuse(
        self,
        results_lists: List[List[Dict[str, Any]]],
        top_k: int = 5,
        doc_id_key: str = "chunk_id"
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple result lists using RRF.
        
        Args:
            results_lists: List of result lists from different retrievers
                          Each result should have a doc_id_key field
            top_k: Number of final results to return
            doc_id_key: Key to use for document identification
            
        Returns:
            Fused and re-ranked results
        """
        if not results_lists:
            logger.warning("No results lists provided for fusion")
            return []
        
        # Filter out empty lists
        results_lists = [r for r in results_lists if r]
        
        if not results_lists:
            logger.warning("All results lists are empty")
            return []
        
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, Dict[str, Any]] = {}
        doc_ranks: Dict[str, List[int]] = defaultdict(list)
        doc_sources: Dict[str, List[str]] = defaultdict(list)
        
        for retriever_idx, results in enumerate(results_lists):
            retriever_name = results[0].get("retrieval_method", f"retriever_{retriever_idx}") if results else f"retriever_{retriever_idx}"
            
            for rank, result in enumerate(results, start=1):
                # Get document ID
                doc_id = result.get(doc_id_key) or result.get("doc_id") or result.get("document_id")
                
                if not doc_id:
                    logger.warning(f"Result missing document ID: {result}")
                    continue
                
                # Calculate RRF score contribution
                rrf_score = 1.0 / (self.k + rank)
                rrf_scores[doc_id] += rrf_score
                
                # Store document data (use first occurrence)
                if doc_id not in doc_data:
                    doc_data[doc_id] = result.copy()
                
                # Track ranks and sources
                doc_ranks[doc_id].append(rank)
                doc_sources[doc_id].append(retriever_name)
        
        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create final results
        fused_results = []
        for final_rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], start=1):
            result = doc_data[doc_id].copy()
            
            # Add RRF metadata
            result["rrf_score"] = rrf_score
            result["final_rank"] = final_rank
            result["source_ranks"] = doc_ranks[doc_id]
            result["source_retrievers"] = doc_sources[doc_id]
            result["num_retrievers"] = len(doc_sources[doc_id])
            
            # Update score to RRF score
            result["score"] = rrf_score
            
            fused_results.append(result)
        
        logger.info(
            f"Fused {len(results_lists)} result lists into {len(fused_results)} results"
        )
        
        return fused_results
    
    def fuse_with_weights(
        self,
        results_lists: List[List[Dict[str, Any]]],
        weights: List[float],
        top_k: int = 5,
        doc_id_key: str = "chunk_id"
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple result lists using weighted RRF.
        
        Args:
            results_lists: List of result lists from different retrievers
            weights: Weight for each retriever (must match length of results_lists)
            top_k: Number of final results to return
            doc_id_key: Key to use for document identification
            
        Returns:
            Fused and re-ranked results
        """
        if len(results_lists) != len(weights):
            raise ValueError(
                f"Number of result lists ({len(results_lists)}) must match "
                f"number of weights ({len(weights)})"
            )
        
        if not results_lists:
            logger.warning("No results lists provided for fusion")
            return []
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            logger.warning("Total weight is 0, using equal weights")
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]
        
        # Filter out empty lists and adjust weights
        filtered_lists = []
        filtered_weights = []
        for results, weight in zip(results_lists, weights):
            if results:
                filtered_lists.append(results)
                filtered_weights.append(weight)
        
        if not filtered_lists:
            logger.warning("All results lists are empty")
            return []
        
        # Calculate weighted RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, Dict[str, Any]] = {}
        doc_ranks: Dict[str, List[int]] = defaultdict(list)
        doc_sources: Dict[str, List[str]] = defaultdict(list)
        doc_weights: Dict[str, List[float]] = defaultdict(list)
        
        for retriever_idx, (results, weight) in enumerate(zip(filtered_lists, filtered_weights)):
            retriever_name = results[0].get("retrieval_method", f"retriever_{retriever_idx}") if results else f"retriever_{retriever_idx}"
            
            for rank, result in enumerate(results, start=1):
                # Get document ID
                doc_id = result.get(doc_id_key) or result.get("doc_id") or result.get("document_id")
                
                if not doc_id:
                    continue
                
                # Calculate weighted RRF score contribution
                rrf_score = weight * (1.0 / (self.k + rank))
                rrf_scores[doc_id] += rrf_score
                
                # Store document data
                if doc_id not in doc_data:
                    doc_data[doc_id] = result.copy()
                
                # Track ranks, sources, and weights
                doc_ranks[doc_id].append(rank)
                doc_sources[doc_id].append(retriever_name)
                doc_weights[doc_id].append(weight)
        
        # Sort by weighted RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create final results
        fused_results = []
        for final_rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], start=1):
            result = doc_data[doc_id].copy()
            
            # Add RRF metadata
            result["rrf_score"] = rrf_score
            result["final_rank"] = final_rank
            result["source_ranks"] = doc_ranks[doc_id]
            result["source_retrievers"] = doc_sources[doc_id]
            result["source_weights"] = doc_weights[doc_id]
            result["num_retrievers"] = len(doc_sources[doc_id])
            
            # Update score to RRF score
            result["score"] = rrf_score
            
            fused_results.append(result)
        
        logger.info(
            f"Fused {len(filtered_lists)} weighted result lists into "
            f"{len(fused_results)} results"
        )
        
        return fused_results
    
    def get_overlap_stats(
        self,
        results_lists: List[List[Dict[str, Any]]],
        doc_id_key: str = "chunk_id"
    ) -> Dict[str, Any]:
        """
        Get statistics about overlap between result lists.
        
        Args:
            results_lists: List of result lists
            doc_id_key: Key to use for document identification
            
        Returns:
            Dictionary with overlap statistics
        """
        if not results_lists:
            return {"error": "No results lists provided"}
        
        # Get document IDs from each list
        doc_sets = []
        for results in results_lists:
            doc_ids = set()
            for result in results:
                doc_id = result.get(doc_id_key) or result.get("doc_id")
                if doc_id:
                    doc_ids.add(doc_id)
            doc_sets.append(doc_ids)
        
        # Calculate statistics
        all_docs = set.union(*doc_sets) if doc_sets else set()
        common_docs = set.intersection(*doc_sets) if doc_sets else set()
        
        stats = {
            "num_lists": len(results_lists),
            "total_unique_docs": len(all_docs),
            "common_docs": len(common_docs),
            "overlap_percentage": (len(common_docs) / len(all_docs) * 100) if all_docs else 0,
            "list_sizes": [len(results) for results in results_lists],
            "unique_per_list": [len(doc_set) for doc_set in doc_sets]
        }
        
        return stats


# Made with Bob