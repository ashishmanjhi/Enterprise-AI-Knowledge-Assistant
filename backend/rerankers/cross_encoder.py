"""
Cross-Encoder Reranker (Phase 4).

Re-scores retrieved chunks by reading the query and each chunk together,
producing a true joint relevance score rather than an independent embedding
similarity. This is more accurate than RRF ranking alone.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (default)
  - ~22 MB download on first use
  - Runs on CPU, ~10-50ms per chunk
  - Trained on MS MARCO passage ranking
"""

from typing import List, Optional, TYPE_CHECKING
import time

from backend.core.logging import get_logger

if TYPE_CHECKING:
    from backend.retrievers.hybrid_retriever import HybridRetrievalResult

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.

    A cross-encoder reads (query, passage) pairs jointly and outputs a
    relevance score. This is more accurate than bi-encoder similarity
    (FAISS) but too slow to run over the entire corpus — so we run it
    only on the top-K candidates already returned by the hybrid retriever.

    Pipeline position:
        Hybrid Retriever → [CrossEncoderReranker] → LLM
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_n: int = 5,
        batch_size: int = 16,
    ):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model ID for the cross-encoder.
            top_n: How many top results to return after reranking.
            batch_size: Number of pairs to score in one forward pass.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.top_n = top_n
        self.batch_size = batch_size
        self._model = None  # lazy-loaded on first use

        logger.info(
            f"Initialized CrossEncoderReranker: model={self.model_name}, "
            f"top_n={top_n}, batch_size={batch_size}"
        )

    def _load_model(self):
        """Lazy-load the cross-encoder model (downloads ~22 MB on first call)."""
        if self._model is None:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Run: pip install sentence-transformers"
                )

    def rerank(
        self,
        query: str,
        results: List["HybridRetrievalResult"],
    ) -> List["HybridRetrievalResult"]:
        """
        Rerank retrieval results by joint query-passage relevance.

        Args:
            query: The user query (or reformulated query).
            results: Candidate results from the hybrid retriever.

        Returns:
            Results re-sorted by cross-encoder score, truncated to top_n.
            Each result gets a new ``score`` (cross-encoder logit) and a
            ``rerank_score`` attribute for display.
        """
        if not results:
            return results

        start_time = time.time()
        self._load_model()

        # Build (query, passage) pairs
        pairs = [[query, r.content] for r in results]

        # Score in batches
        scores = self._score_in_batches(pairs)

        # Attach scores and sort descending
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
            result.retrieval_score_before_rerank = result.score
            result.score = float(score)  # overwrite score with reranker score

        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        top_results = reranked[: self.top_n]

        elapsed = time.time() - start_time
        logger.info(
            f"Reranked {len(results)} results → top {len(top_results)} "
            f"in {elapsed:.3f}s using {self.model_name}"
        )

        if results:
            top_score = top_results[0].score if top_results else 0
            logger.debug(f"Top reranked score: {top_score:.4f}")

        return top_results

    def _score_in_batches(self, pairs: List[List[str]]) -> List[float]:
        """Score query-passage pairs in batches for memory efficiency."""
        all_scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            batch_scores = self._model.predict(batch)
            all_scores.extend(batch_scores.tolist())
        return all_scores

    def get_stats(self) -> dict:
        """Return reranker configuration."""
        return {
            "model": self.model_name,
            "top_n": self.top_n,
            "batch_size": self.batch_size,
            "model_loaded": self._model is not None,
        }


# Made with Bob
