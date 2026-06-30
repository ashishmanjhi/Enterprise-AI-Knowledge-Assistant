"""
Phase 4 Tests: Cross-Encoder Reranker

Tests for CrossEncoderReranker using a mock cross-encoder model so no
model download is required during CI.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from backend.rerankers.cross_encoder import CrossEncoderReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, content: str, score: float = 0.5):
    """Build a minimal HybridRetrievalResult-like object."""
    result = MagicMock()
    result.chunk_id = chunk_id
    result.content = content
    result.score = score
    result.metadata = {"document_id": "doc1", "filename": "test.pdf"}
    result.retrieval_method = "hybrid"
    return result


def _make_reranker(top_n: int = 3) -> CrossEncoderReranker:
    """Return a reranker with a mocked cross-encoder model."""
    reranker = CrossEncoderReranker(model_name="mock-model", top_n=top_n)
    mock_model = MagicMock()
    # predict() returns scores as a numpy array
    mock_model.predict = MagicMock(
        return_value=np.array([0.9, 0.3, 0.7, 0.1, 0.5])
    )
    reranker._model = mock_model
    return reranker


# ===========================================================================
# CrossEncoderReranker Tests
# ===========================================================================

class TestCrossEncoderReranker:

    def test_init_defaults(self):
        r = CrossEncoderReranker()
        assert r.model_name == CrossEncoderReranker.DEFAULT_MODEL
        assert r.top_n == 5
        assert r.batch_size == 16
        assert r._model is None  # lazy-loaded

    def test_init_custom(self):
        r = CrossEncoderReranker(model_name="custom/model", top_n=3, batch_size=8)
        assert r.model_name == "custom/model"
        assert r.top_n == 3
        assert r.batch_size == 8

    def test_rerank_empty_returns_empty(self):
        r = CrossEncoderReranker()
        assert r.rerank("any query", []) == []

    def test_rerank_sorts_by_score_descending(self):
        reranker = _make_reranker(top_n=5)
        # Mock returns scores [0.9, 0.3, 0.7, 0.1, 0.5] for chunks 0-4
        reranker._model.predict = MagicMock(
            return_value=np.array([0.9, 0.3, 0.7, 0.1, 0.5])
        )
        results = [_make_result(f"chunk_{i}", f"content {i}") for i in range(5)]

        reranked = reranker.rerank("test query", results)

        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True), "Results should be sorted descending"

    def test_rerank_truncates_to_top_n(self):
        reranker = _make_reranker(top_n=2)
        reranker._model.predict = MagicMock(
            return_value=np.array([0.9, 0.3, 0.7])
        )
        results = [_make_result(f"chunk_{i}", f"content {i}") for i in range(3)]

        reranked = reranker.rerank("test query", results)

        assert len(reranked) == 2

    def test_rerank_attaches_rerank_score_attribute(self):
        reranker = _make_reranker(top_n=3)
        reranker._model.predict = MagicMock(
            return_value=np.array([0.8, 0.4, 0.6])
        )
        results = [_make_result(f"chunk_{i}", f"content {i}") for i in range(3)]

        reranked = reranker.rerank("test query", results)

        for r in reranked:
            assert hasattr(r, "rerank_score")
            assert hasattr(r, "retrieval_score_before_rerank")

    def test_rerank_overwrites_score_with_reranker_score(self):
        reranker = _make_reranker(top_n=3)
        reranker._model.predict = MagicMock(
            return_value=np.array([0.8, 0.4, 0.6])
        )
        results = [_make_result(f"chunk_{i}", f"content {i}", score=0.5) for i in range(3)]

        reranked = reranker.rerank("test query", results)

        # Scores should now be the cross-encoder scores, not 0.5
        cross_encoder_scores = {0.8, 0.4, 0.6}
        result_scores = {round(r.score, 5) for r in reranked}
        assert result_scores == cross_encoder_scores

    def test_rerank_passes_query_passage_pairs(self):
        reranker = _make_reranker(top_n=2)
        reranker._model.predict = MagicMock(return_value=np.array([0.9, 0.2]))
        results = [
            _make_result("c1", "first passage"),
            _make_result("c2", "second passage"),
        ]

        reranker.rerank("my query", results)

        called_pairs = reranker._model.predict.call_args[0][0]
        assert called_pairs[0] == ["my query", "first passage"]
        assert called_pairs[1] == ["my query", "second passage"]

    def test_score_in_batches_splits_correctly(self):
        reranker = CrossEncoderReranker(model_name="mock", top_n=5, batch_size=2)
        reranker._model = MagicMock()
        # Each batch call returns 2 scores
        reranker._model.predict = MagicMock(
            side_effect=[
                np.array([0.9, 0.8]),
                np.array([0.7, 0.6]),
                np.array([0.5]),
            ]
        )

        pairs = [["q", f"p{i}"] for i in range(5)]
        scores = reranker._score_in_batches(pairs)

        assert len(scores) == 5
        assert reranker._model.predict.call_count == 3  # ceil(5/2) = 3 batches

    def test_get_stats(self):
        reranker = CrossEncoderReranker(model_name="test/model", top_n=4, batch_size=8)
        stats = reranker.get_stats()
        assert stats["model"] == "test/model"
        assert stats["top_n"] == 4
        assert stats["batch_size"] == 8
        assert stats["model_loaded"] is False

    def test_load_model_raises_on_missing_dependency(self):
        reranker = CrossEncoderReranker()
        with patch.dict("sys.modules", {"sentence_transformers": None,
                                         "sentence_transformers.cross_encoder": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                reranker._load_model()


# Made with Bob
