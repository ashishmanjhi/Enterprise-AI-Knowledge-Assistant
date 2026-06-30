"""
Evaluation metrics definitions (Phase 5).

Defines the four core RAGAS metrics used to measure RAG quality,
plus the data structures for evaluation samples and results.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class MetricName(str, Enum):
    """Supported evaluation metrics."""
    FAITHFULNESS          = "faithfulness"           # Does the answer stick to the context?
    ANSWER_RELEVANCY      = "answer_relevancy"       # Does the answer address the question?
    CONTEXT_PRECISION     = "context_precision"      # Are retrieved chunks relevant?
    CONTEXT_RECALL        = "context_recall"         # Was all needed info retrieved?
    FACTUAL_CORRECTNESS   = "factual_correctness"    # Is the answer factually correct?


# Metrics that require a ground-truth answer to compute
REQUIRES_GROUND_TRUTH = {
    MetricName.CONTEXT_RECALL,
    MetricName.FACTUAL_CORRECTNESS,
}

# Default set run when no metrics are specified
DEFAULT_METRICS = [
    MetricName.FAITHFULNESS,
    MetricName.ANSWER_RELEVANCY,
    MetricName.CONTEXT_PRECISION,
]


@dataclass
class EvaluationSample:
    """
    A single question-answer-context tuple for evaluation.

    Fields:
        question:       The user question asked.
        answer:         The LLM-generated answer.
        contexts:       List of retrieved chunk texts passed to the LLM.
        ground_truth:   Optional reference answer (needed for recall/correctness).
        metadata:       Optional extra info (document_id, retrieval_method, etc.).
    """
    question:     str
    answer:       str
    contexts:     List[str]
    ground_truth: Optional[str] = None
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def requires_ground_truth(self, metrics: List[MetricName]) -> bool:
        return any(m in REQUIRES_GROUND_TRUTH for m in metrics)

    def to_dict(self) -> dict:
        return {
            "question":     self.question,
            "answer":       self.answer,
            "contexts":     self.contexts,
            "ground_truth": self.ground_truth,
            "metadata":     self.metadata,
        }


@dataclass
class MetricScore:
    """Score for a single metric on a single sample."""
    metric:  MetricName
    score:   Optional[float]          # None if evaluation failed for this metric
    error:   Optional[str] = None     # Error message if scoring failed


@dataclass
class EvaluationResult:
    """
    Evaluation result for one or more samples.

    Attributes:
        sample_scores:  Per-sample per-metric scores.
        aggregated:     Mean score per metric across all samples.
        sample_count:   Number of samples evaluated.
        metrics_used:   Which metrics were computed.
        llm_model:      LLM used as judge.
        duration_s:     Total wall-clock time for the evaluation run.
    """
    sample_scores: List[List[MetricScore]]
    aggregated:    Dict[str, Optional[float]]
    sample_count:  int
    metrics_used:  List[MetricName]
    llm_model:     str
    duration_s:    float

    def to_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "metrics_used": [m.value for m in self.metrics_used],
            "aggregated":   self.aggregated,
            "llm_model":    self.llm_model,
            "duration_s":   round(self.duration_s, 3),
            "samples": [
                [{"metric": s.metric.value, "score": s.score, "error": s.error}
                 for s in sample]
                for sample in self.sample_scores
            ],
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = [f"{k}={v:.3f}" for k, v in self.aggregated.items() if v is not None]
        return f"[{self.sample_count} samples] " + " | ".join(parts)


# Made with Bob
