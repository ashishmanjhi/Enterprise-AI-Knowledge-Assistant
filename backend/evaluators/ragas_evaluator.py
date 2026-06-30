"""
RAGAS Evaluation Engine (Phase 5).

Wraps RAGAS 0.4.x to evaluate RAG quality using LLM-as-judge metrics.
Works with Ollama (local) or OpenAI — no external API key required if
Ollama is running.

Metrics computed:
  - Faithfulness:        Answer is grounded in the retrieved context
  - Answer Relevancy:    Answer addresses the actual question
  - Context Precision:   Retrieved chunks are relevant to the question
  - Context Recall:      Retrieved chunks contain what was needed (needs ground truth)
  - Factual Correctness: Answer is factually correct vs reference (needs ground truth)
"""

from __future__ import annotations

import time
from typing import List, Optional, Dict, Any

from backend.evaluators.metrics import (
    EvaluationSample,
    EvaluationResult,
    MetricName,
    MetricScore,
    DEFAULT_METRICS,
    REQUIRES_GROUND_TRUTH,
)
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """
    Evaluates RAG quality using the RAGAS framework.

    Uses the LLM configured in settings as the judge. On first call
    the RAGAS metric objects are instantiated (lazy init).
    """

    def __init__(
        self,
        judge_model: Optional[str] = None,
        judge_provider: Optional[str] = None,
    ):
        """
        Args:
            judge_model:    Model name for the LLM judge.
                            Defaults to settings.ollama_default_model.
            judge_provider: Provider for the LLM judge ("ollama" or "openai").
                            Defaults to settings.default_provider.
        """
        self.judge_model    = judge_model    or settings.ollama_default_model
        self.judge_provider = judge_provider or settings.default_provider
        self._llm           = None   # lazy-loaded
        self._embeddings    = None   # lazy-loaded
        logger.info(
            f"Initialized RAGASEvaluator: "
            f"judge={self.judge_provider}/{self.judge_model}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        samples: List[EvaluationSample],
        metrics: Optional[List[MetricName]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a list of RAG samples.

        Args:
            samples: Question/answer/context tuples to evaluate.
            metrics: Metrics to compute. Defaults to DEFAULT_METRICS.

        Returns:
            EvaluationResult with per-sample scores and aggregated means.
        """
        if not samples:
            raise ValueError("At least one sample is required for evaluation.")

        metrics = metrics or DEFAULT_METRICS
        start   = time.time()

        logger.info(
            f"Starting RAGAS evaluation: {len(samples)} samples, "
            f"metrics={[m.value for m in metrics]}"
        )

        # Warn if ground-truth metrics requested but not provided
        missing_gt = [
            s.question for s in samples
            if s.requires_ground_truth(metrics) and not s.ground_truth
        ]
        if missing_gt:
            logger.warning(
                f"{len(missing_gt)} sample(s) are missing ground_truth "
                f"required by {[m.value for m in metrics if m in REQUIRES_GROUND_TRUTH]}. "
                f"Those metrics will be skipped for those samples."
            )

        try:
            ragas_dataset = self._build_ragas_dataset(samples, metrics)
            ragas_metrics = self._build_ragas_metrics(metrics)
            llm           = self._get_llm()
            embeddings    = self._get_embeddings()

            from ragas import evaluate as ragas_evaluate

            # Configure judge LLM and embeddings on each metric object
            for metric in ragas_metrics:
                if hasattr(metric, "llm"):
                    metric.llm = llm
                if hasattr(metric, "embeddings"):
                    metric.embeddings = embeddings

            result_df = ragas_evaluate(
                dataset=ragas_dataset,
                metrics=ragas_metrics,
            )

            sample_scores, aggregated = self._parse_results(result_df, metrics, samples)

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"RAGAS evaluation failed: {e}") from e

        duration = time.time() - start
        logger.info(
            f"RAGAS evaluation complete in {duration:.2f}s. "
            f"Results: {aggregated}"
        )

        return EvaluationResult(
            sample_scores=sample_scores,
            aggregated=aggregated,
            sample_count=len(samples),
            metrics_used=metrics,
            llm_model=self.judge_model,
            duration_s=duration,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ragas_dataset(
        self,
        samples: List[EvaluationSample],
        metrics: List[MetricName],
    ):
        """Convert our EvaluationSample list into a RAGAS EvaluationDataset."""
        from ragas import EvaluationDataset, SingleTurnSample

        ragas_samples = []
        for s in samples:
            kwargs: Dict[str, Any] = {
                "user_input":          s.question,
                "response":            s.answer,
                "retrieved_contexts":  s.contexts,
            }
            # Only include reference when we have it and a metric needs it
            if s.ground_truth and s.requires_ground_truth(metrics):
                kwargs["reference"] = s.ground_truth

            ragas_samples.append(SingleTurnSample(**kwargs))

        return EvaluationDataset(samples=ragas_samples)

    def _build_ragas_metrics(self, metrics: List[MetricName]) -> list:
        """Instantiate the requested RAGAS metric objects."""
        from ragas.metrics.collections import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
            FactualCorrectness,
        )

        metric_map = {
            MetricName.FAITHFULNESS:       Faithfulness,
            MetricName.ANSWER_RELEVANCY:   ResponseRelevancy,
            MetricName.CONTEXT_PRECISION:  LLMContextPrecisionWithoutReference,
            MetricName.CONTEXT_RECALL:     LLMContextRecall,
            MetricName.FACTUAL_CORRECTNESS: FactualCorrectness,
        }

        return [metric_map[m]() for m in metrics if m in metric_map]

    def _get_llm(self):
        """Return a RAGAS-compatible LLM wrapper (lazy-loaded)."""
        if self._llm is not None:
            return self._llm

        if self.judge_provider == "ollama":
            try:
                from ragas.llms import LangchainLLMWrapper
                from langchain_community.llms import Ollama as LangchainOllama
                self._llm = LangchainLLMWrapper(
                    LangchainOllama(
                        base_url=settings.ollama_host,
                        model=self.judge_model,
                    )
                )
                logger.info(
                    f"Using Ollama LLM as RAGAS judge: "
                    f"{settings.ollama_host}/{self.judge_model}"
                )
            except ImportError as e:
                raise ImportError(
                    "langchain-community is required for Ollama RAGAS evaluation. "
                    f"Original error: {e}"
                ) from e

        elif self.judge_provider == "openai":
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Set it or switch DEFAULT_PROVIDER to 'ollama'."
                )
            from ragas.llms import LangchainLLMWrapper
            from langchain_openai import ChatOpenAI
            self._llm = LangchainLLMWrapper(ChatOpenAI(model=self.judge_model))
            logger.info(f"Using OpenAI LLM as RAGAS judge: {self.judge_model}")

        else:
            raise ValueError(
                f"Unsupported judge provider: {self.judge_provider}. "
                f"Use 'ollama' or 'openai'."
            )

        return self._llm

    def _get_embeddings(self):
        """Return RAGAS-compatible embeddings wrapper (lazy-loaded)."""
        if self._embeddings is not None:
            return self._embeddings

        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=settings.embedding_model)
            )
            logger.info(f"Using embeddings model: {settings.embedding_model}")
        except ImportError as e:
            raise ImportError(
                f"Failed to load embeddings for RAGAS: {e}"
            ) from e

        return self._embeddings

    def _parse_results(
        self,
        result_df,
        metrics: List[MetricName],
        samples: List[EvaluationSample],
    ):
        """Convert RAGAS result DataFrame to our types."""
        import pandas as pd

        # Map MetricName → RAGAS column name
        col_map = {
            MetricName.FAITHFULNESS:        "faithfulness",
            MetricName.ANSWER_RELEVANCY:    "answer_relevancy",
            MetricName.CONTEXT_PRECISION:   "llm_context_precision_without_reference",
            MetricName.CONTEXT_RECALL:      "llm_context_recall",
            MetricName.FACTUAL_CORRECTNESS: "factual_correctness",
        }

        # RAGAS evaluate() returns a dict-like; convert to DataFrame
        if not isinstance(result_df, pd.DataFrame):
            df = result_df.to_pandas()
        else:
            df = result_df

        sample_scores: List[List[MetricScore]] = []
        aggregated: Dict[str, Optional[float]] = {}

        for metric in metrics:
            col = col_map.get(metric)
            scores_for_metric: List[Optional[float]] = []

            if col and col in df.columns:
                for val in df[col]:
                    try:
                        scores_for_metric.append(float(val) if val is not None else None)
                    except (TypeError, ValueError):
                        scores_for_metric.append(None)
                valid = [s for s in scores_for_metric if s is not None]
                aggregated[metric.value] = sum(valid) / len(valid) if valid else None
            else:
                scores_for_metric = [None] * len(samples)
                aggregated[metric.value] = None

            # Build per-sample score list (initialise on first metric)
            if not sample_scores:
                sample_scores = [[] for _ in samples]

            for i, score in enumerate(scores_for_metric):
                sample_scores[i].append(MetricScore(metric=metric, score=score))

        return sample_scores, aggregated

    def get_stats(self) -> dict:
        return {
            "judge_model":    self.judge_model,
            "judge_provider": self.judge_provider,
            "llm_loaded":     self._llm is not None,
        }


# Made with Bob
