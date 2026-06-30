"""
Evaluation API routes (Phase 5).

POST /api/v1/evaluate   — run RAGAS evaluation on provided samples
GET  /api/v1/evaluate/metrics — list available metrics
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.evaluators.metrics import (
    EvaluationSample,
    MetricName,
    DEFAULT_METRICS,
)
from backend.evaluators.ragas_evaluator import RAGASEvaluator
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/evaluate", tags=["evaluation"])


# ── Request / Response models ──────────────────────────────────────────────

class EvaluationSampleRequest(BaseModel):
    question:     str = Field(..., description="User question")
    answer:       str = Field(..., description="LLM-generated answer")
    contexts:     List[str] = Field(..., description="Retrieved context chunks passed to the LLM")
    ground_truth: Optional[str] = Field(None, description="Reference answer (needed for recall/correctness)")
    metadata:     dict = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "question":  "What is EVSE uptime?",
                "answer":    "EVSE uptime refers to the percentage of time charging stations are operational.",
                "contexts":  ["EVSE connectors experience downtime due to network failures..."],
                "ground_truth": None,
            }
        }


class EvaluationRequest(BaseModel):
    samples:  List[EvaluationSampleRequest] = Field(..., min_length=1)
    metrics:  Optional[List[MetricName]] = Field(
        None,
        description="Metrics to compute. Defaults to faithfulness, answer_relevancy, context_precision."
    )
    judge_model:    Optional[str] = Field(None, description="Override judge LLM model")
    judge_provider: Optional[str] = Field(None, description="Override judge LLM provider (ollama/openai)")

    class Config:
        json_schema_extra = {
            "example": {
                "samples": [
                    {
                        "question":  "What is EVSE uptime?",
                        "answer":    "EVSE uptime refers to operational availability of chargers.",
                        "contexts":  ["EVSE connectors experience downtime due to network failures..."],
                    }
                ],
                "metrics": ["faithfulness", "answer_relevancy", "context_precision"],
            }
        }


class EvaluationResponse(BaseModel):
    sample_count: int
    metrics_used: List[str]
    aggregated:   dict
    llm_model:    str
    duration_s:   float
    samples:      list

    class Config:
        json_schema_extra = {
            "example": {
                "sample_count": 1,
                "metrics_used": ["faithfulness", "answer_relevancy", "context_precision"],
                "aggregated": {
                    "faithfulness": 0.92,
                    "answer_relevancy": 0.87,
                    "context_precision": 0.78,
                },
                "llm_model": "gemma3:4b",
                "duration_s": 12.4,
                "samples": [],
            }
        }


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    """
    Run RAGAS evaluation on a set of question/answer/context samples.

    The LLM configured in settings acts as the judge. Faithfulness,
    answer relevancy, and context precision are computed by default.
    Context recall and factual correctness additionally require a
    ground_truth reference answer.
    """
    # Enforce sample cap
    if len(request.samples) > settings.eval_max_samples:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many samples: {len(request.samples)}. "
                f"Maximum per run: {settings.eval_max_samples}."
            )
        )

    # Convert request samples
    samples = [
        EvaluationSample(
            question=s.question,
            answer=s.answer,
            contexts=s.contexts,
            ground_truth=s.ground_truth,
            metadata=s.metadata,
        )
        for s in request.samples
    ]

    # Build evaluator — use request overrides or fall back to settings
    judge_model    = request.judge_model    or (settings.eval_judge_model    or None)
    judge_provider = request.judge_provider or (settings.eval_judge_provider or None)

    evaluator = RAGASEvaluator(
        judge_model=judge_model,
        judge_provider=judge_provider,
    )

    try:
        result = await evaluator.evaluate(
            samples=samples,
            metrics=request.metrics,
        )
    except Exception as e:
        logger.error(f"Evaluation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return EvaluationResponse(**result.to_dict())


@router.get("/metrics")
async def list_metrics():
    """
    List all available evaluation metrics.

    Returns metric names, descriptions, and whether a ground-truth
    reference answer is required.
    """
    from backend.evaluators.metrics import REQUIRES_GROUND_TRUTH

    return {
        "metrics": [
            {
                "name": m.value,
                "requires_ground_truth": m in REQUIRES_GROUND_TRUTH,
                "description": _METRIC_DESCRIPTIONS.get(m.value, ""),
            }
            for m in MetricName
        ],
        "default_metrics": [m.value for m in DEFAULT_METRICS],
    }


_METRIC_DESCRIPTIONS = {
    "faithfulness":       "Measures whether the answer is grounded in the retrieved context (no hallucination).",
    "answer_relevancy":   "Measures whether the answer addresses the actual question asked.",
    "context_precision":  "Measures whether the retrieved chunks are relevant to the question.",
    "context_recall":     "Measures whether the retrieved chunks contain the information needed to answer (requires ground_truth).",
    "factual_correctness":"Measures whether the answer is factually correct against a reference answer (requires ground_truth).",
}


# Made with Bob
