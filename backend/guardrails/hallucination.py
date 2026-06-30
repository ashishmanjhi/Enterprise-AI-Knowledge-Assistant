"""
Hallucination Detector (Phase 7) — LLM-as-judge grounding check.

Asks the judge LLM whether the answer is fully supported by the provided
context.  Falls back to a heuristic keyword-overlap check when no LLM is
available so the pipeline always produces a result.
"""

from __future__ import annotations

import re
from typing import List, Optional

from backend.guardrails.detectors import DetectionResult, Severity
from backend.core.logging import get_logger

logger = get_logger(__name__)

_JUDGE_PROMPT = """\
You are a strict fact-checker.  Given the CONTEXT and the ANSWER below, \
decide whether the ANSWER contains any claims that are NOT supported by the \
CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with exactly one word: GROUNDED if all claims in the answer are \
supported by the context, or HALLUCINATION if the answer contains \
unsupported claims.

Verdict:"""


class HallucinationDetector:
    """
    Check whether an LLM answer is grounded in its source context.

    Primary path: LLM-as-judge (requires ``llm_service``).
    Fallback path: heuristic token-overlap ratio.
    """

    def __init__(
        self,
        llm_service=None,
        overlap_threshold: float = 0.15,
    ) -> None:
        """
        Args:
            llm_service:        LLMService instance.  None → heuristic only.
            overlap_threshold:  Minimum token-overlap ratio for the heuristic
                                to consider an answer grounded (0–1).
        """
        self._llm               = llm_service
        self.overlap_threshold  = overlap_threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def check(
        self,
        answer: str,
        contexts: List[str],
    ) -> DetectionResult:
        """
        Check whether ``answer`` is grounded in ``contexts``.

        Returns a DetectionResult where ``detected=True`` means a potential
        hallucination was found.
        """
        if not contexts or not answer.strip():
            return DetectionResult(
                detected=False,
                detector="hallucination",
                details={"reason": "no_context_or_empty_answer"},
            )

        context_text = "\n\n".join(contexts)

        if self._llm is not None:
            return await self._llm_check(answer, context_text)

        return self._heuristic_check(answer, context_text)

    # ------------------------------------------------------------------ #
    # LLM-as-judge
    # ------------------------------------------------------------------ #

    async def _llm_check(self, answer: str, context: str) -> DetectionResult:
        prompt = _JUDGE_PROMPT.format(
            context=context[:3000],   # cap to avoid prompt overflow
            answer=answer[:1000],
        )
        try:
            result = await self._llm.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=10,
            )
            verdict = result.get("text", "").strip().upper()
            is_hallucination = "HALLUCINATION" in verdict

            logger.debug(f"Hallucination judge verdict: {verdict!r}")
            return DetectionResult(
                detected=is_hallucination,
                detector="hallucination",
                severity=Severity.HIGH if is_hallucination else None,
                details={
                    "method":  "llm_judge",
                    "verdict": verdict,
                },
            )
        except Exception as e:
            logger.warning(f"LLM hallucination check failed, falling back to heuristic: {e}")
            return self._heuristic_check(answer, context)

    # ------------------------------------------------------------------ #
    # Heuristic fallback: token-overlap ratio
    # ------------------------------------------------------------------ #

    def _heuristic_check(self, answer: str, context: str) -> DetectionResult:
        answer_tokens  = set(re.findall(r"\b\w+\b", answer.lower()))
        context_tokens = set(re.findall(r"\b\w+\b", context.lower()))

        # Remove stop words for a cleaner signal
        stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                "at", "to", "of", "and", "or", "but", "it", "this", "that",
                "for", "with", "be", "been", "by", "from", "as", "not"}
        answer_tokens  -= stop
        context_tokens -= stop

        if not answer_tokens:
            return DetectionResult(
                detected=False,
                detector="hallucination",
                details={"method": "heuristic", "reason": "empty_answer_tokens"},
            )

        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        is_low  = overlap < self.overlap_threshold

        return DetectionResult(
            detected=is_low,
            detector="hallucination",
            severity=Severity.MEDIUM if is_low else None,
            details={
                "method":             "heuristic_overlap",
                "overlap_ratio":      round(overlap, 3),
                "threshold":          self.overlap_threshold,
                "answer_token_count": len(answer_tokens),
            },
        )


# Made with Bob
