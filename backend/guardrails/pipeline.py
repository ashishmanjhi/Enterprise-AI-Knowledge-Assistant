"""
Guardrails Pipeline (Phase 7) — orchestrates all safety checks.

The pipeline runs a series of checks on the user input (pre-generation)
and optionally on the LLM output (post-generation).  Each check is
independent; failures are collected and summarised in a ``GuardrailsResult``.

Usage::

    pipeline = GuardrailsPipeline()

    # Before generation
    pre = await pipeline.check_input(user_message)
    if pre.blocked:
        return {"error": pre.block_reason}

    # After generation
    post = await pipeline.check_output(answer, contexts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.guardrails.detectors import (
    DetectionResult,
    PIIDetector,
    PromptInjectionDetector,
    Severity,
    ToxicityDetector,
)
from backend.guardrails.hallucination import HallucinationDetector
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GuardrailsResult:
    """Aggregated result from all guardrail checks."""
    blocked:      bool                    = False
    block_reason: Optional[str]           = None
    checks:       List[DetectionResult]   = field(default_factory=list)
    redacted_text: Optional[str]          = None

    def to_dict(self) -> dict:
        return {
            "blocked":       self.blocked,
            "block_reason":  self.block_reason,
            "redacted_text": self.redacted_text,
            "checks": [c.to_dict() for c in self.checks],
        }

    @property
    def warnings(self) -> List[DetectionResult]:
        """Checks that detected something but did not trigger a block."""
        return [c for c in self.checks if c.detected]


class GuardrailsPipeline:
    """
    Central pipeline that wires together all safety detectors.

    Input checks (run on user message before generation):
      - Prompt injection detection
      - PII detection
      - Toxicity detection

    Output checks (run on LLM answer after generation):
      - Hallucination detection
      - PII detection (ensure no PII leaked from context)

    Blocking behaviour is controlled by settings:
      - ``guardrails_block_on_injection``  (default True)
      - ``guardrails_block_on_toxicity``   (default True)
      - ``guardrails_block_on_pii_input``  (default False — warn only)
      - ``guardrails_block_on_hallucination`` (default False — warn only)
    """

    def __init__(self, llm_service=None) -> None:
        self._injection   = PromptInjectionDetector()
        self._pii         = PIIDetector()
        self._toxicity    = ToxicityDetector()
        self._hallucinate = HallucinationDetector(llm_service=llm_service)
        logger.info("GuardrailsPipeline initialised")

    # ------------------------------------------------------------------ #
    # Input checks
    # ------------------------------------------------------------------ #

    async def check_input(self, text: str) -> GuardrailsResult:
        """
        Run all input-side checks on a user message.

        Returns a ``GuardrailsResult``.  If ``blocked=True`` the caller
        should return an error to the user without calling the LLM.
        """
        result = GuardrailsResult()

        # 1. Prompt injection
        if settings.guardrails_enable_injection:
            inj = self._injection.check(text)
            result.checks.append(inj)
            if inj.detected and settings.guardrails_block_on_injection:
                result.blocked      = True
                result.block_reason = "prompt_injection_detected"
                logger.warning(f"Input blocked: prompt injection | matched={inj.matched}")
                return result   # short-circuit

        # 2. Toxicity
        if settings.guardrails_enable_toxicity:
            tox = self._toxicity.check(text)
            result.checks.append(tox)
            if tox.detected and settings.guardrails_block_on_toxicity:
                result.blocked      = True
                result.block_reason = "toxic_content_detected"
                logger.warning(f"Input blocked: toxicity | matched={tox.matched}")
                return result

        # 3. PII in input — warn and optionally redact
        if settings.guardrails_enable_pii:
            pii = self._pii.check(text)
            result.checks.append(pii)
            if pii.detected:
                result.redacted_text = self._pii.redact(text)
                logger.info(f"PII detected in input: {pii.details.get('pii_types', {})}")
                if settings.guardrails_block_on_pii_input:
                    result.blocked      = True
                    result.block_reason = "pii_in_input"
                    return result

        return result

    # ------------------------------------------------------------------ #
    # Output checks
    # ------------------------------------------------------------------ #

    async def check_output(
        self,
        answer: str,
        contexts: Optional[List[str]] = None,
    ) -> GuardrailsResult:
        """
        Run output-side checks on the LLM-generated answer.

        Args:
            answer:   The LLM response text.
            contexts: Retrieved context chunks (needed for hallucination check).

        Returns a ``GuardrailsResult``.  If ``blocked=True`` the caller
        should substitute a safety message for the answer.
        """
        result = GuardrailsResult()

        # 1. Hallucination check
        if settings.guardrails_enable_hallucination and contexts:
            hall = await self._hallucinate.check(answer, contexts)
            result.checks.append(hall)
            if hall.detected:
                logger.warning(
                    f"Potential hallucination detected: "
                    f"{hall.details.get('method')} | overlap={hall.details.get('overlap_ratio')}"
                )
                if settings.guardrails_block_on_hallucination:
                    result.blocked      = True
                    result.block_reason = "hallucination_detected"
                    return result

        # 2. PII in output — warn only (never block)
        if settings.guardrails_enable_pii:
            pii = self._pii.check(answer)
            result.checks.append(pii)
            if pii.detected:
                result.redacted_text = self._pii.redact(answer)
                logger.info(f"PII detected in output: {pii.details.get('pii_types', {})}")

        return result

    def get_stats(self) -> dict:
        return {
            "injection_enabled":    settings.guardrails_enable_injection,
            "toxicity_enabled":     settings.guardrails_enable_toxicity,
            "pii_enabled":          settings.guardrails_enable_pii,
            "hallucination_enabled": settings.guardrails_enable_hallucination,
        }


# Made with Bob
