"""
Safety Detectors (Phase 7) — prompt injection, PII, and toxicity.

All detectors are regex/heuristic-based and work fully offline with no
external model downloads or API keys required.  Each detector returns a
``DetectionResult`` with a boolean flag, severity level, and details dict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ── Severity ────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ── Result type ─────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Outcome of a single detector pass."""
    detected:  bool
    detector:  str
    severity:  Optional[Severity] = None
    details:   Dict              = field(default_factory=dict)
    matched:   List[str]         = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "detector": self.detector,
            "severity": self.severity.value if self.severity else None,
            "details":  self.details,
            "matched":  self.matched,
        }


# ════════════════════════════════════════════════════════════════════════════
# Prompt Injection Detector
# ════════════════════════════════════════════════════════════════════════════

# Patterns that suggest an attempt to override system instructions
_INJECTION_PATTERNS: List[Tuple[str, Severity]] = [
    # Direct instruction overrides
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|constraints?)", Severity.HIGH),
    (r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", Severity.HIGH),
    (r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)", Severity.HIGH),
    # Role override attempts
    (r"you\s+are\s+now\s+(a|an|the)\s+\w+", Severity.MEDIUM),
    (r"act\s+as\s+(a|an|the)\s+\w+\s+(without|with no)\s+(restriction|limit|filter|guideline)", Severity.HIGH),
    (r"pretend\s+(you\s+are|to\s+be)\s+(a|an|the|unrestricted|unfiltered)", Severity.MEDIUM),
    (r"jailbreak", Severity.HIGH),
    (r"DAN\s+mode", Severity.HIGH),
    # System prompt extraction
    (r"(reveal|show|print|output|repeat|tell me)\s+(your|the)\s+(system\s+prompt|instructions?|directives?)", Severity.HIGH),
    (r"what\s+(are|were)\s+your\s+(instructions?|system\s+prompt|directives?)", Severity.MEDIUM),
    # Delimiter injection
    (r"(</?(system|user|assistant|human|prompt)>)", Severity.MEDIUM),
    (r"(\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)", Severity.MEDIUM),
    # Command injection via context
    (r"translate\s+the\s+above\s+(as|to)\s+", Severity.LOW),
    (r"(new|updated|override)\s+(instruction|directive|rule|command)\s*:", Severity.HIGH),
]

_COMPILED_INJECTION = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), sev)
    for p, sev in _INJECTION_PATTERNS
]


class PromptInjectionDetector:
    """Detect prompt injection and jailbreak attempts in user input."""

    def check(self, text: str) -> DetectionResult:
        matched_patterns: List[str] = []
        max_severity: Optional[Severity] = None
        sev_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]

        for pattern, severity in _COMPILED_INJECTION:
            m = pattern.search(text)
            if m:
                matched_patterns.append(m.group(0).strip())
                if max_severity is None or sev_order.index(severity) > sev_order.index(max_severity):
                    max_severity = severity

        detected = len(matched_patterns) > 0
        return DetectionResult(
            detected=detected,
            detector="prompt_injection",
            severity=max_severity,
            details={"pattern_count": len(matched_patterns)},
            matched=matched_patterns[:5],  # cap returned matches
        )


# ════════════════════════════════════════════════════════════════════════════
# PII Detector
# ════════════════════════════════════════════════════════════════════════════

_PII_PATTERNS: List[Tuple[str, str, Severity]] = [
    # US SSN
    (r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b", "ssn", Severity.HIGH),
    # Credit card (Luhn-style shape only)
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "credit_card", Severity.HIGH),
    # Email
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "email", Severity.MEDIUM),
    # US phone (various formats)
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b", "phone_us", Severity.MEDIUM),
    # IP address (v4)
    (r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b", "ip_address", Severity.LOW),
    # Passport-like codes
    (r"\b[A-Z]{1,2}\d{6,9}\b", "passport_like", Severity.MEDIUM),
    # Date of birth patterns
    (r"\b(DOB|date\s+of\s+birth|born\s+on)\s*:?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", "dob", Severity.MEDIUM),
    # API keys / tokens (generic high-entropy strings)
    (r"\b(?:sk|pk|api|key|token|secret)[-_]?[A-Za-z0-9]{20,}\b", "api_key_like", Severity.HIGH),
]

_COMPILED_PII = [
    (re.compile(p, re.IGNORECASE), label, sev)
    for p, label, sev in _PII_PATTERNS
]


class PIIDetector:
    """Detect personally identifiable information in text."""

    def check(self, text: str) -> DetectionResult:
        found_types: Dict[str, int] = {}
        matched_values: List[str] = []
        max_severity: Optional[Severity] = None
        sev_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]

        for pattern, label, severity in _COMPILED_PII:
            hits = pattern.findall(text)
            if hits:
                found_types[label] = len(hits)
                # Redact actual values — store only label+count
                matched_values.append(f"{label}:{len(hits)}")
                if max_severity is None or sev_order.index(severity) > sev_order.index(max_severity):
                    max_severity = severity

        detected = len(found_types) > 0
        return DetectionResult(
            detected=detected,
            detector="pii",
            severity=max_severity,
            details={"pii_types": found_types},
            matched=matched_values,
        )

    def redact(self, text: str) -> str:
        """Return a copy of ``text`` with detected PII replaced by placeholders."""
        redacted = text
        for pattern, label, _ in _COMPILED_PII:
            redacted = pattern.sub(f"[REDACTED:{label.upper()}]", redacted)
        return redacted


# ════════════════════════════════════════════════════════════════════════════
# Toxicity Detector
# ════════════════════════════════════════════════════════════════════════════

# Lightweight keyword-based toxicity detection.
# NOT a replacement for a real classifier — used as a fast first pass.
_TOXIC_PHRASES: List[Tuple[str, Severity]] = [
    # Violence / threats
    (r"\b(kill|murder|assassinate|shoot|bomb|explode|blow\s+up)\s+(you|him|her|them|everyone|all)\b", Severity.HIGH),
    (r"\b(i\s+will|i'm\s+going\s+to|i\s+want\s+to)\s+(kill|hurt|harm|attack)\b", Severity.HIGH),
    # Self-harm
    (r"\b(suicide|self.harm|self-harm|cut\s+myself|end\s+my\s+life)\b", Severity.HIGH),
    # Hate speech markers (very coarse)
    (r"\b(racial\s+slur|ethnic\s+cleansing|genocide\s+of)\b", Severity.HIGH),
    # Severe harassment
    (r"\b(doxx|doxxing|swat|swatting)\s+(you|him|her|them)\b", Severity.HIGH),
    # Moderate toxicity
    (r"\b(you\s+are\s+an?\s+(idiot|moron|imbecile|loser|trash))\b", Severity.MEDIUM),
    (r"\b(go\s+(die|kill\s+yourself|f\*\*\*\s+yourself))\b", Severity.HIGH),
]

_COMPILED_TOXIC = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), sev)
    for p, sev in _TOXIC_PHRASES
]


class ToxicityDetector:
    """Detect toxic or harmful language in text."""

    def check(self, text: str) -> DetectionResult:
        matched_patterns: List[str] = []
        max_severity: Optional[Severity] = None
        sev_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]

        for pattern, severity in _COMPILED_TOXIC:
            m = pattern.search(text)
            if m:
                matched_patterns.append(m.group(0).strip())
                if max_severity is None or sev_order.index(severity) > sev_order.index(max_severity):
                    max_severity = severity

        detected = len(matched_patterns) > 0
        return DetectionResult(
            detected=detected,
            detector="toxicity",
            severity=max_severity,
            details={"pattern_count": len(matched_patterns)},
            matched=matched_patterns[:3],
        )


# Made with Bob
