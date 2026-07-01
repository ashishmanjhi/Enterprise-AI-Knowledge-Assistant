"""
EntityExtractor (Phase 12) — LLM + regex Named Entity Recognition.

Extracts typed entities from text and populates the GraphStore.

Entity types (configurable via settings.kg_entity_types):
  PERSON, ORG, PRODUCT, LOCATION, CONCEPT, TECHNICAL

Strategy
────────
1. If LLM extraction is enabled (settings.kg_llm_extraction_enabled),
   prompt the LLM with a structured NER prompt and parse "TYPE: text" lines.
2. On any LLM failure (or when disabled), fall back to heuristic regex patterns
   that cover capitalised proper-noun sequences and common technical patterns.

Public API
──────────
  async extract(text, source_doc, graph_store) → list[dict]
      Returns list of {"id": str, "text": str, "entity_type": str} dicts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


# ── Regex fallback patterns ────────────────────────────────────────────────

# Capitalised proper-noun sequences (2–4 words) — catches PERSON, ORG, LOCATION
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b")

# Uppercase acronyms (2–5 letters) — catches ORG abbreviations, TECHNICAL terms
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,5})\b")

# Version-tagged technical terms: "Python 3.12", "GPT-4", "LangGraph 0.1"
_TECH_VERSION_RE = re.compile(
    r"\b([A-Za-z][\w\-\.]*(?:\s+v?[\d][\d\.\-]*))\b"
)

# Quoted strings — may be product names, concepts
_QUOTED_RE = re.compile(r'"([^"]{3,60})"')


# ── LLM extraction prompt ─────────────────────────────────────────────────

_NER_SYSTEM = (
    "You are a precise named-entity extractor. "
    "Output ONLY entity lines in the format  TYPE: entity text  "
    "(one per line, no extra commentary). "
    "Valid types: PERSON, ORG, PRODUCT, LOCATION, CONCEPT, TECHNICAL. "
    "Skip anything that is not a clear named entity."
)

_NER_PROMPT = """\
Extract all named entities from the following text.
Format: TYPE: entity text
Text:
\"\"\"
{text}
\"\"\""""


# ── Parser ─────────────────────────────────────────────────────────────────

def _parse_llm_output(raw: str, valid_types: set) -> List[Dict[str, str]]:
    """Parse LLM NER output into list of {text, entity_type} dicts."""
    entities: List[Dict[str, str]] = []
    seen: set = set()
    for line in raw.splitlines():
        line = line.strip().lstrip("•-*").strip()
        if ":" not in line:
            continue
        parts = line.split(":", 1)
        etype = parts[0].strip().upper()
        etext = parts[1].strip().strip('"').strip("'")
        if not etext or len(etext) < 2:
            continue
        if etype not in valid_types:
            # Try to salvage lines like "• TECHNICAL: NetworkX"
            for vt in valid_types:
                if vt in etype:
                    etype = vt
                    break
            else:
                etype = "CONCEPT"
        key = (etype, etext.lower())
        if key not in seen:
            seen.add(key)
            entities.append({"text": etext, "entity_type": etype})
    return entities


def _regex_extract(text: str) -> List[Dict[str, str]]:
    """Heuristic regex fallback — returns {text, entity_type} dicts."""
    seen: set = set()
    results: List[Dict[str, str]] = []

    def _add(t: str, et: str) -> None:
        t = t.strip()
        if len(t) < 2:
            return
        key = t.lower()
        if key not in seen:
            seen.add(key)
            results.append({"text": t, "entity_type": et})

    for m in _TECH_VERSION_RE.findall(text):
        _add(m, "TECHNICAL")
    for m in _QUOTED_RE.findall(text):
        _add(m, "CONCEPT")
    for m in _PROPER_NOUN_RE.findall(text):
        _add(m, "PERSON")  # coarse — KG builder can refine later
    for m in _ACRONYM_RE.findall(text):
        # Skip common English words and single-character noise
        if m not in {"I", "A", "IT", "AI", "ML", "API", "URL", "HTTP", "JSON"}:
            _add(m, "ORG")
    return results


# ── EntityExtractor ────────────────────────────────────────────────────────

class EntityExtractor:
    """
    Extracts typed named entities from text using LLM (primary) or regex (fallback).

    Writes discovered entities directly into the supplied GraphStore.
    """

    def __init__(self, llm=None) -> None:
        """
        Args:
            llm: LLMService instance (optional).  When None, LLM extraction is
                 disabled and regex is always used.
        """
        self._llm = llm
        self._valid_types = set(settings.kg_entity_types)
        logger.info(f"EntityExtractor initialised — "
                    f"llm={'enabled' if llm else 'disabled'}, "
                    f"types={self._valid_types}")

    async def extract(
        self,
        text: str,
        source_doc: str = "",
        graph_store=None,
        max_entities: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract named entities from text and (optionally) add to graph_store.

        Args:
            text:         Input text to analyse.
            source_doc:   Document identifier to attach to each entity node.
            graph_store:  GraphStore instance — entities are upserted when provided.
            max_entities: Cap on number of entities returned.

        Returns:
            List of dicts with keys: id, text, entity_type.
        """
        cap = max_entities or settings.kg_max_entities_per_doc
        truncated = text[:2000]  # keep prompt manageable

        entities: List[Dict[str, str]] = []

        if self._llm and settings.kg_llm_extraction_enabled:
            entities = await self._llm_extract(truncated)

        if not entities:
            # LLM disabled, failed, or returned nothing — use regex
            entities = _regex_extract(truncated)

        entities = entities[:cap]

        # Persist to graph store
        results: List[Dict[str, Any]] = []
        for ent in entities:
            node_id: Optional[str] = None
            if graph_store is not None:
                node_id = graph_store.add_entity(
                    text=ent["text"],
                    entity_type=ent["entity_type"],
                    source_doc=source_doc,
                )
            results.append({
                "id":          node_id,
                "text":        ent["text"],
                "entity_type": ent["entity_type"],
            })

        logger.debug(f"EntityExtractor: {len(results)} entities from {len(text)} chars")
        return results

    async def _llm_extract(self, text: str) -> List[Dict[str, str]]:
        """Call the LLM and parse the entity list output."""
        try:
            result = await self._llm.generate(
                prompt=_NER_PROMPT.format(text=text),
                system_prompt=_NER_SYSTEM,
                temperature=0.0,
                max_tokens=settings.kg_extraction_max_tokens,
            )
            raw = result.get("text", "")
            entities = _parse_llm_output(raw, self._valid_types)
            logger.debug(f"EntityExtractor LLM: {len(entities)} entities parsed")
            return entities
        except Exception as exc:
            logger.warning(f"EntityExtractor LLM failed: {exc} — falling back to regex")
            return []


# Made with Bob
