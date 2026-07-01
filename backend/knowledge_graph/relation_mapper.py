"""
RelationMapper (Phase 12) — LLM relation triple extraction.

Extracts (subject, predicate, object) triples from text and stores them
as directed edges in the GraphStore.

Strategy
────────
1. LLM is prompted to output triples in the format:
       subject | predicate | object
   one per line.
2. Both subject and object are matched against existing GraphStore entities
   (substring search).  Unmatched ends are added as CONCEPT entities.
3. On LLM failure, a lightweight regex heuristic is applied to detect
   simple "X is/has/uses Y" patterns.

Public API
──────────
  async map(text, source_doc, graph_store, entity_ids) → list[dict]
      Returns list of {"subject": str, "predicate": str, "object": str,
                        "subject_id": str, "object_id": str} dicts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


# ── LLM extraction prompt ──────────────────────────────────────────────────

_REL_SYSTEM = (
    "You are a precise relation extractor. "
    "Given text, output ONLY relation triples in the format:  "
    "subject | predicate | object  "
    "(one triple per line, no extra commentary). "
    "Use short, lowercase predicates (e.g. 'is a', 'has', 'uses', 'part of', 'located in'). "
    "Only include factual relationships clearly stated or strongly implied by the text."
)

_REL_PROMPT = """\
Extract all factual relations from the following text as triples.
Format: subject | predicate | object
Text:
\"\"\"
{text}
\"\"\""""


# ── Regex heuristic fallback ───────────────────────────────────────────────

# Patterns: "X is a Y", "X uses Y", "X has Y", "X is part of Y"
_HEURISTIC_RE = re.compile(
    r"([A-Z][A-Za-z\s\-\.]{1,40}?)"
    r"\s+(is a|is an|is|uses|has|contains|belongs to|part of|located in|works for|"
    r"developed by|created by|built on|based on|related to)"
    r"\s+([A-Z][A-Za-z\s\-\.]{1,40})",
    re.IGNORECASE,
)


def _parse_llm_triples(raw: str) -> List[Dict[str, str]]:
    """Parse LLM triple output into list of {subject, predicate, object} dicts."""
    triples: List[Dict[str, str]] = []
    seen: set = set()
    for line in raw.splitlines():
        line = line.strip().lstrip("•-*").strip()
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        subj, pred, obj = parts
        if not subj or not pred or not obj:
            continue
        key = (subj.lower(), pred.lower(), obj.lower())
        if key not in seen:
            seen.add(key)
            triples.append({
                "subject":   subj,
                "predicate": pred.lower(),
                "object":    obj,
            })
    return triples


def _heuristic_triples(text: str) -> List[Dict[str, str]]:
    """Regex-based heuristic triple extraction."""
    triples: List[Dict[str, str]] = []
    seen: set = set()
    for m in _HEURISTIC_RE.finditer(text):
        subj = m.group(1).strip()
        pred = m.group(2).strip().lower()
        obj  = m.group(3).strip()
        key  = (subj.lower(), pred, obj.lower())
        if key not in seen:
            seen.add(key)
            triples.append({"subject": subj, "predicate": pred, "object": obj})
    return triples


# ── RelationMapper ─────────────────────────────────────────────────────────

class RelationMapper:
    """
    Extracts relation triples from text and maps them to GraphStore edges.

    Subject and object texts are matched against existing GraphStore entity
    nodes via substring search.  Unmatched ends are auto-created as CONCEPT nodes.
    """

    def __init__(self, llm=None) -> None:
        """
        Args:
            llm: LLMService instance (optional).  When None, only heuristic
                 regex extraction is used.
        """
        self._llm = llm
        logger.info(f"RelationMapper initialised — llm={'enabled' if llm else 'disabled'}")

    async def map(
        self,
        text: str,
        source_doc: str = "",
        graph_store=None,
        max_relations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract triples from text and persist edges to graph_store.

        Args:
            text:          Input text.
            source_doc:    Document identifier to attach to each edge.
            graph_store:   GraphStore instance — edges are added when provided.
            max_relations: Cap on triples processed.

        Returns:
            List of dicts: subject, predicate, object, subject_id, object_id.
        """
        cap = max_relations or settings.kg_max_relations_per_doc
        truncated = text[:2000]

        raw_triples: List[Dict[str, str]] = []

        if self._llm and settings.kg_llm_extraction_enabled:
            raw_triples = await self._llm_extract(truncated)

        if not raw_triples:
            raw_triples = _heuristic_triples(truncated)

        raw_triples = raw_triples[:cap]

        results: List[Dict[str, Any]] = []
        for triple in raw_triples:
            subj_text = triple["subject"]
            pred      = triple["predicate"]
            obj_text  = triple["object"]

            subj_id: Optional[str] = None
            obj_id:  Optional[str] = None

            if graph_store is not None:
                # Resolve subject
                subj_matches = graph_store.search_entities(subj_text, top_k=1)
                if subj_matches:
                    subj_id = subj_matches[0]["id"]
                else:
                    subj_id = graph_store.add_entity(
                        text=subj_text,
                        entity_type="CONCEPT",
                        source_doc=source_doc,
                    )

                # Resolve object
                obj_matches = graph_store.search_entities(obj_text, top_k=1)
                if obj_matches:
                    obj_id = obj_matches[0]["id"]
                else:
                    obj_id = graph_store.add_entity(
                        text=obj_text,
                        entity_type="CONCEPT",
                        source_doc=source_doc,
                    )

                # Add the edge
                graph_store.add_relation(
                    subject_id=subj_id,
                    predicate=pred,
                    object_id=obj_id,
                    confidence=0.8 if self._llm else 0.5,
                    source_doc=source_doc,
                )

            results.append({
                "subject":    subj_text,
                "predicate":  pred,
                "object":     obj_text,
                "subject_id": subj_id,
                "object_id":  obj_id,
            })

        logger.debug(f"RelationMapper: {len(results)} triples from {len(text)} chars")
        return results

    async def _llm_extract(self, text: str) -> List[Dict[str, str]]:
        """Call LLM and parse triple output."""
        try:
            result = await self._llm.generate(
                prompt=_REL_PROMPT.format(text=text),
                system_prompt=_REL_SYSTEM,
                temperature=0.0,
                max_tokens=settings.kg_extraction_max_tokens,
            )
            raw = result.get("text", "")
            triples = _parse_llm_triples(raw)
            logger.debug(f"RelationMapper LLM: {len(triples)} triples parsed")
            return triples
        except Exception as exc:
            logger.warning(f"RelationMapper LLM failed: {exc} — falling back to heuristic")
            return []


# Made with Bob
