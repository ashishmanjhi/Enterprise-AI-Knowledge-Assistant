"""
Research Agent (Phase 11) — Multi-Agent Ecosystem.

The Research Agent decomposes a complex question into a numbered plan of
sub-questions, answers each one independently using RAG retrieval, then
synthesises all findings into a coherent summary paragraph.

Flow
────
    decompose_query  →  (for each sub-question) retrieve + answer  →  synthesise
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from backend.agents.multi.state import MultiAgentState
from backend.core.settings import settings
from backend.core.logging import get_logger
from backend.core.tracing import trace_span

logger = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
You are a research planner.  Break the question below into at most {max_sub} \
focused sub-questions that together fully answer it.  Return them as a \
numbered list (one per line, no extra text).

Question: {query}

Sub-questions:"""

_ANSWER_PROMPT = """\
Using only the context below, answer the following question concisely. \
If the context does not contain the answer, say "Not found in documents."

Context:
{context}

Question: {question}

Answer:"""

_SYNTHESISE_PROMPT = """\
You are a research synthesiser.  Given the findings below, write a single \
coherent paragraph that answers the original question.  Cite finding numbers \
where relevant (e.g. [1], [2]).

Original question: {query}

Findings:
{findings}

Summary:"""


# ── Node functions ────────────────────────────────────────────────────────

async def decompose_query(state: MultiAgentState, llm) -> Dict[str, Any]:
    """Use the LLM to decompose the user query into sub-questions."""
    query     = state["query"]
    max_sub   = settings.research_agent_max_sub_questions
    start     = time.time()

    try:
        result = await llm.generate(
            prompt=_DECOMPOSE_PROMPT.format(query=query, max_sub=max_sub),
            temperature=0.3,
            max_tokens=200,
        )
        raw = result.get("text", "").strip()
        # Parse numbered list — accept "1. ...", "1) ...", or plain lines
        lines = [
            line.lstrip("0123456789.)- ").strip()
            for line in raw.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        plan = lines[:max_sub] if lines else [query]
    except Exception as exc:
        logger.warning(f"decompose_query failed ({exc}), using original query")
        plan = [query]

    elapsed = time.time() - start
    logger.info(f"research.decompose_query: {len(plan)} sub-questions ({elapsed:.2f}s)")
    return {
        "research_plan": plan,
        "active_agents": ["research"],
        "trace": [f"research.decompose: {len(plan)} sub-questions"],
    }


async def research_retrieve_answer(state: MultiAgentState, llm, retriever) -> Dict[str, Any]:
    """For each sub-question: retrieve relevant chunks and generate an answer."""
    plan     = state.get("research_plan", [state["query"]])
    top_k    = state.get("top_k", settings.top_k_retrieval)
    findings = []
    start    = time.time()

    for sub_q in plan:
        # Retrieve
        try:
            docs = await retriever.retrieve(query=sub_q, top_k=top_k, method="hybrid")
            context = "\n\n".join(
                f"[{i+1}] {d.filename}: {d.content[:400]}"
                for i, d in enumerate(docs[:3])
            )
        except Exception as exc:
            logger.warning(f"research retrieval failed for '{sub_q[:40]}': {exc}")
            context = ""
            docs    = []

        # Answer
        try:
            ans = await llm.generate(
                prompt=_ANSWER_PROMPT.format(context=context or "No context.", question=sub_q),
                temperature=state.get("temperature", 0.7),
                max_tokens=state.get("max_tokens", 300),
            )
            answer_text = ans.get("text", "").strip()
        except Exception as exc:
            logger.warning(f"research generation failed for '{sub_q[:40]}': {exc}")
            answer_text = "Unable to answer."

        findings.append({
            "question": sub_q,
            "answer":   answer_text,
            "sources":  [
                {"filename": d.filename, "chunk_id": d.chunk_id, "score": round(d.score, 3)}
                for d in (docs or [])
            ],
        })

    elapsed = time.time() - start
    logger.info(f"research.retrieve_answer: {len(findings)} findings ({elapsed:.2f}s)")
    return {
        "research_findings": findings,
        "trace": [f"research.retrieve_answer: {len(findings)} findings in {elapsed:.1f}s"],
    }


async def synthesise(state: MultiAgentState, llm) -> Dict[str, Any]:
    """Synthesise all findings into a single coherent answer paragraph."""
    query    = state["query"]
    findings = state.get("research_findings", [])
    start    = time.time()

    if not findings:
        return {
            "research_summary": "No findings to synthesise.",
            "trace": ["research.synthesise: no findings"],
        }

    findings_text = "\n".join(
        f"[{i+1}] Q: {f['question']}\n    A: {f['answer']}"
        for i, f in enumerate(findings)
    )

    try:
        result = await llm.generate(
            prompt=_SYNTHESISE_PROMPT.format(query=query, findings=findings_text),
            temperature=state.get("temperature", 0.5),
            max_tokens=state.get("max_tokens", 500),
        )
        summary = result.get("text", "").strip()
    except Exception as exc:
        logger.warning(f"synthesise failed ({exc}), concatenating findings")
        summary = " ".join(f["answer"] for f in findings)

    elapsed = time.time() - start
    logger.info(f"research.synthesise: {len(summary)} chars ({elapsed:.2f}s)")
    return {
        "research_summary": summary,
        "final_response":   summary,
        "trace": [f"research.synthesise: {len(summary)} chars in {elapsed:.1f}s"],
    }


# ── Graph builder ─────────────────────────────────────────────────────────

class ResearchAgent:
    """
    LangGraph sub-graph for the Research Agent.

    Accepts ``llm`` (LLMService) and ``retriever`` (HybridRetriever).
    Call ``await agent.run(state)`` with a MultiAgentState dict.
    """

    def __init__(self, llm, retriever) -> None:
        self.llm       = llm
        self.retriever = retriever
        self._graph    = self._build()
        logger.info("ResearchAgent compiled")

    def _build(self):
        import functools
        builder = StateGraph(MultiAgentState)

        builder.add_node("decompose",       functools.partial(decompose_query,          llm=self.llm))
        builder.add_node("retrieve_answer", functools.partial(research_retrieve_answer, llm=self.llm, retriever=self.retriever))
        builder.add_node("synthesise",      functools.partial(synthesise,               llm=self.llm))

        builder.set_entry_point("decompose")
        builder.add_edge("decompose",       "retrieve_answer")
        builder.add_edge("retrieve_answer", "synthesise")
        builder.add_edge("synthesise",      END)

        return builder.compile()

    async def run(self, state: Dict[str, Any]) -> MultiAgentState:
        async with trace_span("research_agent.run", {"query": state.get("query", "")[:80]}):
            return await self._graph.ainvoke(state)  # type: ignore[return-value]


# Made with Bob
