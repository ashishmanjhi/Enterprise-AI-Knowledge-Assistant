"""
Multi-Agent Ecosystem UI — Phase 11
Runs the full 5-agent pipeline:
  classify → (research | retrieval) → knowledge → evaluation → governance → generate

Shows per-agent results, research plan, quality scores, governance status,
and extracted entities alongside the final answer.
"""

import json
import time
import requests
import streamlit as st
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Multi-Agent - RAG Platform",
    page_icon="🌐",
    layout="wide",
)

st.title("🌐 Multi-Agent Ecosystem")
st.markdown(
    "Five specialised agents collaborate on every query: "
    "**Research · Retrieval · Knowledge · Evaluation · Governance**"
)

# ── Session state ──────────────────────────────────────────────────────────
if "ma_messages" not in st.session_state:
    st.session_state.ma_messages = []
if "ma_conv_id" not in st.session_state:
    st.session_state.ma_conv_id = f"ma_{int(time.time())}"
if "tenant_id" not in st.session_state:
    st.session_state["tenant_id"] = "default"


def _tenant_headers() -> dict:
    """Return X-Tenant-ID header for the active tenant (set on the Settings page)."""
    return {"X-Tenant-ID": st.session_state.get("tenant_id", "default")}

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Settings")

    top_k       = st.slider("Docs per retrieval", 1, 20, 5)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens  = st.slider("Max tokens", 100, 2000, 600, 100)

    st.divider()
    st.subheader("Display")
    show_agents  = st.toggle("Show agent breakdown",  value=True)
    show_scores  = st.toggle("Show quality scores",   value=True)
    show_trace   = st.toggle("Show execution trace",  value=False)
    show_entities = st.toggle("Show extracted entities", value=True)

    st.divider()
    try:
        h = requests.get(f"{API_BASE_URL}/api/v1/multi-agent/health", timeout=4).json()
        if h.get("status") == "healthy":
            st.success("✅ Multi-agent ready")
            st.caption(
                f"Research sub-q: {h.get('research_max_sub')} | "
                f"Expansions: {h.get('retrieval_expansions')} | "
                f"Pass threshold: {h.get('eval_pass_threshold')}"
            )
        else:
            st.error(f"❌ {h.get('error','unhealthy')}")
    except Exception:
        st.error("❌ Backend unreachable")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.ma_messages = []
        st.session_state.ma_conv_id  = f"ma_{int(time.time())}"
        st.rerun()

    st.caption(f"Conv: `{st.session_state.ma_conv_id}`")

    st.divider()
    _tid = st.session_state.get("tenant_id", "default")
    if _tid != "default":
        st.info(f"🏢 Tenant: **{_tid}**")
    else:
        st.caption("🏢 Tenant: default — [change in Settings](0_⚙️_Settings)")


# ── Helpers ────────────────────────────────────────────────────────────────

_AGENT_ICONS = {
    "research":   "🔬",
    "retrieval":  "🔎",
    "knowledge":  "🧠",
    "evaluation": "📊",
    "governance": "🛡️",
}

_INTENT_ICONS = {"research": "🔬", "retrieval": "🔎", "general": "💬"}


def _render_agent_cards(data: dict) -> None:
    results = data.get("agent_results", [])
    if not results:
        return
    cols = st.columns(len(results))
    for col, ar in zip(cols, results):
        icon = _AGENT_ICONS.get(ar["name"], "🤖")
        with col:
            if ar.get("ran"):
                st.success(f"{icon} **{ar['name'].title()}**")
                if ar.get("summary"):
                    st.caption(ar["summary"])
            else:
                st.info(f"{icon} {ar['name'].title()} (skipped)")


def _render_quality_scores(scores: dict) -> None:
    if not scores:
        return
    cols = st.columns(len(scores))
    for col, (name, val) in zip(cols, scores.items()):
        color = "🟢" if val >= 0.6 else ("🟡" if val >= 0.3 else "🔴")
        col.metric(f"{color} {name.replace('_', ' ').title()}", f"{val:.2f}")


def _render_research_plan(plan: list) -> None:
    if not plan:
        return
    with st.expander(f"🗂️ Research plan ({len(plan)} sub-questions)", expanded=False):
        for i, q in enumerate(plan, 1):
            st.markdown(f"**{i}.** {q}")


def _render_entities(entities: list) -> None:
    if not entities:
        return
    with st.expander(f"🏷️ Extracted entities ({len(entities)})", expanded=False):
        st.markdown("  ".join(f"`{e}`" for e in entities))


def _render_governance(data: dict) -> None:
    passed  = data.get("governance_passed")
    n_issues = data.get("governance_issues", 0) or 0
    if passed is None:
        return
    if passed:
        st.success(f"🛡️ Governance: passed ({n_issues} warning{'s' if n_issues != 1 else ''})")
    else:
        st.error(f"🛡️ Governance: FAILED — {n_issues} issue{'s' if n_issues != 1 else ''}")


def _render_trace(trace: list) -> None:
    if not trace:
        return
    with st.expander("🔍 Execution trace", expanded=False):
        for step in trace:
            st.code(step, language="text")


# ── Render existing messages ───────────────────────────────────────────────

for msg in st.session_state.ma_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            data = msg.get("data", {})
            if show_agents:
                _render_agent_cards(data)
            if show_scores and data.get("eval_scores"):
                _render_quality_scores(data["eval_scores"])
            if data.get("research_plan"):
                _render_research_plan(data["research_plan"])
            if show_entities and data.get("entities"):
                _render_entities(data["entities"])
            _render_governance(data)
            if show_trace:
                _render_trace(data.get("trace", []))
            # Meta footer
            intent = data.get("intent", "")
            iicon  = _INTENT_ICONS.get(intent, "🤖")
            st.caption(
                f"{iicon} {intent} | ⏱ {data.get('processing_time', 0):.2f}s | "
                f"🤖 {data.get('model','?')} | 🔢 {data.get('tokens_used',0)} tokens"
            )


# ── Chat input ─────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask the multi-agent ecosystem anything about your documents…"):
    st.session_state.ma_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.ma_messages[:-1]
        if m["role"] in ("user", "assistant")
    ][-10:]

    payload = {
        "message":              prompt,
        "conversation_id":      st.session_state.ma_conv_id,
        "top_k":               top_k,
        "temperature":         temperature,
        "max_tokens":          max_tokens,
        "conversation_history": history,
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Running multi-agent pipeline…_")

        try:
            resp = requests.post(
                f"{API_BASE_URL}/api/v1/multi-agent/chat",
                json=payload,
                headers=_tenant_headers(),
                timeout=600,
            )

            if resp.status_code == 200:
                data   = resp.json()
                answer = data.get("answer", "")
                placeholder.markdown(answer)

                if show_agents:
                    _render_agent_cards(data)
                if show_scores and data.get("eval_scores"):
                    _render_quality_scores(data["eval_scores"])
                if data.get("research_plan"):
                    _render_research_plan(data["research_plan"])
                if show_entities and data.get("entities"):
                    _render_entities(data["entities"])
                _render_governance(data)
                if show_trace:
                    _render_trace(data.get("trace", []))

                intent = data.get("intent", "")
                iicon  = _INTENT_ICONS.get(intent, "🤖")
                st.caption(
                    f"{iicon} {intent} | ⏱ {data.get('processing_time',0):.2f}s | "
                    f"🤖 {data.get('model','?')} | 🔢 {data.get('tokens_used',0)} tokens"
                )

                st.session_state.ma_messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "data":    data,
                })

            else:
                placeholder.error(f"❌ Error {resp.status_code}: {resp.text[:200]}")

        except requests.exceptions.ConnectionError:
            placeholder.error("❌ Cannot reach backend.")
        except requests.exceptions.Timeout:
            placeholder.error("⏱ Request timed out — multi-agent pipeline can take several minutes on CPU.")
        except Exception as exc:
            placeholder.error(f"❌ {exc}")

# ── Info panel ─────────────────────────────────────────────────────────────

st.divider()
with st.expander("ℹ️ How the multi-agent pipeline works", expanded=False):
    st.markdown("""
    | Step | Agent | What it does |
    |------|-------|-------------|
    | 1 | **Orchestrator** | Classifies query intent: `research` (complex) or `retrieval` (factual) |
    | 2a | **Research Agent** | Decomposes complex questions into sub-questions, answers each independently, then synthesises |
    | 2b | **Retrieval Agent** | Expands query, runs multi-strategy parallel retrieval, deduplicates, re-ranks |
    | 3 | **Knowledge Agent** | Extracts named entities, performs targeted entity lookups for richer context |
    | 4 | **Evaluation Agent** | Scores answer quality (faithfulness, length, sources) and optionally runs RAGAS |
    | 5 | **Governance Agent** | Runs safety guardrails, appends low-confidence disclaimers, blocks unsafe answers |
    | 6 | **Generate** | Assembles all agent outputs into the final answer |

    **vs. Standard Chat (Phase 9 Agent):**
    - Phase 9 uses a *single* adaptive RAG graph
    - Phase 11 uses *five* specialised agents coordinated by an orchestrator
    - Better for complex research tasks; higher latency than single-agent mode
    """)

# Made with Bob
