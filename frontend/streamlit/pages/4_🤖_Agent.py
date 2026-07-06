"""
Agentic RAG Chat Page — Phase 9
Runs the full LangGraph pipeline:
  route_query → (rewrite_query) → retrieve → grade_documents
              → generate → check_grounding

Shows live node-progress events and word-by-word generation via SSE streaming.
Falls back to the non-streaming /chat endpoint on SSE failure.
"""

import json
import time
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic Chat - RAG Platform",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic RAG Chat")
st.markdown(
    "Powered by **LangGraph** — the agent automatically selects retrieval strategy, "
    "rewrites vague queries, grades documents for relevance, and verifies answer grounding."
)

# ── Session state ──────────────────────────────────────────────────────────
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "agent_conversation_id" not in st.session_state:
    st.session_state.agent_conversation_id = f"agent_{int(time.time())}"

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Agent Settings")

    st.subheader("Retrieval")
    retrieval_override = st.radio(
        "Strategy override",
        ["🤖 Auto (agent decides)", "🔀 Hybrid", "🧠 FAISS (semantic)", "🔤 BM25 (keyword)"],
        index=0,
        help="'Auto' lets the LangGraph router pick the best strategy for each query.",
    )
    method_map = {
        "🤖 Auto (agent decides)": None,
        "🔀 Hybrid": "hybrid",
        "🧠 FAISS (semantic)": "faiss",
        "🔤 BM25 (keyword)": "bm25",
    }
    retrieval_method = method_map[retrieval_override]

    top_k = st.slider("Documents to retrieve", 1, 20, 5)

    st.divider()
    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens  = st.slider("Max tokens", 100, 2000, 600, 100)
    use_reranking = st.toggle("Cross-encoder reranking", value=False)

    st.divider()
    st.subheader("Agent behaviour")
    show_trace = st.toggle("Show graph trace", value=True,
                           help="Display each node execution step in the response.")
    show_grounding = st.toggle("Show grounding verdict", value=True)

    st.divider()
    # Agent health
    try:
        h = requests.get(f"{API_BASE_URL}/api/v1/agent/health", timeout=4).json()
        if h.get("status") == "healthy":
            st.success("✅ Agent ready")
            st.caption(
                f"Max rewrites: {h.get('max_rewrites', '?')}  |  "
                f"Doc grading: {'on' if h.get('document_grading') else 'off'}  |  "
                f"Grounding check: {'on' if h.get('grounding_check') else 'off'}"
            )
        else:
            st.error(f"❌ Agent unhealthy: {h.get('error', 'unknown')}")
    except Exception:
        st.error("❌ Backend unreachable")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.agent_messages = []
        st.session_state.agent_conversation_id = f"agent_{int(time.time())}"
        st.rerun()

    st.caption(f"Conversation ID: `{st.session_state.agent_conversation_id}`")


# ── Helper: render source cards ────────────────────────────────────────────

def _render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)} chunks)", expanded=False):
        for i, src in enumerate(sources, 1):
            method_icon = {"hybrid": "🔀", "faiss": "🧠", "bm25": "🔤"}.get(
                src.get("retrieval_method", ""), "🔍"
            )
            score_str = f"{src.get('score', 0.0):.3f}"
            pg = f" · p.{src['page_number']}" if src.get("page_number") else ""
            extra = []
            if src.get("faiss_score") is not None:
                extra.append(f"FAISS={src['faiss_score']:.3f}")
            if src.get("bm25_score") is not None:
                extra.append(f"BM25={src['bm25_score']:.3f}")
            extra_str = f" ({', '.join(extra)})" if extra else ""
            st.markdown(
                f"**[{i}] {src.get('filename', 'unknown')}{pg}**  "
                f"{method_icon} score={score_str}{extra_str}"
            )
            st.caption(src.get("content", ""))
            if i < len(sources):
                st.divider()


# ── Helper: render agent trace ─────────────────────────────────────────────

def _render_trace(trace: list) -> None:
    if not trace:
        return
    with st.expander("🔍 Graph trace", expanded=False):
        for step in trace:
            st.code(step, language="text")


# ── Helper: grounding badge ────────────────────────────────────────────────

def _grounding_badge(is_grounded: bool | None) -> str:
    if is_grounded is None:
        return ""
    return "✅ grounded" if is_grounded else "⚠️ unverified"


# ── Render existing messages ───────────────────────────────────────────────

for msg in st.session_state.agent_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))
            if show_trace:
                _render_trace(msg.get("trace", []))
            meta = msg.get("meta", {})
            if meta:
                parts = []
                rm = meta.get("retrieval_method")
                if rm:
                    icon = {"hybrid": "🔀", "faiss": "🧠", "bm25": "🔤"}.get(rm, "🔍")
                    parts.append(f"{icon} {rm}")
                rc = meta.get("rewrite_count", 0)
                if rc:
                    parts.append(f"✏️ {rc} rewrite{'s' if rc > 1 else ''}")
                if show_grounding:
                    gb = _grounding_badge(meta.get("is_grounded"))
                    if gb:
                        parts.append(gb)
                rq = meta.get("rewritten_query")
                if rq:
                    parts.append(f'rewrote → *"{rq[:60]}"*')
                parts.append(f"⏱ {meta.get('processing_time', 0.0):.2f}s")
                parts.append(f"🤖 {meta.get('model', 'unknown')}")
                st.caption(" | ".join(parts))


# ── Chat input ─────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents (agent mode)…"):
    # Show user message immediately
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build payload
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.agent_messages[:-1]  # exclude the just-added user msg
        if m["role"] in ("user", "assistant")
    ]

    payload: dict = {
        "message": prompt,
        "conversation_id": st.session_state.agent_conversation_id,
        "top_k": top_k,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "use_reranking": use_reranking,
        "conversation_history": history[-10:],  # last 10 messages for context
    }
    if retrieval_method:
        payload["retrieval_method"] = retrieval_method

    with st.chat_message("assistant"):
        node_status   = st.empty()   # live node-progress line
        placeholder   = st.empty()   # streaming answer text
        node_status.markdown("_Starting agent…_")

        # Accumulated state from SSE events
        streamed_text   = ""
        done_data: dict = {}
        sse_ok          = False

        try:
            with requests.post(
                f"{API_BASE_URL}/api/v1/agent/chat/stream",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                if resp.status_code == 429:
                    node_status.empty()
                    placeholder.warning("⚠️ Rate limit reached. Please wait a moment.")
                elif resp.status_code != 200:
                    node_status.empty()
                    placeholder.error(f"❌ Error {resp.status_code}: {resp.text[:200]}")
                else:
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        # SSE lines begin with "data: "
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        if not line.startswith("data: "):
                            continue
                        payload_str = line[len("data: "):]
                        try:
                            event = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue

                        etype = event.get("type")

                        if etype == "node":
                            node_name = event.get("node", "")
                            _NODE_LABELS = {
                                "route_query":      "🔀 Routing query…",
                                "rewrite_query":    "✏️ Rewriting query…",
                                "retrieve":         "🔍 Retrieving documents…",
                                "grade_documents":  "📋 Grading documents…",
                                "generate":         "💬 Generating answer…",
                                "check_grounding":  "✅ Checking grounding…",
                            }
                            node_status.markdown(
                                _NODE_LABELS.get(node_name, f"⚙️ {node_name}…")
                            )

                        elif etype == "token":
                            streamed_text += event.get("content", "")
                            placeholder.markdown(streamed_text + "▌")

                        elif etype == "done":
                            done_data = event.get("data", {})
                            sse_ok = True
                            # Remove cursor on completion
                            placeholder.markdown(streamed_text)
                            node_status.empty()

                        elif etype == "error":
                            node_status.empty()
                            placeholder.error(
                                f"❌ Agent error: {event.get('detail', 'unknown')}"
                            )

        except requests.exceptions.ConnectionError:
            placeholder.error(
                "❌ Cannot reach backend. Is `uvicorn backend.api.main:app --reload` running?"
            )
        except requests.exceptions.Timeout:
            placeholder.error(
                "⏱ Request timed out (LLM may be slow on CPU). Try a shorter query."
            )
        except Exception as exc:
            placeholder.error(f"❌ Unexpected error: {exc}")

        if sse_ok and done_data:
            answer          = done_data.get("answer", streamed_text)
            sources         = done_data.get("sources", [])
            trace           = done_data.get("trace", [])
            is_grounded     = done_data.get("is_grounded")
            rewrite_count   = done_data.get("rewrite_count", 0)
            rewritten_query = done_data.get("rewritten_query")
            ret_method      = done_data.get("retrieval_method", "")
            model           = done_data.get("model", "unknown")
            tokens_used     = done_data.get("tokens_used", 0)
            proc_t          = done_data.get("processing_time", 0.0)

            # Render sources and trace below the answer
            _render_sources(sources)
            if show_trace:
                _render_trace(trace)

            # Meta caption
            parts = []
            if ret_method:
                icon = {"hybrid": "🔀", "faiss": "🧠", "bm25": "🔤"}.get(ret_method, "🔍")
                parts.append(f"{icon} {ret_method}")
            if rewrite_count:
                parts.append(f"✏️ {rewrite_count} rewrite{'s' if rewrite_count > 1 else ''}")
            if show_grounding:
                gb = _grounding_badge(is_grounded)
                if gb:
                    parts.append(gb)
            if rewritten_query:
                parts.append(f'rewrote → *"{rewritten_query[:60]}"*')
            parts.append(f"⏱ {proc_t:.2f}s")
            parts.append(f"🤖 {model}")
            if tokens_used:
                parts.append(f"🔢 {tokens_used} tokens")
            st.caption(" | ".join(parts))

            # Persist message in session
            st.session_state.agent_messages.append({
                "role":    "assistant",
                "content": answer,
                "sources": sources,
                "trace":   trace,
                "meta": {
                    "retrieval_method": ret_method,
                    "rewrite_count":    rewrite_count,
                    "rewritten_query":  rewritten_query,
                    "is_grounded":      is_grounded,
                    "processing_time":  proc_t,
                    "model":            model,
                    "tokens_used":      tokens_used,
                },
            })

# ── Footer info ────────────────────────────────────────────────────────────

st.divider()
with st.expander("ℹ️ How the agent works", expanded=False):
    st.markdown("""
    **LangGraph pipeline — nodes execute in this order:**

    | Node | What it does |
    |------|-------------|
    | `route_query` | LLM classifies query → picks `hybrid`, `faiss`, or `bm25` |
    | `rewrite_query` | If query is vague, LLM rewrites it for better retrieval (≤ max_rewrites) |
    | `retrieve` | Runs the chosen retrieval strategy against the vector/BM25 index |
    | `grade_documents` | LLM scores each chunk yes/no for relevance; drops irrelevant ones |
    | `generate` | Builds a grounded answer from the kept chunks |
    | `check_grounding` | Verifies the answer only uses information from the retrieved context |

    If grounding fails and rewrite budget remains, the graph loops back to `route_query` for another attempt.

    **Differences from standard Chat page:**
    - The retrieval strategy is **decided per-query** rather than fixed
    - Low-quality documents are **filtered before generation**
    - Every answer is **grounding-verified** against the source chunks
    - The full decision trace is available for inspection
    """)

# Made with Bob
