"""
RAG Chat Interface — Phase 8: User Experience
- Phase 6: Auto-persists conversation history in Redis
- Phase 7: Guardrail safety badges on responses
- Phase 8: Streaming responses, history restore on page load
"""

import json
import time
import requests
import streamlit as st
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Chat - RAG Platform",
    page_icon="💬",
    layout="wide"
)

st.title("💬 RAG Chat Assistant")
st.markdown("Ask questions about your uploaded documents")

# ── Session state ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "_history_loaded" not in st.session_state:
    st.session_state._history_loaded = False

if "tenant_id" not in st.session_state:
    st.session_state["tenant_id"] = "default"


def _tenant_headers() -> dict:
    """Return X-Tenant-ID header for the active tenant (set on the Settings page)."""
    return {"X-Tenant-ID": st.session_state.get("tenant_id", "default")}


# ── Phase 8: Restore history from server on page load ────────────────────

def _restore_history_from_server(conversation_id: str) -> None:
    """Fetch persisted messages from the memory API and rebuild session state."""
    if not conversation_id or st.session_state._history_loaded:
        return
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/v1/memory/{conversation_id}",
            params={"limit": 40},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            msgs = data.get("messages", [])
            # Skip system messages (summaries)
            rebuilt = []
            for m in msgs:
                if m.get("role") in ("user", "assistant"):
                    rebuilt.append({
                        "role":      m["role"],
                        "content":   m["content"],
                        "timestamp": m.get("timestamp", ""),
                        "sources":   [],
                    })
            if rebuilt:
                st.session_state.messages = rebuilt
                st.toast(f"Restored {len(rebuilt)} messages from server", icon="💾")
    except Exception:
        pass
    finally:
        st.session_state._history_loaded = True


if st.session_state.conversation_id and not st.session_state._history_loaded:
    _restore_history_from_server(st.session_state.conversation_id)


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Chat Settings")

    # RAG settings
    use_rag = st.toggle("Use RAG", value=True, help="Enable document retrieval")

    if use_rag:
        st.subheader("🔍 Retrieval Method")
        retrieval_method = st.radio(
            "Select method",
            options=["hybrid", "faiss", "bm25"],
            index=0,
            help="Hybrid combines semantic (FAISS) and keyword (BM25) search",
            horizontal=True,
        )
        method_descriptions = {
            "hybrid": "🔀 **Hybrid**: Best of both — combines semantic understanding with exact keyword matching",
            "faiss":  "🧠 **Semantic**: Understands meaning and context, finds conceptually similar content",
            "bm25":   "🔤 **Keyword**: Exact term matching, great for technical terms and specific phrases",
        }
        st.info(method_descriptions[retrieval_method])

        top_k = st.slider(
            "Documents to retrieve",
            min_value=1, max_value=10, value=5,
            help="Number of relevant document chunks to retrieve",
        )
    else:
        top_k = 5
        retrieval_method = "hybrid"

    # Query Understanding (Phase 3)
    st.subheader("🧠 Query Understanding")
    with st.expander("Configure query techniques", expanded=False):
        enable_query_understanding = st.toggle(
            "Enable Query Understanding",
            value=False,
            help="Use LLM to improve your query before retrieval (adds latency)",
        )
        if enable_query_understanding:
            enable_reformulation = st.checkbox("Reformulation", value=True,
                help="Clarify vague or ambiguous queries")
            enable_expansion = st.checkbox("Expansion", value=True,
                help="Generate alternative phrasings to increase recall")
            num_expansions = st.slider("Expansion variants", min_value=1, max_value=5, value=3)
            enable_hyde = st.checkbox("HyDE", value=True,
                help="Generate a hypothetical answer and use it for semantic search")
        else:
            enable_reformulation = enable_expansion = enable_hyde = False
            num_expansions = 3

    st.divider()

    # Reranking (Phase 4)
    st.subheader("🎯 Reranking")
    use_reranking = st.toggle(
        "Enable Reranking",
        value=False,
        help="Re-score retrieved chunks with a cross-encoder. Downloads ~22 MB on first use.",
    )
    if use_reranking:
        st.info("cross-encoder/ms-marco-MiniLM-L-6-v2", icon="🎯")

    st.divider()

    # Phase 8: Streaming toggle
    st.subheader("⚡ Streaming")
    use_streaming = st.toggle(
        "Stream responses",
        value=True,
        help="Show tokens as they are generated instead of waiting for the full answer",
    )

    st.divider()

    # Generation settings
    st.subheader("🎛️ Generation")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
        help="Higher = more creative, Lower = more focused")
    max_tokens = st.slider("Max tokens", 300, 2000, 800, 100,
        help="Maximum length of response")

    st.divider()

    # Tenant indicator
    _tid = st.session_state.get("tenant_id", "default")
    if _tid != "default":
        st.info(f"🏢 Tenant: **{_tid}**")
    else:
        st.caption("🏢 Tenant: default — [change in Settings](0_⚙️_Settings)")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        if st.session_state.conversation_id:
            try:
                requests.delete(
                    f"{API_BASE_URL}/api/v1/memory/{st.session_state.conversation_id}",
                    timeout=5,
                )
            except Exception:
                pass
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state._history_loaded = False
        st.rerun()

    st.divider()

    # Phase 6: Memory panel
    st.subheader("🧠 Conversation Memory")
    if st.session_state.get("conversation_id"):
        cid = st.session_state.conversation_id
        try:
            info_resp = requests.get(f"{API_BASE_URL}/api/v1/memory/{cid}/info", timeout=5)
            if info_resp.status_code == 200:
                info = info_resp.json()
                st.caption(
                    f"ID: `{cid[:16]}…`  "
                    f"| {info.get('message_count', 0)} messages stored"
                )
                if info.get("has_summary"):
                    st.caption("📝 History compacted with summary")
        except Exception:
            st.caption(f"ID: `{cid[:16]}…`")
    else:
        st.caption("No active conversation yet.")

    st.divider()

    # Phase 7: Safety settings
    st.subheader("🛡️ Safety Settings")
    try:
        safety_resp = requests.get(f"{API_BASE_URL}/api/v1/guardrails/status", timeout=5)
        if safety_resp.status_code == 200:
            s = safety_resp.json()
            checks = {
                "Injection":     s.get("injection_enabled"),
                "Toxicity":      s.get("toxicity_enabled"),
                "PII":           s.get("pii_enabled"),
                "Hallucination": s.get("hallucination_enabled"),
            }
            for name, enabled in checks.items():
                icon = "✅" if enabled else "⬜"
                st.caption(f"{icon} {name}")
    except Exception:
        st.caption("Safety status unavailable")

    st.divider()

    # API Status
    st.subheader("🔌 Status")
    try:
        if requests.get(f"{API_BASE_URL}/health", timeout=5).status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")

        chat_health = requests.get(f"{API_BASE_URL}/api/v1/chat/health", timeout=5)
        if chat_health.status_code == 200:
            hd = chat_health.json()
            st.success("✅ LLM Ready") if hd.get("llm_available") else st.warning("⚠️ LLM Not Available")
            vec = hd.get("vector_store", {})
            st.info(f"📊 {vec.get('total_vectors', 0)} chunks indexed")
    except Exception:
        st.error("❌ Disconnected")

    st.divider()
    st.markdown("""
    ### 💡 Tips
    - Upload documents first in the Documents page
    - **Hybrid** retrieval combines semantic + keyword search
    - Enable **Streaming** for token-by-token output
    - Sources show retrieval method and relevance scores
    """)


# ── Helpers ───────────────────────────────────────────────────────────────

def _render_sources(sources: list) -> None:
    """Render a collapsible source citations block."""
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for idx, src in enumerate(sources, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{idx}. {src.get('filename', '?')}**")
                if src.get("page_number"):
                    st.caption(f"📄 Page {src['page_number']}")
            with col2:
                method = src.get("retrieval_method", "unknown")
                badge = {"hybrid": "🔀 Hybrid", "faiss": "🧠 Semantic", "bm25": "🔤 Keyword"}.get(method, f"🔍 {method}")
                st.markdown(f"**{badge}**")

            score_parts = [f"**Score:** {src.get('score', 0):.3f}"]
            if src.get("faiss_score") is not None:
                rk = f" #{src.get('faiss_rank', '?')}" if src.get("faiss_rank") else ""
                score_parts.append(f"Semantic: {src['faiss_score']:.3f}{rk}")
            if src.get("bm25_score") is not None:
                rk = f" #{src.get('bm25_rank', '?')}" if src.get("bm25_rank") else ""
                score_parts.append(f"Keyword: {src['bm25_score']:.3f}{rk}")
            st.text(" | ".join(score_parts))

            # Phase 8: full content preview
            content = src.get("content", "")
            if content:
                with st.expander("📄 View chunk", expanded=False):
                    st.markdown(f"> {content}")
            st.divider()


def _render_safety_badges(warnings: list) -> None:
    """Phase 8: inline safety warning badges per message."""
    if not warnings:
        return
    with st.expander("⚠️ Safety notices", expanded=False):
        for w in warnings:
            detector = w.get("detector", "unknown")
            severity = w.get("severity") or "info"
            matched  = w.get("matched", [])
            label    = f"🔶 {detector.replace('_', ' ').title()}"
            if severity == "high":
                label = f"🔴 {detector.replace('_', ' ').title()}"
            elif severity == "medium":
                label = f"🟡 {detector.replace('_', ' ').title()}"
            st.caption(label + (f": {matched[0][:60]}…" if matched else ""))


# ── Display existing messages ──────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))
            _render_safety_badges(msg.get("guardrail_warnings", []))
            if msg.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
                    st.caption(f"🕐 {ts.strftime('%H:%M:%S')}")
                except Exception:
                    pass


# ── Streaming consumer (Phase 8) ──────────────────────────────────────────

def _stream_response(payload: dict):
    """
    Consume the SSE stream from /api/v1/chat/stream and yield text tokens.
    Also extracts sources from the [SOURCES]{...} sentinel line.
    Returns (full_text, sources) after the generator is exhausted.
    """
    sources    = []
    full_text  = []

    try:
        with requests.post(
            f"{API_BASE_URL}/api/v1/chat/stream",
            json=payload,
            headers=_tenant_headers(),
            stream=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                yield f"❌ Stream error {resp.status_code}: {resp.text}"
                return

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data = line[6:]

                if data == "[DONE]":
                    break
                elif data.startswith("[ERROR]"):
                    yield f"❌ {data[7:]}"
                    break
                elif data.startswith("[SOURCES]"):
                    try:
                        sources = json.loads(data[9:])
                    except Exception:
                        pass
                else:
                    full_text.append(data)
                    yield data

    except requests.exceptions.ConnectionError:
        yield "❌ Cannot connect to API. Make sure the backend is running."
    except requests.exceptions.Timeout:
        yield "⏱️ Stream timed out."

    # Attach sources to the generator so the caller can retrieve them
    _stream_response._last_sources = sources


_stream_response._last_sources = []


# ── Chat input ────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Build common payload
        request_data: dict = {
            "message":          prompt,
            "conversation_id":  st.session_state.conversation_id,
            "use_rag":          use_rag,
            "top_k":            top_k,
            "temperature":      temperature,
            "max_tokens":       max_tokens,
            "retrieval_method": retrieval_method if use_rag else None,
        }
        if use_rag and enable_query_understanding:
            request_data["query_understanding"] = {
                "enable_reformulation": enable_reformulation,
                "enable_expansion":     enable_expansion,
                "enable_hyde":          enable_hyde,
                "num_expansions":       num_expansions,
            }
        if use_rag:
            request_data["use_reranking"] = use_reranking

        # ── Streaming path ────────────────────────────────────────────
        if use_streaming:
            try:
                stream_payload = {
                    "message":          prompt,
                    "use_rag":          use_rag,
                    "top_k":            top_k,
                    "temperature":      temperature,
                    "max_tokens":       max_tokens,
                    "retrieval_method": retrieval_method if use_rag else None,
                }
                with st.spinner("Retrieving context…"):
                    # prime the request before showing streamed text
                    pass

                # st.write_stream consumes a generator and renders tokens live
                full_text = st.write_stream(_stream_response(stream_payload))
                sources   = _stream_response._last_sources

                _render_sources(sources)
                st.session_state.messages.append({
                    "role":               "assistant",
                    "content":            full_text,
                    "sources":            sources,
                    "guardrail_warnings": [],
                    "timestamp":          datetime.utcnow().isoformat(),
                })

                # Update conversation ID — streaming doesn't return one, so
                # use the non-streaming route to create / reuse the conversation
                if not st.session_state.conversation_id:
                    st.session_state.conversation_id = f"conv_{abs(hash(prompt))%10**10:010d}"

            except Exception as e:
                message_placeholder.error(f"❌ Streaming error: {e}")

        # ── Non-streaming path ────────────────────────────────────────
        else:
            try:
                with st.spinner("Thinking…"):
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/chat",
                        json=request_data,
                        headers=_tenant_headers(),
                        timeout=180,
                    )

                if response.status_code == 200:
                    data = response.json()

                    # Extract answer text
                    if "message" in data and isinstance(data["message"], dict):
                        answer = data["message"].get("content", "")
                    elif "response" in data:
                        answer = data["response"]
                    else:
                        answer = str(data.get("message", "No response generated"))

                    sources = data.get("sources", [])

                    # Update conversation ID from server
                    if data.get("conversation_id"):
                        st.session_state.conversation_id = data["conversation_id"]
                        st.session_state._history_loaded  = True  # already synced

                    # Phase 8: guardrail warnings from response metadata
                    guardrail_warnings = (
                        data.get("metadata", {}).get("guardrail_warnings", [])
                        if isinstance(data.get("metadata"), dict) else []
                    )

                    message_placeholder.markdown(answer)
                    _render_sources(sources)
                    _render_safety_badges(guardrail_warnings)

                    # Query understanding panel
                    qm = data.get("query_metadata")
                    if qm:
                        techniques = qm.get("techniques_applied", {})
                        if any(techniques.values()):
                            with st.expander("🧠 Query Understanding applied", expanded=False):
                                if techniques.get("reformulation") and qm.get("reformulated_query"):
                                    st.markdown(f"**Reformulated:** {qm['reformulated_query']}")
                                if techniques.get("expansion") and qm.get("expanded_queries"):
                                    st.markdown("**Expanded queries:**")
                                    for eq in qm["expanded_queries"]:
                                        st.markdown(f"- {eq}")
                                if techniques.get("hyde") and qm.get("hyde_answer"):
                                    st.markdown("**HyDE answer** *(used for semantic search)*")
                                    st.caption(qm["hyde_answer"])
                                st.caption(
                                    f"Query processing: {qm.get('processing_time', 0):.2f}s · "
                                    f"{qm.get('total_queries', 1)} total queries"
                                )

                    # Metadata footer
                    meta = data.get("metadata") or {}
                    model    = meta.get("model", data.get("model", "unknown"))
                    tokens   = meta.get("tokens_used", data.get("tokens_used", 0))
                    proc_t   = meta.get("total_time", data.get("processing_time", 0))
                    ret_m    = meta.get("retrieval_method", data.get("retrieval_method", ""))
                    method_icons = {"hybrid": "🔀", "faiss": "🧠", "bm25": "🔤"}
                    micon    = method_icons.get(ret_m, "🔍")
                    reranked = meta.get("reranking_applied", False)
                    rerank_badge = " | 🎯 reranked" if reranked else ""
                    guard_badge  = " | 🛡️ safety checked" if guardrail_warnings else ""
                    st.caption(
                        f"⏱️ {proc_t:.2f}s | 🤖 {model} | 📊 {tokens} tokens | "
                        f"{micon} {ret_m}{rerank_badge}{guard_badge}"
                    )

                    # Feedback widget (Phase 10)
                    fb_col1, fb_col2, _ = st.columns([1, 1, 8])
                    with fb_col1:
                        if st.button("👍", key=f"up_{len(st.session_state.messages)}", help="Good answer"):
                            try:
                                requests.post(f"{API_BASE_URL}/api/v1/feedback", json={
                                    "conversation_id": conversation_id,
                                    "message": prompt,
                                    "answer": answer,
                                    "rating": "up",
                                    "pipeline": "rag",
                                    "retrieval_method": ret_m,
                                }, timeout=5)
                                st.toast("Thanks for the feedback! 👍")
                            except Exception:
                                pass
                    with fb_col2:
                        if st.button("👎", key=f"dn_{len(st.session_state.messages)}", help="Poor answer"):
                            try:
                                requests.post(f"{API_BASE_URL}/api/v1/feedback", json={
                                    "conversation_id": conversation_id,
                                    "message": prompt,
                                    "answer": answer,
                                    "rating": "down",
                                    "pipeline": "rag",
                                    "retrieval_method": ret_m,
                                }, timeout=5)
                                st.toast("Thanks for the feedback! 👎")
                            except Exception:
                                pass

                    st.session_state.messages.append({
                        "role":               "assistant",
                        "content":            answer,
                        "sources":            sources,
                        "guardrail_warnings": guardrail_warnings,
                        "timestamp":          datetime.utcnow().isoformat(),
                    })

                elif response.status_code == 400:
                    # Guardrail block
                    try:
                        detail = response.json().get("detail", {})
                        block_reason = detail.get("block_reason", "unknown")
                        st.error(
                            f"🛡️ **Request blocked by safety guardrails**\n\n"
                            f"Reason: `{block_reason}`\n\n"
                            "Please rephrase your message and try again."
                        )
                    except Exception:
                        st.error(f"❌ Request blocked ({response.status_code})")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                message_placeholder.error("❌ Cannot connect to API. Make sure the backend is running.")
            except requests.exceptions.Timeout:
                message_placeholder.error("⏱️ Request timed out. Try reducing max_tokens or using a faster model.")
            except Exception as e:
                message_placeholder.error(f"❌ Error: {e}")


# ── Welcome message ───────────────────────────────────────────────────────

if not st.session_state.messages:
    st.info("""
    👋 **Welcome to RAG Chat!**

    Ask questions about your uploaded documents and get AI-powered answers with source citations.

    **Example questions:**
    - "What is the main topic of the documents?"
    - "Summarise the key points from the report"
    - "What are the recommendations mentioned?"

    💡 Upload documents in the **Documents** page first. Enable **Streaming** in the sidebar for real-time output.
    """)

# Made with Bob
