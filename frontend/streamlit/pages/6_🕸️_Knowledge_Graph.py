"""
Knowledge Graph Explorer UI — Phase 12
Allows users to:
  1. Build / update the knowledge graph from free text
  2. Query / explore entities and their graph neighbourhoods
  3. Inspect entity types and relation triples
  4. View live graph statistics
"""

import time
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"
KG_BASE = f"{API_BASE_URL}/api/v1/kg"

st.set_page_config(
    page_title="Knowledge Graph - RAG Platform",
    page_icon="🕸️",
    layout="wide",
)

st.title("🕸️ Knowledge Graph Explorer")
st.markdown(
    "Build, query, and explore the **entity–relation graph** that enriches "
    "your RAG pipeline with structured knowledge."
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs):
    """Thin API helper — returns (data_dict | None, error_str | None)."""
    try:
        r = requests.request(method, f"{KG_BASE}{path}", timeout=60, **kwargs)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach API — is the backend running on port 8000?"
    except Exception as exc:
        return None, str(exc)


# ── Sidebar — Stats & Controls ─────────────────────────────────────────────

with st.sidebar:
    st.header("📊 Graph Statistics")

    if st.button("🔄 Refresh stats", key="refresh_stats"):
        st.session_state.pop("kg_stats", None)

    if "kg_stats" not in st.session_state:
        stats, err = _api("GET", "/stats")
        if stats:
            st.session_state.kg_stats = stats
        else:
            st.session_state.kg_stats = None
            if err:
                st.warning(err)

    stats = st.session_state.get("kg_stats")
    if stats:
        st.metric("Entities", stats.get("num_entities", 0))
        st.metric("Relations", stats.get("num_relations", 0))
        st.caption(f"Persist path: `{stats.get('persist_path', '?')}`")
        et = stats.get("entity_types", {})
        if et:
            st.markdown("**Entity type breakdown**")
            for etype, cnt in sorted(et.items(), key=lambda x: -x[1]):
                st.progress(
                    min(cnt / max(et.values(), default=1), 1.0),
                    text=f"{etype}: {cnt}",
                )
    else:
        st.info("No stats available yet.")

    st.divider()
    st.header("⚠️ Danger Zone")
    if st.button("🗑️ Clear knowledge graph", type="secondary"):
        with st.spinner("Clearing…"):
            _, err = _api("DELETE", "/clear")
        if err:
            st.error(err)
        else:
            st.success("Knowledge graph cleared.")
            st.session_state.pop("kg_stats", None)
            st.rerun()


# ── Tab layout ─────────────────────────────────────────────────────────────

tab_build, tab_query, tab_entities, tab_relations = st.tabs([
    "🔨 Build Graph",
    "🔍 Query / Explore",
    "📋 Entities",
    "🔗 Relations",
])


# ── Tab 1: Build Graph ─────────────────────────────────────────────────────

with tab_build:
    st.subheader("Extract entities & relations from text")
    st.markdown(
        "Paste any document excerpt, article, or notes below. "
        "The pipeline uses an LLM (with regex fallback) to extract "
        "named entities and relation triples into the graph."
    )

    col_input, col_options = st.columns([3, 1])

    with col_options:
        source_doc = st.text_input(
            "Source document label",
            value="manual_input",
            help="Identifier attached to extracted nodes/edges (e.g. filename)",
        )
        extract_relations = st.toggle("Extract relations", value=True)
        persist = st.toggle("Save to disk", value=True)

    with col_input:
        text_input = st.text_area(
            "Text to ingest",
            height=260,
            placeholder=(
                "Paste text here, e.g.:\n\n"
                "LangGraph is a framework developed by LangChain for building "
                "stateful multi-agent applications. It uses NetworkX under the "
                "hood and integrates with OpenAI, Anthropic, and Ollama providers."
            ),
        )

    if st.button("⚡ Extract & Build", type="primary", disabled=not text_input.strip()):
        payload = {
            "text": text_input,
            "source_doc": source_doc,
            "extract_relations": extract_relations,
            "persist": persist,
        }
        with st.spinner("Extracting entities and relations…"):
            result, err = _api("POST", "/build", json=payload)

        if err:
            st.error(err)
        elif result:
            elapsed = result.get("elapsed_seconds", 0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Entities added", result.get("entities_added", 0))
            c2.metric("Relations added", result.get("relations_added", 0))
            c3.metric("Elapsed (s)", f"{elapsed:.2f}")
            gs = result.get("graph_stats", {})
            st.success(
                f"Graph now has **{gs.get('num_entities', '?')} entities** "
                f"and **{gs.get('num_relations', '?')} relations**."
            )
            # Invalidate cached stats
            st.session_state.pop("kg_stats", None)


# ── Tab 2: Query / Explore ─────────────────────────────────────────────────

with tab_query:
    st.subheader("Search entities and explore their graph neighbourhood")

    qcol, _ = st.columns([3, 1])
    with qcol:
        query_text = st.text_input(
            "Search query",
            placeholder="e.g. LangGraph, neural network, Ashish Manjhi …",
        )
        top_k_q = st.slider("Max results", 1, 50, 10, key="query_topk")

    if st.button("🔍 Search", type="primary", disabled=not query_text.strip()):
        with st.spinner("Querying knowledge graph…"):
            data, err = _api("GET", "/query", params={"q": query_text, "top_k": top_k_q})

        if err:
            st.error(err)
        elif data:
            results = data.get("results", [])
            st.markdown(
                f"Found **{data.get('count', 0)}** results in "
                f"`{data.get('elapsed', '?')}s`"
            )
            if not results:
                st.info("No matching entities found.")
            else:
                for i, r in enumerate(results):
                    with st.expander(
                        f"**{r.get('entity_text', '?')}** "
                        f"[{r.get('entity_type', '?')}] — score {r.get('score', 0):.4f}",
                        expanded=(i == 0),
                    ):
                        st.markdown(f"**ID:** `{r.get('entity_id', '')}`")
                        src = r.get("source_docs", [])
                        if src:
                            st.markdown(f"**Sources:** {', '.join(src)}")

                        rels = r.get("relations", [])
                        if rels:
                            st.markdown("**Relations:**")
                            for rel in rels[:10]:
                                st.markdown(
                                    f"- {r.get('entity_text', '?')} "
                                    f"*{rel.get('predicate', '')}* "
                                    f"**{rel.get('object_text', '')}**"
                                )

                        nbrs = r.get("neighbours", [])
                        if nbrs:
                            st.markdown("**Neighbours:**")
                            nbr_labels = [
                                f"{n.get('text', n.get('id', '?'))} "
                                f"[{n.get('entity_type', '')}]"
                                for n in nbrs[:8]
                            ]
                            st.markdown(" · ".join(nbr_labels))

                        ctx = r.get("context_text", "")
                        if ctx:
                            st.markdown("**Context:**")
                            st.code(ctx, language="text")


# ── Tab 3: Entities ────────────────────────────────────────────────────────

with tab_entities:
    st.subheader("Browse all entities")

    col_filter, col_limit = st.columns([2, 1])
    with col_filter:
        type_options = ["(all)", "PERSON", "ORG", "PRODUCT", "LOCATION", "CONCEPT", "TECHNICAL"]
        selected_type = st.selectbox("Filter by type", type_options)
    with col_limit:
        ent_limit = st.number_input("Max entities", min_value=1, max_value=1000, value=100)

    if st.button("📋 Load entities", type="secondary"):
        params: dict = {"limit": ent_limit}
        if selected_type != "(all)":
            params["entity_type"] = selected_type

        with st.spinner("Loading entities…"):
            data, err = _api("GET", "/entities", params=params)

        if err:
            st.error(err)
        elif data is not None:
            if not data:
                st.info("No entities found.")
            else:
                st.caption(f"Showing {len(data)} entities")
                rows = []
                for e in data:
                    rows.append({
                        "ID":          e.get("id", ""),
                        "Text":        e.get("text", ""),
                        "Type":        e.get("entity_type", ""),
                        "Sources":     ", ".join(e.get("source_docs", [])[:3]),
                    })
                st.dataframe(rows, hide_index=True)


# ── Tab 4: Relations ───────────────────────────────────────────────────────

with tab_relations:
    st.subheader("Inspect relations for an entity")

    entity_id_input = st.text_input(
        "Entity ID",
        placeholder="Paste an entity ID from the Entities tab or a query result…",
    )

    if st.button("🔗 Load relations", type="secondary", disabled=not entity_id_input.strip()):
        with st.spinner("Loading relations…"):
            data, err = _api("GET", "/relations", params={"entity_id": entity_id_input})

        if err:
            st.error(err)
        elif data is not None:
            if not data:
                st.info("No outgoing relations for this entity.")
            else:
                st.caption(f"{len(data)} relation(s) found")
                rows = [
                    {
                        "Subject ID":    r["subject_id"],
                        "Predicate":     r["predicate"],
                        "Object":        r["object_text"],
                        "Object ID":     r["object_id"],
                        "Confidence":    round(r["confidence"], 2),
                        "Source Doc":    r["source_doc"],
                    }
                    for r in data
                ]
                st.dataframe(rows, hide_index=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption("Phase 12 — Knowledge Graph Enhancement · Enterprise Agentic RAG Platform")

# Made with Bob
