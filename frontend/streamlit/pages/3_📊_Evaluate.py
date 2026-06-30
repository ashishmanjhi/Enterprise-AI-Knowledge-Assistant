"""
RAG Evaluation Page (Phase 5)
Evaluate answer quality using RAGAS metrics.
"""

import streamlit as st
import requests
import json

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Evaluation - RAG Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Quality Evaluation")
st.markdown("Measure answer quality using RAGAS metrics (LLM-as-judge)")


# ── Shared evaluation runner ───────────────────────────────────────────────
# Defined here (before tabs call it) to avoid NameError.

def _run_evaluation(samples, metrics, judge_model=None):
    payload = {
        "samples": samples,
        "metrics": metrics,
    }
    if judge_model:
        payload["judge_model"] = judge_model

    with st.spinner(f"Running RAGAS evaluation on {len(samples)} sample(s)..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/evaluate",
                json=payload,
                timeout=600,  # evaluation can take a while
            )
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the backend is running.")
            return
        except requests.exceptions.Timeout:
            st.error("⏱️ Evaluation timed out. Try fewer samples or a faster model.")
            return

    if response.status_code == 200:
        data = response.json()
        _display_results(data)
    else:
        st.error(f"❌ Evaluation failed ({response.status_code}): {response.text}")


def _display_results(data):
    st.success(
        f"✅ Evaluation complete — {data['sample_count']} sample(s) in {data['duration_s']:.1f}s"
    )
    st.caption(f"Judge model: `{data['llm_model']}`")

    # ── Aggregated scores ──────────────────────────────────────────────
    st.subheader("📈 Aggregated Scores")
    agg = {k: v for k, v in data["aggregated"].items() if v is not None}
    if agg:
        cols = st.columns(len(agg))
        for col, (metric, score) in zip(cols, agg.items()):
            col.metric(
                label=metric.replace("_", " ").title(),
                value=f"{score:.3f}",
                delta=_score_label(score),
            )
    else:
        st.warning("No scores returned — check backend logs for errors.")

    # ── Per-sample breakdown ───────────────────────────────────────────
    if data.get("samples"):
        st.subheader("🔍 Per-Sample Scores")
        for i, sample_scores in enumerate(data["samples"]):
            with st.expander(f"Sample {i+1}"):
                for s in sample_scores:
                    val = f"{s['score']:.3f}" if s["score"] is not None else "N/A"
                    st.markdown(f"- **{s['metric']}**: {val}")

    # ── Raw JSON ───────────────────────────────────────────────────────
    with st.expander("Raw JSON response"):
        st.json(data)


def _score_label(score: float) -> str:
    if score >= 0.8: return "🟢 Good"
    if score >= 0.6: return "🟡 Fair"
    return "🔴 Poor"


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Evaluation Settings")

    # Fetch available metrics from API
    available_metrics = []
    default_metrics   = []
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/evaluate/metrics", timeout=5)
        if resp.status_code == 200:
            data              = resp.json()
            available_metrics = data.get("metrics", [])
            default_metrics   = data.get("default_metrics", [])
    except Exception:
        pass

    if available_metrics:
        metric_options = {m["name"]: m for m in available_metrics}
        selected_metrics = st.multiselect(
            "Metrics to compute",
            options=list(metric_options.keys()),
            default=default_metrics,
            help="Metrics marked * require a ground-truth answer"
        )
        # Show descriptions
        for name in selected_metrics:
            m = metric_options[name]
            gt_note = " *(requires ground truth)*" if m["requires_ground_truth"] else ""
            st.caption(f"**{name}**{gt_note}: {m['description']}")
    else:
        selected_metrics = ["faithfulness", "answer_relevancy", "context_precision"]
        st.info("Could not load metric list from API — using defaults")

    st.divider()

    judge_override = st.text_input(
        "Judge model override",
        value="",
        placeholder="Leave blank to use server default",
        help="Ollama model name to use as judge, e.g. gemma3:4b"
    )

    st.divider()
    st.markdown("""
    ### 💡 Tips
    - Each sample = one question + its answer + retrieved chunks
    - **Faithfulness**: does the answer only use what's in the context?
    - **Answer relevancy**: does the answer actually address the question?
    - **Context precision**: were the right chunks retrieved?
    - Context recall & factual correctness need a **ground truth** answer
    """)

# ── Main area ──────────────────────────────────────────────────────────────

tab_manual, tab_json = st.tabs(["✏️ Manual Entry", "📂 Paste JSON"])

# ── Tab 1: Manual entry ────────────────────────────────────────────────────
with tab_manual:
    st.subheader("Add evaluation samples manually")

    if "eval_samples" not in st.session_state:
        st.session_state.eval_samples = []

    with st.form("add_sample_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            question     = st.text_area("Question *", height=80, placeholder="What is EVSE uptime?")
            answer       = st.text_area("LLM Answer *", height=120, placeholder="EVSE uptime refers to...")
        with col2:
            context_text = st.text_area(
                "Retrieved context (one chunk per line) *",
                height=120,
                placeholder="EVSE connectors experience downtime due to...\nNetwork failures account for 40% of incidents..."
            )
            ground_truth = st.text_input("Ground truth answer (optional)", placeholder="Leave blank if not available")

        if st.form_submit_button("➕ Add sample"):
            if question and answer and context_text:
                contexts = [c.strip() for c in context_text.strip().split("\n") if c.strip()]
                st.session_state.eval_samples.append({
                    "question":     question,
                    "answer":       answer,
                    "contexts":     contexts,
                    "ground_truth": ground_truth or None,
                })
                st.success(f"Sample added ({len(st.session_state.eval_samples)} total)")
            else:
                st.error("Question, answer, and context are required.")

    # Show staged samples
    if st.session_state.eval_samples:
        st.markdown(f"**{len(st.session_state.eval_samples)} sample(s) staged:**")
        for i, s in enumerate(st.session_state.eval_samples):
            with st.expander(f"Sample {i+1}: {s['question'][:60]}..."):
                st.markdown(f"**Answer:** {s['answer'][:200]}...")
                st.markdown(f"**Contexts:** {len(s['contexts'])} chunk(s)")
                if s.get("ground_truth"):
                    st.markdown(f"**Ground truth:** {s['ground_truth'][:100]}...")
        col_run, col_clear = st.columns([3, 1])
        with col_clear:
            if st.button("🗑️ Clear all", use_container_width=True):
                st.session_state.eval_samples = []
                st.rerun()
        with col_run:
            run_manual = st.button("▶️ Run Evaluation", type="primary", use_container_width=True)
    else:
        st.info("Add at least one sample above, then run evaluation.")
        run_manual = False

    if run_manual and st.session_state.eval_samples:
        _run_evaluation(
            samples=st.session_state.eval_samples,
            metrics=selected_metrics,
            judge_model=judge_override or None,
        )

# ── Tab 2: JSON paste ──────────────────────────────────────────────────────
with tab_json:
    st.subheader("Paste samples as JSON")
    st.markdown("""
    Expected format:
    ```json
    [
      {
        "question": "What is EVSE uptime?",
        "answer":   "EVSE uptime refers to...",
        "contexts": ["EVSE connectors experience downtime..."],
        "ground_truth": null
      }
    ]
    ```
    """)

    json_input = st.text_area("JSON samples", height=250, placeholder="Paste your JSON array here")

    if st.button("▶️ Run Evaluation from JSON", type="primary"):
        try:
            samples_json = json.loads(json_input)
            if not isinstance(samples_json, list) or not samples_json:
                st.error("Input must be a non-empty JSON array.")
            else:
                _run_evaluation(
                    samples=samples_json,
                    metrics=selected_metrics,
                    judge_model=judge_override or None,
                )
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")


# Made with Bob
