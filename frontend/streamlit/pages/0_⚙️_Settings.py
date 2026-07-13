"""
Platform Settings Page — Phase 15 Multi-Tenancy

Sets app-wide preferences that persist for the browser-tab session:
  • Tenant ID  — which X-Tenant-ID header to send on every API request.
                 All pages read st.session_state["tenant_id"] and pass it
                 as a header automatically.
  • Other UI prefs may be added here in future phases.

How multi-tenancy works
───────────────────────
When MULTI_TENANCY_ENABLED=true on the backend, every upload, chat, and
agent request is scoped to the tenant named by the X-Tenant-ID header.
Each tenant gets its own FAISS + BM25 index under
  data/vectorstore/<tenant_id>/

When the feature is disabled (default) all requests behave as before —
the "default" tenant bucket is used for everyone.
"""

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Settings - RAG Platform",
    page_icon="⚙️",
    layout="wide",
)

# ── Initialise session defaults ───────────────────────────────────────────

if "tenant_id" not in st.session_state:
    st.session_state["tenant_id"] = "default"


# ── Helper: read backend multi-tenancy flag ────────────────────────────────

def _mt_enabled() -> bool:
    """Return True if the backend has MULTI_TENANCY_ENABLED=true."""
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/status", timeout=4)
        if r.status_code == 200:
            return r.json().get("multi_tenancy_enabled", False)
    except Exception:
        pass
    return False


# ── Page ──────────────────────────────────────────────────────────────────

st.title("⚙️ Platform Settings")
st.markdown("Configure app-wide preferences. Changes apply immediately to all pages in this tab.")

# ── Tenant configuration ──────────────────────────────────────────────────

st.header("🏢 Tenant Configuration")

mt_on = _mt_enabled()

if mt_on:
    st.success(
        "✅ Multi-tenancy is **enabled** on the backend.  "
        "Each tenant's documents and indexes are fully isolated."
    )
else:
    st.info(
        "ℹ️ Multi-tenancy is **disabled** on the backend (default).  "
        "The Tenant ID field below has no effect until you set "
        "`MULTI_TENANCY_ENABLED=true` in your `.env` and restart the server."
    )

col_input, col_btn = st.columns([3, 1])

with col_input:
    new_tenant = st.text_input(
        "Tenant ID",
        value=st.session_state["tenant_id"],
        placeholder="e.g. acme-corp, team-alpha, default",
        help=(
            "Sent as the `X-Tenant-ID` header on every upload, chat, and agent request.  "
            "Allowed characters: letters, digits, hyphens, underscores (1–64 chars)."
        ),
    )

with col_btn:
    st.write("")  # vertical alignment
    st.write("")
    if st.button("Apply", type="primary", use_container_width=True):
        # Validate slug client-side before saving
        import re
        if re.fullmatch(r"[a-zA-Z0-9_\-]{1,64}", new_tenant):
            st.session_state["tenant_id"] = new_tenant
            st.success(f"Tenant set to **{new_tenant}**")
        else:
            st.error(
                "Invalid tenant ID. Use only letters, digits, hyphens and underscores "
                "(1–64 characters)."
            )

st.caption(
    f"Current active tenant: `{st.session_state['tenant_id']}`  ·  "
    "This value is stored in your browser-tab session and resets on page refresh."
)

# ── Live tenant index stats (when MT enabled) ──────────────────────────────

if mt_on:
    st.divider()
    st.subheader("📊 Tenant Index Overview")
    st.caption("Known tenants are those with an existing data directory on the server.")

    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/status", timeout=4)
        tenants = r.json().get("tenants", []) if r.status_code == 200 else []
    except Exception:
        tenants = []

    if tenants:
        st.write(f"**{len(tenants)} tenant(s) found:**")
        for t in sorted(tenants):
            active_badge = " ← **(active)**" if t == st.session_state["tenant_id"] else ""
            st.markdown(f"- `{t}`{active_badge}")
    else:
        st.info("No tenant data directories found yet (upload a document to create one).")

# ── How it works (expandable) ─────────────────────────────────────────────

st.divider()
with st.expander("ℹ️ How multi-tenancy works", expanded=False):
    st.markdown("""
    **Backend flow**

    1. You set a Tenant ID here (e.g. `acme-corp`).
    2. Every API call from Documents, Chat, Agent, and Multi-Agent pages
       automatically includes the header `X-Tenant-ID: acme-corp`.
    3. The backend resolves the header in every route and directs all
       FAISS and BM25 index operations to
       `data/vectorstore/acme-corp/`.
    4. Documents uploaded by `acme-corp` are **never visible** to other tenants.

    **Enabling multi-tenancy**

    Add to your `.env` file and restart the server:
    ```
    MULTI_TENANCY_ENABLED=true
    ```

    **Default behaviour (disabled)**

    All requests use the `default` tenant bucket — identical to the
    pre-multi-tenancy behaviour.  The Tenant ID header is sent but
    silently ignored by the backend.

    **Tenant ID rules**

    - 1–64 characters
    - Letters, digits, hyphens (`-`), underscores (`_`)
    - No spaces, slashes, or dots
    """)

# Made with Bob
