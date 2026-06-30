"""
Streamlit frontend for Enterprise Agentic RAG Platform.
"""

import streamlit as st
import requests
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.core.settings import settings


# Page configuration
st.set_page_config(
    page_title=settings.app_name,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = f"http://localhost:{settings.api_port}"


def check_backend_status():
    """Check if backend is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_status():
    """Get detailed status from backend."""
    try:
        response = requests.get(f"{API_BASE_URL}{settings.api_prefix}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Failed to get status: {e}")
        return None


def get_providers():
    """Get available providers."""
    try:
        response = requests.get(
            f"{API_BASE_URL}{settings.api_prefix}/status/providers",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


# Main title
st.title(f"🤖 {settings.app_name}")
st.caption(f"Version {settings.app_version} | Environment: {settings.environment}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Backend status
    st.subheader("Backend Status")
    if check_backend_status():
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Disconnected")
        st.warning("Please start the FastAPI backend server")
        st.code("uvicorn backend.api.main:app --reload", language="bash")
    
    st.divider()
    
    # Provider selection
    st.subheader("LLM Provider")
    providers_info = get_providers()
    
    if providers_info:
        available_providers = providers_info.get("available_providers", [])
        default_provider = providers_info.get("default_provider", "ollama")
        
        provider = st.selectbox(
            "Select Provider",
            available_providers,
            index=available_providers.index(default_provider) if default_provider in available_providers else 0
        )
        
        # Model selection based on provider
        if provider == "ollama":
            model = st.selectbox(
                "Model",
                ["qwen3:4b", "gemma3:4b", "phi4-mini"],
                index=0
            )
        elif provider == "huggingface":
            model = st.text_input(
                "Model",
                value=settings.hf_default_model
            )
        else:
            model = st.text_input("Model", value="gpt-4")
            st.info(f"{provider} provider will be available in Phase 10")
    else:
        provider = "ollama"
        model = "qwen3:4b"
        st.warning("Could not fetch provider information")
    
    st.divider()
    
    # Generation parameters
    st.subheader("Generation Parameters")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=settings.default_temperature,
        step=0.1,
        help="Higher values make output more random"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=2048,
        value=settings.default_max_tokens,
        step=50,
        help="Maximum number of tokens to generate"
    )
    
    st.divider()
    
    # System status
    st.subheader("System Status")
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
    
    status = get_status()
    if status:
        services = status.get("services", {})
        
        # PostgreSQL
        pg_status = services.get("postgresql", "unknown")
        if pg_status == "connected":
            st.success("✅ PostgreSQL")
        else:
            st.error("❌ PostgreSQL")
        
        # Redis
        redis_status = services.get("redis", "unknown")
        if redis_status == "connected":
            st.success("✅ Redis")
        else:
            st.error("❌ Redis")
        
        # Ollama
        ollama_status = services.get("ollama", "unknown")
        if ollama_status == "connected":
            st.success("✅ Ollama")
            models = services.get("ollama_models", [])
            if models:
                with st.expander("Available Models"):
                    for m in models:
                        st.text(f"• {m}")
        else:
            st.error("❌ Ollama")

# Main content area
st.header("🎉 Enterprise Agentic RAG Platform")

# Phase status banner
st.success("""
✅ **Phases 0–7 Complete** — Evaluation, Conversational Memory, and Safety & Governance are all live.
🔄 **Phase 8: User Experience** — Streaming responses, history restore, and safety badges now active.
""")

# Feature overview — 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📄 Documents")
    st.markdown("""
    - ✅ PDF & DOCX support
    - ✅ Intelligent chunking
    - ✅ BAAI/bge-small-en-v1.5 embeddings
    - ✅ FAISS + BM25 hybrid indexing
    - ✅ Filename search filter *(Phase 8)*

    👉 **Documents** page
    """)

with col2:
    st.subheader("💬 Chat")
    st.markdown("""
    - ✅ Hybrid / Semantic / Keyword retrieval
    - ✅ Query Understanding (HyDE, expansion)
    - ✅ Cross-encoder reranking
    - ✅ Conversational memory (Redis)
    - ✅ Streaming responses *(Phase 8)*
    - ✅ Safety guardrails (injection, PII, toxicity)

    👉 **Chat** page
    """)

with col3:
    st.subheader("📊 Evaluate")
    st.markdown("""
    - ✅ RAGAS faithfulness
    - ✅ Answer relevancy
    - ✅ Context precision/recall
    - ✅ Manual + JSON batch input
    - ✅ LLM-as-judge metrics

    👉 **Evaluate** page
    """)

st.divider()

# Quick start guide
st.subheader("🚀 Quick Start")

tab1, tab2, tab3 = st.tabs(["1️⃣ Setup", "2️⃣ Upload", "3️⃣ Chat"])

with tab1:
    st.markdown("""
    ### Prerequisites
    
    1. **Start Ollama** (if not running):
       ```bash
       ollama serve
       ```
    
    2. **Pull a model** (if not already done):
       ```bash
       ollama pull qwen3:4b
       ```
    
    3. **Start the backend** (in project root):
       ```bash
       uvicorn backend.api.main:app --reload
       ```
    
    4. **Start Streamlit** (in another terminal):
       ```bash
       streamlit run frontend/streamlit/app.py
       ```
    """)

with tab2:
    st.markdown("""
    ### Upload Documents
    
    1. Navigate to the **📄 Documents** page (sidebar)
    2. Click on the **Upload** tab
    3. Select PDF or DOCX files (max 10MB each)
    4. Click **🚀 Upload** button
    5. Wait for processing to complete
    6. Check the **Library** tab to see your documents
    """)

with tab3:
    st.markdown("""
    ### Start Chatting
    
    1. Navigate to the **💬 Chat** page (sidebar)
    2. Ensure RAG is enabled (toggle in sidebar)
    3. Adjust settings if needed (temperature, max tokens)
    4. Type your question in the chat input
    5. Press Enter or click Send
    6. View the response with source citations
    
    **Example questions:**
    - "What is the main topic of the documents?"
    - "Summarize the key findings"
    - "What recommendations are mentioned?"
    """)

st.divider()

# System status
st.subheader("📊 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    if check_backend_status():
        st.metric("Backend API", "✅ Online", delta="Ready")
    else:
        st.metric("Backend API", "❌ Offline", delta="Not Ready")

with col2:
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/chat/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("llm_available"):
                st.metric("LLM Service", "✅ Ready", delta="Ollama")
            else:
                st.metric("LLM Service", "⚠️ Unavailable", delta="Check Ollama")
        else:
            st.metric("LLM Service", "❌ Error", delta="Check logs")
    except:
        st.metric("LLM Service", "❌ Offline", delta="Not Ready")

with col3:
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/documents/stats/overview", timeout=5)
        if response.status_code == 200:
            data = response.json()
            total_docs = data.get("documents", {}).get("total", 0)
            st.metric("Documents", total_docs, delta="Indexed")
        else:
            st.metric("Documents", "0", delta="None")
    except:
        st.metric("Documents", "?", delta="Unknown")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("📚 [Documentation](../docs)")
with col2:
    st.caption(f"🔧 API: {API_BASE_URL}/docs")
with col3:
    st.caption("🐛 [GitHub Issues](#)")

# Made with Bob
