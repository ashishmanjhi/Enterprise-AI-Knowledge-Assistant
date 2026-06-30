"""
RAG Chat Interface
Chat with your documents using RAG-powered responses.
"""

import streamlit as st
import requests
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Chat - RAG Platform",
    page_icon="💬",
    layout="wide"
)

st.title("💬 RAG Chat Assistant")
st.markdown("Ask questions about your uploaded documents")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None

# Sidebar configuration
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
            horizontal=True
        )
        
        # Show method description
        method_descriptions = {
            "hybrid": "🔀 **Hybrid**: Best of both - combines semantic understanding with exact keyword matching",
            "faiss": "🧠 **Semantic**: Understands meaning and context, finds conceptually similar content",
            "bm25": "🔤 **Keyword**: Exact term matching, great for technical terms and specific phrases"
        }
        st.info(method_descriptions[retrieval_method])
        
        top_k = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant document chunks to retrieve"
        )
    else:
        top_k = 5
        retrieval_method = "hybrid"
    
    # Generation settings
    st.subheader("🎛️ Generation")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    max_tokens = st.slider(
        "Max tokens",
        min_value=300,
        max_value=2000,
        value=800,
        step=100,
        help="Maximum length of response (qwen3 needs 500+ for thinking + answer)"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()
    
    st.divider()
    
    # API Status
    st.subheader("🔌 Status")
    try:
        # Check API health
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
        
        # Check chat health
        chat_health = requests.get(f"{API_BASE_URL}/api/v1/chat/health", timeout=5)
        if chat_health.status_code == 200:
            health_data = chat_health.json()
            if health_data.get("llm_available"):
                st.success("✅ LLM Ready")
            else:
                st.warning("⚠️ LLM Not Available")
            
            # Show vector store info
            vector_info = health_data.get("vector_store", {})
            total_vectors = vector_info.get("total_vectors", 0)
            st.info(f"📊 {total_vectors} chunks indexed")
        
    except:
        st.error("❌ Disconnected")
    
    st.divider()
    
    # Info
    st.markdown("""
    ### 💡 Tips
    - Upload documents first in the Documents page
    - **Hybrid** retrieval combines semantic + keyword search
    - **Semantic** finds conceptually similar content
    - **Keyword** matches exact terms and phrases
    - Adjust temperature for creativity
    - Sources show retrieval method and scores
    """)

# Main chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for idx, source in enumerate(sources, 1):
                            # Create columns for source display
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{idx}. {source['filename']}**")
                                if source.get('page_number'):
                                    st.caption(f"📄 Page {source['page_number']}")
                            
                            with col2:
                                # Show retrieval method badge
                                method = source.get('retrieval_method', 'unknown')
                                if method == 'hybrid':
                                    st.markdown("🔀 **Hybrid**")
                                elif method == 'faiss':
                                    st.markdown("🧠 **Semantic**")
                                elif method == 'bm25':
                                    st.markdown("🔤 **Keyword**")
                            
                            # Show scores
                            score_text = f"**Overall Score:** {source['score']:.3f}"
                            
                            # Show individual scores for hybrid results
                            if source.get('faiss_score') is not None or source.get('bm25_score') is not None:
                                score_parts = []
                                if source.get('faiss_score') is not None:
                                    rank_text = f"#{source.get('faiss_rank', '?')}" if source.get('faiss_rank') else ""
                                    score_parts.append(f"Semantic: {source['faiss_score']:.3f} {rank_text}")
                                if source.get('bm25_score') is not None:
                                    rank_text = f"#{source.get('bm25_rank', '?')}" if source.get('bm25_rank') else ""
                                    score_parts.append(f"Keyword: {source['bm25_score']:.3f} {rank_text}")
                                score_text += f" ({', '.join(score_parts)})"
                            
                            st.text(score_text)
                            
                            # Show content
                            st.markdown(f"> {source['content']}")
                            st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        try:
            # Prepare request
            request_data = {
                "message": prompt,
                "conversation_id": st.session_state.conversation_id,
                "use_rag": use_rag,
                "top_k": top_k,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "retrieval_method": retrieval_method if use_rag else None
            }
            
            # Show loading
            with st.spinner("Thinking..."):
                # Call chat API
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/chat",
                    json=request_data,
                    timeout=120
                )
            
            if response.status_code == 200:
                data = response.json()
                
                # Debug: Show raw response structure
                # st.write("DEBUG - Response data:", data)
                
                # Extract response - handle different response structures
                if "message" in data and isinstance(data["message"], dict):
                    assistant_message = data["message"].get("content", "")
                elif "response" in data:
                    assistant_message = data["response"]
                else:
                    assistant_message = str(data.get("message", "No response generated"))
                
                sources = data.get("sources", [])
                conversation_id = data.get("conversation_id")
                
                # Update conversation ID
                if not st.session_state.conversation_id:
                    st.session_state.conversation_id = conversation_id
                
                # Display response
                if assistant_message:
                    message_placeholder.markdown(assistant_message)
                else:
                    message_placeholder.warning("⚠️ No response text generated. Check backend logs.")
                
                # Display sources with hybrid metadata
                if sources:
                    with sources_placeholder.expander(f"📚 Sources ({len(sources)})"):
                        for idx, source in enumerate(sources, 1):
                            # Create columns for source display
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{idx}. {source['filename']}**")
                                if source.get('page_number'):
                                    st.caption(f"📄 Page {source['page_number']}")
                            
                            with col2:
                                # Show retrieval method badge
                                method = source.get('retrieval_method', 'unknown')
                                if method == 'hybrid':
                                    st.markdown("🔀 **Hybrid**")
                                elif method == 'faiss':
                                    st.markdown("🧠 **Semantic**")
                                elif method == 'bm25':
                                    st.markdown("🔤 **Keyword**")
                            
                            # Show scores
                            score_text = f"**Overall Score:** {source['score']:.3f}"
                            
                            # Show individual scores for hybrid results
                            if source.get('faiss_score') is not None or source.get('bm25_score') is not None:
                                score_parts = []
                                if source.get('faiss_score') is not None:
                                    rank_text = f"#{source.get('faiss_rank', '?')}" if source.get('faiss_rank') else ""
                                    score_parts.append(f"Semantic: {source['faiss_score']:.3f} {rank_text}")
                                if source.get('bm25_score') is not None:
                                    rank_text = f"#{source.get('bm25_rank', '?')}" if source.get('bm25_rank') else ""
                                    score_parts.append(f"Keyword: {source['bm25_score']:.3f} {rank_text}")
                                score_text += f" ({', '.join(score_parts)})"
                            
                            st.text(score_text)
                            
                            # Show content
                            st.markdown(f"> {source['content']}")
                            st.divider()
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message,
                    "sources": sources
                })
                
                # Show metadata
                model = data.get("model", "unknown")
                tokens = data.get("tokens_used", 0)
                proc_time = data.get("processing_time", 0)
                ret_method = data.get("retrieval_method", "unknown")
                
                # Format retrieval method display
                method_icons = {
                    "hybrid": "🔀",
                    "faiss": "🧠",
                    "bm25": "🔤"
                }
                method_icon = method_icons.get(ret_method, "🔍")
                
                st.caption(
                    f"⏱️ {proc_time:.2f}s | "
                    f"🤖 {model} | "
                    f"📊 {tokens} tokens | "
                    f"{method_icon} {ret_method}"
                )
                
            else:
                error_msg = f"❌ Error: {response.text}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
        
        except requests.exceptions.ConnectionError:
            error_msg = "❌ Cannot connect to API. Make sure the backend is running."
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
        
        except requests.exceptions.Timeout:
            error_msg = "⏱️ Request timed out. Try reducing max_tokens or using a faster model."
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# Welcome message if no messages
if len(st.session_state.messages) == 0:
    st.info("""
    👋 **Welcome to RAG Chat!**
    
    Ask questions about your uploaded documents and get AI-powered answers with source citations.
    
    **Example questions:**
    - "What is the main topic of the documents?"
    - "Summarize the key points from the report"
    - "What are the recommendations mentioned?"
    
    💡 Make sure you've uploaded documents in the Documents page first!
    """)

# Made with Bob