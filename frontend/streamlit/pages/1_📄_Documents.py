"""
Document Management Page
Upload, view, and manage documents for RAG.
"""

import streamlit as st
import requests
from pathlib import Path
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Documents - RAG Platform",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Management")
st.markdown("Upload and manage documents for the RAG knowledge base")

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0
if "tenant_id" not in st.session_state:
    st.session_state["tenant_id"] = "default"


def _tenant_headers() -> dict:
    """Return X-Tenant-ID header for the active tenant (set on the Settings page)."""
    return {"X-Tenant-ID": st.session_state.get("tenant_id", "default")}

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["📤 Upload", "📚 Library", "📊 Statistics"])

# Tab 1: Upload Documents
with tab1:
    st.header("Upload Documents")
    st.markdown("Upload PDF or DOCX files to add them to the knowledge base")
    
    # File uploader with dynamic key to reset after upload
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload PDF or DOCX files (max 10MB each)",
        key=f"file_uploader_{st.session_state.upload_key}"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        upload_button = st.button("🚀 Upload", type="primary", use_container_width=True)
    
    if upload_button and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(uploaded_files)
        successful = 0
        failed = 0
        
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Uploading {file.name}...")
            
            try:
                # Prepare file for upload
                files = {"file": (file.name, file.getvalue(), file.type)}
                
                # Upload to API
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/documents/upload",
                    files=files,
                    headers=_tenant_headers(),
                    timeout=30
                )
                
                if response.status_code == 200:
                    successful += 1
                    st.success(f"✅ {file.name} uploaded successfully!")
                else:
                    failed += 1
                    st.error(f"❌ Failed to upload {file.name}: {response.text}")
                    
            except Exception as e:
                failed += 1
                st.error(f"❌ Error uploading {file.name}: {str(e)}")
            
            # Update progress
            progress_bar.progress((idx + 1) / total_files)
        
        status_text.text("Upload complete!")
        
        # Summary
        st.info(f"📊 Upload Summary: {successful} successful, {failed} failed out of {total_files} files")
        
        if successful > 0:
            st.balloons()
            # Increment key to reset file uploader
            st.session_state.upload_key += 1
            time.sleep(1)
            st.rerun()
    
    elif upload_button and not uploaded_files:
        st.warning("⚠️ Please select files to upload")
    
    # Upload tips — Phase 13 + 14: describe enhanced extraction
    with st.expander("💡 Upload Tips"):
        st.markdown("""
        - **Supported formats**: PDF, DOCX
        - **Max file size**: 10 MB per file
        - **Processing**: Files are processed in the background
        - **Chunking**: Documents are automatically split into 1 000-character chunks
        - **Embedding**: Chunks are embedded using BAAI/bge-small-en-v1.5
        - **Indexing**: Embeddings are indexed in FAISS + BM25 for hybrid retrieval
        """)
        st.info(
            "🆕 **Phase 13 — Tables**  \n"
            "Tables extracted as Markdown → dedicated **[TABLE]** chunks.  \n\n"
            "🆕 **Phase 14 — Charts & Diagrams**  \n"
            "Charts and images described by **llava:7b** → dedicated **[CHART]** chunks.  \n"
            "Scanned pages → OCR fallback when pytesseract is installed."
        )

# Tab 2: Document Library
with tab2:
    st.header("Document Library")

    # Refresh + search (Phase 8)
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col3:
        doc_search = st.text_input(
            "🔍 Filter by filename",
            placeholder="Type to filter…",
            label_visibility="collapsed",
        )

    try:
        # Fetch documents from API
        response = requests.get(
            f"{API_BASE_URL}/api/v1/documents",
            params={"skip": 0, "limit": 100},
            headers=_tenant_headers(),
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            documents = data.get("documents", [])
            total = data.get("total", 0)

            # Phase 8: client-side filename filter
            if doc_search:
                documents = [d for d in documents if doc_search.lower() in d.get("filename", "").lower()]

            if total == 0:
                st.info("📭 No documents uploaded yet. Upload some documents to get started!")
            elif not documents:
                st.warning(f"No documents match '{doc_search}'.")
            else:
                match_note = f" (showing {len(documents)} matching '{doc_search}')" if doc_search else ""
                st.success(f"📚 {total} document(s) in library{match_note}")

                # Display documents in a table
                for doc in documents:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{doc['filename']}**")
                            st.caption(f"ID: {doc['document_id']}")
                        
                        with col2:
                            file_type = doc['file_type'].upper()
                            st.text(f"📎 {file_type}")
                        
                        with col3:
                            size_mb = doc['file_size'] / (1024 * 1024)
                            st.text(f"{size_mb:.2f} MB")
                        
                        with col4:
                            if st.button("🗑️", key=f"delete_{doc['document_id']}", help="Delete document"):
                                try:
                                    del_response = requests.delete(
                                        f"{API_BASE_URL}/api/v1/documents/{doc['document_id']}",
                                        headers=_tenant_headers(),
                                        timeout=10
                                    )
                                    if del_response.status_code == 200:
                                        st.success(f"Deleted {doc['filename']}")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to delete: {del_response.text}")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        
                        st.divider()
        else:
            st.error(f"❌ Failed to fetch documents: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the backend is running at http://localhost:8000")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Tab 3: Statistics
with tab3:
    st.header("System Statistics")

    # Phase 13: PDF extraction capability card
    try:
        stats_resp = requests.get(
            f"{API_BASE_URL}/api/v1/documents/stats/overview", timeout=5
        )
        if stats_resp.status_code == 200:
            pdf_info = stats_resp.json().get("pdf_extraction", {})
            if pdf_info:
                backend = pdf_info.get("backend", "pypdf2")
                table_support = pdf_info.get("table_support", False)
                ocr_fallback = pdf_info.get("ocr_fallback", False)

                backend_label = "pdfplumber ✅" if backend == "pdfplumber" else "PyPDF2 (basic)"
                table_label = "✅ Enabled" if table_support else "❌ Not available"
                ocr_label = "✅ Enabled" if ocr_fallback else "⚠️ pytesseract not installed"

                chart_support = pdf_info.get("chart_support", False)
                chart_model   = pdf_info.get("chart_model", "")
                chart_label   = f"✅ {chart_model}" if chart_support else "❌ Disabled"

                st.subheader("📑 PDF Extraction Engine (Phase 13 + 14)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Backend", backend_label)
                c2.metric("Table extraction", table_label)
                c3.metric("Chart / image AI", chart_label)
                c4.metric("OCR fallback", ocr_label)
                st.divider()
    except Exception:
        pass
    
    try:
        # Fetch statistics
        response = requests.get(
            f"{API_BASE_URL}/api/v1/documents/stats/overview",
            timeout=10
        )
        
        if response.status_code == 200:
            stats = response.json()
            
            # Document stats
            st.subheader("📄 Documents")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", stats['documents']['total'])
            with col2:
                st.metric("PDF Files", stats['documents']['pdf'])
            with col3:
                st.metric("DOCX Files", stats['documents']['docx'])
            
            st.divider()
            
            # Vector store stats
            st.subheader("🔍 Vector Store")
            col1, col2 = st.columns(2)
            
            vector_stats = stats.get('vector_store', {})
            with col1:
                st.metric("Total Vectors", vector_stats.get('total_vectors', 0))
            with col2:
                st.metric("Dimension", vector_stats.get('dimension', 0))
            
            st.divider()
            
            # Embedding model info
            st.subheader("🤖 Embedding Model")
            embedding_info = stats.get('embedding_model', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"Model: {embedding_info.get('model_name', 'N/A')}")
            with col2:
                st.text(f"Dimension: {embedding_info.get('dimension', 'N/A')}")
            
            st.divider()
            
            # Admin actions
            st.subheader("⚙️ Admin Actions")
            st.warning("⚠️ **Danger Zone**: These actions cannot be undone!")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
                    st.session_state.confirm_clear = True
            
            # Confirmation dialog
            if st.session_state.get('confirm_clear', False):
                st.error("⚠️ **Are you absolutely sure?**")
                st.markdown("""
                This will permanently delete:
                - All FAISS vectors
                - All BM25 indices
                - All document metadata
                
                You will need to re-upload all documents!
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ Yes, Clear Everything", type="primary"):
                        try:
                            with st.spinner("Clearing vector stores..."):
                                clear_response = requests.post(
                                    f"{API_BASE_URL}/api/v1/admin/clear-vector-stores",
                                    timeout=10
                                )
                            
                            if clear_response.status_code == 200:
                                result = clear_response.json()
                                st.success("✅ Vector stores cleared successfully!")
                                st.json(result)
                                st.info("🔄 Please restart the backend to reload empty indices")
                                st.session_state.confirm_clear = False
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(f"❌ Failed: {clear_response.text}")
                                st.session_state.confirm_clear = False
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.session_state.confirm_clear = False
                
                with col2:
                    if st.button("❌ No, Cancel"):
                        st.session_state.confirm_clear = False
                        st.rerun()
            
        else:
            st.error(f"❌ Failed to fetch statistics: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the backend is running.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Sidebar
with st.sidebar:
    # Tenant indicator
    _tid = st.session_state.get("tenant_id", "default")
    if _tid != "default":
        st.info(f"🏢 Tenant: **{_tid}**")
    else:
        st.caption("🏢 Tenant: default — [change in Settings](0_⚙️_Settings)")

    st.divider()
    st.header("ℹ️ Information")

    st.markdown("""
    ### Document Processing

    When you upload a document:
    1. **Upload**: File is saved to storage
    2. **Processing**: Document is processed in background
    3. **Chunking**: Text is split into 1 000-char chunks
    4. **Embedding**: Chunks are embedded (BGE-small)
    5. **Indexing**: Embeddings → FAISS + BM25

    ### Phase 13 + 14 Enhancements
    - **Tables** → Markdown `[TABLE]` chunks
    - **Charts / Diagrams** → llava:7b description → `[CHART]` chunks
    - **Scanned pages** → OCR fallback (pytesseract)
    - Extraction engine details in Statistics tab

    ### Supported Formats
    - PDF (.pdf) — text + tables + OCR
    - Microsoft Word (.docx)

    ### Tips
    - Upload multiple files at once
    - Processing happens in background
    - Check Statistics tab for extraction engine details
    """)
    
    st.divider()
    
    # API Status
    st.subheader("🔌 API Status")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ Disconnected")

# Made with Bob