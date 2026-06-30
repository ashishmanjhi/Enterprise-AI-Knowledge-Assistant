# Phase 1 Bug Fixes

## Critical Bugs Found During Manual Testing

### Bug #1: Documents Not Retrieved in RAG Chat (CRITICAL)

**Severity:** Critical  
**Status:** Fix Identified  
**Impact:** RAG system doesn't work - uploaded documents are not used in chat responses

#### Symptoms

- Document uploads successfully
- Document appears in library
- Chat returns: "No documents were provided in the context"
- LLM responds without using document content

#### Root Cause

The `DocumentRetriever` class initializes a new empty `FAISSVectorStore()` but never loads the saved index file from disk. This means:

1. Documents are uploaded and indexed correctly
2. FAISS index is saved to `data/vectorstore/faiss_index.bin`
3. When chat is initiated, retriever creates a NEW empty vector store
4. Retriever searches the empty store and finds nothing
5. RAG chain has no context to provide to LLM

#### Fix

**File:** `backend/retrievers/retriever.py`

**Location:** `__init__` method (lines 72-87)

**Current Code:**

```python
def __init__(
    self,
    embedding_service: Optional[EmbeddingService] = None,
    vector_store: Optional[FAISSVectorStore] = None
):
    self.embedding_service = embedding_service or EmbeddingService()
    self.vector_store = vector_store or FAISSVectorStore()

    logger.info("Initialized DocumentRetriever")
```

**Fixed Code:**

```python
def __init__(
    self,
    embedding_service: Optional[EmbeddingService] = None,
    vector_store: Optional[FAISSVectorStore] = None
):
    from pathlib import Path
    from backend.core.settings import settings

    self.embedding_service = embedding_service or EmbeddingService()
    self.vector_store = vector_store or FAISSVectorStore()

    # Load existing index if it exists
    index_path = Path(settings.faiss_index_path)
    if index_path.exists():
        try:
            self.vector_store.load(str(index_path))
            logger.info(f"Loaded existing FAISS index from {index_path}")
            logger.info(f"Index contains {self.vector_store.index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            logger.warning("Starting with empty vector store")
    else:
        logger.info("No existing FAISS index found - starting with empty store")

    logger.info("Initialized DocumentRetriever")
```

#### Testing the Fix

1. Apply the fix to `backend/retrievers/retriever.py`
2. Restart the backend server
3. Upload a test document
4. Go to chat and ask about the document
5. Verify the response includes document content and sources

---

### Bug #2: File Uploader Not Clearing After Upload (MINOR)

**Severity:** Minor (UX Issue)  
**Status:** Fix Identified  
**Impact:** Confusing UX - users might think upload failed or try to upload again

#### Symptoms

- Document uploads successfully
- Success message appears
- File still shows in the "selected files" area
- User might think they need to upload again

#### Root Cause

Streamlit's `file_uploader` widget maintains its state even after `st.rerun()`. The widget needs a unique key that changes after upload to force it to reset.

#### Fix

**File:** `frontend/streamlit/pages/1_📄_Documents.py`

**Location:** File uploader section (lines 24-41)

**Current Code:**

```python
# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# ... tabs ...

# File uploader
uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    help="Upload PDF or DOCX files (max 10MB each)"
)
```

**Fixed Code:**

```python
# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0

# ... tabs ...

# File uploader with dynamic key
uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    help="Upload PDF or DOCX files (max 10MB each)",
    key=f"file_uploader_{st.session_state.upload_key}"
)
```

**And update the upload success section (after line 89):**

```python
if successful > 0:
    st.balloons()
    # Increment key to reset file uploader
    st.session_state.upload_key += 1
    time.sleep(1)
    st.rerun()
```

#### Testing the Fix

1. Apply the fix to `frontend/streamlit/pages/1_📄_Documents.py`
2. Refresh the Streamlit page
3. Upload a document
4. Verify the file uploader clears after successful upload

---

## How to Apply All Fixes

### Step 1: Apply Backend Fix (Critical)

```bash
# Edit backend/retrievers/retriever.py
# Update the __init__ method as shown in Bug #1

# Restart backend
# Stop with Ctrl+C, then:
cd backend
uvicorn api.main:app --reload --port 8000
```

### Step 2: Apply Frontend Fix (Optional)

```bash
# Edit frontend/streamlit/pages/1_📄_Documents.py
# Update as shown in Bug #2

# Streamlit will auto-reload
# Or manually refresh the browser
```

### Step 3: Verify Fixes

**Test Bug #1 Fix:**

```bash
# 1. Upload a simple PDF with clear content
# 2. Go to chat page
# 3. Ask: "What is the main topic of the document?"
# 4. Verify response includes document content
# 5. Check that sources are shown
```

**Test Bug #2 Fix:**

```bash
# 1. Go to Documents page
# 2. Select a file
# 3. Click Upload
# 4. Verify file disappears from selected files after success
```

---

## Additional Recommendations

### 1. Add Index Reload Method

Add a method to reload the index without restarting:

**File:** `backend/retrievers/retriever.py`

```python
def reload_index(self) -> bool:
    """
    Reload the FAISS index from disk.

    Returns:
        True if reload successful, False otherwise
    """
    from pathlib import Path
    from backend.core.settings import settings

    index_path = Path(settings.faiss_index_path)
    if index_path.exists():
        try:
            self.vector_store.load(str(index_path))
            logger.info(f"Reloaded FAISS index: {self.vector_store.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to reload index: {e}")
            return False
    return False
```

### 2. Add Index Status Endpoint

Add an endpoint to check index status:

**File:** `backend/api/routes/documents.py`

```python
@router.get("/index/status")
async def get_index_status():
    """Get FAISS index status."""
    from pathlib import Path
    from backend.core.settings import settings

    index_path = Path(settings.faiss_index_path)

    if not index_path.exists():
        return {
            "exists": False,
            "message": "No index file found"
        }

    try:
        # Get file size
        size_bytes = index_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        # Try to load and get vector count
        from backend.retrievers.vector_store import FAISSVectorStore
        vs = FAISSVectorStore()
        vs.load(str(index_path))

        return {
            "exists": True,
            "path": str(index_path),
            "size_mb": round(size_mb, 2),
            "vector_count": vs.index.ntotal,
            "dimension": vs.dimension
        }
    except Exception as e:
        return {
            "exists": True,
            "error": str(e)
        }
```

### 3. Add Better Error Messages

Update chat endpoint to provide better error messages:

**File:** `backend/api/routes/chat.py`

```python
# In the chat endpoint, add:
if len(result.get("sources", [])) == 0:
    logger.warning("No documents retrieved for query")
    # Could add a note to the response
    result["response"] += "\n\n*Note: No relevant documents were found. Make sure documents are uploaded and indexed.*"
```

---

## Prevention Checklist

To prevent similar issues in future:

- [ ] Always load persistent state (indices, databases) in `__init__`
- [ ] Add logging for state loading (success/failure)
- [ ] Test with empty state (no documents)
- [ ] Test with populated state (after uploads)
- [ ] Clear Streamlit widgets after operations using dynamic keys
- [ ] Add status endpoints for debugging
- [ ] Include reload methods for runtime updates

---

## Testing Checklist

After applying fixes:

- [ ] Backend starts without errors
- [ ] Upload a document successfully
- [ ] Check backend logs for "Loaded existing FAISS index"
- [ ] Verify vector count in logs
- [ ] Ask question in chat
- [ ] Verify response includes document content
- [ ] Verify sources are shown
- [ ] Verify file uploader clears after upload
- [ ] Test with multiple documents
- [ ] Test with different file types (PDF, DOCX)

---

**Last Updated:** 2026-06-24  
**Phase:** 1 - Basic RAG  
**Status:** Fixes Identified and Documented
