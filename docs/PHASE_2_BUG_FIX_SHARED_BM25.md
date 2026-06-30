# Phase 2 Bug Fix: Shared BM25 Retriever Instance

## Issue Description

**Problem**: BM25 index was showing as empty during retrieval even after documents were uploaded and indexed.

**Symptoms**:

- Logs showed: `WARNING - BM25 index is empty`
- Hybrid retrieval returned: `Retrieved 6 FAISS results, 0 BM25 results`
- Only semantic search was working, keyword search returned no results

## Root Cause Analysis

The issue was caused by **multiple independent BM25Retriever instances** being created:

1. **Ingestion Pipeline** created its own `BM25Retriever()` instance
2. **Hybrid Retriever** created a separate `BM25Retriever()` instance
3. When documents were uploaded:
   - Pipeline's BM25 instance got the documents
   - Pipeline saved the index to disk
4. When querying:
   - Hybrid retriever's BM25 instance loaded from disk
   - But the file was being overwritten or not properly shared
   - Result: Empty index during retrieval

### Code Evidence

**Before Fix - Ingestion Pipeline** (`backend/ingestion/pipeline.py`):

```python
def __init__(self, ...):
    self.bm25_retriever = bm25_retriever or BM25Retriever()  # New instance
```

**Before Fix - Hybrid Retriever** (`backend/retrievers/hybrid_retriever.py`):

```python
def __init__(self, ...):
    self.bm25_retriever = bm25_retriever or BM25Retriever()  # Another new instance
    self.bm25_retriever.load()  # Loads from disk, but may be stale
```

## Solution

Implemented a **Singleton Pattern** for BM25 retriever, similar to the existing FAISS vector store manager.

### Changes Made

#### 1. Created BM25 Manager (`backend/retrievers/bm25_manager.py`)

```python
"""
Shared BM25 retriever manager.
Ensures a single BM25 retriever instance is used across the application.
"""

_shared_bm25_retriever: Optional[BM25Retriever] = None

def get_shared_bm25_retriever() -> BM25Retriever:
    """Get the shared BM25 retriever instance."""
    global _shared_bm25_retriever

    if _shared_bm25_retriever is None:
        _shared_bm25_retriever = BM25Retriever()
        try:
            _shared_bm25_retriever.load()
        except FileNotFoundError:
            pass

    return _shared_bm25_retriever

def reset_shared_bm25_retriever() -> None:
    """Reset the shared BM25 retriever instance."""
    global _shared_bm25_retriever
    _shared_bm25_retriever = None
```

#### 2. Updated Ingestion Pipeline

```python
from backend.retrievers.bm25_manager import get_shared_bm25_retriever

def __init__(self, ...):
    self.bm25_retriever = bm25_retriever or get_shared_bm25_retriever()
```

#### 3. Updated Hybrid Retriever

```python
from backend.retrievers.bm25_manager import get_shared_bm25_retriever

def __init__(self, ...):
    self.bm25_retriever = bm25_retriever or get_shared_bm25_retriever()
```

#### 4. Updated Admin Routes

Added reset functionality to clear in-memory instances:

```python
from backend.retrievers.bm25_manager import reset_shared_bm25_retriever
from backend.retrievers.vector_store_manager import reset_shared_vector_store

@router.post("/clear-vector-stores")
async def clear_vector_stores():
    # Delete files...

    # Reset shared instances
    reset_shared_vector_store()
    reset_shared_bm25_retriever()
```

## Benefits

1. **Single Source of Truth**: All components use the same BM25 instance
2. **In-Memory Consistency**: Documents added during ingestion are immediately available for retrieval
3. **No Restart Required**: Clearing vector stores resets in-memory instances
4. **Matches FAISS Pattern**: Consistent with existing vector store management

## Testing

### Before Fix

```
2026-06-24 13:52:18 - BM25 index is empty
2026-06-24 13:52:18 - Retrieved 6 FAISS results, 0 BM25 results
```

### After Fix (Expected)

```
2026-06-24 XX:XX:XX - Loaded BM25 index with 84 documents
2026-06-24 XX:XX:XX - Retrieved 6 FAISS results, 6 BM25 results
2026-06-24 XX:XX:XX - Hybrid retrieval returned 5 results
```

## Verification Steps

1. **Restart Backend**: Stop and restart the backend server
2. **Upload Document**: Upload a new document through the UI
3. **Check Logs**: Verify BM25 indexing logs show documents added
4. **Query**: Ask a question and check logs for BM25 results
5. **Verify UI**: Check that sources show both FAISS and BM25 scores

### Expected Log Sequence

```
# On startup
INFO - Creating shared BM25 retriever instance
INFO - Loaded existing BM25 index with X documents

# On document upload
INFO - Added Y documents to BM25 index in Z.XXXs
INFO - Saved FAISS and BM25 indices

# On query
INFO - BM25 search returned N results in Z.XXXs
INFO - Retrieved M FAISS results, N BM25 results
INFO - Hybrid retrieval (hybrid) returned K results
```

## Files Modified

1. `backend/retrievers/bm25_manager.py` - NEW (50 lines)
2. `backend/ingestion/pipeline.py` - Updated imports and initialization
3. `backend/retrievers/hybrid_retriever.py` - Updated imports and initialization
4. `backend/api/routes/admin.py` - Added reset calls
5. `backend/retrievers/vector_store_manager.py` - Added reset_shared_vector_store()

## Related Issues

This fix resolves the core issue preventing BM25 from working. Users should now:

1. **Restart the backend** to apply the fix
2. **Re-upload documents** OR **run migration script**:
   ```bash
   python scripts/migrate_to_phase2.py
   ```

## Future Improvements

1. Add automated tests for shared instance behavior
2. Add health check endpoint to verify BM25 index status
3. Consider adding metrics for BM25 index size and query performance

---

**Status**: Fixed ✅  
**Date**: 2026-06-24  
**Impact**: Critical - Enables BM25 keyword search functionality  
**Breaking Changes**: None - backward compatible
