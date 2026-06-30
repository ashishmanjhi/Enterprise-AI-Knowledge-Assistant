# Phase 2 Migration Guide

## Issue: BM25 Index Empty

If you see this warning in your logs:

```
WARNING - BM25 index is empty
```

This means your existing documents are only indexed in FAISS, not in BM25. You need to run the migration script.

## Solution: Run Migration Script

### Step 1: Stop the Backend

```bash
# Press Ctrl+C to stop the backend
```

### Step 2: Run Migration Script

```bash
python scripts/migrate_to_phase2.py
```

### Expected Output:

```
============================================================
Phase 2 Migration: Re-indexing documents for BM25
============================================================
INFO - Found 150 vectors in FAISS index
INFO - Found 150 document chunks
INFO - Indexing documents in BM25...
INFO - Saving BM25 index...
============================================================
Migration Complete!
============================================================
INFO - FAISS vectors: 150
INFO - BM25 documents: 150
INFO - BM25 index saved to: data/vectorstore/bm25_index.pkl
============================================================
INFO - You can now use hybrid retrieval!
INFO - Restart your backend to load the new BM25 index.
============================================================
```

### Step 3: Restart Backend

```bash
uvicorn backend.api.main:app --reload
```

### Step 4: Verify

Check logs for:

```
INFO - Loaded existing BM25 index
INFO - BM25 retriever initialized with 150 documents
```

## Alternative: Re-upload Documents

If migration fails, you can simply re-upload your documents:

1. Delete old indices:

```bash
rm data/vectorstore/faiss_index.bin
rm data/vectorstore/metadata.json
rm data/vectorstore/bm25_index.pkl
```

2. Restart backend

3. Re-upload documents via UI
   - Documents will be automatically indexed in both FAISS and BM25

## Verification

### Check BM25 Index

```python
from backend.retrievers.bm25_retriever import BM25Retriever

retriever = BM25Retriever()
retriever.load()
stats = retriever.get_stats()
print(f"BM25 documents: {stats['total_documents']}")
```

### Test Hybrid Retrieval

1. Open Chat page
2. Select "Hybrid" retrieval method
3. Ask a question
4. Check sources - should show both FAISS and BM25 scores

## Troubleshooting

### Error: "No FAISS index found"

**Solution**: Upload documents first, then run migration

### Error: "FAISS index is empty"

**Solution**: Upload documents first, then run migration

### Error: "Permission denied" when saving BM25 index

**Solution**: Check write permissions on `data/vectorstore/` directory

### BM25 still shows 0 results after migration

**Solution**:

1. Check if `data/vectorstore/bm25_index.pkl` exists
2. Restart backend to reload index
3. Check logs for loading errors

## What the Migration Does

1. **Reads** existing FAISS index and metadata
2. **Extracts** all document chunks
3. **Indexes** chunks in BM25 retriever
4. **Saves** BM25 index to disk
5. **Preserves** existing FAISS index (no changes)

## Performance Impact

- **Migration time**: ~1-2 seconds per 100 documents
- **Disk space**: +5-10MB for BM25 index (depends on document count)
- **No downtime**: Can run while backend is stopped

## Future Uploads

After migration, all new document uploads will automatically:

- Index in FAISS (semantic search)
- Index in BM25 (keyword search)
- No manual migration needed

---

**Last Updated**: 2026-06-24  
**Status**: Ready to use

# Made with Bob
