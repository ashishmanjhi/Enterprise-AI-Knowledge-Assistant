# Troubleshooting: Documents Not Being Retrieved in Chat

## Issue Description

**Problem:** Document uploaded successfully but when asking questions in chat, the system responds with "No documents were provided in the context."

**Expected Behavior:** The RAG system should retrieve relevant chunks from uploaded documents and use them to answer questions.

## Root Cause Analysis

This issue indicates that the document retrieval chain is broken. Possible causes:

1. **Document not being processed** - Upload succeeded but processing failed
2. **Vector store not being created** - Embeddings not generated or not saved
3. **Retriever not loading vector store** - Chat endpoint can't access indexed documents
4. **RAG chain not using retriever** - Integration issue between components

## Diagnostic Steps

### Step 1: Check Backend Logs

Look for errors during document upload:

```bash
# Check if backend is running
# Look for errors in terminal where backend is running

# Common errors to look for:
# - "Failed to process document"
# - "Error generating embeddings"
# - "Failed to save vector store"
# - "FAISS index error"
```

### Step 2: Verify Document Storage

Check if files are being saved:

```bash
# Windows
dir data\raw
dir data\vectorstore

# Linux/Mac
ls -la data/raw
ls -la data/vectorstore
```

**Expected:**

- `data/raw/` should contain your uploaded PDF/DOCX files
- `data/vectorstore/` should contain:
  - `faiss_index.bin` (FAISS index file)
  - `metadata.json` (document metadata)

### Step 3: Check Vector Store Files

```bash
# Check if FAISS index exists
# Windows
dir data\vectorstore\faiss_index.bin

# Linux/Mac
ls -lh data/vectorstore/faiss_index.bin
```

**If file is missing:** Documents are not being indexed properly.

**If file exists:** Check file size - should be > 0 bytes.

### Step 4: Verify Metadata

```bash
# Check metadata file
# Windows
type data\vectorstore\metadata.json

# Linux/Mac
cat data/vectorstore/metadata.json
```

**Expected:** JSON file with document information including:

- document_id
- filename
- num_chunks
- embeddings info

### Step 5: Test Embedding Service

The embedding service might not be initialized. Check if the model is downloaded:

```python
# Run in Python console
from backend.llm.embeddings import EmbeddingService

service = EmbeddingService()
print(f"Model: {service.model_name}")
print(f"Dimension: {service.dimension}")

# Test embedding
test_embedding = service.embed_query("test")
print(f"Embedding shape: {test_embedding.shape}")
```

**Expected:** Should print model info and generate a 384-dimensional embedding.

## Common Issues & Solutions

### Issue 1: FAISS Index Not Being Created

**Symptoms:**

- Document uploads successfully
- No `faiss_index.bin` file created
- Chat returns "no documents"

**Solution:**

Check if the ingestion pipeline is saving the index. The issue might be in `backend/ingestion/pipeline.py`:

```python
# Verify this line exists in pipeline.py
await self.vector_store.save(settings.faiss_index_path)
```

**Fix:** Ensure the pipeline calls `save()` after adding vectors.

### Issue 2: Retriever Not Loading Index

**Symptoms:**

- FAISS index file exists
- Chat still returns "no documents"
- No errors in logs

**Solution:**

The retriever might not be loading the index on initialization. Check `backend/retrievers/retriever.py`:

```python
# Retriever should load index in __init__
def __init__(self):
    self.vector_store = FAISSVectorStore()
    # Should load existing index
    if Path(settings.faiss_index_path).exists():
        self.vector_store.load(settings.faiss_index_path)
```

**Fix:** Add index loading in retriever initialization.

### Issue 3: RAG Chain Not Using Retriever

**Symptoms:**

- Documents indexed correctly
- Retriever works in isolation
- Chat doesn't use documents

**Solution:**

Check `backend/llm/rag_chain.py` - ensure it's calling the retriever:

```python
async def generate_answer(self, question: str, top_k: int = 3):
    # Should retrieve documents
    retrieved_docs = await self.retriever.search(question, top_k=top_k)

    # Should include docs in context
    context = self._format_context(retrieved_docs)

    # Should pass context to LLM
    prompt = self._create_prompt(question, context)
```

**Fix:** Ensure RAG chain retrieves and uses documents.

### Issue 4: Chat Endpoint Not Using RAG

**Symptoms:**

- RAG chain works in isolation
- Chat endpoint doesn't use it

**Solution:**

Check `backend/api/routes/chat.py`:

```python
@router.post("/")
async def chat(request: ChatRequest):
    # Should use RAG chain when use_rag=True
    if request.use_rag:
        response = await rag_chain.generate_answer(
            question=request.message,
            top_k=request.top_k
        )
```

**Fix:** Ensure chat endpoint calls RAG chain with `use_rag=True`.

### Issue 5: Streamlit Not Enabling RAG

**Symptoms:**

- Backend RAG works
- Streamlit chat doesn't retrieve documents

**Solution:**

Check `frontend/streamlit/pages/2_💬_Chat.py`:

```python
# Ensure RAG is enabled in request
response = requests.post(
    f"{API_URL}/chat/",
    json={
        "message": user_message,
        "use_rag": True,  # Must be True!
        "top_k": top_k
    }
)
```

**Fix:** Ensure Streamlit sends `use_rag: true` in API request.

## Quick Fix Checklist

Try these in order:

1. ☐ **Restart Backend**

   ```bash
   # Stop backend (Ctrl+C)
   # Start again
   cd backend
   uvicorn api.main:app --reload --port 8000
   ```

2. ☐ **Clear and Re-upload**

   ```bash
   # Delete existing data
   rm -rf data/vectorstore/*
   rm -rf data/raw/*

   # Re-upload document through UI
   ```

3. ☐ **Check Ollama**

   ```bash
   # Verify Ollama is running
   curl http://localhost:11434/api/tags

   # If not running, start it
   ollama serve
   ```

4. ☐ **Verify RAG Toggle**
   - In Streamlit chat page
   - Check "Use RAG" checkbox is enabled
   - Try asking question again

5. ☐ **Check Backend Logs**
   - Look for any errors during upload
   - Look for errors during chat request
   - Check if retriever is being called

## Testing the Fix

After applying fixes, test with these steps:

1. **Upload a test document**
   - Use a simple PDF with clear content
   - Example: Create a PDF with "The sky is blue. Grass is green."

2. **Wait for processing**
   - Check backend logs for "Successfully processed document"
   - Verify FAISS index file created

3. **Ask a specific question**
   - Question: "What color is the sky?"
   - Expected: "The sky is blue" (with source citation)

4. **Verify sources**
   - Response should include source document name
   - Should show which chunk was used

## Still Not Working?

If none of the above fixes work, the issue might be in the core implementation. Check:

1. **Document Routes** (`backend/api/routes/documents.py`)
   - Verify upload endpoint calls ingestion pipeline
   - Check if background processing is working

2. **Ingestion Pipeline** (`backend/ingestion/pipeline.py`)
   - Verify all steps execute: load → chunk → embed → index
   - Check if errors are being caught and logged

3. **Vector Store** (`backend/retrievers/vector_store.py`)
   - Verify save/load methods work correctly
   - Check FAISS index operations

4. **RAG Chain** (`backend/llm/rag_chain.py`)
   - Verify retriever integration
   - Check context formatting
   - Verify LLM receives context

## Getting Help

If you're still stuck, provide this information:

1. **Backend logs** (last 50 lines)
2. **File listing:**
   ```bash
   ls -R data/
   ```
3. **FAISS index size:**
   ```bash
   ls -lh data/vectorstore/faiss_index.bin
   ```
4. **Metadata content:**
   ```bash
   cat data/vectorstore/metadata.json
   ```
5. **Test results** from diagnostic steps above

## Prevention

To avoid this issue in future:

1. **Always check logs** after document upload
2. **Verify FAISS index** is created
3. **Test with simple documents** first
4. **Enable debug logging** during development
5. **Monitor file sizes** in data directories

---

**Last Updated:** 2026-06-24  
**Issue:** Documents not being retrieved in RAG chat  
**Status:** Troubleshooting guide
