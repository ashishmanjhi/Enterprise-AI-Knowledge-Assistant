# Phase 1 Testing Guide - Enterprise Agentic RAG Platform

## Overview

This guide provides step-by-step instructions for testing the Phase 1 Basic RAG implementation, including prerequisites, test scenarios, expected results, and troubleshooting.

## Prerequisites Checklist

### 1. System Requirements

- [ ] Python 3.9+ installed
- [ ] Git installed
- [ ] 4GB+ RAM available
- [ ] 2GB+ disk space available

### 2. Ollama Setup (Local LLM)

- [ ] Ollama installed (download from https://ollama.ai)
- [ ] Ollama service running
- [ ] qwen3:4b model pulled

**Verify Ollama:**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull the model if not already available
ollama pull qwen3:4b
```

### 3. Python Environment

- [ ] Virtual environment created and activated
- [ ] All dependencies installed from requirements.txt

**Setup Commands:**

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Project Structure

- [ ] All backend modules present
- [ ] Frontend Streamlit pages created
- [ ] Configuration files in place

**Verify Structure:**

```bash
# Check critical directories
ls backend/ingestion/loaders/
ls backend/api/routes/
ls frontend/streamlit/pages/
```

## Test Scenarios

### Test 1: Backend API Health Check

**Objective:** Verify the FastAPI backend starts correctly and all endpoints are registered.

**Steps:**

1. Start the backend server:

   ```bash
   cd backend
   uvicorn api.main:app --reload --port 8000
   ```

2. Open browser to http://localhost:8000

3. Check the root endpoint response

4. Navigate to http://localhost:8000/docs to view Swagger UI

**Expected Results:**

- ✅ Server starts without errors
- ✅ Root endpoint shows welcome message with Phase 1 status
- ✅ Swagger UI displays 10 endpoints:
  - POST /api/documents/upload
  - GET /api/documents/
  - GET /api/documents/{document_id}
  - DELETE /api/documents/{document_id}
  - POST /api/documents/search
  - GET /api/documents/stats
  - POST /api/chat/
  - POST /api/chat/stream
  - POST /api/chat/direct
  - GET /api/chat/health

**Troubleshooting:**

- If port 8000 is in use: Change port with `--port 8001`
- If import errors: Verify all dependencies installed
- If module not found: Check PYTHONPATH or run from project root

---

### Test 2: Document Upload (PDF)

**Objective:** Test PDF document ingestion pipeline.

**Prerequisites:**

- Backend server running
- Sample PDF file available

**Steps:**

1. Using Swagger UI (http://localhost:8000/docs):
   - Navigate to POST /api/documents/upload
   - Click "Try it out"
   - Upload a PDF file
   - Click "Execute"

2. Check the response

3. Verify document in storage:
   ```bash
   ls data/documents/
   ls data/vector_store/
   ```

**Expected Results:**

- ✅ HTTP 200 response
- ✅ Response includes:
  - document_id (UUID)
  - filename
  - file_type: "pdf"
  - file_size
  - num_chunks
  - status: "completed"
- ✅ PDF file saved in data/documents/
- ✅ FAISS index files created in data/vector_store/
- ✅ Metadata JSON file created

**Troubleshooting:**

- If "Failed to extract text": Check PDF is not image-based or encrypted
- If "Embedding failed": Verify sentence-transformers model downloaded
- If "Vector store error": Check data/vector_store/ directory exists and is writable

---

### Test 3: Document Upload (DOCX)

**Objective:** Test DOCX document ingestion pipeline.

**Prerequisites:**

- Backend server running
- Sample DOCX file available

**Steps:**

1. Using Swagger UI:
   - Navigate to POST /api/documents/upload
   - Upload a DOCX file
   - Click "Execute"

2. Check the response

**Expected Results:**

- ✅ HTTP 200 response
- ✅ file_type: "docx"
- ✅ Text extracted from all paragraphs and tables
- ✅ Document chunked and indexed

**Troubleshooting:**

- If "Unsupported file type": Verify file extension is .docx
- If "Failed to extract text": Check DOCX is not corrupted

---

### Test 4: List Documents

**Objective:** Verify document listing endpoint.

**Prerequisites:**

- At least one document uploaded

**Steps:**

1. Using Swagger UI:
   - Navigate to GET /api/documents/
   - Click "Execute"

2. Review the response

**Expected Results:**

- ✅ HTTP 200 response
- ✅ Array of document objects
- ✅ Each document includes:
  - document_id
  - filename
  - file_type
  - file_size
  - num_chunks
  - upload_date
  - status

---

### Test 5: Document Search

**Objective:** Test semantic search across documents.

**Prerequisites:**

- At least one document uploaded

**Steps:**

1. Using Swagger UI:
   - Navigate to POST /api/documents/search
   - Enter request body:
     ```json
     {
     	"query": "What is the main topic?",
     	"top_k": 3
     }
     ```
   - Click "Execute"

2. Review the results

**Expected Results:**

- ✅ HTTP 200 response
- ✅ Array of search results
- ✅ Each result includes:
  - chunk_text
  - document_id
  - filename
  - page_number (if available)
  - similarity_score
- ✅ Results ordered by relevance (highest score first)

**Troubleshooting:**

- If empty results: Check query matches document content
- If low scores: Try more specific queries
- If error: Verify vector store is properly initialized

---

### Test 6: RAG Chat (Basic)

**Objective:** Test question answering with document context.

**Prerequisites:**

- At least one document uploaded
- Ollama running with qwen3:4b model

**Steps:**

1. Using Swagger UI:
   - Navigate to POST /api/chat/
   - Enter request body:
     ```json
     {
     	"message": "What are the key points in the document?",
     	"top_k": 3,
     	"use_rag": true
     }
     ```
   - Click "Execute"

2. Review the response

**Expected Results:**

- ✅ HTTP 200 response
- ✅ Response includes:
  - answer (generated text)
  - sources (array of retrieved chunks)
  - metadata (retrieval info)
- ✅ Answer is relevant to the question
- ✅ Answer references document content
- ✅ Sources include document names and page numbers

**Troubleshooting:**

- If "LLM service unavailable": Check Ollama is running
- If "Model not found": Run `ollama pull qwen3:4b`
- If generic answer: Verify use_rag=true and documents are indexed
- If timeout: Increase timeout in settings or use smaller model

---

### Test 7: Direct LLM Chat (No RAG)

**Objective:** Test LLM without document retrieval.

**Steps:**

1. Using Swagger UI:
   - Navigate to POST /api/chat/direct
   - Enter request body:
     ```json
     {
     	"message": "What is machine learning?",
     	"model": "qwen3:4b"
     }
     ```
   - Click "Execute"

**Expected Results:**

- ✅ HTTP 200 response
- ✅ Response includes generated answer
- ✅ Answer is general knowledge (not document-specific)

---

### Test 8: Frontend - Document Upload UI

**Objective:** Test Streamlit document management interface.

**Prerequisites:**

- Backend server running on port 8000

**Steps:**

1. Start Streamlit:

   ```bash
   cd frontend/streamlit
   streamlit run app.py
   ```

2. Navigate to "📄 Documents" page

3. Upload a document using the file uploader

4. Wait for processing to complete

5. Check the document library

**Expected Results:**

- ✅ Streamlit app opens in browser
- ✅ File uploader accepts PDF and DOCX
- ✅ Progress bar shows during upload
- ✅ Success message displayed
- ✅ Document appears in library with:
  - Filename
  - Type badge
  - Size
  - Chunks count
  - Upload date
  - Delete button
- ✅ Statistics updated (total documents, total chunks)

**Troubleshooting:**

- If "Connection refused": Verify backend is running on port 8000
- If upload fails: Check backend logs for errors
- If UI doesn't update: Refresh the page

---

### Test 9: Frontend - Chat Interface

**Objective:** Test interactive chat with RAG.

**Prerequisites:**

- Backend running
- Streamlit running
- At least one document uploaded

**Steps:**

1. Navigate to "💬 Chat" page

2. Enter a question in the chat input

3. Submit the question

4. Review the response

5. Test with different parameters:
   - Adjust "Number of sources" slider
   - Toggle "Use RAG" checkbox

**Expected Results:**

- ✅ Chat interface displays
- ✅ Message history shows user and assistant messages
- ✅ Assistant response includes:
  - Generated answer
  - Source citations (when RAG enabled)
  - Document references
- ✅ Adjusting parameters affects results
- ✅ Disabling RAG provides general answers

**Troubleshooting:**

- If no response: Check backend logs
- If no sources: Verify documents are indexed
- If slow response: Normal for local LLM, consider smaller model

---

### Test 10: Document Deletion

**Objective:** Test document removal from system.

**Prerequisites:**

- At least one document uploaded

**Steps:**

1. In Streamlit Documents page:
   - Click "Delete" button on a document
   - Confirm deletion

2. Verify document removed from list

3. Check backend:
   ```bash
   ls data/documents/
   ```

**Expected Results:**

- ✅ Document removed from UI
- ✅ File deleted from data/documents/
- ✅ Metadata removed
- ✅ Statistics updated

---

### Test 11: System Statistics

**Objective:** Verify statistics endpoint accuracy.

**Steps:**

1. Using Swagger UI:
   - Navigate to GET /api/documents/stats
   - Click "Execute"

2. Compare with actual files:
   ```bash
   ls data/documents/ | wc -l
   ```

**Expected Results:**

- ✅ total_documents matches file count
- ✅ total_chunks is sum of all document chunks
- ✅ total_size is sum of all file sizes
- ✅ Statistics update after upload/delete

---

### Test 12: Error Handling

**Objective:** Verify graceful error handling.

**Test Cases:**

1. **Invalid File Type:**
   - Upload .txt or .jpg file
   - Expected: HTTP 400 with error message

2. **Empty File:**
   - Upload 0-byte file
   - Expected: HTTP 400 with error message

3. **Large File:**
   - Upload file > 50MB
   - Expected: HTTP 413 or processing with warning

4. **Invalid Document ID:**
   - GET /api/documents/invalid-uuid
   - Expected: HTTP 404 with error message

5. **Ollama Not Running:**
   - Stop Ollama service
   - Try chat request
   - Expected: HTTP 503 with clear error message

**Expected Results:**

- ✅ All errors return appropriate HTTP status codes
- ✅ Error messages are clear and actionable
- ✅ No server crashes
- ✅ Logs contain detailed error information

---

## Performance Benchmarks

### Document Processing

- **Small PDF (1-5 pages):** < 5 seconds
- **Medium PDF (10-50 pages):** < 30 seconds
- **Large PDF (100+ pages):** < 2 minutes

### Embedding Generation

- **100 chunks:** < 10 seconds
- **1000 chunks:** < 60 seconds

### Search Performance

- **Query time:** < 1 second
- **Top-k retrieval:** < 500ms

### Chat Response

- **With RAG (local LLM):** 5-15 seconds
- **Without RAG:** 3-10 seconds

_Note: Times vary based on hardware and model size_

---

## Integration Testing Checklist

- [ ] Upload multiple documents (PDF and DOCX)
- [ ] Search across all documents
- [ ] Chat with questions spanning multiple documents
- [ ] Delete documents and verify cleanup
- [ ] Restart services and verify persistence
- [ ] Test concurrent uploads
- [ ] Test with various document sizes
- [ ] Test with different content types (technical, general, etc.)

---

## Troubleshooting Guide

### Common Issues

#### 1. "Module not found" errors

**Solution:**

```bash
pip install -r requirements.txt --upgrade
```

#### 2. Ollama connection refused

**Solution:**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (if not running)
ollama serve
```

#### 3. FAISS index errors

**Solution:**

```bash
# Clear vector store and re-index
rm -rf data/vector_store/*
# Re-upload documents
```

#### 4. Slow embedding generation

**Solution:**

- Use GPU if available
- Reduce batch size in settings.py
- Use smaller embedding model

#### 5. Out of memory errors

**Solution:**

- Reduce chunk size in settings.py
- Process fewer documents at once
- Use smaller LLM model

#### 6. Streamlit connection errors

**Solution:**

```bash
# Verify backend URL in Streamlit
# Default: http://localhost:8000
# Update if backend runs on different port
```

---

## Test Data Recommendations

### Sample Documents

1. **Technical Document:** API documentation, user manual
2. **Research Paper:** Academic paper with citations
3. **Business Document:** Report, proposal, presentation
4. **Mixed Content:** Document with text, tables, and lists

### Sample Questions

1. **Factual:** "What is the main purpose of this document?"
2. **Specific:** "What are the system requirements?"
3. **Comparative:** "How does X compare to Y?"
4. **Summarization:** "Summarize the key findings."
5. **Multi-document:** "What common themes appear across documents?"

---

## Success Criteria

Phase 1 is considered successfully tested when:

- ✅ All 12 test scenarios pass
- ✅ No critical errors in logs
- ✅ Performance meets benchmarks
- ✅ Error handling works correctly
- ✅ UI is responsive and intuitive
- ✅ Documents persist across restarts
- ✅ RAG provides accurate, sourced answers
- ✅ System handles edge cases gracefully

---

## Next Steps After Testing

1. Document any issues found
2. Create bug reports for failures
3. Optimize performance bottlenecks
4. Gather user feedback
5. Plan Phase 2 enhancements

---

## Support

For issues or questions:

1. Check logs in `backend/logs/`
2. Review error messages in Swagger UI
3. Verify all prerequisites are met
4. Consult troubleshooting guide above

---

**Last Updated:** 2026-06-23
**Phase:** 1 - Basic RAG
**Status:** Ready for Testing
