# Phase 2: Hybrid Retrieval - Testing Guide

## Quick Start Testing

### Prerequisites

1. Backend running: `uvicorn backend.api.main:app --reload`
2. Frontend running: `streamlit run frontend/streamlit/app.py`
3. At least one document uploaded and indexed

### Manual Testing Steps

#### 1. Test Hybrid Retrieval (Default)

**Goal**: Verify hybrid retrieval combines semantic and keyword search

**Steps**:

1. Open Chat page
2. Ensure "Hybrid" is selected in retrieval method
3. Upload a technical document (e.g., API documentation)
4. Ask: "What is the authentication method?"
5. Check sources:
   - Should show "🔀 Hybrid" badge
   - Should display both Semantic and Keyword scores
   - Should show ranks from both retrievers

**Expected Result**:

```
Source 1: api-docs.pdf (Page 5) [🔀 Hybrid]
Overall Score: 0.85 (Semantic: 0.82 #2, Keyword: 0.88 #1)
> "Authentication uses OAuth 2.0 with JWT tokens..."
```

#### 2. Test Semantic-Only Retrieval

**Goal**: Verify FAISS semantic search works independently

**Steps**:

1. Select "faiss" retrieval method
2. Ask: "How do I authenticate?" (paraphrased query)
3. Check sources:
   - Should show "🧠 Semantic" badge
   - Should only show overall score (no keyword score)

**Expected Result**:

- Finds conceptually similar content even with different wording
- No BM25 scores shown

#### 3. Test Keyword-Only Retrieval

**Goal**: Verify BM25 keyword search works independently

**Steps**:

1. Select "bm25" retrieval method
2. Ask: "OAuth 2.0 JWT authentication" (exact terms)
3. Check sources:
   - Should show "🔤 Keyword" badge
   - Should only show overall score (no semantic score)

**Expected Result**:

- Finds exact term matches
- No FAISS scores shown

#### 4. Compare Retrieval Methods

**Goal**: Understand differences between methods

**Test Query**: "What is the API rate limit?"

| Method   | Expected Behavior                                |
| -------- | ------------------------------------------------ |
| Hybrid   | Finds both "rate limit" and "request throttling" |
| Semantic | Finds "request throttling", "quota management"   |
| Keyword  | Finds exact "rate limit" mentions                |

#### 5. Test Edge Cases

**Empty Index**:

- Delete all documents
- Try to chat
- Should handle gracefully with "No documents found"

**Long Query**:

- Ask a very long question (200+ words)
- Should still retrieve relevant documents

**Technical Terms**:

- Query: "What is the OAuth2 client_credentials flow?"
- Keyword should excel at finding exact technical terms

**Conceptual Query**:

- Query: "How do I secure my API calls?"
- Semantic should excel at understanding intent

### Automated Testing

#### Run Existing Tests

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test file
pytest backend/tests/test_bm25_retriever.py -v
pytest backend/tests/test_fusion.py -v
pytest backend/tests/test_hybrid_retriever.py -v
```

#### Expected Test Coverage

- BM25 Retriever: 10+ tests
- RRF Fusion: 8+ tests
- Hybrid Retriever: 12+ tests
- Integration: 5+ tests

### Performance Testing

#### Retrieval Speed Comparison

```python
import time
import requests

def test_retrieval_speed(method, query):
    start = time.time()
    response = requests.post(
        "http://localhost:8000/api/v1/chat",
        json={
            "message": query,
            "retrieval_method": method,
            "top_k": 5
        }
    )
    elapsed = time.time() - start
    return elapsed, response.json()

# Test each method
for method in ["hybrid", "faiss", "bm25"]:
    elapsed, result = test_retrieval_speed(method, "What is the API rate limit?")
    print(f"{method}: {elapsed:.3f}s")
```

**Expected Times**:

- FAISS: 0.2-0.3s
- BM25: 0.1-0.2s
- Hybrid: 0.3-0.5s (parallel execution)

### Quality Testing

#### Retrieval Quality Metrics

**Test Dataset**: Create 10 question-answer pairs from your documents

**Metrics to Calculate**:

1. **Recall@5**: How many relevant docs in top 5?
2. **Precision@5**: How many top 5 docs are relevant?
3. **MRR**: Mean Reciprocal Rank of first relevant doc

**Example Test**:

```python
test_cases = [
    {
        "query": "What is the authentication method?",
        "relevant_chunks": ["chunk_doc_123_5", "chunk_doc_123_6"],
        "method": "hybrid"
    },
    # ... more test cases
]

def calculate_recall_at_k(retrieved, relevant, k=5):
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)
```

### UI Testing Checklist

- [ ] Retrieval method selector displays correctly
- [ ] Method descriptions show for each option
- [ ] Sources display retrieval method badge
- [ ] Hybrid sources show both scores
- [ ] Single-method sources show only one score
- [ ] Ranks display correctly (#1, #2, etc.)
- [ ] Metadata shows retrieval method used
- [ ] Chat history preserves source metadata
- [ ] Method switching works without errors

### API Testing

#### Test Endpoints

**1. Chat Endpoint with Retrieval Method**

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the API rate limit?",
    "retrieval_method": "hybrid",
    "top_k": 5
  }'
```

**Expected Response**:

```json
{
	"conversation_id": "conv_abc123",
	"message": {
		"role": "assistant",
		"content": "The API rate limit is..."
	},
	"sources": [
		{
			"document_id": "doc_123",
			"filename": "api-docs.pdf",
			"score": 0.85,
			"retrieval_method": "hybrid",
			"faiss_score": 0.82,
			"bm25_score": 0.88,
			"faiss_rank": 2,
			"bm25_rank": 1
		}
	],
	"retrieval_method": "hybrid"
}
```

**2. Test Each Method**

```bash
# Hybrid
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "retrieval_method": "hybrid"}'

# FAISS only
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "retrieval_method": "faiss"}'

# BM25 only
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "retrieval_method": "bm25"}'
```

### Troubleshooting

#### Issue: BM25 index not found

**Solution**:

```bash
# Re-upload documents to trigger BM25 indexing
# Or manually trigger indexing:
python -c "
from backend.ingestion.pipeline import IngestionPipeline
from pathlib import Path
pipeline = IngestionPipeline()
# Upload your documents
"
```

#### Issue: Hybrid retrieval returns only FAISS results

**Symptoms**: All sources show only faiss_score, no bm25_score

**Solution**:

1. Check BM25 index exists: `data/vectorstore/bm25_index.pkl`
2. Check logs for BM25 errors
3. Verify BM25 retriever initialization in pipeline

#### Issue: Scores don't make sense

**Example**: BM25 score higher than FAISS but lower rank

**Explanation**:

- Scores are normalized differently
- RRF uses ranks, not raw scores
- Higher score doesn't always mean better rank

#### Issue: UI not showing retrieval method

**Solution**:

1. Clear browser cache
2. Restart Streamlit
3. Check API response includes `retrieval_method` field

### Benchmarking

#### Create Benchmark Script

```python
# scripts/benchmark_phase2.py
import requests
import time
from typing import List, Dict

def benchmark_retrieval(
    queries: List[str],
    methods: List[str] = ["hybrid", "faiss", "bm25"]
) -> Dict:
    results = {}

    for method in methods:
        times = []
        for query in queries:
            start = time.time()
            response = requests.post(
                "http://localhost:8000/api/v1/chat",
                json={
                    "message": query,
                    "retrieval_method": method,
                    "top_k": 5
                }
            )
            elapsed = time.time() - start
            times.append(elapsed)

        results[method] = {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times)
        }

    return results

# Run benchmark
queries = [
    "What is the authentication method?",
    "How do I handle errors?",
    "What are the rate limits?",
    # ... more queries
]

results = benchmark_retrieval(queries)
print(results)
```

#### Expected Benchmark Results

**Phase 1 (FAISS only)**:

- Avg retrieval time: 0.25s
- Recall@5: 65%
- Precision@5: 70%

**Phase 2 (Hybrid)**:

- Avg retrieval time: 0.40s (+60%)
- Recall@5: 80% (+15%)
- Precision@5: 75% (+5%)

**Trade-off**: Slightly slower but significantly better quality

### Success Criteria

#### Functional

- [x] All three retrieval methods work
- [x] UI displays method selector
- [x] Sources show correct metadata
- [x] API accepts retrieval_method parameter
- [ ] All tests pass

#### Performance

- [ ] Hybrid retrieval < 0.5s
- [ ] Recall@5 ≥ 80%
- [ ] Precision@5 ≥ 75%
- [ ] No errors in logs

#### Quality

- [ ] Hybrid outperforms single methods
- [ ] Semantic finds conceptual matches
- [ ] Keyword finds exact terms
- [ ] RRF properly combines results

### Next Steps After Testing

1. **If tests pass**:
   - Document results
   - Create benchmark report
   - Move to Phase 3 planning

2. **If tests fail**:
   - Identify failure patterns
   - Debug specific components
   - Adjust parameters (weights, k, etc.)
   - Re-test

3. **Performance issues**:
   - Profile slow components
   - Optimize BM25 tokenization
   - Consider caching
   - Parallelize more operations

### Test Data Recommendations

**Good Test Documents**:

1. Technical documentation (APIs, SDKs)
2. Research papers (academic content)
3. Business reports (mixed content)
4. FAQs (question-answer format)

**Good Test Queries**:

1. Exact term queries: "OAuth 2.0 authentication"
2. Conceptual queries: "How do I secure my API?"
3. Multi-term queries: "rate limiting and throttling"
4. Paraphrased queries: "What's the request quota?"

---

**Last Updated**: 2026-06-24  
**Status**: Ready for Testing

# Made with Bob
