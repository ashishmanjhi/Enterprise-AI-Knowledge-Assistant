# Phase 2: Hybrid Retrieval - Implementation Plan

## Overview

Phase 2 enhances the RAG system with hybrid retrieval, combining semantic search (FAISS) with keyword-based search (BM25) using Reciprocal Rank Fusion (RRF).

## Goals

- **Improve Recall**: Catch documents missed by semantic search alone
- **Better Keyword Matching**: Handle exact terms, names, codes, IDs
- **Increased Accuracy**: Combine strengths of both retrieval methods
- **Expected Improvement**: 15-30% better retrieval accuracy

## Architecture

```
Query
  │
  ├─→ FAISS Retriever (Semantic)
  │     └─→ Top K results with scores
  │
  ├─→ BM25 Retriever (Keyword)
  │     └─→ Top K results with scores
  │
  └─→ Reciprocal Rank Fusion (RRF)
        └─→ Merged & Re-ranked results
              └─→ Final Top K to LLM
```

## Components to Build

### 1. BM25 Retriever (`backend/retrievers/bm25_retriever.py`)

**Purpose**: Keyword-based document retrieval using BM25 algorithm

**Key Features**:

- Tokenization and preprocessing
- BM25 index creation and persistence
- Fast keyword search
- Score normalization

**Interface**:

```python
class BM25Retriever:
    def __init__(self, index_path: str)
    async def index_documents(self, documents: List[Document])
    async def search(self, query: str, top_k: int) -> List[RetrievalResult]
    def save(self, path: str)
    def load(self, path: str)
```

### 2. Reciprocal Rank Fusion (`backend/retrievers/fusion.py`)

**Purpose**: Merge and re-rank results from multiple retrievers

**Algorithm**:

```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
where:
  - k = 60 (constant)
  - rank_i(doc) = rank of doc in retriever i
  - Σ = sum across all retrievers
```

**Interface**:

```python
class ReciprocalRankFusion:
    def __init__(self, k: int = 60)
    def fuse(
        self,
        results_list: List[List[RetrievalResult]],
        top_k: int
    ) -> List[RetrievalResult]
```

### 3. Hybrid Retriever (`backend/retrievers/hybrid_retriever.py`)

**Purpose**: Orchestrate FAISS + BM25 + RRF

**Key Features**:

- Parallel retrieval from both sources
- Configurable retrieval weights
- Fallback strategies
- Performance monitoring

**Interface**:

```python
class HybridRetriever:
    def __init__(
        self,
        faiss_retriever: DocumentRetriever,
        bm25_retriever: BM25Retriever,
        fusion: ReciprocalRankFusion,
        faiss_weight: float = 0.5,
        bm25_weight: float = 0.5
    )
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid"  # "hybrid", "faiss", "bm25"
    ) -> List[RetrievalResult]
```

## Implementation Steps

### Step 1: BM25 Retriever (Day 1-2)

**Files to Create**:

- `backend/retrievers/bm25_retriever.py`

**Tasks**:

1. Implement BM25 indexing
2. Add tokenization (simple whitespace + lowercase)
3. Implement search functionality
4. Add persistence (pickle/json)
5. Add logging and error handling

**Testing**:

- Unit tests for indexing
- Unit tests for search
- Test with sample documents

### Step 2: Reciprocal Rank Fusion (Day 2-3)

**Files to Create**:

- `backend/retrievers/fusion.py`

**Tasks**:

1. Implement RRF algorithm
2. Handle score normalization
3. Handle duplicate documents
4. Add configurable k parameter

**Testing**:

- Unit tests with mock results
- Test with overlapping results
- Test with non-overlapping results

### Step 3: Hybrid Retriever (Day 3-4)

**Files to Create**:

- `backend/retrievers/hybrid_retriever.py`

**Tasks**:

1. Implement parallel retrieval
2. Add retrieval method selection
3. Add weight configuration
4. Integrate with existing retriever

**Testing**:

- Integration tests
- Test all retrieval modes
- Performance benchmarks

### Step 4: Update Ingestion Pipeline (Day 4-5)

**Files to Modify**:

- `backend/ingestion/pipeline.py`

**Tasks**:

1. Add BM25 indexing to pipeline
2. Save BM25 index alongside FAISS
3. Update document processing flow
4. Add BM25 index to shared storage

**Testing**:

- Test document upload
- Verify both indices created
- Test index persistence

### Step 5: Update RAG Chain (Day 5-6)

**Files to Modify**:

- `backend/llm/rag_chain.py`

**Tasks**:

1. Replace DocumentRetriever with HybridRetriever
2. Add retrieval method configuration
3. Update response metadata
4. Add retrieval method to sources

**Testing**:

- End-to-end RAG tests
- Compare with Phase 1 results
- Measure improvement

### Step 6: Update Settings (Day 6)

**Files to Modify**:

- `backend/core/settings.py`

**Tasks**:

1. Add BM25 configuration
2. Add hybrid retrieval settings
3. Add RRF k parameter
4. Add retrieval weights

**Configuration**:

```python
# BM25 Settings
bm25_index_path: str = "data/vectorstore/bm25_index.pkl"
bm25_k1: float = 1.5
bm25_b: float = 0.75

# Hybrid Retrieval Settings
default_retrieval_method: str = "hybrid"
faiss_weight: float = 0.5
bm25_weight: float = 0.5
rrf_k: int = 60
```

### Step 7: Update UI (Day 7-8)

**Files to Modify**:

- `frontend/streamlit/pages/2_💬_Chat.py`

**Tasks**:

1. Add retrieval method selector
2. Show retrieval method in sources
3. Display BM25 vs FAISS scores
4. Add retrieval statistics

**UI Enhancements**:

```python
# Sidebar
retrieval_method = st.selectbox(
    "Retrieval Method",
    ["Hybrid (FAISS + BM25)", "Semantic Only (FAISS)", "Keyword Only (BM25)"]
)

# Sources display
st.caption(f"Retrieved via: {source['retrieval_method']}")
st.caption(f"FAISS Score: {source.get('faiss_score', 'N/A')}")
st.caption(f"BM25 Score: {source.get('bm25_score', 'N/A')}")
st.caption(f"RRF Score: {source['score']}")
```

### Step 8: Testing & Benchmarking (Day 8-10)

**Files to Create**:

- `backend/tests/test_bm25_retriever.py`
- `backend/tests/test_fusion.py`
- `backend/tests/test_hybrid_retriever.py`
- `scripts/benchmark_phase2.py`

**Test Coverage**:

- Unit tests for each component
- Integration tests for hybrid retrieval
- Performance benchmarks
- Accuracy comparison with Phase 1

**Benchmark Metrics**:

- Retrieval time (FAISS vs BM25 vs Hybrid)
- Recall@K (K=1,3,5,10)
- Precision@K
- MRR (Mean Reciprocal Rank)
- Answer quality (manual evaluation)

### Step 9: Documentation (Day 10)

**Files to Create/Update**:

- `docs/PHASE_2_HYBRID_RETRIEVAL.md`
- `docs/PHASE_2_BENCHMARKS.md`
- `README.md`

**Documentation Topics**:

- Hybrid retrieval overview
- BM25 algorithm explanation
- RRF algorithm explanation
- Configuration guide
- Performance comparison
- Usage examples

## File Structure

```
backend/
├── retrievers/
│   ├── __init__.py
│   ├── retriever.py              # Existing FAISS retriever
│   ├── vector_store.py           # Existing
│   ├── vector_store_manager.py   # Existing
│   ├── bm25_retriever.py         # NEW - BM25 implementation
│   ├── fusion.py                 # NEW - RRF algorithm
│   └── hybrid_retriever.py       # NEW - Hybrid orchestration
│
├── ingestion/
│   └── pipeline.py               # MODIFIED - Add BM25 indexing
│
├── llm/
│   └── rag_chain.py              # MODIFIED - Use hybrid retriever
│
├── core/
│   └── settings.py               # MODIFIED - Add hybrid settings
│
└── tests/
    ├── test_bm25_retriever.py    # NEW
    ├── test_fusion.py            # NEW
    └── test_hybrid_retriever.py  # NEW

frontend/streamlit/pages/
└── 2_💬_Chat.py                  # MODIFIED - Add retrieval UI

data/vectorstore/
├── faiss_index.bin               # Existing
├── faiss_metadata.json           # Existing
└── bm25_index.pkl                # NEW - BM25 index

docs/
├── PHASE_2_IMPLEMENTATION_PLAN.md  # This file
├── PHASE_2_HYBRID_RETRIEVAL.md     # NEW - Technical docs
└── PHASE_2_BENCHMARKS.md           # NEW - Performance results

scripts/
└── benchmark_phase2.py             # NEW - Benchmarking script
```

## Configuration Examples

### Environment Variables

```bash
# Hybrid Retrieval Settings
RETRIEVAL_METHOD=hybrid          # hybrid, faiss, bm25
FAISS_WEIGHT=0.5
BM25_WEIGHT=0.5
RRF_K=60

# BM25 Settings
BM25_K1=1.5                      # Term frequency saturation
BM25_B=0.75                      # Length normalization
```

### Code Usage

```python
# Initialize hybrid retriever
from backend.retrievers.hybrid_retriever import HybridRetriever

hybrid_retriever = HybridRetriever(
    faiss_weight=0.5,
    bm25_weight=0.5,
    rrf_k=60
)

# Retrieve with different methods
results_hybrid = await hybrid_retriever.retrieve(
    query="What is the project timeline?",
    top_k=5,
    method="hybrid"
)

results_faiss = await hybrid_retriever.retrieve(
    query="What is the project timeline?",
    top_k=5,
    method="faiss"
)

results_bm25 = await hybrid_retriever.retrieve(
    query="What is the project timeline?",
    top_k=5,
    method="bm25"
)
```

## Expected Improvements

### Retrieval Quality

| Metric      | Phase 1 (FAISS Only) | Phase 2 (Hybrid) | Improvement |
| ----------- | -------------------- | ---------------- | ----------- |
| Recall@5    | 65%                  | 80-85%           | +15-20%     |
| Precision@5 | 70%                  | 75-80%           | +5-10%      |
| MRR         | 0.72                 | 0.80-0.85        | +8-13%      |

### Use Cases Where Hybrid Excels

1. **Exact Term Matching**
   - Query: "What is the project ID ABC-123?"
   - BM25 catches exact ID match
   - FAISS might miss due to semantic distance

2. **Named Entity Queries**
   - Query: "What did John Smith say about the budget?"
   - BM25 finds exact name matches
   - FAISS provides semantic context

3. **Technical Terms**
   - Query: "Explain the OAuth2 implementation"
   - BM25 matches technical term exactly
   - FAISS finds related concepts

4. **Mixed Queries**
   - Query: "How does the authentication system work in module XYZ?"
   - BM25 finds "module XYZ" exactly
   - FAISS finds "authentication system" semantically

## Success Criteria

- [ ] BM25 retriever implemented and tested
- [ ] RRF fusion working correctly
- [ ] Hybrid retriever orchestrating both methods
- [ ] Ingestion pipeline updated
- [ ] RAG chain using hybrid retriever
- [ ] UI showing retrieval methods
- [ ] All tests passing (unit + integration)
- [ ] Benchmarks showing improvement
- [ ] Documentation complete
- [ ] Phase 2 ready for production

## Timeline

**Total Duration**: 10 days

- **Days 1-2**: BM25 Retriever
- **Days 2-3**: RRF Fusion
- **Days 3-4**: Hybrid Retriever
- **Days 4-5**: Update Ingestion
- **Days 5-6**: Update RAG Chain
- **Days 6-7**: Update Settings & Config
- **Days 7-8**: Update UI
- **Days 8-10**: Testing & Benchmarking
- **Day 10**: Documentation

## Next Steps

1. ✅ Install rank-bm25 library
2. ⏭️ Implement BM25 retriever
3. ⏭️ Implement RRF fusion
4. ⏭️ Create hybrid retriever
5. ⏭️ Update ingestion pipeline
6. ⏭️ Update RAG chain
7. ⏭️ Update UI
8. ⏭️ Create tests
9. ⏭️ Run benchmarks
10. ⏭️ Write documentation

---

**Status**: Planning Complete - Ready to Start Implementation
**Date**: 2026-06-24
**Phase**: 2 - Hybrid Retrieval
