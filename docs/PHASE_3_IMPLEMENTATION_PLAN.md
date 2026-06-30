# Phase 3: Query Understanding - Implementation Plan

## Overview

Phase 3 enhances retrieval quality by improving user queries before they reach the retrieval system. This addresses the semantic gap between how users ask questions and how information is stored in documents.

## Objectives

1. **Increase Recall**: Find more relevant documents through query expansion
2. **Improve Precision**: Better match user intent through reformulation
3. **Bridge Semantic Gap**: Use HyDE to match document language patterns
4. **Maintain Performance**: Keep query processing under 1 second

## Three Core Techniques

### 1. Query Expansion

**Purpose**: Generate multiple query variations to increase recall

**How It Works**:

```python
Original Query: "What is EVSE uptime?"

LLM Prompt:
"Generate 3 alternative ways to ask this question, focusing on different aspects:
- Synonyms and related terms
- Different perspectives
- More specific or general versions

Original: What is EVSE uptime?

Alternatives:"

Generated Queries:
1. "How reliable are electric vehicle charging stations?"
2. "What percentage of time are EV chargers operational?"
3. "EVSE connector availability statistics"
```

**Benefits**:

- Captures different terminology (EVSE vs EV charger vs charging station)
- Addresses different aspects (uptime vs reliability vs availability)
- Increases recall by 15-20%

**Implementation**:

- Use LLM to generate 2-4 alternative queries
- Retrieve documents for each query
- Merge results using RRF (already implemented in Phase 2)

### 2. Query Reformulation

**Purpose**: Clarify ambiguous or poorly-formed queries

**How It Works**:

```python
Original Query: "How does it work?"

LLM Prompt:
"Rewrite this vague query to be more specific and clear.
Consider:
- What 'it' refers to (use context if available)
- What aspect of 'work' is being asked about
- Add necessary context

Original: How does it work?
Context: Previous question was about EVSE monitoring

Reformulated:"

Generated Query:
"How does the EVSE connector uptime monitoring system work?"
```

**Benefits**:

- Resolves pronouns and vague references
- Adds missing context from conversation history
- Improves precision by 10-15%

**Implementation**:

- Detect vague queries (pronouns, short length, ambiguous terms)
- Use LLM with conversation context to reformulate
- Replace original query with reformulated version

### 3. HyDE (Hypothetical Document Embeddings)

**Purpose**: Bridge semantic gap between questions and answers

**How It Works**:

```python
Query: "What causes EVSE downtime?"

LLM Prompt:
"Write a detailed, factual answer to this question as if you were
an expert. Include specific details, statistics, and technical terms
that would appear in a real document.

Question: What causes EVSE downtime?

Answer:"

Generated Hypothetical Answer:
"EVSE connectors experience downtime due to several factors:
1. Network connectivity issues (40% of incidents)
2. Hardware failures including cable damage and connector wear (25%)
3. Power outages and electrical grid issues (20%)
4. Scheduled maintenance windows (10%)
5. Software bugs and firmware updates (5%)

The average downtime per incident is 2.3 hours, with network issues
typically resolved within 1 hour while hardware failures may require
4-6 hours for replacement..."

→ Embed this hypothetical answer instead of the query
→ Retrieve documents similar to this answer
```

**Benefits**:

- Answers use similar language to documents (technical terms, statistics)
- Questions often use different vocabulary than answers
- Improves retrieval quality by 20-25%

**Implementation**:

- Generate hypothetical answer using LLM
- Embed the answer instead of the query
- Use for semantic (FAISS) retrieval only
- Original query still used for BM25 (keyword matching)

## Architecture

### Module Structure

```
backend/query_understanding/
├── __init__.py
├── base.py                 # Base classes and interfaces
├── query_expander.py       # Query expansion (200 lines)
├── query_reformulator.py   # Query reformulation (180 lines)
├── hyde_generator.py       # HyDE implementation (220 lines)
└── query_processor.py      # Orchestrator (250 lines)
```

### Data Flow

```
User Query
    ↓
QueryProcessor.process()
    ↓
    ├─→ [Optional] Reformulation
    │   └─→ Clarified Query
    ↓
    ├─→ [Optional] Expansion
    │   └─→ Multiple Query Variants
    ↓
    └─→ [Optional] HyDE
        └─→ Hypothetical Answer
    ↓
Enhanced Queries
    ↓
Hybrid Retrieval
    ├─→ FAISS (uses HyDE answer if enabled)
    └─→ BM25 (uses original/reformulated query)
    ↓
RRF Fusion
    ↓
Final Results
```

### Integration Points

1. **RAG Chain** (`backend/llm/rag_chain.py`)
   - Add query processing before retrieval
   - Pass processed queries to hybrid retriever

2. **Settings** (`backend/core/settings.py`)
   - Add configuration for each technique
   - Enable/disable flags
   - LLM parameters (temperature, max_tokens)

3. **API Models** (`backend/api/models/chat.py`)
   - Add query understanding options to ChatRequest
   - Add metadata to ChatResponse

4. **API Routes** (`backend/api/routes/chat.py`)
   - Pass query understanding options to RAG chain

5. **UI** (`frontend/streamlit/pages/2_💬_Chat.py`)
   - Add toggles for each technique
   - Show which techniques were used
   - Display reformulated/expanded queries

## Implementation Details

### 1. Query Expander

**File**: `backend/query_understanding/query_expander.py`

**Key Features**:

- Generate 2-4 alternative queries
- Use LLM with specific prompt template
- Cache results to avoid redundant calls
- Configurable number of expansions

**Prompt Template**:

```python
EXPANSION_PROMPT = """Generate {num_queries} alternative ways to ask this question.
Focus on:
1. Using different terminology and synonyms
2. Asking from different perspectives
3. Being more specific or more general

Original question: {query}

Generate {num_queries} alternative questions (one per line):"""
```

**API**:

```python
class QueryExpander:
    async def expand(
        self,
        query: str,
        num_expansions: int = 3
    ) -> List[str]:
        """
        Expand query into multiple variations.

        Returns:
            List of expanded queries (including original)
        """
```

### 2. Query Reformulator

**File**: `backend/query_understanding/query_reformulator.py`

**Key Features**:

- Detect vague queries (pronouns, short length)
- Use conversation context if available
- Clarify ambiguous terms
- Preserve user intent

**Prompt Template**:

```python
REFORMULATION_PROMPT = """Rewrite this query to be more specific and clear.

Original query: {query}

{context_section}

Provide a single, clear, specific question that captures the user's intent:"""
```

**API**:

```python
class QueryReformulator:
    async def reformulate(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Reformulate vague query into clear question.

        Returns:
            Reformulated query (or original if already clear)
        """
```

### 3. HyDE Generator

**File**: `backend/query_understanding/hyde_generator.py`

**Key Features**:

- Generate hypothetical answer
- Use document-like language
- Include technical details
- Configurable answer length

**Prompt Template**:

```python
HYDE_PROMPT = """Write a detailed, factual answer to this question as if you were
an expert writing documentation. Include:
- Specific technical terms
- Relevant statistics or numbers
- Clear explanations
- Professional tone

Question: {query}

Detailed answer:"""
```

**API**:

```python
class HyDEGenerator:
    async def generate(
        self,
        query: str,
        max_length: int = 500
    ) -> str:
        """
        Generate hypothetical document for query.

        Returns:
            Hypothetical answer text
        """
```

### 4. Query Processor (Orchestrator)

**File**: `backend/query_understanding/query_processor.py`

**Key Features**:

- Coordinate all query understanding techniques
- Apply techniques based on configuration
- Return structured results
- Track which techniques were used

**API**:

```python
class QueryProcessor:
    def __init__(
        self,
        llm_service: LLMService,
        enable_reformulation: bool = True,
        enable_expansion: bool = True,
        enable_hyde: bool = True
    ):
        self.reformulator = QueryReformulator(llm_service)
        self.expander = QueryExpander(llm_service)
        self.hyde = HyDEGenerator(llm_service)

    async def process(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        options: Optional[QueryUnderstandingOptions] = None
    ) -> QueryProcessingResult:
        """
        Process query through all enabled techniques.

        Returns:
            QueryProcessingResult with:
            - original_query
            - reformulated_query (if enabled)
            - expanded_queries (if enabled)
            - hyde_answer (if enabled)
            - techniques_used
        """
```

## Configuration

### Settings (`backend/core/settings.py`)

```python
# Query Understanding
enable_query_reformulation: bool = True
enable_query_expansion: bool = True
enable_hyde: bool = True

# Query Expansion
num_query_expansions: int = 3
expansion_temperature: float = 0.7

# Query Reformulation
reformulation_temperature: float = 0.3

# HyDE
hyde_temperature: float = 0.7
hyde_max_tokens: int = 500
```

### API Models (`backend/api/models/chat.py`)

```python
class QueryUnderstandingOptions(BaseModel):
    """Options for query understanding."""
    enable_reformulation: Optional[bool] = None
    enable_expansion: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    num_expansions: Optional[int] = None

class ChatRequest(BaseModel):
    message: str
    # ... existing fields ...
    query_understanding: Optional[QueryUnderstandingOptions] = None

class ChatResponse(BaseModel):
    # ... existing fields ...
    query_metadata: Optional[Dict[str, Any]] = None  # Shows techniques used
```

## Performance Considerations

### Latency Impact

| Technique     | Additional Latency | Mitigation             |
| ------------- | ------------------ | ---------------------- |
| Reformulation | +0.5-1.0s          | Only for vague queries |
| Expansion     | +1.0-1.5s          | Parallel LLM calls     |
| HyDE          | +1.0-2.0s          | Cache results          |
| **Total**     | +2.5-4.5s          | Async execution        |

### Optimization Strategies

1. **Parallel Execution**: Run expansion and HyDE simultaneously
2. **Caching**: Cache LLM responses for identical queries
3. **Conditional Application**: Only reformulate when needed
4. **Batch Processing**: Generate all expansions in one LLM call

## Testing Strategy

### Unit Tests

1. **Query Expander**
   - Test expansion generation
   - Test number of expansions
   - Test quality of expansions

2. **Query Reformulator**
   - Test vague query detection
   - Test reformulation quality
   - Test context integration

3. **HyDE Generator**
   - Test answer generation
   - Test answer quality
   - Test length constraints

4. **Query Processor**
   - Test orchestration logic
   - Test option handling
   - Test result structure

### Integration Tests

1. Test with RAG chain
2. Test with different retrieval methods
3. Test performance impact
4. Test with real queries

### Evaluation Metrics

1. **Recall@10**: Should increase by 8-10%
2. **MRR**: Should increase by 5-8%
3. **Latency**: Should stay under 5 seconds total
4. **User Satisfaction**: Subjective improvement

## UI Enhancements

### Chat Page Updates

```python
# Add query understanding controls
st.sidebar.subheader("🧠 Query Understanding")

enable_reformulation = st.sidebar.checkbox(
    "Reformulate vague queries",
    value=True,
    help="Clarify ambiguous questions"
)

enable_expansion = st.sidebar.checkbox(
    "Expand query",
    value=True,
    help="Generate alternative phrasings"
)

enable_hyde = st.sidebar.checkbox(
    "Use HyDE",
    value=True,
    help="Generate hypothetical answer for better retrieval"
)

# Show query processing results
if response.get("query_metadata"):
    with st.expander("🔍 Query Processing"):
        metadata = response["query_metadata"]

        if metadata.get("reformulated_query"):
            st.write("**Reformulated:**", metadata["reformulated_query"])

        if metadata.get("expanded_queries"):
            st.write("**Expanded Queries:**")
            for i, q in enumerate(metadata["expanded_queries"], 1):
                st.write(f"{i}. {q}")

        if metadata.get("hyde_used"):
            st.write("✅ HyDE enabled")
```

## Implementation Phases

### Phase 3.1: Core Modules (Week 1)

- [ ] Implement QueryExpander
- [ ] Implement QueryReformulator
- [ ] Implement HyDEGenerator
- [ ] Implement QueryProcessor
- [ ] Add unit tests

### Phase 3.2: Integration (Week 1)

- [ ] Update settings
- [ ] Update API models
- [ ] Integrate with RAG chain
- [ ] Update API routes

### Phase 3.3: UI & Testing (Week 2)

- [ ] Update Streamlit UI
- [ ] Add integration tests
- [ ] Performance testing
- [ ] User acceptance testing

### Phase 3.4: Documentation (Week 2)

- [ ] Update README
- [ ] Create usage guide
- [ ] Document best practices
- [ ] Create troubleshooting guide

## Expected Outcomes

### Quantitative Improvements

| Metric        | Phase 2   | Phase 3 Target | Improvement |
| ------------- | --------- | -------------- | ----------- |
| Recall@10     | 80-85%    | 88-92%         | +8-10%      |
| Precision@10  | 75-80%    | 78-83%         | +3-5%       |
| MRR           | 0.80-0.85 | 0.85-0.90      | +5-8%       |
| Query Latency | 0.03s     | 0.05s          | +0.02s      |
| Total Latency | 68s       | 72s            | +4s         |

### Qualitative Improvements

1. **Better handling of vague queries**
   - "How does it work?" → Clear, specific question
2. **Increased terminology coverage**
   - Finds documents using different terms
3. **Improved semantic matching**
   - HyDE bridges question-answer gap

## Success Criteria

- ✅ All three techniques implemented and working
- ✅ Integration with existing RAG chain seamless
- ✅ UI provides clear controls and feedback
- ✅ Performance impact acceptable (<5s additional latency)
- ✅ Measurable improvement in retrieval quality
- ✅ Comprehensive documentation

## Risks & Mitigation

| Risk              | Impact           | Mitigation                     |
| ----------------- | ---------------- | ------------------------------ |
| High latency      | User experience  | Parallel execution, caching    |
| LLM hallucination | Poor expansions  | Temperature tuning, validation |
| Over-expansion    | Noise in results | Limit to 3-4 expansions        |
| Cost (cloud LLMs) | Budget           | Use local LLMs, caching        |

## Next Steps

1. Create module directory structure
2. Implement QueryExpander
3. Implement QueryReformulator
4. Implement HyDEGenerator
5. Implement QueryProcessor
6. Integrate with RAG chain
7. Update API and UI
8. Test and validate
9. Document and deploy

---

**Status**: Ready for implementation  
**Estimated Effort**: 2 weeks  
**Dependencies**: Phase 2 complete  
**Priority**: High
