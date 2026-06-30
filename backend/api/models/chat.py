"""
Pydantic models for chat API endpoints.
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    """Model for a chat message."""
    
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")
    
    @validator('role')
    def validate_role(cls, v):
        """Validate message role."""
        allowed_roles = ['user', 'assistant', 'system']
        if v.lower() not in allowed_roles:
            raise ValueError(f"Role must be one of: {', '.join(allowed_roles)}")
        return v.lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is the revenue for Q4?",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class SourceReference(BaseModel):
    """Model for a source reference in chat response."""
    
    document_id: str = Field(..., description="Source document ID")
    filename: str = Field(..., description="Source filename")
    chunk_id: str = Field(..., description="Source chunk ID")
    content: str = Field(..., description="Relevant content excerpt")
    score: float = Field(..., description="Relevance score")
    page_number: Optional[int] = Field(None, description="Page number")
    
    # Hybrid retrieval metadata
    retrieval_method: Optional[str] = Field(None, description="Retrieval method used")
    faiss_score: Optional[float] = Field(None, description="FAISS similarity score")
    bm25_score: Optional[float] = Field(None, description="BM25 relevance score")
    faiss_rank: Optional[int] = Field(None, description="Rank in FAISS results")
    bm25_rank: Optional[int] = Field(None, description="Rank in BM25 results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123abc",
                "filename": "report.pdf",
                "chunk_id": "chunk_doc_123abc_5",
                "content": "Q4 revenue increased by 15% to $2.5M...",
                "score": 0.85,
                "page_number": 3,
                "retrieval_method": "hybrid",
                "faiss_score": 0.82,
                "bm25_score": 0.88,
                "faiss_rank": 2,
                "bm25_rank": 1
            }
        }


class QueryUnderstandingOptions(BaseModel):
    """Options for query understanding techniques."""
    
    enable_reformulation: bool = Field(True, description="Enable query reformulation")
    enable_expansion: bool = Field(True, description="Enable query expansion")
    enable_hyde: bool = Field(True, description="Enable HyDE generation")
    num_expansions: int = Field(3, description="Number of query expansions", ge=1, le=5)
    
    class Config:
        json_schema_extra = {
            "example": {
                "enable_reformulation": True,
                "enable_expansion": True,
                "enable_hyde": True,
                "num_expansions": 3
            }
        }


class QueryMetadata(BaseModel):
    """Metadata about query processing."""
    
    original_query: str = Field(..., description="Original user query")
    reformulated_query: Optional[str] = Field(None, description="Reformulated query")
    expanded_queries: List[str] = Field(default_factory=list, description="Expanded query variants")
    hyde_answer: Optional[str] = Field(None, description="Hypothetical answer")
    processing_time: float = Field(..., description="Query processing time in seconds")
    techniques_applied: Dict[str, bool] = Field(..., description="Which techniques were applied")
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_query": "What is it?",
                "reformulated_query": "What is the revenue for Q4?",
                "expanded_queries": [
                    "What are the Q4 earnings?",
                    "How much revenue was generated in Q4?"
                ],
                "hyde_answer": "The Q4 revenue was $2.5M...",
                "processing_time": 1.2,
                "techniques_applied": {
                    "reformulation": True,
                    "expansion": True,
                    "hyde": True
                }
            }
        }


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    message: str = Field(..., description="User message", min_length=1)
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    history: Optional[List[ChatMessage]] = Field(None, description="Conversation history")
    
    # RAG parameters
    use_rag: bool = Field(True, description="Whether to use RAG for response")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    retrieval_method: Optional[Literal["hybrid", "faiss", "bm25"]] = Field(
        None,
        description="Retrieval method (hybrid/faiss/bm25), uses default if not specified"
    )
    
    # Query understanding (Phase 3)
    query_understanding: Optional[QueryUnderstandingOptions] = Field(
        None,
        description="Query understanding options"
    )

    # Reranking (Phase 4)
    use_reranking: Optional[bool] = Field(
        None,
        description="Enable cross-encoder reranking. None = use server default (ENABLE_RERANKING setting)"
    )
    
    # Generation parameters
    temperature: float = Field(0.7, description="Generation temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(500, description="Maximum tokens in response", ge=50, le=2000)
    
    @validator('history')
    def validate_history(cls, v):
        """Validate conversation history."""
        if v and len(v) > 20:
            raise ValueError("Conversation history cannot exceed 20 messages")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the revenue for Q4?",
                "conversation_id": "conv_123",
                "use_rag": True,
                "top_k": 5,
                "temperature": 0.7,
                "max_tokens": 500
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    conversation_id: str = Field(..., description="Conversation ID")
    message: ChatMessage = Field(..., description="Assistant's response message")
    sources: List[SourceReference] = Field(..., description="Source references")
    
    # Metadata
    model: str = Field(..., description="Model used for generation")
    tokens_used: Optional[int] = Field(None, description="Tokens used in generation")
    processing_time: float = Field(..., description="Processing time in seconds")
    retrieval_method: Optional[str] = Field(None, description="Retrieval method used")
    query_metadata: Optional[QueryMetadata] = Field(None, description="Query processing metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "message": {
                    "role": "assistant",
                    "content": "According to the Q4 report, revenue increased by 15% to $2.5M...",
                    "timestamp": "2024-01-15T10:30:05Z"
                },
                "sources": [
                    {
                        "document_id": "doc_123abc",
                        "filename": "report.pdf",
                        "chunk_id": "chunk_doc_123abc_5",
                        "content": "Q4 revenue increased by 15%...",
                        "score": 0.85,
                        "page_number": 3
                    }
                ],
                "model": "gpt-3.5-turbo",
                "tokens_used": 150,
                "processing_time": 2.3
            }
        }


class ConversationSummary(BaseModel):
    """Model for conversation summary."""
    
    conversation_id: str = Field(..., description="Conversation ID")
    title: Optional[str] = Field(None, description="Conversation title")
    message_count: int = Field(..., description="Number of messages")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "title": "Q4 Revenue Discussion",
                "message_count": 6,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z"
            }
        }


class ConversationDetail(BaseModel):
    """Model for detailed conversation."""
    
    conversation_id: str = Field(..., description="Conversation ID")
    title: Optional[str] = Field(None, description="Conversation title")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "title": "Q4 Revenue Discussion",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the revenue for Q4?",
                        "timestamp": "2024-01-15T10:30:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "Q4 revenue increased by 15%...",
                        "timestamp": "2024-01-15T10:30:05Z"
                    }
                ],
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z"
            }
        }


class ConversationListResponse(BaseModel):
    """Response model for conversation listing."""
    
    total: int = Field(..., description="Total number of conversations")
    conversations: List[ConversationSummary] = Field(..., description="List of conversations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total": 2,
                "conversations": [
                    {
                        "conversation_id": "conv_123",
                        "title": "Q4 Revenue Discussion",
                        "message_count": 6,
                        "created_at": "2024-01-15T10:30:00Z"
                    },
                    {
                        "conversation_id": "conv_456",
                        "title": "Product Roadmap",
                        "message_count": 4,
                        "created_at": "2024-01-14T15:20:00Z"
                    }
                ]
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for response feedback."""
    
    conversation_id: str = Field(..., description="Conversation ID")
    message_index: int = Field(..., description="Message index in conversation")
    rating: int = Field(..., description="Rating (1-5)", ge=1, le=5)
    feedback: Optional[str] = Field(None, description="Optional feedback text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv_123",
                "message_index": 1,
                "rating": 5,
                "feedback": "Very helpful and accurate response"
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    
    status: str = Field(..., description="Submission status")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Feedback recorded successfully"
            }
        }


class StreamChatRequest(BaseModel):
    """Request model for streaming chat endpoint."""
    
    message: str = Field(..., description="User message", min_length=1)
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    
    # RAG parameters
    use_rag: bool = Field(True, description="Whether to use RAG")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    retrieval_method: Optional[Literal["hybrid", "faiss", "bm25"]] = Field(
        None,
        description="Retrieval method (hybrid/faiss/bm25)"
    )
    
    # Generation parameters
    temperature: float = Field(0.7, description="Generation temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(500, description="Maximum tokens", ge=50, le=2000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the revenue for Q4?",
                "conversation_id": "conv_123",
                "use_rag": True,
                "top_k": 5,
                "temperature": 0.7
            }
        }


class StreamChunk(BaseModel):
    """Model for streaming response chunk."""
    
    type: str = Field(..., description="Chunk type (token, sources, done)")
    content: Optional[str] = Field(None, description="Token content")
    sources: Optional[List[SourceReference]] = Field(None, description="Source references")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "token",
                "content": "According to",
                "sources": None,
                "metadata": None
            }
        }


class RAGMetrics(BaseModel):
    """Model for RAG performance metrics."""
    
    retrieval_time: float = Field(..., description="Retrieval time in seconds")
    generation_time: float = Field(..., description="Generation time in seconds")
    total_time: float = Field(..., description="Total processing time")
    documents_retrieved: int = Field(..., description="Number of documents retrieved")
    tokens_used: int = Field(..., description="Tokens used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "retrieval_time": 0.5,
                "generation_time": 1.8,
                "total_time": 2.3,
                "documents_retrieved": 5,
                "tokens_used": 150
            }
        }


# Made with Bob