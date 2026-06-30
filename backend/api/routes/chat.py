"""
Chat API routes with RAG support.
Handles conversational interactions with document context.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import uuid
from datetime import datetime

from backend.api.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    QueryMetadata,
    SourceReference,
    StreamChatRequest,
    StreamChunk
)
from backend.llm.rag_chain import RAGChain
from backend.query_understanding.query_processor import QueryUnderstandingOptions as QUOptions
from backend.memory.conversation_manager import conversation_manager
from backend.guardrails.pipeline import GuardrailsPipeline
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Initialize RAG chain and guardrails pipeline
rag_chain  = RAGChain()
_guardrails = GuardrailsPipeline()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Generate a chat response with optional RAG.

    Retrieves relevant documents and generates a contextual response.
    History is automatically loaded from / persisted to Redis memory
    (Phase 6) if no explicit history is supplied in the request.

    Args:
        request: Chat request with message and parameters

    Returns:
        Chat response with answer and sources
    """
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

        # Phase 7: input guardrails (injection, toxicity, PII)
        input_guard = await _guardrails.check_input(request.message)
        if input_guard.blocked:
            raise HTTPException(
                status_code=400,
                detail={
                    "error":        "request_blocked_by_guardrails",
                    "block_reason": input_guard.block_reason,
                    "checks":       [c.to_dict() for c in input_guard.checks],
                },
            )

        # Use redacted text if PII was found in input (warn-only mode)
        effective_message = input_guard.redacted_text or request.message

        # Phase 6: load persisted history when client doesn't send it
        history_dicts = None
        if request.history:
            history_dicts = [
                {"role": msg.role, "content": msg.content}
                for msg in request.history
            ]
        else:
            # Load from Redis/in-process memory
            loaded = conversation_manager.get_prompt_history(conversation_id)
            if loaded:
                history_dicts = loaded

        # Build query understanding options if provided in the request
        qu_options = None
        if request.query_understanding is not None:
            qu_options = QUOptions(
                enable_reformulation=request.query_understanding.enable_reformulation,
                enable_expansion=request.query_understanding.enable_expansion,
                enable_hyde=request.query_understanding.enable_hyde,
                num_expansions=request.query_understanding.num_expansions
            )

        # Generate response using RAG
        result = await rag_chain.generate_response(
            query=effective_message,
            top_k=request.top_k,
            document_ids=request.document_ids,
            conversation_history=history_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            retrieval_method=request.retrieval_method,
            query_understanding_options=qu_options,
            use_reranking=request.use_reranking
        )
        
        # Format sources with proper defaults
        sources = []
        for source in result.get("sources", []):
            sources.append(SourceReference(
                document_id=source.get("document_id") or "unknown",
                filename=source.get("filename") or "unknown",
                chunk_id=source.get("chunk_id") or "unknown",
                content=source.get("content") or "",
                score=source.get("score", 0.0),
                page_number=source.get("page_number"),
                retrieval_method=source.get("retrieval_method"),
                faiss_score=source.get("faiss_score"),
                bm25_score=source.get("bm25_score"),
                faiss_rank=source.get("faiss_rank"),
                bm25_rank=source.get("bm25_rank")
            ))
        
        # Phase 7: output guardrails (hallucination, PII in output)
        context_texts = [s.get("content", "") for s in result.get("sources", [])]
        output_guard  = await _guardrails.check_output(result["response"], context_texts)

        final_answer = result["response"]
        guardrail_warnings = [c.to_dict() for c in output_guard.warnings]

        if output_guard.blocked:
            # Replace hallucinated answer with a safety message
            final_answer = (
                "I'm sorry, my answer could not be verified against the source documents. "
                "Please rephrase your question or consult the original documents directly."
            )
            logger.warning(f"Output blocked by guardrails: {output_guard.block_reason}")
        elif output_guard.redacted_text:
            # PII found in output — use redacted version
            final_answer = output_guard.redacted_text

        # Create response message
        message = ChatMessage(
            role="assistant",
            content=final_answer,
            timestamp=datetime.utcnow()
        )

        # Get metadata
        metadata = result.get("metadata", {})

        # Build query_metadata if query understanding was applied
        query_metadata = None
        raw_qm = result.get("query_metadata")
        if raw_qm:
            query_metadata = QueryMetadata(
                original_query=raw_qm.get("original_query", request.message),
                reformulated_query=raw_qm.get("reformulated_query"),
                expanded_queries=raw_qm.get("expanded_queries", []),
                hyde_answer=raw_qm.get("hyde_answer"),
                processing_time=raw_qm.get("processing_time", 0.0),
                techniques_applied=raw_qm.get("techniques_applied", {})
            )
        
        # Phase 6: persist this turn to memory (use original message, final answer)
        await conversation_manager.record_turn(
            conversation_id,
            request.message,
            final_answer,
        )

        resp = ChatResponse(
            conversation_id=conversation_id,
            message=message,
            sources=sources,
            model=metadata.get("model", "unknown"),
            tokens_used=metadata.get("tokens_used", 0),
            processing_time=metadata.get("total_time", 0.0),
            retrieval_method=metadata.get("retrieval_method"),
            query_metadata=query_metadata
        )

        # Attach guardrail warnings as extra metadata (non-breaking addition)
        if guardrail_warnings:
            resp_dict = resp.model_dump()
            resp_dict.setdefault("metadata", {})
            resp_dict["metadata"]["guardrail_warnings"] = guardrail_warnings
            logger.info(f"Guardrail warnings on response: {[w['detector'] for w in guardrail_warnings]}")

        return resp

    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {str(e)}"
        )


@router.post("/stream")
async def chat_stream(request: StreamChatRequest):
    """
    Generate a streaming chat response with RAG.
    
    Streams tokens as they are generated for better UX.
    
    Args:
        request: Streaming chat request
        
    Returns:
        Server-sent events stream
    """
    try:
        async def generate():
            """Generate streaming response."""
            try:
                # Stream response from RAG chain
                async for chunk in rag_chain.generate_response_stream(
                    query=request.message,
                    top_k=request.top_k,
                    document_ids=None,
                    conversation_history=None,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    retrieval_method=request.retrieval_method
                ):
                    # Format chunk as SSE
                    if chunk.get("type") == "token":
                        yield f"data: {chunk['content']}\n\n"
                    elif chunk.get("type") == "sources":
                        # Send sources at the end
                        import json
                        sources_data = json.dumps(chunk.get("sources", []))
                        yield f"data: [SOURCES]{sources_data}\n\n"
                    elif chunk.get("type") == "done":
                        yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Streaming failed: {e}")
                yield f"data: [ERROR]{str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Chat streaming failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat streaming failed: {str(e)}"
        )


@router.post("/direct")
async def chat_direct(
    message: str,
    temperature: float = 0.7,
    max_tokens: int = 500
):
    """
    Generate a direct response without RAG.
    
    Useful for general questions that don't require document context.
    
    Args:
        message: User message
        temperature: Generation temperature
        max_tokens: Maximum tokens
        
    Returns:
        Direct LLM response
    """
    try:
        result = await rag_chain.answer_question(
            question=message,
            use_rag=False,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "response": result["response"],
            "model": result["metadata"].get("model", "unknown"),
            "tokens_used": result["metadata"].get("tokens_used", 0)
        }
        
    except Exception as e:
        logger.error(f"Direct chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Direct chat failed: {str(e)}"
        )


@router.get("/health")
async def chat_health():
    """
    Check chat service health.
    
    Returns:
        Health status of chat components
    """
    try:
        # Check if LLM service is available
        llm_available = rag_chain.llm_service.client is not None
        
        # Get retriever stats
        retriever_stats = rag_chain.retriever.get_stats()
        
        return {
            "status": "healthy" if llm_available else "degraded",
            "llm_available": llm_available,
            "vector_store": retriever_stats.get("vector_store", {}),
            "embedding_model": retriever_stats.get("embedding_service", {})
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Made with Bob