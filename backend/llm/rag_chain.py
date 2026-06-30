"""
RAG (Retrieval-Augmented Generation) chain.
Combines document retrieval with LLM generation.
"""

from typing import List, Dict, Any, Optional, Literal
import time
from backend.retrievers.retriever import DocumentRetriever, RetrievalResult
from backend.retrievers.hybrid_retriever import HybridRetriever, HybridRetrievalResult
from backend.llm.llm_service import LLMService
from backend.query_understanding.query_processor import QueryProcessor, QueryUnderstandingOptions
from backend.rerankers.cross_encoder import CrossEncoderReranker
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class RAGChain:
    """
    RAG chain that combines retrieval and generation.
    
    Retrieves relevant documents and uses them as context for LLM generation.
    """
    
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_service: Optional[LLMService] = None,
        use_hybrid: bool = True,
        query_processor: Optional[QueryProcessor] = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        """
        Initialize RAG chain.
        
        Args:
            retriever: Hybrid retriever (or basic DocumentRetriever for backward compatibility)
            llm_service: LLM service for generation
            use_hybrid: Whether to use hybrid retrieval (default: True)
            query_processor: Query understanding processor (Phase 3)
            reranker: Cross-encoder reranker (Phase 4). None = use settings default.
        """
        if use_hybrid:
            self.retriever = retriever or HybridRetriever()
            self.is_hybrid = True
        else:
            # Backward compatibility with basic retriever
            self.retriever = retriever or DocumentRetriever()
            self.is_hybrid = False

        self.llm_service = llm_service or LLMService()
        self.query_processor = query_processor or QueryProcessor(llm_service=self.llm_service)

        # Reranker: instantiate with settings defaults; None means disabled
        if reranker is not None:
            self.reranker: Optional[CrossEncoderReranker] = reranker
        elif settings.enable_reranking:
            self.reranker = CrossEncoderReranker(
                model_name=settings.reranker_model,
                top_n=settings.reranker_top_n,
                batch_size=settings.reranker_batch_size,
            )
        else:
            self.reranker = None

        retriever_type = "HybridRetriever" if self.is_hybrid else "DocumentRetriever"
        reranker_info = f"reranker={settings.reranker_model}" if self.reranker else "reranker=off"
        logger.info(f"Initialized RAGChain with {retriever_type}, QueryProcessor, {reranker_info}")
    
    async def generate_response(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        retrieval_method: Optional[Literal["hybrid", "faiss", "bm25"]] = None,
        query_understanding_options: Optional[QueryUnderstandingOptions] = None,
        use_reranking: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            document_ids: Filter by document IDs
            conversation_history: Previous conversation messages
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            retrieval_method: Retrieval method (hybrid/faiss/bm25), uses default if None
            query_understanding_options: Phase 3 query processing options
            use_reranking: Override reranking on/off per request. None = use instance default.
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        # Use default retrieval method if not specified
        if retrieval_method is None:
            retrieval_method = settings.default_retrieval_method
        start_time = time.time()
        
        try:
            # Step 1: Query Understanding (Phase 3)
            query_processing_result = None
            if query_understanding_options is not None:
                try:
                    query_processing_result = await self.query_processor.process(
                        query=query,
                        options=query_understanding_options,
                        conversation_history=conversation_history
                    )
                    logger.info(
                        f"Query understanding complete in "
                        f"{query_processing_result.processing_time:.3f}s"
                    )
                except Exception as qe:
                    logger.error(f"Query understanding failed (continuing without it): {qe}")

            # Determine effective retrieval query
            retrieval_query = query
            faiss_query = None
            if query_processing_result is not None:
                retrieval_query = query_processing_result.get_primary_query()
                # Use HyDE answer for FAISS semantic search when available
                if query_processing_result.hyde_applied and query_processing_result.hyde_answer:
                    faiss_query = query_processing_result.hyde_answer

            # Step 2: Retrieve relevant documents
            retrieval_start = time.time()
            
            if self.is_hybrid:
                retrieval_results = await self.retriever.retrieve(  # type: ignore
                    query=retrieval_query,
                    top_k=top_k,
                    document_ids=document_ids,
                    method=retrieval_method,
                    faiss_query=faiss_query
                )
                # Format context using hybrid retriever's method
                context = self.retriever.format_context(  # type: ignore
                    retrieval_results,
                    max_length=3000,
                    include_metadata=True
                )
            else:
                # Basic retriever doesn't support method/faiss_query parameters
                retrieval_results = await self.retriever.retrieve(  # type: ignore
                    query=retrieval_query,
                    top_k=top_k,
                    document_ids=document_ids
                )
                # Format context using basic retriever's method
                context = self.retriever.format_context(  # type: ignore
                    retrieval_results,
                    max_length=3000,
                    include_metadata=True
                )
            
            retrieval_time = time.time() - retrieval_start

            logger.info(
                f"Retrieved {len(retrieval_results)} documents "
                f"using {retrieval_method} in {retrieval_time:.3f}s"
            )

            # Step 3: Rerank (Phase 4) — optional second-pass scoring
            reranking_applied = False
            should_rerank = (
                use_reranking if use_reranking is not None else self.reranker is not None
            )
            if should_rerank and self.reranker is not None and retrieval_results:
                rerank_start = time.time()
                retrieval_results = self.reranker.rerank(
                    query=retrieval_query,
                    results=list(retrieval_results),
                )
                reranking_applied = True
                logger.info(
                    f"Reranking complete in {time.time() - rerank_start:.3f}s, "
                    f"{len(retrieval_results)} results kept"
                )
                # Re-format context with reranked order
                if self.is_hybrid:
                    context = self.retriever.format_context(  # type: ignore
                        retrieval_results, max_length=3000, include_metadata=True
                    )

            # Step 4: Build prompt with context
            # Use original query for the prompt so the answer stays on-topic
            prompt = self._build_prompt(
                query=query,
                context=context,
                conversation_history=conversation_history
            )
            
            # Step 4: Generate response
            generation_start = time.time()
            
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            generation_time = time.time() - generation_start
            
            logger.info(f"Generated response in {generation_time:.3f}s")
            
            # Step 5: Format sources
            sources = self._format_sources(retrieval_results)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            result = {
                "response": response["text"],
                "sources": sources,
                "metadata": {
                    "model": response.get("model", "unknown"),
                    "tokens_used": response.get("tokens_used", 0),
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "documents_retrieved": len(retrieval_results),
                    "retrieval_method": retrieval_method,
                    "reranking_applied": reranking_applied,
                }
            }

            # Include query understanding metadata if processing was performed
            if query_processing_result is not None:
                result["query_metadata"] = query_processing_result.to_dict()

            return result
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            raise
    
    async def generate_response_stream(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        retrieval_method: Optional[Literal["hybrid", "faiss", "bm25"]] = None,
        query_understanding_options: Optional[QueryUnderstandingOptions] = None,
        use_reranking: Optional[bool] = None
    ):
        """
        Generate a streaming response using RAG.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            document_ids: Filter by document IDs
            conversation_history: Previous conversation messages
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            retrieval_method: Retrieval method (hybrid/faiss/bm25)
            query_understanding_options: Phase 3 query processing options
            
        Yields:
            Response chunks
        """
        # Use default retrieval method if not specified
        if retrieval_method is None:
            retrieval_method = settings.default_retrieval_method
        try:
            # Step 1: Query Understanding (Phase 3)
            retrieval_query = query
            faiss_query = None
            if query_understanding_options is not None:
                try:
                    qp_result = await self.query_processor.process(
                        query=query,
                        options=query_understanding_options,
                        conversation_history=conversation_history
                    )
                    retrieval_query = qp_result.get_primary_query()
                    if qp_result.hyde_applied and qp_result.hyde_answer:
                        faiss_query = qp_result.hyde_answer
                except Exception as qe:
                    logger.error(f"Query understanding failed in stream (continuing): {qe}")

            # Step 2: Retrieve relevant documents
            if self.is_hybrid:
                retrieval_results = await self.retriever.retrieve(  # type: ignore
                    query=retrieval_query,
                    top_k=top_k,
                    document_ids=document_ids,
                    method=retrieval_method,
                    faiss_query=faiss_query
                )
                context = self.retriever.format_context(  # type: ignore
                    retrieval_results, max_length=3000, include_metadata=True
                )
            else:
                retrieval_results = await self.retriever.retrieve(  # type: ignore
                    query=retrieval_query,
                    top_k=top_k,
                    document_ids=document_ids
                )
                context = self.retriever.format_context(  # type: ignore
                    retrieval_results, max_length=3000, include_metadata=True
                )

            # Step 3: Rerank (Phase 4)
            should_rerank = (
                use_reranking if use_reranking is not None else self.reranker is not None
            )
            if should_rerank and self.reranker is not None and retrieval_results:
                retrieval_results = self.reranker.rerank(
                    query=retrieval_query,
                    results=list(retrieval_results),
                )
                if self.is_hybrid:
                    context = self.retriever.format_context(  # type: ignore
                        retrieval_results, max_length=3000, include_metadata=True
                    )

            # Step 4: Build prompt
            prompt = self._build_prompt(
                query=query,
                context=context,
                conversation_history=conversation_history
            )
            
            # Step 4: Stream response
            async for chunk in self.llm_service.generate_stream(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                yield chunk
            
            # Step 5: Send sources at the end
            sources = self._format_sources(retrieval_results)
            yield {
                "type": "sources",
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"RAG streaming failed: {e}")
            raise
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build prompt with context and conversation history.
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous messages
            
        Returns:
            Formatted prompt
        """
        # System message
        system_message = (
            "You are a helpful AI assistant that answers questions based on the provided context. "
            "Always cite your sources by referring to the document names and page numbers when available. "
            "If the context doesn't contain enough information to answer the question, say so clearly. "
            "Be concise and accurate in your responses."
        )
        
        # Build prompt parts
        prompt_parts = [system_message, "\n\n"]
        
        # Add conversation history if provided
        if conversation_history:
            prompt_parts.append("Previous conversation:\n")
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}\n")
            prompt_parts.append("\n")
        
        # Add context
        if context:
            prompt_parts.append("Context from documents:\n")
            prompt_parts.append(context)
            prompt_parts.append("\n\n")
        
        # Add current query
        prompt_parts.append(f"Question: {query}\n\n")
        prompt_parts.append("Answer:")
        
        return "".join(prompt_parts)
    
    def _format_sources(
        self,
        retrieval_results: List
    ) -> List[Dict[str, Any]]:
        """
        Format retrieval results as source references.
        
        Args:
            retrieval_results: List of retrieval results (RetrievalResult or HybridRetrievalResult)
            
        Returns:
            List of formatted sources
        """
        sources = []
        
        for result in retrieval_results:
            source = {
                "document_id": result.document_id,
                "filename": result.filename,
                "chunk_id": result.chunk_id,
                "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "score": round(result.score, 3),
                "page_number": result.page_number
            }
            
            # Add hybrid-specific metadata if available
            if hasattr(result, 'retrieval_method'):
                source["retrieval_method"] = result.retrieval_method
            if hasattr(result, 'faiss_score') and result.faiss_score is not None:
                source["faiss_score"] = round(result.faiss_score, 3)
            if hasattr(result, 'bm25_score') and result.bm25_score is not None:
                source["bm25_score"] = round(result.bm25_score, 3)
            if hasattr(result, 'faiss_rank') and result.faiss_rank is not None:
                source["faiss_rank"] = result.faiss_rank
            if hasattr(result, 'bm25_rank') and result.bm25_rank is not None:
                source["bm25_rank"] = result.bm25_rank
            
            sources.append(source)
        
        return sources
    
    async def answer_question(
        self,
        question: str,
        use_rag: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Answer a question with or without RAG.
        
        Args:
            question: User question
            use_rag: Whether to use RAG (retrieval)
            **kwargs: Additional arguments for generation
            
        Returns:
            Response dictionary
        """
        if use_rag:
            return await self.generate_response(query=question, **kwargs)
        else:
            # Direct LLM generation without retrieval
            response = await self.llm_service.generate(
                prompt=question,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 500)
            )
            
            return {
                "response": response["text"],
                "sources": [],
                "metadata": {
                    "model": response.get("model", "unknown"),
                    "tokens_used": response.get("tokens_used", 0),
                    "documents_retrieved": 0
                }
            }


# Made with Bob