"""
Query Processor - Orchestrates all query understanding techniques.

Coordinates query expansion, reformulation, and HyDE generation.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import time
import asyncio

from backend.query_understanding.query_expander import QueryExpander
from backend.query_understanding.query_reformulator import QueryReformulator
from backend.query_understanding.hyde_generator import HyDEGenerator
from backend.llm.llm_service import LLMService
from backend.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryUnderstandingOptions:
    """
    Configuration options for query understanding.
    """
    enable_reformulation: bool = True
    enable_expansion: bool = True
    enable_hyde: bool = True
    num_expansions: int = 3
    use_technical_hyde: bool = False


@dataclass
class QueryProcessingResult:
    """
    Result from query processing.
    
    Contains all processed query variants and metadata.
    """
    original_query: str
    reformulated_query: Optional[str] = None
    expanded_queries: List[str] = field(default_factory=list)
    hyde_answer: Optional[str] = None
    
    # Metadata
    processing_time: float = 0.0
    reformulation_applied: bool = False
    expansion_applied: bool = False
    hyde_applied: bool = False
    
    def get_primary_query(self) -> str:
        """
        Get the primary query to use for retrieval.
        
        Returns reformulated query if available, otherwise original.
        """
        return self.reformulated_query or self.original_query
    
    def get_all_queries(self) -> List[str]:
        """
        Get all query variants for retrieval.
        
        Returns:
            List of all queries (primary + expansions)
        """
        queries = [self.get_primary_query()]
        if self.expanded_queries:
            queries.extend(self.expanded_queries)
        return queries
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for API responses.
        
        Returns:
            Dictionary representation
        """
        return {
            "original_query": self.original_query,
            "reformulated_query": self.reformulated_query,
            "expanded_queries": self.expanded_queries,
            "hyde_answer": self.hyde_answer,
            "processing_time": round(self.processing_time, 3),
            "techniques_applied": {
                "reformulation": self.reformulation_applied,
                "expansion": self.expansion_applied,
                "hyde": self.hyde_applied
            },
            "total_queries": len(self.get_all_queries())
        }
    
    def __repr__(self) -> str:
        return (
            f"QueryProcessingResult("
            f"original='{self.original_query[:50]}...', "
            f"reformulated={self.reformulation_applied}, "
            f"expanded={self.expansion_applied}, "
            f"hyde={self.hyde_applied}, "
            f"time={self.processing_time:.3f}s)"
        )


class QueryProcessor:
    """
    Orchestrates all query understanding techniques.
    
    Coordinates:
    - Query reformulation (clarify vague queries)
    - Query expansion (generate alternative phrasings)
    - HyDE generation (create hypothetical answers)
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        expander: Optional[QueryExpander] = None,
        reformulator: Optional[QueryReformulator] = None,
        hyde_generator: Optional[HyDEGenerator] = None
    ):
        """
        Initialize Query Processor.
        
        Args:
            llm_service: Shared LLM service
            expander: Query expander instance
            reformulator: Query reformulator instance
            hyde_generator: HyDE generator instance
        """
        self.llm_service = llm_service or LLMService()
        
        # Initialize components
        self.expander = expander or QueryExpander(llm_service=self.llm_service)
        self.reformulator = reformulator or QueryReformulator(llm_service=self.llm_service)
        self.hyde_generator = hyde_generator or HyDEGenerator(llm_service=self.llm_service)
        
        logger.info("Initialized QueryProcessor with all components")
    
    async def process(
        self,
        query: str,
        options: Optional[QueryUnderstandingOptions] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> QueryProcessingResult:
        """
        Process query using enabled techniques.
        
        Args:
            query: Original user query
            options: Configuration options
            conversation_history: Previous conversation for context
            
        Returns:
            QueryProcessingResult with all processed variants
        """
        start_time = time.time()
        
        # Use default options if not provided
        if options is None:
            options = QueryUnderstandingOptions()
        
        logger.info(
            f"Processing query with options: "
            f"reformulation={options.enable_reformulation}, "
            f"expansion={options.enable_expansion}, "
            f"hyde={options.enable_hyde}"
        )
        
        # Initialize result
        result = QueryProcessingResult(original_query=query)
        
        # Step 1: Reformulation (if enabled)
        # This should happen first to clarify the query
        if options.enable_reformulation:
            try:
                reformulated = await self.reformulator.reformulate(
                    query=query,
                    conversation_history=conversation_history
                )
                if reformulated != query:
                    result.reformulated_query = reformulated
                    result.reformulation_applied = True
                    logger.info(f"Query reformulated: {query} -> {reformulated}")
            except Exception as e:
                logger.error(f"Reformulation failed: {e}")
        
        # Get the primary query (reformulated or original)
        primary_query = result.get_primary_query()
        
        # Step 2 & 3: Run expansion and HyDE in parallel (if enabled)
        tasks = []
        
        if options.enable_expansion:
            tasks.append(self._expand_query(primary_query, options.num_expansions))
        
        if options.enable_hyde:
            tasks.append(self._generate_hyde(primary_query))
        
        # Execute parallel tasks
        if tasks:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process expansion result
                if options.enable_expansion:
                    expansion_result = results[0]
                    if isinstance(expansion_result, list):
                        result.expanded_queries = expansion_result
                        result.expansion_applied = len(expansion_result) > 0
                    elif isinstance(expansion_result, Exception):
                        logger.error(f"Expansion failed: {expansion_result}")
                
                # Process HyDE result
                if options.enable_hyde:
                    hyde_idx = 1 if options.enable_expansion else 0
                    hyde_result = results[hyde_idx]
                    if isinstance(hyde_result, str):
                        result.hyde_answer = hyde_result
                        result.hyde_applied = hyde_result != primary_query
                    elif isinstance(hyde_result, Exception):
                        logger.error(f"HyDE generation failed: {hyde_result}")
                        
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}")
        
        # Calculate total processing time
        result.processing_time = time.time() - start_time
        
        logger.info(
            f"Query processing complete in {result.processing_time:.3f}s: "
            f"reformulated={result.reformulation_applied}, "
            f"expanded={result.expansion_applied} ({len(result.expanded_queries)} variants), "
            f"hyde={result.hyde_applied}"
        )
        
        return result
    
    async def _expand_query(
        self,
        query: str,
        num_expansions: int
    ) -> List[str]:
        """
        Expand query (internal helper).
        
        Args:
            query: Query to expand
            num_expansions: Number of expansions
            
        Returns:
            List of expanded queries (excluding original)
        """
        try:
            all_queries = await self.expander.expand(
                query=query,
                num_expansions=num_expansions
            )
            # Return only the expansions (exclude original which is first)
            return all_queries[1:] if len(all_queries) > 1 else []
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []
    
    async def _generate_hyde(self, query: str) -> str:
        """
        Generate HyDE answer (internal helper).
        
        Args:
            query: Query to generate answer for
            
        Returns:
            Hypothetical answer
        """
        try:
            return await self.hyde_generator.generate(query)
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query
    
    def get_stats(self) -> dict:
        """
        Get processor statistics.
        
        Returns:
            Dictionary with component stats
        """
        return {
            "expander": self.expander.get_stats(),
            "reformulator": self.reformulator.get_stats(),
            "hyde_generator": self.hyde_generator.get_stats()
        }


# Made with Bob