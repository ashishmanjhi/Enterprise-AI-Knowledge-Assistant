"""
Query Expander for generating multiple query variations.

Increases recall by creating alternative phrasings that capture different
terminology, perspectives, and specificity levels.
"""

from typing import List, Optional
import time
from backend.llm.llm_service import LLMService
from backend.core.logging import get_logger

logger = get_logger(__name__)


class QueryExpander:
    """
    Expands queries into multiple variations to increase retrieval recall.
    
    Uses LLM to generate alternative phrasings that:
    - Use different terminology and synonyms
    - Ask from different perspectives
    - Vary in specificity (more general or more specific)
    """
    
    # Prompt template for query expansion
    EXPANSION_PROMPT = """Generate {num_queries} alternative ways to ask this question.
Focus on:
1. Using different terminology and synonyms
2. Asking from different perspectives  
3. Being more specific or more general

Original question: {query}

Generate exactly {num_queries} alternative questions (one per line, no numbering):"""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        default_num_expansions: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 200
    ):
        """
        Initialize Query Expander.
        
        Args:
            llm_service: LLM service for generation
            default_num_expansions: Default number of expansions to generate
            temperature: LLM temperature for diversity
            max_tokens: Maximum tokens for generation
        """
        self.llm_service = llm_service or LLMService()
        self.default_num_expansions = default_num_expansions
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(
            f"Initialized QueryExpander with "
            f"num_expansions={default_num_expansions}, "
            f"temperature={temperature}"
        )
    
    async def expand(
        self,
        query: str,
        num_expansions: Optional[int] = None
    ) -> List[str]:
        """
        Expand query into multiple variations.
        
        Args:
            query: Original query to expand
            num_expansions: Number of expansions (uses default if None)
            
        Returns:
            List of expanded queries (including original as first item)
        """
        if num_expansions is None:
            num_expansions = self.default_num_expansions
        
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self.EXPANSION_PROMPT.format(
                query=query,
                num_queries=num_expansions
            )
            
            # Generate expansions
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse response
            expanded_queries = self._parse_expansions(
                response["text"],
                num_expansions
            )
            
            # Always include original query first
            all_queries = [query] + expanded_queries
            
            elapsed = time.time() - start_time
            
            logger.info(
                f"Expanded query into {len(all_queries)} variations "
                f"in {elapsed:.3f}s"
            )
            logger.debug(f"Original: {query}")
            for i, exp_query in enumerate(expanded_queries, 1):
                logger.debug(f"Expansion {i}: {exp_query}")
            
            return all_queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            # Return original query on failure
            return [query]
    
    def _parse_expansions(
        self,
        response_text: str,
        expected_count: int
    ) -> List[str]:
        """
        Parse LLM response into list of queries.
        
        Args:
            response_text: Raw LLM response
            expected_count: Expected number of expansions
            
        Returns:
            List of parsed queries
        """
        # Split by newlines and clean
        lines = [
            line.strip()
            for line in response_text.strip().split('\n')
            if line.strip()
        ]
        
        # Remove numbering if present (e.g., "1. ", "- ", etc.)
        cleaned_queries = []
        for line in lines:
            # Remove common prefixes
            for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            
            # Remove quotes if present
            line = line.strip('"\'')
            
            if line and len(line) > 10:  # Minimum query length
                cleaned_queries.append(line)
        
        # Limit to expected count
        result = cleaned_queries[:expected_count]
        
        # Warn if we got fewer than expected
        if len(result) < expected_count:
            logger.warning(
                f"Expected {expected_count} expansions but got {len(result)}"
            )
        
        return result
    
    def get_stats(self) -> dict:
        """
        Get expander statistics.
        
        Returns:
            Dictionary with configuration
        """
        return {
            "default_num_expansions": self.default_num_expansions,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "llm_provider": self.llm_service.provider_name
        }


# Made with Bob