"""
HyDE (Hypothetical Document Embeddings) Generator.

Generates hypothetical answers to queries for improved semantic retrieval.
"""

from typing import Optional
import time
from backend.llm.llm_service import LLMService
from backend.core.logging import get_logger

logger = get_logger(__name__)


class HyDEGenerator:
    """
    Generates hypothetical document answers for queries.
    
    HyDE improves retrieval by:
    1. Generating a hypothetical answer to the query
    2. Embedding the answer instead of the query
    3. Using the answer embedding to find similar documents
    
    This bridges the semantic gap between questions and answers.
    """
    
    # Prompt template for generating hypothetical answers
    HYDE_PROMPT = """Write a detailed, informative paragraph that would answer this question.
Write as if you are an expert providing a comprehensive answer.
Use technical terminology and specific details that would appear in a relevant document.

Question: {query}

Write a single paragraph (3-5 sentences) that directly answers this question:"""
    
    # Alternative prompt for technical queries
    TECHNICAL_HYDE_PROMPT = """Generate a technical explanation that would answer this question.
Include specific terminology, concepts, and details that would appear in technical documentation.

Question: {query}

Technical explanation:"""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        temperature: float = 0.7,  # Higher temperature for diverse answers
        max_tokens: int = 500,  # Longer for detailed answers
        use_technical_prompt: bool = False
    ):
        """
        Initialize HyDE Generator.
        
        Args:
            llm_service: LLM service for generation
            temperature: LLM temperature (higher for more diverse answers)
            max_tokens: Maximum tokens for generation
            use_technical_prompt: Use technical prompt variant
        """
        self.llm_service = llm_service or LLMService()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_technical_prompt = use_technical_prompt
        
        logger.info(
            f"Initialized HyDEGenerator with temperature={temperature}, "
            f"max_tokens={max_tokens}, technical={use_technical_prompt}"
        )
    
    async def generate(self, query: str) -> str:
        """
        Generate hypothetical answer for query.
        
        Args:
            query: Original query
            
        Returns:
            Hypothetical answer text (or original query on failure)
        """
        start_time = time.time()
        
        try:
            # Select prompt template
            prompt_template = (
                self.TECHNICAL_HYDE_PROMPT if self.use_technical_prompt
                else self.HYDE_PROMPT
            )
            
            # Build prompt
            prompt = prompt_template.format(query=query)
            
            # Generate hypothetical answer
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse and clean response
            hypothetical_answer = self._parse_answer(response["text"])
            
            elapsed = time.time() - start_time
            
            logger.info(
                f"Generated HyDE answer in {elapsed:.3f}s "
                f"({len(hypothetical_answer)} chars)"
            )
            logger.debug(f"Query: {query}")
            logger.debug(f"HyDE answer: {hypothetical_answer[:200]}...")
            
            return hypothetical_answer
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            # Return original query on failure
            return query
    
    def _parse_answer(self, response_text: str) -> str:
        """
        Parse and clean LLM response.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Cleaned hypothetical answer
        """
        # Remove leading/trailing whitespace
        answer = response_text.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            'answer:',
            'explanation:',
            'technical explanation:',
            'response:',
            'here is',
            'here\'s'
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                # Remove colon if present after prefix
                if answer.startswith(':'):
                    answer = answer[1:].strip()
                break
        
        # Remove quotes if entire answer is quoted
        if (answer.startswith('"') and answer.endswith('"')) or \
           (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1].strip()
        
        return answer
    
    def get_stats(self) -> dict:
        """
        Get generator statistics.
        
        Returns:
            Dictionary with configuration
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_technical_prompt": self.use_technical_prompt
        }


class HyDEResult:
    """
    Result from HyDE generation.
    
    Contains both the original query and hypothetical answer.
    """
    
    def __init__(
        self,
        original_query: str,
        hypothetical_answer: str,
        generation_time: float
    ):
        """
        Initialize HyDE result.
        
        Args:
            original_query: Original user query
            hypothetical_answer: Generated hypothetical answer
            generation_time: Time taken to generate (seconds)
        """
        self.original_query = original_query
        self.hypothetical_answer = hypothetical_answer
        self.generation_time = generation_time
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "original_query": self.original_query,
            "hypothetical_answer": self.hypothetical_answer,
            "generation_time": self.generation_time,
            "answer_length": len(self.hypothetical_answer)
        }
    
    def __repr__(self) -> str:
        return (
            f"HyDEResult(query='{self.original_query[:50]}...', "
            f"answer_length={len(self.hypothetical_answer)}, "
            f"time={self.generation_time:.3f}s)"
        )


# Made with Bob