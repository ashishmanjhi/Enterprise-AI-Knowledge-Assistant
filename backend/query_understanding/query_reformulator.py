"""
Query Reformulator for clarifying vague or ambiguous queries.

Improves precision by rewriting unclear queries into specific, well-formed questions.
"""

from typing import List, Dict, Optional
import time
from backend.llm.llm_service import LLMService
from backend.core.logging import get_logger

logger = get_logger(__name__)


class QueryReformulator:
    """
    Reformulates vague or ambiguous queries into clear, specific questions.
    
    Handles:
    - Vague pronouns ("it", "this", "that")
    - Missing context
    - Ambiguous terms
    - Poorly formed questions
    """
    
    # Prompt template for query reformulation
    REFORMULATION_PROMPT = """Rewrite this query to be more specific and clear.

Original query: {query}

{context_section}

Provide a single, clear, specific question that captures the user's intent.
Do not add information that isn't implied by the original query or context.

Reformulated query:"""
    
    # Prompt for reformulation with conversation history
    CONTEXT_TEMPLATE = """Previous conversation context:
{conversation_context}

Use this context to resolve any ambiguous references."""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        temperature: float = 0.3,  # Lower temperature for more focused reformulation
        max_tokens: int = 100
    ):
        """
        Initialize Query Reformulator.
        
        Args:
            llm_service: LLM service for generation
            temperature: LLM temperature (lower for more focused output)
            max_tokens: Maximum tokens for generation
        """
        self.llm_service = llm_service or LLMService()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(
            f"Initialized QueryReformulator with temperature={temperature}"
        )
    
    async def reformulate(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Reformulate vague query into clear question.
        
        Args:
            query: Original query to reformulate
            conversation_history: Previous conversation messages for context
            
        Returns:
            Reformulated query (or original if already clear)
        """
        # Check if reformulation is needed
        if not self._needs_reformulation(query):
            logger.debug(f"Query is already clear, skipping reformulation: {query}")
            return query
        
        start_time = time.time()
        
        try:
            # Build context section if conversation history provided
            context_section = ""
            if conversation_history:
                context_section = self._build_context_section(conversation_history)
            
            # Build prompt
            prompt = self.REFORMULATION_PROMPT.format(
                query=query,
                context_section=context_section
            )
            
            # Generate reformulation
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse and clean response
            reformulated = self._parse_reformulation(response["text"])
            
            elapsed = time.time() - start_time
            
            logger.info(
                f"Reformulated query in {elapsed:.3f}s"
            )
            logger.debug(f"Original: {query}")
            logger.debug(f"Reformulated: {reformulated}")
            
            return reformulated
            
        except Exception as e:
            logger.error(f"Query reformulation failed: {e}")
            # Return original query on failure
            return query
    
    def _needs_reformulation(self, query: str) -> bool:
        """
        Determine if query needs reformulation.
        
        Args:
            query: Query to check
            
        Returns:
            True if reformulation is recommended
        """
        query_lower = query.lower()
        
        # Check for vague pronouns
        vague_pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'them']
        has_vague_pronoun = any(
            f' {pronoun} ' in f' {query_lower} ' or
            query_lower.startswith(f'{pronoun} ') or
            query_lower.endswith(f' {pronoun}')
            for pronoun in vague_pronouns
        )
        
        # Check for very short queries (likely missing context)
        is_very_short = len(query.split()) < 4
        
        # Check for question words without clear subject
        starts_with_how = query_lower.startswith('how') and 'how does' not in query_lower
        
        # Needs reformulation if any condition is met
        needs_reform = has_vague_pronoun or (is_very_short and starts_with_how)
        
        if needs_reform:
            logger.debug(
                f"Query needs reformulation: "
                f"vague_pronoun={has_vague_pronoun}, "
                f"very_short={is_very_short}, "
                f"unclear_how={starts_with_how}"
            )
        
        return needs_reform
    
    def _build_context_section(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Build context section from conversation history.
        
        Args:
            conversation_history: List of previous messages
            
        Returns:
            Formatted context string
        """
        if not conversation_history:
            return ""
        
        # Use last 3 messages for context
        recent_messages = conversation_history[-3:]
        
        context_lines = []
        for msg in recent_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                context_lines.append(f"{role.capitalize()}: {content}")
        
        if not context_lines:
            return ""
        
        conversation_context = "\n".join(context_lines)
        
        return self.CONTEXT_TEMPLATE.format(
            conversation_context=conversation_context
        )
    
    def _parse_reformulation(self, response_text: str) -> str:
        """
        Parse and clean LLM response.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Cleaned reformulated query
        """
        # Take first line if multiple lines
        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
        
        if not lines:
            logger.warning("Empty reformulation response")
            return response_text.strip()
        
        reformulated = lines[0]
        
        # Remove quotes if present
        reformulated = reformulated.strip('"\'')
        
        # Remove common prefixes
        prefixes_to_remove = [
            'reformulated query:',
            'reformulated:',
            'query:',
            'answer:',
            'a:',
            'q:'
        ]
        
        reformulated_lower = reformulated.lower()
        for prefix in prefixes_to_remove:
            if reformulated_lower.startswith(prefix):
                reformulated = reformulated[len(prefix):].strip()
                break
        
        return reformulated
    
    def get_stats(self) -> dict:
        """
        Get reformulator statistics.
        
        Returns:
            Dictionary with configuration
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


# Made with Bob