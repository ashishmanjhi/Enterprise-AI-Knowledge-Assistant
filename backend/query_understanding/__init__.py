"""
Query Understanding Module

Provides query processing techniques to improve retrieval:
- Query Expansion: Generate alternative query phrasings
- Query Reformulation: Clarify vague or ambiguous queries
- HyDE: Generate hypothetical answers for semantic matching
"""

from backend.query_understanding.query_expander import QueryExpander
from backend.query_understanding.query_reformulator import QueryReformulator
from backend.query_understanding.hyde_generator import HyDEGenerator, HyDEResult
from backend.query_understanding.query_processor import (
    QueryProcessor,
    QueryProcessingResult,
    QueryUnderstandingOptions
)

__all__ = [
    # Core processor
    "QueryProcessor",
    "QueryProcessingResult",
    "QueryUnderstandingOptions",
    
    # Individual techniques
    "QueryExpander",
    "QueryReformulator",
    "HyDEGenerator",
    "HyDEResult",
]

# Made with Bob
