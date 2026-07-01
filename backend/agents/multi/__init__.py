"""
Phase 11 — Multi-Agent Ecosystem.

Public surface:
    MultiAgentState         — shared TypedDict
    MultiAgentOrchestrator  — top-level LangGraph orchestrator
    ResearchAgent           — research sub-graph
    RetrievalAgent          — retrieval sub-graph
    EvaluationAgent         — evaluation sub-graph
    GovernanceAgent         — governance sub-graph
    KnowledgeAgent          — knowledge sub-graph
"""

from backend.agents.multi.state import MultiAgentState
from backend.agents.multi.orchestrator import MultiAgentOrchestrator
from backend.agents.multi.research_agent import ResearchAgent
from backend.agents.multi.retrieval_agent import RetrievalAgent
from backend.agents.multi.evaluation_agent import EvaluationAgent
from backend.agents.multi.governance_agent import GovernanceAgent
from backend.agents.multi.knowledge_agent import KnowledgeAgent

__all__ = [
    "MultiAgentState",
    "MultiAgentOrchestrator",
    "ResearchAgent",
    "RetrievalAgent",
    "EvaluationAgent",
    "GovernanceAgent",
    "KnowledgeAgent",
]

# Made with Bob
