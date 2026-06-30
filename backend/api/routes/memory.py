"""
Memory API routes (Phase 6).

GET    /api/v1/memory/{conversation_id}        — fetch history
GET    /api/v1/memory/{conversation_id}/info   — metadata + message count
DELETE /api/v1/memory/{conversation_id}        — delete conversation history
"""

from fastapi import APIRouter, HTTPException

from backend.memory.conversation_manager import conversation_manager
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


@router.get("/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    limit: int = 50,
):
    """
    Retrieve the message history for a conversation.

    Args:
        conversation_id: Unique conversation identifier.
        limit: Maximum number of recent messages to return (0 = all).
    """
    history = conversation_manager.get_full_history(conversation_id)
    if limit > 0:
        history = history[-limit:]

    return {
        "conversation_id": conversation_id,
        "message_count":   len(history),
        "messages":        history,
    }


@router.get("/{conversation_id}/info")
async def get_conversation_info(conversation_id: str):
    """Return metadata for a conversation without fetching full history."""
    return conversation_manager.get_conversation_info(conversation_id)


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete all stored history for a conversation.

    Returns a confirmation dict. Does not raise 404 if the conversation
    did not exist — deletion is idempotent.
    """
    deleted = conversation_manager.delete_conversation(conversation_id)
    return {
        "conversation_id": conversation_id,
        "deleted":         deleted,
        "message":         "Conversation history deleted." if deleted
                           else "No history found for this conversation.",
    }


# Made with Bob
