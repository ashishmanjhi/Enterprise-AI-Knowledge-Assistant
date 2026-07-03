"""
Conversation Manager (Phase 6) — orchestrates session + summary memory.

Responsibilities:
  1. Append new user/assistant turns to Redis via SessionMemory.
  2. Keep a sliding window for the prompt (``max_history_messages``).
  3. When history exceeds ``summary_threshold``, summarise the oldest half
     using the LLM and store the summary as a compressed system message.
  4. Expose ``get_prompt_history()`` for the RAG chain — returns the list
     of dicts that should be included in the prompt context.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from backend.memory.session_memory import session_memory
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class ConversationManager:
    """
    High-level conversation memory manager.

    Usage::

        mgr = ConversationManager()
        history = await mgr.get_prompt_history(conversation_id)
        # … generate response …
        await mgr.record_turn(conversation_id, question, answer)
    """

    def __init__(self, llm_service=None) -> None:
        """
        Args:
            llm_service: Optional LLMService instance used for summarisation.
                         If None, summarisation is disabled.
        """
        self._llm = llm_service

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def record_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Persist one Q-A turn to session memory."""
        session_memory.append(conversation_id, "user",      user_message)
        session_memory.append(conversation_id, "assistant", assistant_message)

        # Check if we should compress old history
        count = session_memory.message_count(conversation_id)
        if (
            self._llm is not None
            and settings.memory_enable_summarisation
            and count >= settings.memory_summary_threshold * 2  # threshold in *turns*
        ):
            await self._maybe_summarise(conversation_id)

    def get_prompt_history(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Return the last ``max_messages`` messages ready for prompt injection.

        Returns list of {"role": ..., "content": ...} dicts, summary
        prepended if one exists.
        """
        limit = max_messages or settings.memory_max_history_messages
        msgs  = session_memory.get_history(conversation_id, max_messages=limit)
        # Strip timestamps — the prompt builder only needs role+content
        return [{"role": m["role"], "content": m["content"]} for m in msgs]

    def get_full_history(self, conversation_id: str) -> List[Dict]:
        """Return complete history (with timestamps) for API responses."""
        return session_memory.get_history(conversation_id, max_messages=0)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete all memory for a conversation."""
        return session_memory.delete(conversation_id)

    def get_conversation_info(self, conversation_id: str) -> Dict:
        """Return conversation metadata + message count."""
        meta  = session_memory.get_meta(conversation_id)
        count = session_memory.message_count(conversation_id)
        return {
            "conversation_id": conversation_id,
            "message_count":   count,
            "created_at":      meta.get("created_at"),
            "updated_at":      meta.get("updated_at"),
            "has_summary":     count > 0 and session_memory.get_history(
                                   conversation_id, max_messages=1
                               )[0].get("role") == "system",
        }

    # ------------------------------------------------------------------ #
    # Summarisation (long-term memory compaction)
    # ------------------------------------------------------------------ #

    async def _maybe_summarise(self, conversation_id: str) -> None:
        """
        Summarise the oldest half of the conversation and replace it with a
        single compressed system message so the context window stays small.
        """
        all_msgs = session_memory.get_history(conversation_id)
        if len(all_msgs) < settings.memory_summary_threshold * 2:
            return

        # Split: summarise the first half, keep the second half verbatim
        half     = len(all_msgs) // 2
        to_sum   = all_msgs[:half]
        to_keep  = all_msgs[half:]

        summary = await self._summarise_messages(to_sum)
        if not summary:
            return

        # Rebuild history: [system summary] + recent messages
        session_memory.delete(conversation_id)
        session_memory.append(
            conversation_id,
            role="system",
            content=f"[Conversation summary] {summary}",
        )
        for msg in to_keep:
            session_memory.append(conversation_id, msg["role"], msg["content"])

        logger.info(
            f"Summarised {half} messages for conversation {conversation_id}"
        )

    async def _summarise_messages(self, messages: List[Dict]) -> Optional[str]:
        """Call the LLM to produce a brief summary of the given messages."""
        if not self._llm:
            return None

        # Build a compact transcript
        transcript_parts = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            transcript_parts.append(f"{role.capitalize()}: {text}")

        transcript = "\n".join(transcript_parts)
        prompt = (
            "Summarise the following conversation in 3–5 sentences, "
            "preserving the key facts and decisions:\n\n"
            f"{transcript}\n\nSummary:"
        )

        try:
            result = await self._llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200,
            )
            return result.get("text", "").strip() or None
        except Exception as e:
            logger.error(f"Summarisation LLM call failed: {e}")
            return None


# ── Module-level singleton with lazy LLM injection (F-07) ──────────────
# Previously instantiated with no LLM, so summarisation was silently disabled.
# Now creates the LLM on first call so memory_enable_summarisation actually works.

_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Return (or lazily create) the module-level ConversationManager singleton.

    The LLMService is imported and instantiated on first call — not at module
    import time — so startup is unaffected when Ollama is unavailable.
    """
    global _conversation_manager
    if _conversation_manager is None:
        from backend.llm.llm_service import LLMService
        _conversation_manager = ConversationManager(llm_service=LLMService())
    return _conversation_manager


# Backward-compatible alias — all existing ``from backend.memory.conversation_manager
# import conversation_manager`` imports continue to work because the name resolves
# to the lazily-created singleton on first attribute access via the module.
conversation_manager = get_conversation_manager()

# Made with Bob
