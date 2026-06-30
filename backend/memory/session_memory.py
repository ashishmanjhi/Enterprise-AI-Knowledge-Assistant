"""
Session Memory (Phase 6) — Redis-backed per-conversation message store.

Each conversation is stored as a Redis list under the key
  conv:<conversation_id>:messages
with an optional TTL (default 24 h).  Each entry is a JSON-encoded dict:
  {"role": "user"|"assistant", "content": "...", "timestamp": "<iso>"}

On Redis unavailability the manager degrades gracefully to an in-process
dict so the rest of the stack keeps working.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

_MSG_KEY   = "conv:{cid}:messages"   # Redis list of JSON-encoded messages
_META_KEY  = "conv:{cid}:meta"       # Redis hash: created_at, updated_at, title


class SessionMemory:
    """
    Thin wrapper around a Redis connection that stores conversation history
    as a Redis list.  Falls back to an in-process dict when Redis is not
    available.
    """

    def __init__(self) -> None:
        self._redis  = None          # lazy-loaded
        self._fallback: Dict[str, list] = {}   # {conversation_id: [msg, ...]}
        self._use_redis = True

    # ------------------------------------------------------------------ #
    # Redis helpers
    # ------------------------------------------------------------------ #

    def _get_redis(self):
        if not self._use_redis:
            return None
        if self._redis is not None:
            return self._redis
        try:
            import redis as _redis
            r = _redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                socket_connect_timeout=2,
                decode_responses=True,
            )
            r.ping()
            self._redis = r
            logger.info("SessionMemory connected to Redis")
            return self._redis
        except Exception as e:
            logger.warning(f"SessionMemory: Redis unavailable ({e}), using in-process fallback")
            self._use_redis = False
            return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def append(
        self,
        conversation_id: str,
        role: str,
        content: str,
        ttl_seconds: int = 0,
    ) -> None:
        """Append a message to the conversation history."""
        ttl = ttl_seconds or settings.memory_session_ttl
        entry = json.dumps({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        r = self._get_redis()
        if r:
            key = _MSG_KEY.format(cid=conversation_id)
            try:
                r.rpush(key, entry)
                if ttl > 0:
                    r.expire(key, ttl)
                # Update meta
                mkey = _META_KEY.format(cid=conversation_id)
                now  = datetime.now(timezone.utc).isoformat()
                r.hset(mkey, mapping={"updated_at": now})
                if not r.hexists(mkey, "created_at"):
                    r.hset(mkey, "created_at", now)
                if ttl > 0:
                    r.expire(mkey, ttl)
                return
            except Exception as e:
                logger.error(f"Redis append failed: {e}")

        # Fallback
        if conversation_id not in self._fallback:
            self._fallback[conversation_id] = []
        self._fallback[conversation_id].append(json.loads(entry))

    def get_history(
        self,
        conversation_id: str,
        max_messages: int = 0,
    ) -> List[Dict]:
        """
        Return message history for a conversation.

        Args:
            conversation_id: Unique conversation identifier.
            max_messages: If > 0, return only the last N messages.

        Returns:
            List of {"role", "content", "timestamp"} dicts, oldest first.
        """
        r = self._get_redis()
        if r:
            key = _MSG_KEY.format(cid=conversation_id)
            try:
                start = -max_messages if max_messages > 0 else 0
                raw   = r.lrange(key, start, -1)
                return [json.loads(m) for m in raw]
            except Exception as e:
                logger.error(f"Redis get_history failed: {e}")

        msgs = self._fallback.get(conversation_id, [])
        if max_messages > 0:
            return msgs[-max_messages:]
        return list(msgs)

    def delete(self, conversation_id: str) -> bool:
        """Delete all history for a conversation. Returns True if anything was deleted."""
        r = self._get_redis()
        if r:
            try:
                deleted = r.delete(
                    _MSG_KEY.format(cid=conversation_id),
                    _META_KEY.format(cid=conversation_id),
                )
                return deleted > 0
            except Exception as e:
                logger.error(f"Redis delete failed: {e}")

        existed = conversation_id in self._fallback
        self._fallback.pop(conversation_id, None)
        return existed

    def get_meta(self, conversation_id: str) -> Dict:
        """Return metadata (created_at, updated_at) for a conversation."""
        r = self._get_redis()
        if r:
            try:
                return r.hgetall(_META_KEY.format(cid=conversation_id)) or {}
            except Exception:
                pass
        return {}

    def message_count(self, conversation_id: str) -> int:
        """Return the number of messages stored for a conversation."""
        r = self._get_redis()
        if r:
            try:
                return r.llen(_MSG_KEY.format(cid=conversation_id))
            except Exception:
                pass
        return len(self._fallback.get(conversation_id, []))


# Module-level singleton
session_memory = SessionMemory()

# Made with Bob
