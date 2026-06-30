"""
Phase 6 Tests: Conversational Memory

Covers SessionMemory (in-process fallback, no Redis required) and
ConversationManager (record_turn, get_prompt_history, delete,
summarisation trigger).  All tests are fully offline — Redis and the LLM
are mocked.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.memory.session_memory import SessionMemory
from backend.memory.conversation_manager import ConversationManager


# ============================================================
# Helpers
# ============================================================

def _memory() -> SessionMemory:
    """Return a fresh SessionMemory instance forced into fallback mode."""
    m = SessionMemory()
    m._use_redis = False          # skip Redis connection attempt
    return m


def _patched_manager(llm=None):
    """
    Context manager that yields (ConversationManager, fresh SessionMemory).

    Patches ``session_memory`` inside ``conversation_manager`` module so all
    method calls use the in-process fallback store — no Redis required.

    Usage::

        with _patched_manager() as (mgr, mem):
            await mgr.record_turn(...)
    """
    from unittest.mock import patch as _patch
    fresh = _memory()
    patcher = _patch("backend.memory.conversation_manager.session_memory", fresh)
    patched = patcher.start()
    mgr = ConversationManager(llm_service=llm)
    # Return a context-manager-like object via a simple namedtuple trick
    class _Ctx:
        def __enter__(self_):
            return mgr, fresh
        def __exit__(self_, *_):
            patcher.stop()
    return _Ctx()


# ============================================================
# SessionMemory — in-process fallback
# ============================================================

class TestSessionMemoryFallback:
    """Tests for SessionMemory using the in-process fallback (no Redis)."""

    def test_append_and_get_history(self):
        m = _memory()
        m.append("c1", "user", "Hello")
        m.append("c1", "assistant", "Hi!")
        history = m.get_history("c1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi!"

    def test_get_history_empty_conversation(self):
        m = _memory()
        assert m.get_history("does_not_exist") == []

    def test_get_history_max_messages(self):
        m = _memory()
        for i in range(10):
            m.append("c2", "user", f"msg {i}")
        recent = m.get_history("c2", max_messages=3)
        assert len(recent) == 3
        assert recent[0]["content"] == "msg 7"
        assert recent[-1]["content"] == "msg 9"

    def test_get_history_max_zero_returns_all(self):
        m = _memory()
        for i in range(5):
            m.append("c3", "user", f"msg {i}")
        all_msgs = m.get_history("c3", max_messages=0)
        assert len(all_msgs) == 5

    def test_message_count(self):
        m = _memory()
        assert m.message_count("cx") == 0
        m.append("cx", "user", "a")
        m.append("cx", "assistant", "b")
        assert m.message_count("cx") == 2

    def test_delete_existing_conversation(self):
        m = _memory()
        m.append("del_me", "user", "test")
        assert m.message_count("del_me") == 1
        deleted = m.delete("del_me")
        assert deleted is True
        assert m.message_count("del_me") == 0

    def test_delete_nonexistent_conversation(self):
        m = _memory()
        deleted = m.delete("ghost")
        assert deleted is False

    def test_multiple_conversations_isolated(self):
        m = _memory()
        m.append("conv_a", "user", "message for A")
        m.append("conv_b", "user", "message for B")
        assert m.message_count("conv_a") == 1
        assert m.message_count("conv_b") == 1
        assert m.get_history("conv_a")[0]["content"] == "message for A"
        assert m.get_history("conv_b")[0]["content"] == "message for B"

    def test_message_has_timestamp(self):
        m = _memory()
        m.append("ts_test", "user", "check timestamp")
        msg = m.get_history("ts_test")[0]
        assert "timestamp" in msg
        assert msg["timestamp"]  # non-empty string

    def test_get_meta_returns_empty_for_fallback(self):
        m = _memory()
        # Fallback doesn't store meta; should return empty dict (not raise)
        meta = m.get_meta("any_conv")
        assert isinstance(meta, dict)

    def test_system_role_allowed(self):
        """Summary messages are stored with role='system'."""
        m = _memory()
        m.append("sys_test", "system", "[Summary] Prior conversation…")
        msgs = m.get_history("sys_test")
        assert msgs[0]["role"] == "system"


# ============================================================
# ConversationManager
# ============================================================

class TestConversationManager:

    @pytest.mark.asyncio
    async def test_record_turn_stores_two_messages(self):
        with _patched_manager() as (mgr, mem):
            await mgr.record_turn("r1", "What is RAG?", "RAG is retrieval-augmented generation.")
            history = mem.get_history("r1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_prompt_history_empty(self):
        with _patched_manager() as (mgr, _):
            result = mgr.get_prompt_history("empty_conv")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_prompt_history_returns_role_content_only(self):
        with _patched_manager() as (mgr, _):
            await mgr.record_turn("ph1", "Q1", "A1")
            history = mgr.get_prompt_history("ph1")
        assert len(history) == 2
        for msg in history:
            assert set(msg.keys()) == {"role", "content"}  # no 'timestamp'

    @pytest.mark.asyncio
    async def test_get_prompt_history_respects_max_messages(self):
        with _patched_manager() as (mgr, _):
            for i in range(6):   # 6 turns = 12 messages
                await mgr.record_turn("trunc", f"Q{i}", f"A{i}")
            limited = mgr.get_prompt_history("trunc", max_messages=4)
        assert len(limited) == 4

    @pytest.mark.asyncio
    async def test_delete_conversation(self):
        with _patched_manager() as (mgr, _):
            await mgr.record_turn("del_conv", "hello", "world")
            deleted = mgr.delete_conversation("del_conv")
            after   = mgr.get_prompt_history("del_conv")
        assert deleted is True
        assert after == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_false(self):
        with _patched_manager() as (mgr, _):
            result = mgr.delete_conversation("no_such_conv")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_conversation_info_message_count(self):
        with _patched_manager() as (mgr, _):
            await mgr.record_turn("info_c", "q", "a")
            info = mgr.get_conversation_info("info_c")
        assert info["conversation_id"] == "info_c"
        assert info["message_count"] == 2

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate(self):
        with _patched_manager() as (mgr, _):
            await mgr.record_turn("acc", "Q1", "A1")
            await mgr.record_turn("acc", "Q2", "A2")
            await mgr.record_turn("acc", "Q3", "A3")
            history = mgr.get_prompt_history("acc")
        assert len(history) == 6

    # -------------------------------------------------------
    # Summarisation
    # -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_summarisation_triggered_when_threshold_exceeded(self):
        """
        When message count reaches summary_threshold*2, _maybe_summarise
        should be called.  We mock the LLM to return a fake summary and
        verify the history is compacted.
        """
        from backend.core.settings import settings

        # Use a very low threshold for the test
        original_threshold = settings.memory_summary_threshold
        original_enabled   = settings.memory_enable_summarisation
        settings.memory_summary_threshold    = 3   # summarise after 3 turns
        settings.memory_enable_summarisation = True

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value={"text": "This is a summary."})

        with _patched_manager(llm=mock_llm) as (mgr, mem):
            # Add exactly threshold*2 turns (6 messages) to trigger summarisation
            for i in range(3):
                await mgr.record_turn("sum_c", f"Q{i}", f"A{i}")
            history = mem.get_history("sum_c")

        # After compaction, the first message should be a system summary
        assert len(history) > 0, "History is empty after summarisation"
        assert history[0]["role"] == "system"
        assert "summary" in history[0]["content"].lower()

        # Restore settings
        settings.memory_summary_threshold    = original_threshold
        settings.memory_enable_summarisation = original_enabled

    @pytest.mark.asyncio
    async def test_summarisation_skipped_when_disabled(self):
        from backend.core.settings import settings

        original_enabled   = settings.memory_enable_summarisation
        original_threshold = settings.memory_summary_threshold
        settings.memory_enable_summarisation = False
        settings.memory_summary_threshold    = 2

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value={"text": "Summary text."})

        with _patched_manager(llm=mock_llm) as (mgr, mem):
            for i in range(2):
                await mgr.record_turn("no_sum", f"Q{i}", f"A{i}")
            history = mem.get_history("no_sum")

        # No system message should be prepended
        assert all(m["role"] != "system" for m in history)

        settings.memory_enable_summarisation = original_enabled
        settings.memory_summary_threshold    = original_threshold

    @pytest.mark.asyncio
    async def test_summarisation_skipped_when_no_llm(self):
        """Without an LLM, summarisation silently skips."""
        from backend.core.settings import settings

        original_threshold = settings.memory_summary_threshold
        original_enabled   = settings.memory_enable_summarisation
        settings.memory_summary_threshold    = 2
        settings.memory_enable_summarisation = True

        with _patched_manager(llm=None) as (mgr, mem):
            for i in range(2):
                await mgr.record_turn("no_llm", f"Q{i}", f"A{i}")
            history = mem.get_history("no_llm")

        # History unchanged — no system message
        assert all(m["role"] != "system" for m in history)

        settings.memory_summary_threshold    = original_threshold
        settings.memory_enable_summarisation = original_enabled

    @pytest.mark.asyncio
    async def test_summarise_messages_returns_none_on_llm_error(self):
        """If the LLM raises, _summarise_messages returns None gracefully."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        mgr = ConversationManager(llm_service=mock_llm)
        result = await mgr._summarise_messages(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        )
        assert result is None


# ============================================================
# SessionMemory — Redis path (mocked)
# ============================================================

class TestSessionMemoryRedis:
    """
    Verify the Redis code path using a mocked redis.Redis client.
    No real Redis connection is made.
    """

    def _memory_with_mock_redis(self):
        """Build a SessionMemory whose _redis attribute is a Mock."""
        m = SessionMemory()
        mock_r = MagicMock()
        mock_r.ping.return_value = True
        mock_r.rpush.return_value = 1
        mock_r.expire.return_value = True
        mock_r.hexists.return_value = False
        mock_r.hset.return_value = 1
        mock_r.lrange.return_value = [
            '{"role":"user","content":"from redis","timestamp":"2024-01-01T00:00:00+00:00"}'
        ]
        mock_r.llen.return_value = 1
        mock_r.delete.return_value = 1
        mock_r.hgetall.return_value = {"created_at": "2024-01-01T00:00:00+00:00"}
        m._redis      = mock_r
        m._use_redis  = True
        return m, mock_r

    def test_append_calls_rpush(self):
        m, r = self._memory_with_mock_redis()
        m.append("cid", "user", "hello", ttl_seconds=3600)
        r.rpush.assert_called_once()
        args = r.rpush.call_args[0]
        assert args[0] == "conv:cid:messages"

    def test_get_history_calls_lrange(self):
        m, r = self._memory_with_mock_redis()
        history = m.get_history("cid", max_messages=5)
        r.lrange.assert_called_once_with("conv:cid:messages", -5, -1)
        assert len(history) == 1
        assert history[0]["content"] == "from redis"

    def test_get_history_all_uses_zero_start(self):
        m, r = self._memory_with_mock_redis()
        m.get_history("cid", max_messages=0)
        r.lrange.assert_called_once_with("conv:cid:messages", 0, -1)

    def test_delete_calls_redis_delete(self):
        m, r = self._memory_with_mock_redis()
        deleted = m.delete("cid")
        r.delete.assert_called_once_with("conv:cid:messages", "conv:cid:meta")
        assert deleted is True

    def test_message_count_uses_llen(self):
        m, r = self._memory_with_mock_redis()
        count = m.message_count("cid")
        r.llen.assert_called_once_with("conv:cid:messages")
        assert count == 1

    def test_get_meta_uses_hgetall(self):
        m, r = self._memory_with_mock_redis()
        meta = m.get_meta("cid")
        r.hgetall.assert_called_once_with("conv:cid:meta")
        assert "created_at" in meta

    def test_redis_failure_falls_back_to_in_process(self):
        """If an rpush call raises, message is stored in the fallback dict."""
        m = SessionMemory()
        mock_r = MagicMock()
        mock_r.rpush.side_effect = Exception("Redis down")
        m._redis     = mock_r
        m._use_redis = True

        m.append("fb", "user", "stored locally")
        # Redis failed — should be in fallback
        assert len(m._fallback.get("fb", [])) == 1


# Made with Bob
