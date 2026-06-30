# Phase 8 — User Experience: Implementation Plan

## Overview

Phase 8 improves the end-user experience of the RAG platform across three areas:

1. **Streaming responses** — Tokens stream to the browser in real time instead of waiting for the full answer
2. **Improved chat history UI** — Restore full session history from server on page load; display timestamps; show safety badges
3. **Document management enhancements** — Search bar, content preview snippets

---

## Goals

| Goal | Success Criteria |
|------|-----------------|
| Streaming SSE | First token appears within 2 s of submit; answer streams token-by-token |
| History restore | Refreshing the Chat page restores the last N messages from Redis |
| Safety badges | Guardrail warnings visible per message inline |
| Document search | Filter document library by filename |
| Content preview | Expand any source citation to see the full chunk |

---

## Architecture Changes

### Streaming (Server → Browser)

The existing `POST /api/v1/chat/stream` endpoint already returns SSE. Phase 8 wires the Streamlit frontend to consume it using `st.write_stream`.

```
Browser                     FastAPI
  │  POST /api/v1/chat/stream │
  │──────────────────────────►│
  │                            │  retrieve docs
  │  data: token1              │  build prompt
  │◄───────────────────────────│  stream LLM tokens
  │  data: token2              │
  │◄───────────────────────────│
  │  data: [SOURCES]{...}      │
  │◄───────────────────────────│
  │  data: [DONE]              │
  │◄───────────────────────────│
```

Because Streamlit reruns on every widget interaction, the streaming consumer uses `st.write_stream` with a generator that wraps the `requests` streaming response.

### History Restore

On Chat page load, if `st.session_state.conversation_id` is set but `st.session_state.messages` is empty, fetch history from `GET /api/v1/memory/{id}` and reconstruct the message list.

### Safety Badges

The chat route already logs guardrail warnings in response metadata. The frontend extracts `metadata.guardrail_warnings` and shows a collapsible warning panel below each assistant message that triggered a detector.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `frontend/streamlit/pages/2_💬_Chat.py` | Streaming toggle, history restore, safety badges |
| `backend/api/routes/chat.py` | Attach `guardrail_warnings` to the non-streaming response body |
| `frontend/streamlit/app.py` | Update home page status to reflect Phase 7 complete |

---

## Implementation Steps

1. **History restore on page load** — detect empty session, call memory API, rebuild `st.session_state.messages`
2. **Streaming toggle** — sidebar toggle; when on, use SSE endpoint via `requests.get(stream=True)` and `st.write_stream`
3. **Safety badges** — show inline warnings when guardrail detectors fire on the output
4. **Document search filter** — add text input above document library table; filter client-side
5. **Source content preview** — full-text expander for each source citation
6. **Home page update** — mark Phase 7 complete, add Phase 8 status

---

*Made with Bob*
