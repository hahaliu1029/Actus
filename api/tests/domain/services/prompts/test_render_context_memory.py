"""M2 PR-4: verify ``build_render_context`` threads ``memory_snapshot`` through.

PR-3 made ``RenderContext.memory_snapshot`` a field and taught the three
memory sections to read it. PR-4 is the last mile — the async nodes in
``main_graph`` build a snapshot upstream and pass it here as a kwarg. If
this plumbing regresses (kwarg dropped, default reverts, snapshot lost),
the sections go silent at runtime even though unit tests for them pass
against ``ctx.memory_snapshot`` directly.

These tests guard the kwarg contract at the boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.domain.services.prompts.memory_snapshot import MemorySnapshot
from app.domain.services.prompts.render_context import build_render_context


class _FakeAgentConfig:
    supports_vision = True
    supports_pdf_input = True


class _FakeLLM:
    provider_name = "openai"


def _base_state() -> dict:
    return {
        "language": "zh",
        "message": "hi",
        "skill_context": "",
        "skill_names_in_context": [],
        "conversation_summaries": [],
    }


def _base_config() -> dict:
    return {
        "configurable": {
            "llm": _FakeLLM(),
            "bound_tool_names": frozenset(),
        }
    }


def test_default_memory_snapshot_is_none() -> None:
    """Absent kwarg → snapshot field stays None (existing callers unaffected)."""
    ctx = build_render_context(_base_state(), _base_config(), _FakeAgentConfig())
    assert ctx.memory_snapshot is None


def test_explicit_none_kwarg_stays_none() -> None:
    ctx = build_render_context(
        _base_state(), _base_config(), _FakeAgentConfig(),
        memory_snapshot=None,
    )
    assert ctx.memory_snapshot is None


def test_snapshot_kwarg_is_threaded_into_render_context() -> None:
    """Non-empty snapshot rides through to ``RenderContext.memory_snapshot``."""
    from app.domain.models.memory_chunk import MemoryChunk

    now = datetime.now(timezone.utc)
    chunk = MemoryChunk(
        id="c1",
        user_id="u1",
        content="prefers concise responses",
        content_hash="h1",
        source="manual",
        metadata={},
        created_at=now,
        updated_at=now,
        session_id=None,
        embedding=None,
        category="user",
        pinned=True,
    )
    snapshot = MemorySnapshot(user_chunks=(chunk,))

    ctx = build_render_context(
        _base_state(), _base_config(), _FakeAgentConfig(),
        memory_snapshot=snapshot,
    )

    assert ctx.memory_snapshot is snapshot
    assert ctx.memory_snapshot.user_chunks[0].id == "c1"


def test_empty_snapshot_sentinel_survives_threading() -> None:
    """``MemorySnapshot.empty()`` should thread through as-is (not be normalized to None)."""
    empty = MemorySnapshot.empty()
    ctx = build_render_context(
        _base_state(), _base_config(), _FakeAgentConfig(),
        memory_snapshot=empty,
    )
    assert ctx.memory_snapshot is empty
    assert ctx.memory_snapshot.is_empty() is True
