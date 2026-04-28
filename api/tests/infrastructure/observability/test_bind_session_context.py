"""B5 PR-S1-2 acceptance: ``bind_session_context`` async context manager (Q8).

The helper has two paths:

- **Replace path** (request scope): a ``TraceContext`` is already bound
  (set by ``ObservabilityMiddleware``), so binding ``session_id`` should
  preserve every other field and only replace ``session_id``.
- **Fresh path** (CLI / background / test scope): no context is bound,
  so the helper auto-generates ``trace_id`` / ``request_id`` /
  ``event_id`` (all v1-format-compliant) and uses the supplied
  ``session_id``.

Both paths must restore the prior binding on scope exit, including when
the body raises an exception.

The project's async test plumbing is anyio (``pytest.mark.anyio`` +
``anyio_backend`` fixture pinned to ``"asyncio"`` in
``api/tests/conftest.py``); we follow that convention here.
"""
from __future__ import annotations

import re

import pytest

from app.domain.external.observability import TraceContext
from app.infrastructure.observability.context import (
    bind_session_context,
    get_trace_context,
    reset_trace_context,
    set_trace_context,
)

pytestmark = pytest.mark.anyio

_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
_REQUEST_ID = "00000000-0000-4000-8000-000000000001"
_EVENT_ID = "00000000-0000-4000-8000-000000000005"
_SESSION_A = "00000000-0000-4000-8000-000000000003"
_SESSION_B = "11111111-1111-4111-8111-111111111111"
_USER_HASH = "a1b2c3d4e5f60789"

_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _root_ctx() -> TraceContext:
    return TraceContext(
        trace_id=_TRACE_ID,
        request_id=_REQUEST_ID,
        event_id=_EVENT_ID,
        user_id_hash=_USER_HASH,
    )


class TestFreshPath:
    async def test_no_context_creates_fresh_with_session_id(self):
        assert get_trace_context() is None

        async with bind_session_context(_SESSION_A) as ctx:
            assert ctx.session_id == _SESSION_A
            assert _HEX32_RE.match(ctx.trace_id)
            assert _UUID4_RE.match(ctx.request_id)
            assert _UUID4_RE.match(ctx.event_id)
            assert get_trace_context() is ctx

        assert get_trace_context() is None

    async def test_no_context_session_id_visible_to_get(self):
        async with bind_session_context(_SESSION_A):
            current = get_trace_context()
            assert current is not None
            assert current.session_id == _SESSION_A


class TestReplacePath:
    async def test_existing_context_preserves_other_fields(self):
        token = set_trace_context(_root_ctx())
        try:
            async with bind_session_context(_SESSION_A) as ctx:
                assert ctx.session_id == _SESSION_A
                assert ctx.trace_id == _TRACE_ID
                assert ctx.request_id == _REQUEST_ID
                assert ctx.event_id == _EVENT_ID
                assert ctx.user_id_hash == _USER_HASH
        finally:
            reset_trace_context(token)

    async def test_existing_context_restored_on_exit(self):
        outer = _root_ctx()
        token = set_trace_context(outer)
        try:
            async with bind_session_context(_SESSION_A):
                assert get_trace_context() is not outer
                current = get_trace_context()
                assert current is not None
                assert current.session_id == _SESSION_A
            assert get_trace_context() is outer
        finally:
            reset_trace_context(token)

    async def test_existing_session_id_overwritten(self):
        outer = TraceContext(
            trace_id=_TRACE_ID,
            request_id=_REQUEST_ID,
            event_id=_EVENT_ID,
            session_id=_SESSION_A,
        )
        token = set_trace_context(outer)
        try:
            async with bind_session_context(_SESSION_B) as inner:
                assert inner.session_id == _SESSION_B
                current = get_trace_context()
                assert current is not None
                assert current.session_id == _SESSION_B
            current = get_trace_context()
            assert current is not None
            assert current.session_id == _SESSION_A
        finally:
            reset_trace_context(token)


class TestExceptionResetsContext:
    async def test_exception_inside_block_still_resets(self):
        outer = _root_ctx()
        token = set_trace_context(outer)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                async with bind_session_context(_SESSION_A):
                    current = get_trace_context()
                    assert current is not None
                    assert current.session_id == _SESSION_A
                    raise RuntimeError("boom")
            assert get_trace_context() is outer
        finally:
            reset_trace_context(token)

    async def test_exception_in_fresh_path_resets_to_none(self):
        assert get_trace_context() is None
        with pytest.raises(ValueError, match="kaboom"):
            async with bind_session_context(_SESSION_A):
                assert get_trace_context() is not None
                raise ValueError("kaboom")
        assert get_trace_context() is None
