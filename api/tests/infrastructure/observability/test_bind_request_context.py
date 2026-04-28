"""B5 PR-S1-2 acceptance: ``bind_request_context`` async context manager (Q8).

Mirror of ``bind_session_context`` for the ``request_id`` field. Common
use case: a sub-flow inside a parent request needs its own ``request_id``
(fan-out, retry sub-spans) without losing the surrounding trace_id.
``ObservabilityMiddleware`` (PR-S1-5) installs the request-root context
via ``set_trace_context`` directly, so this helper is for nested flows
rather than the outermost middleware path.
"""
from __future__ import annotations

import re

import pytest

from app.domain.external.observability import TraceContext
from app.infrastructure.observability.context import (
    bind_request_context,
    get_trace_context,
    reset_trace_context,
    set_trace_context,
)

pytestmark = pytest.mark.anyio

_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
_REQUEST_OUTER = "00000000-0000-4000-8000-000000000001"
_REQUEST_INNER = "11111111-1111-4111-8111-111111111111"
_EVENT_ID = "00000000-0000-4000-8000-000000000005"
_SESSION_ID = "00000000-0000-4000-8000-000000000003"
_USER_HASH = "a1b2c3d4e5f60789"

_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _root_ctx() -> TraceContext:
    return TraceContext(
        trace_id=_TRACE_ID,
        request_id=_REQUEST_OUTER,
        event_id=_EVENT_ID,
        session_id=_SESSION_ID,
        user_id_hash=_USER_HASH,
    )


class TestFreshPath:
    async def test_no_context_creates_fresh_with_request_id(self):
        assert get_trace_context() is None

        async with bind_request_context(_REQUEST_INNER) as ctx:
            assert ctx.request_id == _REQUEST_INNER
            assert _HEX32_RE.match(ctx.trace_id)
            assert _UUID4_RE.match(ctx.event_id)
            assert get_trace_context() is ctx

        assert get_trace_context() is None


class TestReplacePath:
    async def test_existing_context_preserves_other_fields(self):
        token = set_trace_context(_root_ctx())
        try:
            async with bind_request_context(_REQUEST_INNER) as ctx:
                assert ctx.request_id == _REQUEST_INNER
                assert ctx.trace_id == _TRACE_ID
                assert ctx.event_id == _EVENT_ID
                assert ctx.session_id == _SESSION_ID
                assert ctx.user_id_hash == _USER_HASH
        finally:
            reset_trace_context(token)

    async def test_existing_context_restored_on_exit(self):
        outer = _root_ctx()
        token = set_trace_context(outer)
        try:
            async with bind_request_context(_REQUEST_INNER):
                current = get_trace_context()
                assert current is not None
                assert current.request_id == _REQUEST_INNER
            assert get_trace_context() is outer
        finally:
            reset_trace_context(token)


class TestExceptionResetsContext:
    async def test_exception_inside_block_still_resets(self):
        outer = _root_ctx()
        token = set_trace_context(outer)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                async with bind_request_context(_REQUEST_INNER):
                    current = get_trace_context()
                    assert current is not None
                    assert current.request_id == _REQUEST_INNER
                    raise RuntimeError("boom")
            assert get_trace_context() is outer
        finally:
            reset_trace_context(token)

    async def test_exception_in_fresh_path_resets_to_none(self):
        assert get_trace_context() is None
        with pytest.raises(ValueError, match="kaboom"):
            async with bind_request_context(_REQUEST_INNER):
                assert get_trace_context() is not None
                raise ValueError("kaboom")
        assert get_trace_context() is None
