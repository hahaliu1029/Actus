"""B5 PR-S1-2 acceptance: ``set_trace_context`` / ``get_trace_context`` /
``reset_trace_context`` low-level helpers.

Locks the contract that ``set_trace_context`` returns a ``Token`` that
``reset_trace_context`` can use to restore the previous binding,
matching the behavior of ``contextvars.ContextVar.set`` /
``ContextVar.reset``.

The middleware wiring (PR-S1-5) uses these helpers directly because
``BaseHTTPMiddleware`` cannot host a pure ASGI send-wrapper around the
SSE generator — it needs explicit ``set`` / ``reset`` around
``await self.app(...)``.
"""
from __future__ import annotations

from app.domain.external.observability import TraceContext
from app.infrastructure.observability.context import (
    get_trace_context,
    reset_trace_context,
    set_trace_context,
)

_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
_REQUEST_ID = "00000000-0000-4000-8000-000000000001"
_EVENT_ID = "00000000-0000-4000-8000-000000000005"


def _ctx(**overrides) -> TraceContext:
    base: dict[str, object] = {
        "trace_id": _TRACE_ID,
        "request_id": _REQUEST_ID,
        "event_id": _EVENT_ID,
    }
    base.update(overrides)
    return TraceContext(**base)  # type: ignore[arg-type]


class TestGetReturnsNoneByDefault:
    def test_default_is_none(self):
        assert get_trace_context() is None


class TestSetGetCycle:
    def test_set_then_get_returns_same_object(self):
        ctx = _ctx(session_id="00000000-0000-4000-8000-000000000003")
        token = set_trace_context(ctx)
        try:
            assert get_trace_context() is ctx
        finally:
            reset_trace_context(token)

    def test_set_to_none_clears_binding(self):
        ctx = _ctx()
        token1 = set_trace_context(ctx)
        try:
            assert get_trace_context() is ctx
            token2 = set_trace_context(None)
            try:
                assert get_trace_context() is None
            finally:
                reset_trace_context(token2)
            assert get_trace_context() is ctx
        finally:
            reset_trace_context(token1)


class TestResetRestoresPrevious:
    def test_reset_restores_none_default(self):
        ctx = _ctx()
        token = set_trace_context(ctx)
        assert get_trace_context() is ctx

        reset_trace_context(token)
        assert get_trace_context() is None

    def test_reset_restores_previous_binding(self):
        outer = _ctx(session_id="11111111-1111-4111-8111-111111111111")
        inner = _ctx(session_id="22222222-2222-4222-8222-222222222222")

        outer_token = set_trace_context(outer)
        try:
            assert get_trace_context() is outer
            inner_token = set_trace_context(inner)
            try:
                assert get_trace_context() is inner
            finally:
                reset_trace_context(inner_token)
            assert get_trace_context() is outer
        finally:
            reset_trace_context(outer_token)
        assert get_trace_context() is None
