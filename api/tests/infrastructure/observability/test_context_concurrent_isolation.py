"""B5 PR-S1-2 acceptance: contextvars isolation under concurrency.

Locks the cross-task safety invariant from the design doc test plan:
"concurrent requests don't share trace_id". Python's ``ContextVar`` is
async-safe by design — each ``asyncio.Task`` snapshots the contextvar
map at creation time, so a binding made inside one task does not
leak into a sibling task — but we lock this here so a future refactor
that moves the contextvar into a class attribute or module-level
mutable singleton fails fast.
"""
from __future__ import annotations

import asyncio

import pytest

from app.domain.external.observability import TraceContext
from app.infrastructure.observability.context import (
    bind_request_context,
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
_SESSION_C = "22222222-2222-4222-8222-222222222222"
_REQUEST_X = "33333333-3333-4333-8333-333333333333"
_REQUEST_Y = "44444444-4444-4444-8444-444444444444"


def _root_ctx() -> TraceContext:
    return TraceContext(
        trace_id=_TRACE_ID,
        request_id=_REQUEST_ID,
        event_id=_EVENT_ID,
    )


class TestGatherIsolation:
    async def test_two_tasks_see_their_own_session_id(self):
        async def bind_and_observe(session_id: str) -> str:
            async with bind_session_context(session_id):
                # Yield the event loop so the other task interleaves.
                await asyncio.sleep(0)
                current = get_trace_context()
                assert current is not None
                assert current.session_id is not None
                return current.session_id

        results = await asyncio.gather(
            bind_and_observe(_SESSION_A),
            bind_and_observe(_SESSION_B),
            bind_and_observe(_SESSION_C),
        )

        assert results == [_SESSION_A, _SESSION_B, _SESSION_C]
        # No contextvar leak into the test scope.
        assert get_trace_context() is None

    async def test_two_tasks_see_their_own_request_id(self):
        async def bind_and_observe(request_id: str) -> str:
            async with bind_request_context(request_id):
                await asyncio.sleep(0)
                current = get_trace_context()
                assert current is not None
                return current.request_id

        results = await asyncio.gather(
            bind_and_observe(_REQUEST_X),
            bind_and_observe(_REQUEST_Y),
        )

        assert results == [_REQUEST_X, _REQUEST_Y]
        assert get_trace_context() is None


class TestParentBindingPropagatesToChild:
    async def test_child_task_inherits_parent_ctx_snapshot(self):
        """``asyncio.Task`` copies the contextvar map at creation. Child
        tasks should see the parent's binding by default but cannot
        mutate it for the parent.
        """
        token = set_trace_context(_root_ctx())
        try:

            async def child_observes() -> TraceContext | None:
                return get_trace_context()

            child_view = await asyncio.create_task(child_observes())
            assert child_view is not None
            assert child_view.trace_id == _TRACE_ID
            assert child_view.request_id == _REQUEST_ID
        finally:
            reset_trace_context(token)

    async def test_child_mutation_does_not_leak_to_parent(self):
        token = set_trace_context(_root_ctx())
        try:

            async def child_binds_session() -> str:
                async with bind_session_context(_SESSION_A):
                    inner = get_trace_context()
                    assert inner is not None
                    assert inner.session_id is not None
                    return inner.session_id

            child_session = await asyncio.create_task(child_binds_session())
            assert child_session == _SESSION_A

            # Parent's view is unchanged: never had session_id, still
            # does not.
            parent_view = get_trace_context()
            assert parent_view is not None
            assert parent_view.session_id is None
        finally:
            reset_trace_context(token)


class TestNestedBindings:
    async def test_request_then_session_both_visible(self):
        async with bind_request_context(_REQUEST_X) as outer:
            async with bind_session_context(_SESSION_A) as inner:
                # Inner ctx carries both fields; outer trace_id flows through.
                assert inner.request_id == _REQUEST_X
                assert inner.session_id == _SESSION_A
                assert inner.trace_id == outer.trace_id
                current = get_trace_context()
                assert current is inner
            # Inner reset → outer visible again with no session_id.
            current = get_trace_context()
            assert current is not None
            assert current.request_id == _REQUEST_X
            assert current.session_id is None
        assert get_trace_context() is None
