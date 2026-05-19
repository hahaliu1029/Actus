"""Endpoint integration: auth + rate_limit + connection_limit + CS3 SSE wire.

These tests require live PostgreSQL + Redis (per
`CLAUDE.md` Local Test Infrastructure section). They share the
`tests/integration/conftest.py` fixture surface — `asgi_client`,
`sample_user_token`, `other_user_token`, `sample_user`,
`sample_session`, `redis_client`, `db_session`. The non-integration
`client` fixture at `tests/conftest.py:44` is sync `TestClient` and
CANNOT be used with `httpx.AsyncClient.stream(...)`; do not swap.

Per the project's anyio convention these tests use `pytest.mark.anyio`
(not `pytest.mark.asyncio`), and `pytest.mark.integration` so the local
CI gate can scope to `-m "not integration"` when pg/redis are absent.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


# ----------------------------------------------------------------------
# Per-test cleanup of dependency_overrides
# ----------------------------------------------------------------------
#
# ``app`` is the same FastAPI singleton across every test in this
# module — leaving entries in ``app.dependency_overrides`` after a
# test silently leaks the stubs into sibling tests (including the
# auth override the integration conftest installs). Snapshot before
# each test and restore after so the stub installer below can stay
# helper-flavoured instead of forcing every test to remember its own
# cleanup.


@pytest.fixture(autouse=True)
def _restore_dependency_overrides(app):
    snapshot = dict(app.dependency_overrides)
    try:
        yield
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(snapshot)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _install_stub_research_service(app, events=None, raise_exc=None):
    """Override `get_subagent_research_service` with a stub that yields
    a deterministic event sequence (or raises a known exception).

    Returns the stub so tests can assert on calls.
    """
    from app.interfaces.service_dependencies import (
        get_subagent_research_service,
    )

    stub = MagicMock()

    async def _run_research(**_kwargs):
        if raise_exc is not None:
            raise raise_exc
        for ev in events or []:
            yield ev

    stub.run_research = _run_research
    app.dependency_overrides[get_subagent_research_service] = lambda: stub
    return stub


# ----------------------------------------------------------------------
# Auth + ownership
# ----------------------------------------------------------------------


async def test_endpoint_requires_parent_ownership(
    asgi_client, sample_user_token
):
    """POST with non-existent sample_session_id → 404 (not 500)."""
    response = await asgi_client.post(
        "/api/sessions/00000000-0000-0000-0000-000000000000/subagents/research",
        json={"prompts": ["test"], "max_children": 1},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert response.status_code == 404


async def test_endpoint_cross_tenant_returns_404(
    asgi_client, other_user_token, sample_session
):
    """User B accessing User A's parent → 404 (existence not leaked)."""
    response = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/subagents/research",
        json={"prompts": ["test"], "max_children": 1},
        headers={"Authorization": f"Bearer {other_user_token}"},
    )
    assert response.status_code == 404


# ----------------------------------------------------------------------
# Request validation
# ----------------------------------------------------------------------


async def test_endpoint_validates_max_children_upper_bound(
    asgi_client, sample_user_token, sample_session
):
    """max_children > 3 → 422 Unprocessable (Pydantic Field constraint)."""
    response = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/subagents/research",
        json={"prompts": ["test"], "max_children": 10},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert response.status_code == 422


async def test_endpoint_validates_prompts_minimum(
    asgi_client, sample_user_token, sample_session
):
    """Empty prompts list → 422 (Pydantic min_length=1)."""
    response = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/subagents/research",
        json={"prompts": [], "max_children": 1},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert response.status_code == 422


async def test_endpoint_validates_prompts_maximum(
    asgi_client, sample_user_token, sample_session
):
    """prompts list > 3 → 422 (Pydantic max_length=3)."""
    response = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/subagents/research",
        json={"prompts": ["a", "b", "c", "d"], "max_children": 3},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert response.status_code == 422


# ----------------------------------------------------------------------
# CS3 SSE wire contract
# ----------------------------------------------------------------------


async def test_endpoint_sse_frame_id_matches_payload_event_id(
    app, asgi_client, sample_user_token, sample_session
):
    """CS3 invariant: SSE frame.id == payload.event_id (== domain event.id).

    Per `app/interfaces/schemas/event.py:43` the wire payload field is
    named `event_id` (not `id`). The endpoint constructs the SSE frame
    via `ServerSentEvent(id=event.id, ...)`, and the data dict is built
    by `BaseEventData.from_event(event)` which sets `event_id=event.id`.
    Both sides reference the same UUID — that's the invariant.
    """
    from app.interfaces.schemas.subagent import ChildStartedEvent

    started = ChildStartedEvent(
        probe_run_id="run-abc",
        child_session_id="child-1",
        prompt="hi",
    )

    _install_stub_research_service(app, events=[started])

    async with asgi_client.stream(
        "POST",
        f"/api/sessions/{sample_session.id}/subagents/research",
        json={"prompts": ["minimal"], "max_children": 1},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    ) as response:
        assert response.status_code == 200

        seen_frame_id = None
        seen_payload_event_id = None
        async for chunk in response.aiter_lines():
            if chunk.startswith("id:"):
                seen_frame_id = chunk.removeprefix("id:").strip()
            elif chunk.startswith("data:") and seen_frame_id:
                payload = json.loads(chunk.removeprefix("data:").strip())
                seen_payload_event_id = payload.get("event_id")
                break

    assert seen_frame_id is not None, "no SSE frame id observed"
    assert seen_payload_event_id == seen_frame_id, (
        f"CS3 violation: frame id={seen_frame_id} payload event_id="
        f"{seen_payload_event_id}"
    )
    assert seen_frame_id == started.id


async def test_endpoint_streams_all_three_phase1_events(
    app, asgi_client, sample_user_token, sample_session
):
    """Endpoint passes through ChildStarted / ChildDone / JoinedSummary.

    CommonSSEEvent fallback with `extra="allow"` keeps probe-specific
    fields (`probe_run_id`, `child_session_id`, `metrics`, ...) on the
    wire payload — verified by checking event type strings flow through.
    """
    from app.interfaces.schemas.subagent import (
        ChildDoneEvent,
        ChildOutcome,
        ChildStartedEvent,
        JoinedSummaryEvent,
    )

    events = [
        ChildStartedEvent(
            probe_run_id="run-1",
            child_session_id="c1",
            prompt="q1",
        ),
        ChildDoneEvent(
            probe_run_id="run-1",
            child_session_id="c1",
            outcome=ChildOutcome.COMPLETED,
            final_answer="answer-1",
            transcript_tokens=42,
        ),
        JoinedSummaryEvent(
            probe_run_id="run-1",
            summary="integrated summary",
            summary_tokens=120,
            completed_children=["c1"],
            metrics={"latency_ms": 1234, "success_rate": 1.0},
        ),
    ]
    _install_stub_research_service(app, events=events)

    observed_event_types: list[str] = []
    async with asgi_client.stream(
        "POST",
        f"/api/sessions/{sample_session.id}/subagents/research",
        json={"prompts": ["q1"], "max_children": 1},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    ) as response:
        assert response.status_code == 200
        async for chunk in response.aiter_lines():
            if chunk.startswith("event:"):
                observed_event_types.append(
                    chunk.removeprefix("event:").strip()
                )

    assert "child_started" in observed_event_types
    assert "child_done" in observed_event_types
    assert "joined_summary" in observed_event_types


# ----------------------------------------------------------------------
# Quota / preflight errors surface correctly
# ----------------------------------------------------------------------


async def test_endpoint_conflict_when_quota_exceeded(
    app, asgi_client, sample_user_token, sample_session
):
    """ConflictError raised by service before first yield → 409."""
    from app.application.errors.exceptions import ConflictError

    _install_stub_research_service(
        app,
        raise_exc=ConflictError("Active probe quota exceeded"),
    )

    response = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/subagents/research",
        json={"prompts": ["x"], "max_children": 1},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert response.status_code == 409


async def test_endpoint_bad_request_when_classifier_rejects(
    app, asgi_client, sample_user_token, sample_session
):
    """BadRequestError raised by service before first yield → 400."""
    from app.application.errors.exceptions import BadRequestError

    _install_stub_research_service(
        app,
        raise_exc=BadRequestError("Preflight rejected: prompt 1 ..."),
    )

    response = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/subagents/research",
        json={"prompts": ["write file foo.py"], "max_children": 1},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert response.status_code == 400


# ----------------------------------------------------------------------
# Connection lease lifecycle
# ----------------------------------------------------------------------


async def test_endpoint_connection_lease_released_on_normal_close(
    app, asgi_client, sample_user_token, sample_session
):
    """SSE generator's finally must call lease.release() exactly once.

    The endpoint invokes `acquire_connection_limit` directly as a
    free function inside the route body (not via FastAPI `Depends`),
    so we monkey-patch the module-level symbol to inject a stub lease.
    """
    import app.interfaces.endpoints.session_routes as routes_mod
    from app.interfaces.schemas.subagent import ChildStartedEvent

    release_calls: list[None] = []
    captured_lease = MagicMock()
    captured_lease.start_heartbeat = MagicMock()

    async def _release():
        release_calls.append(None)

    captured_lease.release = _release

    async def _fake_acquire(**_kwargs):
        return captured_lease

    original_acquire = routes_mod.acquire_connection_limit
    routes_mod.acquire_connection_limit = _fake_acquire
    try:
        _install_stub_research_service(
            app,
            events=[
                ChildStartedEvent(
                    probe_run_id="r1",
                    child_session_id="c1",
                    prompt="hi",
                ),
            ],
        )

        async with asgi_client.stream(
            "POST",
            f"/api/sessions/{sample_session.id}/subagents/research",
            json={"prompts": ["x"], "max_children": 1},
            headers={"Authorization": f"Bearer {sample_user_token}"},
        ) as response:
            assert response.status_code == 200
            # Drain the stream to completion so the generator's
            # `finally` block runs.
            async for _ in response.aiter_lines():
                pass
    finally:
        routes_mod.acquire_connection_limit = original_acquire

    assert release_calls, "lease.release() was not invoked"
    assert len(release_calls) == 1, (
        f"lease.release() invoked {len(release_calls)}× — must be exactly 1"
    )


async def test_endpoint_lease_released_on_preflight_failure(
    app, asgi_client, sample_user_token, sample_session
):
    """Codex R4 regression: preflight failure (ConflictError) must run
    `_drain_subagent_cleanup` to completion → lease.release() called once.

    Locks in the contract that the priming-failure path drains both
    cleanups before re-raising; if a future refactor removes the call
    to `_drain_subagent_cleanup` from the `except BaseException` block,
    this test goes red even though the HTTP status is still 409.
    """
    import app.interfaces.endpoints.session_routes as routes_mod
    from app.application.errors.exceptions import ConflictError

    release_calls: list[None] = []
    captured_lease = MagicMock()
    captured_lease.start_heartbeat = MagicMock()

    async def _release():
        release_calls.append(None)

    captured_lease.release = _release

    async def _fake_acquire(**_kwargs):
        return captured_lease

    original_acquire = routes_mod.acquire_connection_limit
    routes_mod.acquire_connection_limit = _fake_acquire
    try:
        _install_stub_research_service(
            app,
            raise_exc=ConflictError("Active probe quota exceeded"),
        )

        response = await asgi_client.post(
            f"/api/sessions/{sample_session.id}/subagents/research",
            json={"prompts": ["x"], "max_children": 1},
            headers={"Authorization": f"Bearer {sample_user_token}"},
        )
        assert response.status_code == 409
    finally:
        routes_mod.acquire_connection_limit = original_acquire

    assert release_calls, (
        "lease.release() was not invoked on preflight-failure cleanup path"
    )
    assert len(release_calls) == 1, (
        f"lease.release() invoked {len(release_calls)}× on preflight "
        "failure — must be exactly 1"
    )


# Cancel-storm regression for `_drain_subagent_cleanup` lives at
# tests/app/interfaces/endpoints/test_subagent_drain_helper.py — pure
# asyncio test, kept out of this integration file so it can run without
# pg/redis and without the integration autouse fixture chain.
