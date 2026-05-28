"""E2E happy path: 3 write work_units -> reduce success -> apply success. [r5 P2-2 修订: 3 write 不混 exploration]

依赖 fixtures（在 api/tests/integration/conftest.py 加）：
- async_client: FastAPI httpx test client
- async_session: SQLAlchemy AsyncSession factory
- redis_real: Redis 8.2 container (port-mapped)
- minio_real: MinIO real bucket
- sandbox_real: Docker sandbox per-test isolation
- fixture_mock_llm_3_workers: LLM responses scripted for planner + 3 children
- env_with_coordinator_flag_on: monkeypatch ACTUS_C2_COORDINATOR_ENABLED=true

GAP NOTICE -- PR-9b
===================
This test is currently marked ``@pytest.mark.skip`` because the E2E
infrastructure is incomplete. The body below is the verbatim listing from
``docs/superpowers/plans/2026-05-25-c2-coordinator-task-runner-plan.md``
lines 9669-9806 -- preserved as living documentation of what the happy path
SHOULD verify once PR-9b lands. To enable, both groups must close:

Missing fixtures (none defined in ``api/tests/integration/conftest.py``
beyond ``db_session`` / ``async_session_factory``):
  - ``async_client`` -- FastAPI ``httpx.AsyncClient`` mounted over the app
    with auth header injection for ``X-Test-User-Id``.
  - ``async_session`` -- per-test ``async with async_session() as s:`` factory
    (today only ``async_session_factory`` exists at conftest.py:82, which
    returns the ``async_sessionmaker`` callable -- semantics differ).
  - ``redis_real`` -- real Redis 8.2 container (port-mapped to host) for
    Streams / xreadgroup / xautoclaim paths.
  - ``minio_real`` -- real MinIO bucket (port-mapped) for patch storage
    handoffs.
  - ``sandbox_real`` -- Docker sandbox per-test isolation with
    ``.atomic_write_file(path, bytes)`` and ``.read_file(path)`` async
    methods.
  - ``fixture_mock_llm_3_workers`` -- LLM script harness with
    ``.setup_responses(planner_response, child_1_write_calls,
    child_2_write_calls, child_3_write_calls, before_apply_hook=...)``.
    Must support ``raise_after_iterations`` / ``sleep_seconds`` directives
    used by the sibling-cancel and apply-rollback siblings.
  - ``env_with_coordinator_flag_on`` -- monkeypatch
    ``ACTUS_C2_COORDINATOR_ENABLED=true``.

Deferred coordinator wiring (see ``service_dependencies.py:715-755`` TODO):
  - ``SupervisorContext.cost_rollup_service`` (PR-6 §14.4) -- Protocol-only
    stub today.
  - ``SupervisorContext.coordinator_envelope_store`` (PR-7 §12.4) -- concrete
    impl exists at ``DbCoordinatorResultEnvelopeStoreRepository`` but the
    composition root leaves the field unset for atomic PR-9 flip.
  - ``PatchApplier._emit_event`` (PR-8 §13.5) -- live call site passes None,
    so the ``CoordinatorApplyEvent`` emit silently no-ops.
  - ``CoordinatorRunOrchestrator._emit_event`` (PR-8 §13.6) -- live factory
    call passes no emit_event, so the ``CoordinatorSiblingCancelEvent`` emit
    silently no-ops.
  - ``reducer_node`` cost_total source (PR-8 R2 P2 inline comment in
    ``parallel_execution_subgraph.py:1180-1203``) -- ``ReducerOutput`` does
    not yet carry it; coalesce-to-default applied today, real shape lands
    here.
  - ``CoordinatorApplyEvent`` lineage threading (PR-8 R1 P2 inline comment
    in ``patch_applier.py:790``) -- lineage fields wired through but the
    bind-back to runtime lineage values is deferred to PR-9.

Acceptance: see ``CONTRIBUTING.md`` "C2 PR-9 Acceptance"
(added in Task 9.3).
"""
import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.coordinator_recovery]


@pytest.mark.skip(
    reason="PR-9b: fixture infrastructure pending -- see module docstring GAP NOTICE"
)
async def test_e2e_3_work_units_happy_path(
    async_client, async_session, redis_real, minio_real, sandbox_real,
    fixture_mock_llm_3_workers, env_with_coordinator_flag_on,
):
    """Happy path: 3 WRITE work_units (no exploration) -> all SUCCESS -> reducer
    builds PatchApplyPlan -> applier applies -> audit row=success + sandbox files
    contain new content.
    """
    user_id = "u_e2e_pr9"
    # [r5 P1-1] live API: POST /api/sessions (no body -- auth header is the user); returns Response.success(data=CreateSessionResponse(session_id=...))
    resp = await async_client.post("/api/sessions",
                                     headers={"X-Test-User-Id": user_id})
    assert resp.status_code == 200
    session_id = resp.json()["data"]["session_id"]  # [r5 P1-1] Response.success wraps in .data

    # seed parent sandbox files BEFORE LLM responses (children read these)
    await sandbox_real.atomic_write_file("/workspace/a.py", b"old a")
    await sandbox_real.atomic_write_file("/workspace/b.py", b"old b")
    await sandbox_real.atomic_write_file("/workspace/c.py", b"old c")

    fixture_mock_llm_3_workers.setup_responses(
        planner_response={
            "steps": [{
                "id": "step_e2e_001",
                "description": "patch 3 files in parallel",
                "parallel_work_units": {"work_units": [
                    {"objective": "rewrite /workspace/a.py", "phase": "write",
                     "allowed_tools": ["file_read", "file_write"],
                     "proposed_paths": [{"path": "/workspace/a.py", "op": "modify"}]},
                    {"objective": "rewrite /workspace/b.py", "phase": "write",
                     "allowed_tools": ["file_read", "file_write"],
                     "proposed_paths": [{"path": "/workspace/b.py", "op": "modify"}]},
                    {"objective": "rewrite /workspace/c.py", "phase": "write",
                     "allowed_tools": ["file_read", "file_write"],
                     "proposed_paths": [{"path": "/workspace/c.py", "op": "modify"}]},
                ]},
            }],
        },
        child_1_write_calls=[{"tool": "file_write", "args": {
            "path": "/workspace/a.py", "content": "new a"}}],
        child_2_write_calls=[{"tool": "file_write", "args": {
            "path": "/workspace/b.py", "content": "new b"}}],
        child_3_write_calls=[{"tool": "file_write", "args": {
            "path": "/workspace/c.py", "content": "new c"}}],
    )

    # [r7 P1-1] removed standalone POST -- _collect_sse_until_done helper posts /chat internally;
    # double-POST would trigger 2 coordinator runs and invalidate happy path assertions.
    events = await _collect_sse_until_done(async_client, session_id, timeout_s=120,
                                             message="patch a/b/c.py with new versions")

    # 1. lifecycle events present
    types = [e["type"] for e in events]
    assert "coordinator_dispatch" in types, f"missing dispatch in {set(types)}"
    assert types.count("coordinator_worker_spawned") == 3
    reduce_events = [e for e in events if e["type"] == "coordinator_reduce"]
    assert reduce_events, "no coordinator_reduce emitted"
    assert reduce_events[-1]["data"]["group_outcome"] == "success", (
        f"expect success; got {reduce_events[-1]['data']['group_outcome']}"
    )
    apply_events = [e for e in events if e["type"] == "coordinator_apply"]
    assert apply_events and apply_events[-1]["data"]["apply_status"] == "success"
    assert apply_events[-1]["data"]["file_count"] == 3

    # 2. DB: child rows + audit success + envelope_store has 3 RESULT_READY(success)
    async with async_session() as s:
        child_count = (await s.execute(text("""
            SELECT COUNT(*) FROM sessions
            WHERE parent_session_id=:p AND tool_filter_preset='coordinator_step'
        """), {"p": session_id})).scalar()
        assert child_count == 3

        audit_status = (await s.execute(text("""
            SELECT status FROM coordinator_apply_audit
            WHERE parent_session_id=:p ORDER BY started_at DESC LIMIT 1
        """), {"p": session_id})).scalar()
        assert audit_status == "success"

        envelopes = (await s.execute(text("""
            SELECT envelope_type, payload->>'outcome' FROM coordinator_result_envelope_store
            WHERE coordinator_run_id LIKE :p ORDER BY work_unit_id
        """), {"p": f"{session_id}:%"})).fetchall()
        assert len(envelopes) == 3
        for env_type, outcome in envelopes:
            assert env_type == "RESULT_READY"
            assert outcome == "success"

    # 3. sandbox files actually written with new content (apply real)
    for path, expected in (("/workspace/a.py", b"new a"),
                            ("/workspace/b.py", b"new b"),
                            ("/workspace/c.py", b"new c")):
        content = await sandbox_real.read_file(path)
        assert content == expected, f"{path}: got {content!r}"


async def _collect_sse_until_done(async_client, session_id, *, timeout_s=120,
                                    message="trigger coordinator"):
    """[r5 P1-1] Helper: POST /api/sessions/{id}/chat returns SSE stream directly
    (sse_starlette EventSourceResponse -- see session_routes.py:388). Collect events
    until 'done' (DoneEvent) or timeout.

    Replaces earlier (wrong) /sessions/{id}/events GET -- live `/events` is JSON
    recovery (NOT SSE).
    """
    import asyncio, json
    events = []
    async with async_client.stream(
        "POST", f"/api/sessions/{session_id}/chat",
        json={"message": message},
    ) as resp:
        try:
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    e = json.loads(line[5:].strip())
                    events.append(e)
                    if e.get("type") == "done":
                        return events
        except asyncio.TimeoutError:
            pass
    return events
