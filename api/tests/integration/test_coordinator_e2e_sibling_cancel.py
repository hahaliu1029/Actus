"""E2E: 1 work_unit FAILED -> orchestrator cancels remaining 2 -> reduce FAILED.

GAP NOTICE -- PR-9b
===================
This test is currently marked ``@pytest.mark.skip`` because the E2E
infrastructure is incomplete. The body below is the verbatim listing from
``docs/superpowers/plans/2026-05-25-c2-coordinator-task-runner-plan.md``
lines 9812-9894 -- preserved as living documentation of what the sibling-
cancel path SHOULD verify once PR-9b lands. To enable, both groups must
close:

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
    Must support ``raise_after_iterations`` (child_1 raises RuntimeError)
    and ``sleep_seconds`` (child_2/3 long-sleep so the cancel checkpoint
    converts them into CANCEL_ACK envelopes).
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
    silently no-ops. This is the central deferred wiring for THIS test
    (without it ``coordinator_sibling_cancel`` never appears in the SSE
    stream).
  - ``reducer_node`` cost_total source (PR-8 R2 P2 inline comment in
    ``parallel_execution_subgraph.py:1180-1203``) -- ``ReducerOutput`` does
    not yet carry it.
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
async def test_e2e_first_failed_cancels_siblings(
    async_client, async_session, redis_real, minio_real, sandbox_real,
    fixture_mock_llm_3_workers, env_with_coordinator_flag_on,
):
    user_id = "u_e2e_sib"
    # [r5 P1-1] live API: POST /api/sessions -> Response.success(data=CreateSessionResponse(session_id=...))
    create_resp = await async_client.post("/api/sessions", headers={"X-Test-User-Id": user_id})
    session_id = create_resp.json()["data"]["session_id"]
    await sandbox_real.atomic_write_file("/workspace/x.py", b"orig x")
    await sandbox_real.atomic_write_file("/workspace/y.py", b"orig y")
    await sandbox_real.atomic_write_file("/workspace/z.py", b"orig z")

    fixture_mock_llm_3_workers.setup_responses(
        planner_response={
            "steps": [{
                "id": "step_sib_001",
                "description": "patch x/y/z; child 1 will fail",
                "parallel_work_units": {"work_units": [
                    {"objective": "rewrite x.py (will fail)", "phase": "write",
                     "allowed_tools": ["file_read", "file_write"],
                     "proposed_paths": [{"path": "/workspace/x.py", "op": "modify"}]},
                    {"objective": "rewrite y.py (slow)", "phase": "write",
                     "allowed_tools": ["file_read", "file_write"],
                     "proposed_paths": [{"path": "/workspace/y.py", "op": "modify"}]},
                    {"objective": "rewrite z.py (slow)", "phase": "write",
                     "allowed_tools": ["file_read", "file_write"],
                     "proposed_paths": [{"path": "/workspace/z.py", "op": "modify"}]},
                ]},
            }],
        },
        # child 1: raise unhandled exception -> finalizer FAILED -> orchestrator triggers sibling cancel
        child_1_write_calls=[{"raise_after_iterations": 1, "exc_type": "RuntimeError",
                                "exc_msg": "intentional fail for E2E"}],
        # child 2/3: long sleep -> cancel checkpoint hits -> CANCEL_ACK
        child_2_write_calls=[{"sleep_seconds": 30}],
        child_3_write_calls=[{"sleep_seconds": 30}],
    )
    # [r6 P1-NEW] removed standalone POST -- _collect_sse_until_done posts chat internally
    # [r6 P1-NEW] helper internal posts /chat with given message; don't double-POST
    events = await _collect_sse_until_done(async_client, session_id, timeout_s=60,
                                             message="patch x/y/z in parallel")

    types = [e["type"] for e in events]
    assert "coordinator_sibling_cancel" in types, (
        f"sibling_cancel event not emitted; types={types}"
    )
    reduce_events = [e for e in events if e["type"] == "coordinator_reduce"]
    assert reduce_events and reduce_events[-1]["data"]["group_outcome"] in ("failed", "cancelled"), (
        f"expect failed/cancelled group_outcome; got {reduce_events[-1]['data']}"
    )

    async with async_session() as s:
        # at least 2 CANCEL_ACK envelopes (per siblings cancelled)
        cancelled = (await s.execute(text("""
            SELECT COUNT(*) FROM coordinator_result_envelope_store
            WHERE coordinator_run_id LIKE :p AND envelope_type='CANCEL_ACK'
              AND payload->>'final_state' = 'cancelled'
        """), {"p": f"{session_id}:%"})).scalar()
        assert cancelled >= 2, f"expect >=2 CANCEL_ACK(cancelled); got {cancelled}"

        # at least 1 RESULT_READY(failed)
        failed_count = (await s.execute(text("""
            SELECT COUNT(*) FROM coordinator_result_envelope_store
            WHERE coordinator_run_id LIKE :p AND envelope_type='RESULT_READY'
              AND payload->>'outcome' = 'failed'
        """), {"p": f"{session_id}:%"})).scalar()
        assert failed_count >= 1

    # sandbox files unchanged (all-or-nothing -- apply skipped due to non-success group)
    for path, original in (("/workspace/x.py", b"orig x"),
                            ("/workspace/y.py", b"orig y"),
                            ("/workspace/z.py", b"orig z")):
        content = await sandbox_real.read_file(path)
        assert content == original, f"{path} unexpectedly modified"


async def _collect_sse_until_done(async_client, session_id, *, timeout_s=120,
                                    message="trigger coordinator"):
    """Helper duplicated from ``test_coordinator_e2e_3_work_units.py`` to keep
    each E2E file self-contained while the suite remains skipped.
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
