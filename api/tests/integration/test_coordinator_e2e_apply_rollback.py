"""E2E: 2 work_units success -> during apply DIGEST_DRIFT on file 2 -> rollback file 1.

GAP NOTICE -- PR-9b
===================
This test is currently marked ``@pytest.mark.skip`` because the E2E
infrastructure is incomplete. The body below is the verbatim listing from
``docs/superpowers/plans/2026-05-25-c2-coordinator-task-runner-plan.md``
lines 9899-9979 -- preserved as living documentation of what the apply-
rollback path SHOULD verify once PR-9b lands. To enable, both groups must
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
    methods. ``before_apply_hook`` rewrites file2 between reducer-done
    and apply-start to drive the DIGEST_DRIFT branch.
  - ``fixture_mock_llm_3_workers`` -- LLM script harness with
    ``.setup_responses(planner_response, child_1_write_calls,
    child_2_write_calls, before_apply_hook=...)``. The ``before_apply_hook``
    keyword is required for this test specifically -- the fixture must
    invoke it (e.g. via a redis signal) after the two RESULT_READY
    envelopes land but before the applier preflight runs.
  - ``env_with_coordinator_flag_on`` -- monkeypatch
    ``ACTUS_C2_COORDINATOR_ENABLED=true``.

Deferred coordinator wiring (see ``service_dependencies.py:715-755`` TODO):
  - [BLOCKER -- PR-9b-D discovery] The child-runner PRODUCTION wiring itself is
    incomplete: ``service_dependencies.py:1126`` injects the BARE
    ``AgentTaskRunner`` class; the ``functools.partial(AgentTaskRunner, llm +
    ~10 required deps)`` was deferred (old plan "PR-5") and never shipped, so a
    flag-on child spawn raises ``TypeError``. This is the PRIMARY blocker and is
    OUT OF PR-9b scope -- tracked for the "C2 coordinator finish" follow-up epic.
  - ``SupervisorContext.cost_rollup_service`` (PR-6 §14.4) -- Protocol-only
    stub today.
  - ``SupervisorContext.coordinator_envelope_store`` (PR-7 §12.4) -- concrete
    impl exists at ``DbCoordinatorResultEnvelopeStoreRepository`` but the
    composition root leaves the field unset for atomic PR-9 flip.
  - ``PatchApplier._emit_event`` (PR-8 §13.5) -- live call site passes None,
    so the ``CoordinatorApplyEvent`` emit silently no-ops. This is the
    central deferred wiring for THIS test (without it the
    ``coordinator_apply`` digest_drift event never appears in the SSE
    stream).
  - ``CoordinatorRunOrchestrator._emit_event`` (PR-8 §13.6) -- live factory
    call passes no emit_event, so the ``CoordinatorSiblingCancelEvent`` emit
    silently no-ops.
  - ``reducer_node`` cost_total source (PR-8 R2 P2 inline comment in
    ``parallel_execution_subgraph.py:1180-1203``) -- ``ReducerOutput`` does
    not yet carry it.
  - ``CoordinatorApplyEvent`` lineage threading (PR-8 R1 P2 inline comment
    in ``patch_applier.py:790``) -- lineage fields wired through but the
    bind-back to runtime lineage values is deferred to PR-9. Without
    correct lineage, the audit-row assertion (``rollback_status='complete'``)
    cannot be cross-referenced against the SSE ``apply_status='digest_drift'``
    payload.

Acceptance: see ``CONTRIBUTING.md`` "C2 PR-9 Acceptance"
(added in Task 9.3).
"""
import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.coordinator_recovery]


@pytest.mark.skip(
    reason=(
        "[C2-E2E-DEFERRED] Coordinator live E2E is blocked on UNFINISHED "
        "PRODUCTION wiring, not just test infra: the child-runner dispatch is "
        "cold code -- service_dependencies.py:1126 injects the bare "
        "AgentTaskRunner class (the functools.partial binding llm + ~10 "
        "required deps was deferred to the old plan's 'PR-5' and never shipped), "
        "so a flag-on coordinator child spawn raises TypeError "
        "(child_agent_runner_factory.py:146 calls it with 4 kwargs; "
        "AgentTaskRunner.__init__ needs ~11). Re-enabling requires the dedicated "
        "'C2 coordinator finish' follow-up epic (production child-runner wiring "
        "+ fake-LLM injection seam + setup_responses / X-Test-User-Id / "
        "atomic_write_file harness gaps + CI MinIO/Docker provisioning). "
        "Locked honest-skip by "
        "tests/structure/test_coordinator_e2e_skip_honesty.py."
    )
)
async def test_e2e_apply_digest_drift_rollback(
    async_client, async_session, redis_real, minio_real, sandbox_real,
    fixture_mock_llm_3_workers, env_with_coordinator_flag_on,
):
    """2 work_unit write 完成后，apply 阶段 parent sandbox 已被外部修改 -> DIGEST_DRIFT
    -> file 1 已应用 -> rollback；最终 audit status='failed' + rollback_status='complete'."""
    user_id = "u_e2e_rb"
    # [r5 P1-1] live API: POST /api/sessions -> Response.success(data=CreateSessionResponse(session_id=...))
    create_resp = await async_client.post("/api/sessions", headers={"X-Test-User-Id": user_id})
    session_id = create_resp.json()["data"]["session_id"]

    await sandbox_real.atomic_write_file("/workspace/file1.py", b"orig1")
    await sandbox_real.atomic_write_file("/workspace/file2.py", b"orig2")

    fixture_mock_llm_3_workers.setup_responses(
        planner_response={
            "steps": [{
                "id": "step_rb_001",
                "description": "patch file1 + file2",
                "parallel_work_units": {"work_units": [
                    {"objective": "rewrite file1.py", "phase": "write",
                     "allowed_tools": ["file_read", "file_write"],
                     "proposed_paths": [{"path": "/workspace/file1.py", "op": "modify"}]},
                    {"objective": "rewrite file2.py", "phase": "write",
                     "allowed_tools": ["file_read", "file_write"],
                     "proposed_paths": [{"path": "/workspace/file2.py", "op": "modify"}]},
                ]},
            }],
        },
        child_1_write_calls=[{"tool": "file_write", "args": {
            "path": "/workspace/file1.py", "content": "new1"}}],
        child_2_write_calls=[{"tool": "file_write", "args": {
            "path": "/workspace/file2.py", "content": "new2"}}],
        # 在两个 RESULT_READY 之间但 apply 之前，模拟外部修改 file2 -> 触发 DIGEST_DRIFT
        # （fixture 通过 redis 信号在 reducer 完成后 / apply 开始前 hook 调用）
        before_apply_hook=lambda: sandbox_real.atomic_write_file(
            "/workspace/file2.py", b"EXTERNAL DRIFT"),
    )
    # [r6 P1-NEW] removed standalone POST -- _collect_sse_until_done posts chat internally
    events = await _collect_sse_until_done(async_client, session_id, timeout_s=120,
                                             message="patch file1 and file2")

    apply_events = [e for e in events if e["type"] == "coordinator_apply"]
    assert apply_events, "no coordinator_apply event"
    last_apply = apply_events[-1]["data"]
    assert last_apply["apply_status"] == "digest_drift", (
        f"expect digest_drift; got {last_apply['apply_status']}"
    )

    # rollback complete: file1 restored to "orig1"
    content1 = await sandbox_real.read_file("/workspace/file1.py")
    assert content1 == b"orig1", f"file1 not rolled back; got {content1!r}"

    # file2 keeps external drift (applier never modified it -- DIGEST_DRIFT on preflight)
    content2 = await sandbox_real.read_file("/workspace/file2.py")
    assert content2 == b"EXTERNAL DRIFT"

    # audit assertion: failed + rollback_complete
    async with async_session() as s:
        audit = (await s.execute(text("""
            SELECT status, rollback_status, failed_reason
            FROM coordinator_apply_audit
            WHERE coordinator_run_id LIKE :p ORDER BY started_at DESC LIMIT 1
        """), {"p": f"{session_id}:%"})).fetchone()
        assert audit is not None
        assert audit[0] == "failed", f"audit status={audit[0]}; expect 'failed'"
        assert audit[1] == "complete", f"rollback_status={audit[1]}; expect 'complete'"

    # 不出现 HealthEvent(rollback_partial)（complete rollback path）
    health_events = [e for e in events
                      if e["type"] == "health"
                      and "rollback_partial" in e["data"].get("code", "")]
    assert not health_events, "rollback_partial HealthEvent unexpectedly emitted"


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
