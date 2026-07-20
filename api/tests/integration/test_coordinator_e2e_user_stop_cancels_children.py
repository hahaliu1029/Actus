"""[C2 coordinator-cancel] e2e: user-stop of a coordinator parent immediately
cancels dispatched children. Runs flag-ON in the coordinator-e2e CI job only
(needs pg+redis+minio+sandbox).

Asserts on durable mailbox audit + terminal envelope storage after the raw
mailbox streams have been cleaned: one CANCEL_REQUEST per running child +
CANCEL_ACK(cancelled) present + ZERO RESULT_READY(success) for the cancelled
children; child rows reach a terminal status; resolves well under the 300s
watchdog.
"""
import anyio
import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.coordinator_recovery]


async def _drive_chat(async_client, session_id, headers, message):
    # Best-effort SSE pump. The children sleep, so `done` will not arrive before
    # we stop the parent; the caller cancels this task after issuing /stop.
    try:
        async with async_client.stream(
            "POST", f"/api/sessions/{session_id}/chat",
            json={"message": message}, headers=headers,
        ) as resp:
            async for _line in resp.aiter_lines():
                pass
    except anyio.get_cancelled_exc_class():
        raise
    except Exception:
        return


async def _count_running_children(async_session, root_id):
    async with async_session() as s:
        return (await s.execute(text(
            "SELECT COUNT(*) FROM sessions WHERE parent_session_id = :p "
            "AND worker_type='subagent' AND subagent_control_plane='mailbox' "
            "AND status='running' AND tool_filter_preset='coordinator_step'"
        ), {"p": root_id})).scalar()


async def test_e2e_user_stop_cancels_children(
    inject_routing_fake_llm, coord_async_client, async_session, redis_real, minio_real,
    coord_jwt_headers, fresh_test_user, env_with_coordinator_flag_on,
):
    from tests.integration.coordinator_fixtures import bind_session_sandbox_adapter
    async_client = coord_async_client
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "step_stop_001", "description": "patch y/z; both slow",
            "parallel_work_units": {"work_units": [
                {"objective": "rewrite workspace/y.py (slow)", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/y.py", "op": "modify"}]},
                {"objective": "rewrite workspace/z.py (slow)", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/z.py", "op": "modify"}]},
            ]},
        }]},
        child_responses={
            "rewrite workspace/y.py (slow)": [{"sleep_seconds": 60}],
            "rewrite workspace/z.py (slow)": [{"sleep_seconds": 60}],
        },
    )
    resp = await async_client.post("/api/sessions", headers=coord_jwt_headers)
    session_id = resp.json()["data"]["session_id"]
    import app.main as app_main
    coord_sandbox_adapter = await bind_session_sandbox_adapter(
        app_main.app, session_id, str(fresh_test_user.id),
    )
    for p, c in (("workspace/y.py", b"orig y"), ("workspace/z.py", b"orig z")):
        await coord_sandbox_adapter.atomic_write_file(p, c)

    start = anyio.current_time()
    async with anyio.create_task_group() as tg:
        tg.start_soon(_drive_chat, async_client, session_id, coord_jwt_headers,
                      "patch y/z in parallel")
        # Poll until both children are RUNNING (dispatched), then stop the parent.
        with anyio.fail_after(90):
            while await _count_running_children(async_session, session_id) < 2:
                await anyio.sleep(0.5)
        stop_resp = await async_client.post(
            f"/api/sessions/{session_id}/stop", headers=coord_jwt_headers,
        )
        assert stop_resp.status_code in (200, 202)
        tg.cancel_scope.cancel()  # tear down the SSE pump
    elapsed = anyio.current_time() - start
    assert elapsed < 300  # resolved well under the 300s watchdog

    # Root terminal cleanup intentionally deletes the ephemeral Redis stream.
    # Assert on durable consumer audit + terminal envelope storage, waiting for
    # both independent child finalizers to settle.
    cancel_req_children = 0
    cancel_acks = 0
    with anyio.fail_after(15):
        while cancel_req_children < 2 or cancel_acks < 2:
            async with async_session() as s:
                cancel_req_children = (await s.execute(text("""
                    SELECT COUNT(DISTINCT child_session_id)
                    FROM mailbox_envelope_audit
                    WHERE parent_session_id=:p AND type='CANCEL_REQUEST'
                """), {"p": session_id})).scalar()
                cancel_acks = (await s.execute(text("""
                    SELECT COUNT(*) FROM coordinator_result_envelope_store
                    WHERE coordinator_run_id LIKE :p AND envelope_type='CANCEL_ACK'
                      AND payload->>'final_state' = 'cancelled'
                """), {"p": f"{session_id}:%"})).scalar()
            if cancel_req_children < 2 or cancel_acks < 2:
                await anyio.sleep(0.2)

    async with async_session() as s:
        success = (await s.execute(text("""
            SELECT COUNT(*) FROM coordinator_result_envelope_store
            WHERE coordinator_run_id LIKE :p AND envelope_type='RESULT_READY'
              AND payload->>'outcome' = 'success'
        """), {"p": f"{session_id}:%"})).scalar()
    assert success == 0

    async with async_session() as s:
        statuses = [r[0] for r in (await s.execute(text(
            "SELECT status FROM sessions WHERE parent_session_id=:p "
            "AND worker_type='subagent' AND tool_filter_preset='coordinator_step'"
        ), {"p": session_id})).all()]
    assert statuses and all(st in ("completed", "timed_out") for st in statuses)
