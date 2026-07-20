"""[C2 finish-core PR-F5] Unskipped + enriched: drives the live app via the injected routing fake LLM + real JWT + the session sandbox adapter (subdir-relative paths). Runs flag-ON in the coordinator-e2e CI job only (needs pg+redis+minio+sandbox).

Verifies sibling-cancel: child 1 raises -> orchestrator cancels the 2 slow
siblings -> coordinator_sibling_cancel emitted + reducer group_outcome in
(failed, cancelled) + >=2 CANCEL_ACK(cancelled) + >=1 RESULT_READY(failed)
envelopes; sandbox files stay at their original content (apply skipped).
"""
import anyio
import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.coordinator_recovery]


async def _collect_sse_until_done(async_client, session_id, *, headers=None,
                                    timeout_s=120, message="trigger coordinator"):
    # The real SSE wire is `event: <type>\ndata: <payload-without-type>`
    # (session_routes.py emits ServerSentEvent(event=sse_event.event, data=...);
    # the data model EXCLUDES type). Track the `event:` line and rebuild
    # {"type": <name>, "data": <payload>}.
    import json
    import anyio
    events, current_event = [], None
    # fail_after converts a hung stream (no `done`) into a fast TimeoutError
    # instead of blocking the CI job until its global timeout.
    with anyio.fail_after(timeout_s):
        async with async_client.stream(
            "POST", f"/api/sessions/{session_id}/chat",
            json={"message": message}, headers=headers or {},
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    payload = json.loads(line[len("data:"):].strip())
                    events.append({"type": current_event, "data": payload})
                    if current_event == "done":
                        return events
    return events


async def test_e2e_first_failed_cancels_siblings(
    inject_routing_fake_llm, coord_async_client, async_session, redis_real, minio_real,
    coord_jwt_headers, fresh_test_user, env_with_coordinator_flag_on,
):
    from tests.integration.coordinator_fixtures import bind_session_sandbox_adapter
    async_client = coord_async_client
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "step_sib_001", "description": "patch x/y/z; child 1 fails",
            "parallel_work_units": {"work_units": [
                {"objective": "rewrite workspace/x.py (will fail)", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/x.py", "op": "modify"}]},
                {"objective": "rewrite workspace/y.py (slow)", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/y.py", "op": "modify"}]},
                {"objective": "rewrite workspace/z.py (slow)", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/z.py", "op": "modify"}]},
            ]},
        }]},
        child_responses={
            "rewrite workspace/x.py (will fail)": [{"raise_after_iterations": 1, "exc_type": "RuntimeError", "exc_msg": "intentional fail for E2E"}],
            "rewrite workspace/y.py (slow)": [{"sleep_seconds": 30}],
            "rewrite workspace/z.py (slow)": [{"sleep_seconds": 30}],
        },
    )
    resp = await async_client.post("/api/sessions", headers=coord_jwt_headers)
    session_id = resp.json()["data"]["session_id"]
    import app.main as app_main
    coord_sandbox_adapter = await bind_session_sandbox_adapter(
        app_main.app, session_id, str(fresh_test_user.id),
    )
    for p, c in (("workspace/x.py", b"orig x"), ("workspace/y.py", b"orig y"), ("workspace/z.py", b"orig z")):
        await coord_sandbox_adapter.atomic_write_file(p, c)
    events = await _collect_sse_until_done(
        async_client, session_id, headers=coord_jwt_headers, timeout_s=60,
        message="patch x/y/z in parallel",
    )
    types = [e["type"] for e in events]
    reduce_events = [e for e in events if e["type"] == "coordinator_reduce"]
    assert "coordinator_sibling_cancel" in types
    assert reduce_events[-1]["data"]["group_outcome"] in ("failed", "cancelled")

    # The reduce event may reach SSE while the final sibling is still
    # publishing its CANCEL_ACK. Wait for the durable terminal store instead
    # of racing that independent finalizer task.
    cancelled = 0
    with anyio.fail_after(15):
        while cancelled < 2:
            async with async_session() as s:
                cancelled = (await s.execute(text("""
                    SELECT COUNT(*) FROM coordinator_result_envelope_store
                    WHERE coordinator_run_id LIKE :p AND envelope_type='CANCEL_ACK'
                      AND payload->>'final_state' = 'cancelled'
                """), {"p": f"{session_id}:%"})).scalar()
            if cancelled < 2:
                await anyio.sleep(0.2)

    async with async_session() as s:
        # at least 1 RESULT_READY(failed)
        failed_count = (await s.execute(text("""
            SELECT COUNT(*) FROM coordinator_result_envelope_store
            WHERE coordinator_run_id LIKE :p AND envelope_type='RESULT_READY'
              AND payload->>'outcome' = 'failed'
        """), {"p": f"{session_id}:%"})).scalar()
        assert failed_count >= 1

    for path, original in (("workspace/x.py", b"orig x"), ("workspace/y.py", b"orig y"), ("workspace/z.py", b"orig z")):
        assert await coord_sandbox_adapter.read_file(path) == original
