"""[C2 finish-core PR-F5] Unskipped + enriched: drives the live app via the injected routing fake LLM + real JWT + the session sandbox adapter (subdir-relative paths). Runs flag-ON in the coordinator-e2e CI job only (needs pg+redis+minio+sandbox).

Verifies the happy path: 3 WRITE work_units dispatched in parallel -> 3 workers
spawned -> reducer group_outcome=success (cost_total>0) -> applier writes all 3
files -> DB has 3 child rows + audit=success + 3 RESULT_READY(success) envelopes.
"""
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


async def test_e2e_3_work_units_happy_path(
    inject_routing_fake_llm, coord_async_client, async_session, redis_real, minio_real,
    coord_jwt_headers, fresh_test_user, env_with_coordinator_flag_on,
):
    from tests.integration.coordinator_fixtures import bind_session_sandbox_adapter
    async_client = coord_async_client
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "step_e2e_001", "description": "patch 3 files in parallel",
            "parallel_work_units": {"work_units": [
                {"objective": "rewrite workspace/a.py", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/a.py", "op": "modify"}]},
                {"objective": "rewrite workspace/b.py", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/b.py", "op": "modify"}]},
                {"objective": "rewrite workspace/c.py", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/c.py", "op": "modify"}]},
            ]},
        }]},
        child_responses={
            "rewrite workspace/a.py": [{"tool": "file_write", "args": {"filepath": "workspace/a.py", "content": "new a"}}],
            "rewrite workspace/b.py": [{"tool": "file_write", "args": {"filepath": "workspace/b.py", "content": "new b"}}],
            "rewrite workspace/c.py": [{"tool": "file_write", "args": {"filepath": "workspace/c.py", "content": "new c"}}],
        },
    )

    resp = await async_client.post("/api/sessions", headers=coord_jwt_headers)
    assert resp.status_code == 200
    session_id = resp.json()["data"]["session_id"]

    import app.main as app_main
    coord_sandbox_adapter = await bind_session_sandbox_adapter(
        app_main.app, session_id, str(fresh_test_user.id),
    )
    await coord_sandbox_adapter.atomic_write_file("workspace/a.py", b"old a")
    await coord_sandbox_adapter.atomic_write_file("workspace/b.py", b"old b")
    await coord_sandbox_adapter.atomic_write_file("workspace/c.py", b"old c")

    events = await _collect_sse_until_done(
        async_client, session_id, headers=coord_jwt_headers, timeout_s=120,
        message="patch a/b/c with new versions",
    )

    types = [e["type"] for e in events]
    assert "coordinator_dispatch" in types
    assert types.count("coordinator_worker_spawned") == 3
    reduce_events = [e for e in events if e["type"] == "coordinator_reduce"]
    assert reduce_events and reduce_events[-1]["data"]["group_outcome"] == "success"
    assert reduce_events[-1]["data"]["cost_total"]["total_usd"] > 0
    apply_events = [e for e in events if e["type"] == "coordinator_apply"]
    assert apply_events and apply_events[-1]["data"]["apply_status"] == "success"
    assert apply_events[-1]["data"]["file_count"] == 3
    ad = apply_events[-1]["data"]
    assert ad.get("coordinator_run_id") and ad.get("child_session_id") in (None, "")

    for path, expected in (("workspace/a.py", b"new a"), ("workspace/b.py", b"new b"), ("workspace/c.py", b"new c")):
        assert await coord_sandbox_adapter.read_file(path) == expected

    # KEEP existing DB assertions (child rows + audit success + 3 RESULT_READY):
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
