"""[C2 finish-core PR-F5] Unskipped + enriched: drives the live app via the injected routing fake LLM + real JWT + the session sandbox adapter (subdir-relative paths). Runs flag-ON in the coordinator-e2e CI job only (needs pg+redis+minio+sandbox).

Verifies mid-apply rollback-restore: both children succeed, then the applier's
2nd minio patch fetch is injected to fail -> file-1 is applied then rolled back
-> file-1 restored to original + audit failed/rollback_status=complete + NO
rollback_partial HealthEvent.
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


async def test_e2e_apply_midapply_rollback_restores(
    inject_routing_fake_llm, coord_async_client, async_session, redis_real, minio_real,
    coord_jwt_headers, fresh_test_user, env_with_coordinator_flag_on, monkeypatch,
):
    """Mid-apply fault (file-2 minio fetch fails) -> file-1 applied then rolled
    back -> audit failed + rollback_status='complete' + NO rollback_partial."""
    from tests.integration.coordinator_fixtures import bind_session_sandbox_adapter
    async_client = coord_async_client
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "step_rb_001", "description": "patch file1 + file2",
            "parallel_work_units": {"work_units": [
                {"objective": "rewrite workspace/file1.py", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/file1.py", "op": "modify"}]},
                {"objective": "rewrite workspace/file2.py", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/file2.py", "op": "modify"}]},
            ]},
        }]},
        child_responses={
            "rewrite workspace/file1.py": [{"tool": "file_write", "args": {"filepath": "workspace/file1.py", "content": "new1"}}],
            "rewrite workspace/file2.py": [{"tool": "file_write", "args": {"filepath": "workspace/file2.py", "content": "new2"}}],
        },
    )
    # Inject the mid-apply fault at the INSTANCE the applier actually uses:
    # cfg["artifact_storage"] == app.state.coord_deps.artifact_storage. The frozen
    # dataclass field can't be reassigned, but the instance's get_bytes METHOD can
    # be patched. Seed ("/seed/") fetches pass through; the 2nd "/patch/" fetch
    # raises -> applier applies file-1 then fails on file-2 -> rollback restores file-1.
    import app.main as app_main
    store = app_main.app.state.coord_deps.artifact_storage
    orig_get_bytes = store.get_bytes
    _fetches = {"n": 0}
    async def _faulty_get_bytes(ref):
        if "/patch/" in ref:
            _fetches["n"] += 1
            if _fetches["n"] >= 2:
                raise RuntimeError("injected mid-apply minio fetch failure")
        return await orig_get_bytes(ref)
    monkeypatch.setattr(store, "get_bytes", _faulty_get_bytes)

    resp = await async_client.post("/api/sessions", headers=coord_jwt_headers)
    session_id = resp.json()["data"]["session_id"]
    coord_sandbox_adapter = await bind_session_sandbox_adapter(
        app_main.app, session_id, str(fresh_test_user.id),
    )
    await coord_sandbox_adapter.atomic_write_file("workspace/file1.py", b"orig1")
    await coord_sandbox_adapter.atomic_write_file("workspace/file2.py", b"orig2")
    events = await _collect_sse_until_done(
        async_client, session_id, headers=coord_jwt_headers, message="patch file1+file2",
    )
    apply_events = [e for e in events if e["type"] == "coordinator_apply"]
    assert apply_events and apply_events[-1]["data"]["apply_status"] in ("minio_fetch_failed", "write_io_error")
    assert await coord_sandbox_adapter.read_file("workspace/file1.py") == b"orig1"
    async with async_session() as s:
        audit = (await s.execute(text(
            "SELECT status, rollback_status FROM coordinator_apply_audit "
            "WHERE coordinator_run_id LIKE :p ORDER BY started_at DESC LIMIT 1"
        ), {"p": f"{session_id}:%"})).fetchone()
        assert audit is not None
        # DB stores ApplyStatus.value verbatim (NOT a generic "failed"); the
        # injected get_bytes fault yields minio_fetch_failed, mirroring the SSE
        # apply_status asserted above. (write_io_error allowed for impl variance.)
        assert audit[0] in ("minio_fetch_failed", "write_io_error")
        assert audit[1] == "complete"
    # complete rollback emits NO rollback_partial HealthEvent. The Health SSE code
    # lives in data.metrics["code"], NOT top-level.
    assert not [e for e in events if e["type"] == "health"
               and "rollback_partial" in ((e["data"].get("metrics") or {}).get("code") or "")]
