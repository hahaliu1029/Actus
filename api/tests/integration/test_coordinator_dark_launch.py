"""[C2 finish-core INV-F9.1] Flag-OFF dark-launch: prove production-default-off is
safe even when a planner emits parallel_work_units. assert_coordinator_enabled()
raises -> SSE `error` event + ZERO coordinator_* events. Runs in the coordinator-e2e
CI job (needs app+pg+redis; never dispatches, so no minio/sandbox)."""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.coordinator_recovery]


async def test_flag_off_parallel_step_errors_and_emits_no_coordinator_events(
    inject_routing_fake_llm, coord_async_client, coord_jwt_headers, monkeypatch,
):
    """[INV-F9.1] Flag OFF: a planner parallel_work_units step yields an SSE error
    event + ZERO coordinator_* events (production-default-off is safe)."""
    async_client = coord_async_client  # patched-LLM client (R7 P1)
    monkeypatch.delenv("ACTUS_C2_COORDINATOR_ENABLED", raising=False)  # ensure OFF
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "s1", "description": "parallel",
            "parallel_work_units": {"work_units": [
                {"objective": "do x", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/x.py", "op": "modify"}]},
            ]},
        }]},
        child_responses={},  # no child should ever run
    )
    resp = await async_client.post("/api/sessions", headers=coord_jwt_headers)
    session_id = resp.json()["data"]["session_id"]
    events = await _collect_sse(async_client, session_id, coord_jwt_headers)

    coord_events = [e for e in events if str(e.get("type", "")).startswith("coordinator_")]
    assert coord_events == [], f"flag-off must emit zero coordinator_* events; got {coord_events}"
    errors = [e for e in events if e.get("type") == "error"]
    assert errors, "expected an SSE error event"
    assert any("ACTUS_C2_COORDINATOR_ENABLED" in (e["data"].get("error") or "")
               for e in errors), "error must name the flag (not a stray LLM/config error)"
    # R10 P2 — prove the SAME routing fake was hit and NO child started (spec §5.9):
    assert inject_routing_fake_llm.parent_plan_calls == 1
    assert inject_routing_fake_llm.child_plan_calls == 0


async def _collect_sse(async_client, session_id, headers):
    # R9 P1: SSE type lives on the `event:` line; data payload excludes type.
    import json
    import anyio
    events, current_event = [], None
    # fail_after converts a hung stream into a fast TimeoutError instead of
    # blocking the CI job until its global timeout.
    with anyio.fail_after(60):
        async with async_client.stream(
            "POST", f"/api/sessions/{session_id}/chat",
            json={"message": "do x in parallel"}, headers=headers,
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    payload = json.loads(line[len("data:"):].strip())
                    events.append({"type": current_event, "data": payload})
                    if current_event in ("done", "error"):
                        return events
    return events
