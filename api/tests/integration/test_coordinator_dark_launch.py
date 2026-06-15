"""[C2b rollout WS0 §3A.4] Flag-OFF dark-launch CONTRACT CHANGE: with WS0
parse->Step sanitation, a flag-off planner emitting parallel_work_units no
longer trips the executor gate — the field is cleared at the parse boundary so
the step runs as a normal single ReAct task to `done` with ZERO coordinator_*
events and NO error. The authoritative gate-raises invariant is now pinned by
the fast unit test test_executor_gate_raises_flag_off (+ Task 1.6/1.7/1.8
sanitation tests). Runs in the coordinator-e2e CI job (needs app+pg+redis;
never dispatches, so no minio/sandbox)."""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.coordinator_recovery]


async def test_flag_off_parallel_step_is_sanitized_and_runs_to_done(
    inject_routing_fake_llm, coord_async_client, coord_jwt_headers, monkeypatch,
):
    """[INV-F9.1 + WS0 §3A.4] Flag OFF: a planner emitting parallel_work_units is
    SANITIZED at the parse->Step boundary (field cleared) so it NEVER enters the
    coordinator — it runs as a normal single ReAct step to `done`. Assert: stream
    reaches `done`, NO `error`, ZERO coordinator_* events, planner ran once
    (parent_plan_calls==1), no child spawned (child_plan_calls==0). The authoritative
    gate invariant is pinned by the unit test test_executor_gate_raises_flag_off."""
    async_client = coord_async_client
    monkeypatch.delenv("ACTUS_C2_COORDINATOR_ENABLED", raising=False)
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "s1", "description": "parallel",
            "parallel_work_units": {"work_units": [
                {"objective": "do x", "phase": "write",
                 "allowed_tools": ["file_read", "file_write"],
                 "proposed_paths": [{"path": "workspace/x.py", "op": "modify"}]},
            ]},
        }]},
        child_responses={},
        # [Step 1] sanitized step runs as a normal executor task → terminal response.
        parent_executor_response="done",
    )
    resp = await async_client.post("/api/sessions", headers=coord_jwt_headers)
    session_id = resp.json()["data"]["session_id"]
    events = await _collect_sse(async_client, session_id, coord_jwt_headers)

    coord_events = [e for e in events if str(e.get("type", "")).startswith("coordinator_")]
    errors = [e for e in events if e.get("type") == "error"]
    assert coord_events == [], f"flag-off sanitation must emit zero coordinator_* events; got {coord_events}"
    assert errors == [], f"sanitized step must not error (gate never reached); got {errors}"
    assert any(e.get("type") == "done" for e in events), "stream must reach done"
    assert inject_routing_fake_llm.parent_plan_calls == 1  # planner step actually ran
    assert inject_routing_fake_llm.child_plan_calls == 0   # no child spawned


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
