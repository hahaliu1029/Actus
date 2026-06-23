"""[S2 PR-6] Live flag-ON shell-mode E2E. Drives the live app via the injected
routing fake LLM + real JWT + the session sandbox adapter. Runs flag-ON
(ACTUS_C2_COORDINATOR_ENABLED + ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED) in the
coordinator-e2e CI job ONLY (needs pg+redis+minio+sandbox).

The shell-mode master flag is read from `os.environ` at call time
(`coordinator_feature_flag.is_coordinator_*` / the shell-mode flag template),
so these tests do NOT monkeypatch it on: the `coordinator-e2e` CI job env is the
SOLE flag-ON source (Task 6.3). That makes the dependency real — if the CI job
ever drops `ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED`, dispatch coerces the units
to typed-only and these tests go RED (no shell tools bound → no diff captured).
A separate fast YAML guard (Task 6.3) asserts the job env carries the flag.

Integration unverified locally, CI. Three scenarios:
  1. happy path  — a shell child writes a NEW file via `shell_execute` under a
     tree-add lease; the snapshot differ captures it; the applier writes it to
     the parent sandbox (§6 F1 live; §3.2 capture).
  2. out-of-lease — a shell child writes OUTSIDE all leases; child-finalize
     fail-closes → NEEDS_AUTHORIZATION → reducer non-SUCCESS → apply skipped;
     the parent sandbox is untouched (§6 F7 live; §3.4 group zero-apply).
  3. tree-add collision — under a tree-add lease the parent ALREADY has the
     target file; the shell `add`s the same path → the diff is a tree-`add`
     onto an existing parent regular file → `tree_add_target_exists` →
     group zero-apply; the pre-existing parent file is UNCHANGED
     (§6 F22 live; §3.3 tree leases ADD-only). NOTE: a tree lease never seeds
     the parent file into the child, so this is the F22 add-collision path,
     not the F5 tree-only-modify path (F5 is unit-tested in PR-4).
"""
import pytest
from sqlalchemy import text

# sandbox marker keeps this out of the default local suite (api/pytest.ini
# addopts deselect `not sandbox`); the coordinator-e2e CI job still selects it
# via -m coordinator_recovery (that job does NOT exclude sandbox).
pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.coordinator_recovery,
    pytest.mark.sandbox,
]


async def _collect_sse_until_done(async_client, session_id, *, headers=None,
                                  timeout_s=120, message="trigger coordinator"):
    # Real SSE wire: `event: <type>\ndata: <payload-without-type>`.
    import json
    import anyio
    events, current_event = [], None
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


async def test_shell_mode_e2e_diff_captured_and_applied(
    inject_routing_fake_llm, coord_async_client, async_session, redis_real, minio_real,
    coord_jwt_headers, fresh_test_user, env_with_coordinator_flag_on,
):
    # Shell-mode master flag is supplied ON by the coordinator-e2e CI job env
    # (Task 6.3) — NOT monkeypatched here, so the CI env is the real flag-on
    # source and a missing CI var turns these RED.
    from tests.integration.coordinator_fixtures import bind_session_sandbox_adapter
    async_client = coord_async_client
    # A single shell-mode write child: tree-add lease over `gen/`, raw shell
    # writes a NEW file `gen/out.py` (typed extraction is blind to this — the
    # snapshot differ must catch it).
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "step_shell_001", "description": "shell codegen into gen/",
            "parallel_work_units": {"work_units": [
                {"objective": "codegen gen/out.py via shell", "phase": "write",
                 "shell_mode": True,
                 "allowed_tools": ["file_read", "shell_execute",
                                   "shell_wait_process", "shell_read_output",
                                   "shell_write_input", "shell_kill_process"],
                 "proposed_trees": [{"prefix": "gen", "ops": ["add"]}]},
            ]},
        }]},
        child_responses={
            "codegen gen/out.py via shell": [
                {"tool": "shell_execute",
                 "args": {"command": "mkdir -p /home/ubuntu/gen && printf 'print(1)\\n' > /home/ubuntu/gen/out.py"}},
            ],
        },
    )

    resp = await async_client.post("/api/sessions", headers=coord_jwt_headers)
    assert resp.status_code == 200
    session_id = resp.json()["data"]["session_id"]

    import app.main as app_main
    coord_sandbox_adapter = await bind_session_sandbox_adapter(
        app_main.app, session_id, str(fresh_test_user.id),
    )
    # gen/out.py must be ABSENT in the parent before apply (op=add invariant).
    assert not await coord_sandbox_adapter.exists("gen/out.py")

    events = await _collect_sse_until_done(
        async_client, session_id, headers=coord_jwt_headers, timeout_s=180,
        message="codegen into gen via shell",
    )

    types = [e["type"] for e in events]
    assert "coordinator_dispatch" in types
    assert types.count("coordinator_worker_spawned") == 1
    reduce_events = [e for e in events if e["type"] == "coordinator_reduce"]
    assert reduce_events and reduce_events[-1]["data"]["group_outcome"] == "success"
    apply_events = [e for e in events if e["type"] == "coordinator_apply"]
    assert apply_events and apply_events[-1]["data"]["apply_status"] == "success"
    assert apply_events[-1]["data"]["file_count"] == 1
    # The raw-shell-written file is now in the parent sandbox.
    assert await coord_sandbox_adapter.read_file("gen/out.py") == b"print(1)\n"

    async with async_session() as s:
        audit_status = (await s.execute(text("""
            SELECT status FROM coordinator_apply_audit
            WHERE parent_session_id=:p ORDER BY started_at DESC LIMIT 1
        """), {"p": session_id})).scalar()
        assert audit_status == "success"


async def test_shell_mode_e2e_out_of_lease_zero_applies(
    inject_routing_fake_llm, coord_async_client, async_session, redis_real, minio_real,
    coord_jwt_headers, fresh_test_user, env_with_coordinator_flag_on,
):
    # Shell-mode master flag supplied ON by the CI job env (Task 6.3), not here.
    from tests.integration.coordinator_fixtures import bind_session_sandbox_adapter
    async_client = coord_async_client
    # Tree-add lease over `gen/`, but the shell writes OUTSIDE it (`other/x.py`).
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "step_shell_oot_001", "description": "shell writes out of lease",
            "parallel_work_units": {"work_units": [
                {"objective": "shell write out of lease", "phase": "write",
                 "shell_mode": True,
                 "allowed_tools": ["file_read", "shell_execute",
                                   "shell_wait_process", "shell_read_output",
                                   "shell_write_input", "shell_kill_process"],
                 "proposed_trees": [{"prefix": "gen", "ops": ["add"]}]},
            ]},
        }]},
        child_responses={
            "shell write out of lease": [
                {"tool": "shell_execute",
                 "args": {"command": "mkdir -p /home/ubuntu/other && printf 'x\\n' > /home/ubuntu/other/x.py"}},
            ],
        },
    )

    resp = await async_client.post("/api/sessions", headers=coord_jwt_headers)
    assert resp.status_code == 200
    session_id = resp.json()["data"]["session_id"]

    import app.main as app_main
    coord_sandbox_adapter = await bind_session_sandbox_adapter(
        app_main.app, session_id, str(fresh_test_user.id),
    )
    assert not await coord_sandbox_adapter.exists("other/x.py")

    events = await _collect_sse_until_done(
        async_client, session_id, headers=coord_jwt_headers, timeout_s=180,
        message="shell write out of lease",
    )

    types = [e["type"] for e in events]
    assert "coordinator_dispatch" in types
    reduce_events = [e for e in events if e["type"] == "coordinator_reduce"]
    # Out-of-lease child finalizes NEEDS_AUTHORIZATION → reducer non-SUCCESS.
    assert reduce_events and reduce_events[-1]["data"]["group_outcome"] != "success"
    # Apply gate skips → NO coordinator_apply with status success (zero-apply).
    apply_events = [e for e in events if e["type"] == "coordinator_apply"]
    assert all(e["data"].get("apply_status") != "success" for e in apply_events)
    # Parent sandbox untouched — the out-of-lease write never reached it.
    assert not await coord_sandbox_adapter.exists("other/x.py")

    async with async_session() as s:
        envelopes = (await s.execute(text("""
            SELECT payload->>'outcome' FROM coordinator_result_envelope_store
            WHERE coordinator_run_id LIKE :p
        """), {"p": f"{session_id}:%"})).fetchall()
        assert envelopes
        assert any(o == "needs_authorization" for (o,) in envelopes)


async def test_shell_mode_e2e_tree_add_collision_zero_applies(
    inject_routing_fake_llm, coord_async_client, async_session, redis_real, minio_real,
    coord_jwt_headers, fresh_test_user, env_with_coordinator_flag_on,
):
    # Shell-mode master flag supplied ON by the CI job env (Task 6.3), not here.
    from tests.integration.coordinator_fixtures import bind_session_sandbox_adapter
    async_client = coord_async_client
    # Tree-add lease over `gen/`. The parent ALREADY has gen/keep.py. The shell
    # `add`s the SAME path. Because a tree lease never seeds the parent file into
    # the child, the child PRE-snapshot lacks gen/keep.py → the write diffs as an
    # `add`, but the parent already has a regular file there → tree_add_target_
    # exists → group zero-apply (§6 F22). (This is the add-collision boundary,
    # NOT F5 tree-only-modify, which is unit-tested in PR-4.)
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "step_shell_treeadd_001", "description": "shell adds an already-present tree file",
            "parallel_work_units": {"work_units": [
                {"objective": "shell add colliding tree file", "phase": "write",
                 "shell_mode": True,
                 "allowed_tools": ["file_read", "shell_execute",
                                   "shell_wait_process", "shell_read_output",
                                   "shell_write_input", "shell_kill_process"],
                 "proposed_trees": [{"prefix": "gen", "ops": ["add"]}]},
            ]},
        }]},
        child_responses={
            "shell add colliding tree file": [
                {"tool": "shell_execute",
                 "args": {"command": "mkdir -p /home/ubuntu/gen && printf 'MUTATED\\n' > /home/ubuntu/gen/keep.py"}},
            ],
        },
    )

    resp = await async_client.post("/api/sessions", headers=coord_jwt_headers)
    assert resp.status_code == 200
    session_id = resp.json()["data"]["session_id"]

    import app.main as app_main
    coord_sandbox_adapter = await bind_session_sandbox_adapter(
        app_main.app, session_id, str(fresh_test_user.id),
    )
    # Pre-seed the parent file the child's tree-`add` will collide with.
    await coord_sandbox_adapter.atomic_write_file("gen/keep.py", b"ORIGINAL\n")

    events = await _collect_sse_until_done(
        async_client, session_id, headers=coord_jwt_headers, timeout_s=180,
        message="shell add a file that already exists in the parent",
    )

    reduce_events = [e for e in events if e["type"] == "coordinator_reduce"]
    assert reduce_events and reduce_events[-1]["data"]["group_outcome"] != "success"
    apply_events = [e for e in events if e["type"] == "coordinator_apply"]
    assert all(e["data"].get("apply_status") != "success" for e in apply_events)
    # The pre-existing parent file is UNCHANGED — the colliding tree-add never
    # applied (whole-group zero-apply).
    assert await coord_sandbox_adapter.read_file("gen/keep.py") == b"ORIGINAL\n"
