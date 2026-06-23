"""[S2 PR-6] Flag-OFF dark-launch guard: with the shell-mode master flag OFF
(ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED unset/false) but the coordinator flag
ON, a planner that emits shell_mode=True + proposed_trees is COERCED to
typed-only at dispatch (PR-3 F27 active fail-closed) — the run behaves
byte-for-byte like the pre-S2 typed coordinator path: a typed file_write child
applies, NO shell tool is ever BOUND (asserted via the routing fake's
bound_tool_name_sets recorder — not merely "not called"), and there is no
error. This is the §5 fail-safe-default invariant: shell-mode requires
`flag_on AND wu.shell_mode` (both affirmative); a stale/hand-crafted shell
payload cannot activate the dormant path while the master flag is OFF.

Runs flag-ON-coordinator / flag-OFF-shell in the coordinator-e2e CI job (it
dispatches → needs minio+sandbox). Integration unverified locally, CI.
"""
import pytest
from sqlalchemy import text

# sandbox marker keeps it out of the default suite (api/pytest.ini addopts);
# the coordinator-e2e CI job still selects via -m coordinator_recovery.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.coordinator_recovery,
    pytest.mark.sandbox,
]

# Raw shell tools unblocked only in shell-mode; with the flag OFF none may bind.
_SHELL_TOOL_NAMES = frozenset({
    "shell_execute", "shell_wait_process", "shell_kill_process",
    "shell_write_input", "shell_read_output",
})


async def _collect_sse_until_done(async_client, session_id, *, headers=None,
                                  timeout_s=120, message="trigger coordinator"):
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


async def test_shell_flag_off_coerces_to_typed_only(
    inject_routing_fake_llm, coord_async_client, async_session, redis_real, minio_real,
    coord_jwt_headers, fresh_test_user, env_with_coordinator_flag_on, monkeypatch,
):
    from tests.integration.coordinator_fixtures import bind_session_sandbox_adapter
    # Coordinator ON (env_with_coordinator_flag_on); shell-mode master flag OFF.
    monkeypatch.delenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", raising=False)
    async_client = coord_async_client
    # Planner emits a SHELL-shaped unit (shell_mode + proposed_trees) AND a typed
    # proposed_path so the coerced typed-only unit still has a valid write target.
    # With the flag OFF, dispatch coerces it to typed-only; the child does a
    # normal file_write (NOT shell_execute), captured by the typed extractor.
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{
            "id": "step_dark_001", "description": "shell-shaped but flag off",
            "parallel_work_units": {"work_units": [
                {"objective": "typed write under flag off", "phase": "write",
                 "shell_mode": True,
                 "allowed_tools": ["file_read", "file_write", "shell_execute"],
                 "proposed_trees": [{"prefix": "gen", "ops": ["add"]}],
                 "proposed_paths": [{"path": "workspace/d.py", "op": "modify"}]},
            ]},
        }]},
        child_responses={
            # Typed write only — the coerced unit has shell HARD_BLOCKed.
            "typed write under flag off": [
                {"tool": "file_write", "args": {"filepath": "workspace/d.py", "content": "typed new"}},
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
    await coord_sandbox_adapter.atomic_write_file("workspace/d.py", b"old d")

    events = await _collect_sse_until_done(
        async_client, session_id, headers=coord_jwt_headers, timeout_s=180,
        message="run the shell-shaped unit with the flag off",
    )

    errors = [e for e in events if e["type"] == "error"]
    assert errors == [], f"flag-off coercion must not error; got {errors}"
    # Typed path ran: dispatched, one worker, SUCCESS group, typed file applied.
    types = [e["type"] for e in events]
    assert "coordinator_dispatch" in types
    assert types.count("coordinator_worker_spawned") == 1
    reduce_events = [e for e in events if e["type"] == "coordinator_reduce"]
    assert reduce_events and reduce_events[-1]["data"]["group_outcome"] == "success"
    apply_events = [e for e in events if e["type"] == "coordinator_apply"]
    assert apply_events and apply_events[-1]["data"]["apply_status"] == "success"
    # The typed write applied byte-for-byte like pre-S2.
    assert await coord_sandbox_adapter.read_file("workspace/d.py") == b"typed new"

    # CORE dark-launch assertion: with the flag OFF, the COORDINATOR CHILD binds
    # ZERO shell tools (not merely "doesn't call them"). The routing fake records
    # the tool-name set of every bind_tools call (Step 0).
    #
    # IMPORTANT (codex PR-6 R1 P1): the recorder is GLOBAL — it also captures the
    # ROOT/orchestrator session's bind, which legitimately binds the FULL native
    # tool set (incl. all 5 shell tools + browser + message tools), because
    # shell-mode is about the coordinator CHILD, not the root. So a naive
    # "no bind anywhere includes shell" assertion would FALSE-FAIL on the root.
    #
    # Discriminate the child bind from the root bind by content, using the real
    # allowlist as the single source of truth: the coordinator child binds only
    # the filtered COORDINATOR_STEP preset (∪ shell when flag ON), so a child
    # bind is a non-empty SUBSET of `COORDINATOR_STEP_BASE ∪ shell`. The root bind
    # contains tools OUTSIDE that universe (message_*, browser_*), so it is NOT a
    # subset and is excluded. Under flag OFF the child bind == COORDINATOR_STEP_BASE
    # (no shell); under flag ON it would be the base ∪ shell — so asserting no
    # child-scoped bind contains a shell tool is exactly the flag-off invariant.
    from app.domain.services.tool_filter_presets import (
        COORDINATOR_STEP_BASE_ALLOWED_TOOLS,
    )
    child_universe = COORDINATOR_STEP_BASE_ALLOWED_TOOLS | _SHELL_TOOL_NAMES
    bound_sets = inject_routing_fake_llm.bound_tool_name_sets
    assert bound_sets, "expected at least one bind_tools call"
    child_binds = [s for s in bound_sets if s and s <= child_universe]
    assert child_binds, (
        "expected at least one filtered coordinator-child bind (a non-empty "
        "subset of COORDINATOR_STEP_BASE ∪ shell); got only non-child binds "
        f"{bound_sets} — the dispatch/child-bind path may have changed."
    )
    shell_leaks = [s for s in child_binds if s & _SHELL_TOOL_NAMES]
    assert shell_leaks == [], (
        "flag OFF: a coordinator-child bind leaked shell tools: "
        f"{shell_leaks}. The dispatch fail-closed coercion (F27) is leaking "
        "shell into the dormant child path."
    )

    async with async_session() as s:
        outcome = (await s.execute(text("""
            SELECT payload->>'outcome' FROM coordinator_result_envelope_store
            WHERE coordinator_run_id LIKE :p ORDER BY work_unit_id LIMIT 1
        """), {"p": f"{session_id}:%"})).scalar()
        assert outcome == "success"
