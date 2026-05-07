"""B3-core PR-0: 23 contract anchors (xfail).

These tests assert the contracts defined in spec v3 §3-§7. They fail at
PR-0 ship (supervisor not yet implemented) and flip to PASS as PR-1..PR-4
land their respective features.

Anchor groups (per spec v3 §8.1):

- C-FSM-1..3 (3): FSM transitions T1/T2 admit, T3 promote, T6/T8 suspend
- C-Admission-1..2 (2): Lua 4-key admit; user/system slot enforcement
- C-Lua-Revoke-Idempotent (1): double revoke = no-op
- C-Lua-NoScript (1): EVALSHA NOSCRIPT fallback to EVAL
- C-Restart-1, C-Restart-2, C-Restart-NEW (3): reconciler FINISHING /
  running / new-FINISHING
- C-Repo-Atomic-Terminal (1): update_to_terminal atomic phase+status+reason
- C-Repo-Find-NamedTuple (1): find_running_background returns BgSessionRow
  (4 fields)
- C-FINISHING-1 (1): supervisor_suspend bypass tuple in agent_task_runner
- C-Callback-Compose (1): try/finally guarantees supervisor cleanup runs
- C-Inflight-1, C-Inflight-2 (2): on_llm_end + ainvoke counter
  increment/decrement
- C-Cancel-1 (1): POST /api/sessions/{id}/cancel routes through stop_session
- C-Auth-1 (1): cross-user cancel returns 403
- C-MultiTab-1 (1): Tab2 SSE receives OwnerConflictEvent
- C-Notif-Types, C-Notif-Reuse, C-Notif-Reconcile, C-Notif-Watchdog (4)
                                                              -- 23 total --

Spec basis: docs/superpowers/specs/2026-05-07-b3-core-design.md
Plan basis: docs/superpowers/plans/2026-05-07-b3-core-pr0-plan.md (Task 3).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


# -- C-FSM-1: T1 admit foreground sets execution_mode='foreground' atomically --
@pytest.mark.xfail(strict=False, reason="PR-2: ExecutionSupervisor.admit not yet implemented")
async def test_C_FSM_1_admit_foreground_marks_session_fg_running(
    agent_service_with_redis, sample_user, session_repo,
):
    from app.domain.services.execution_supervisor import ExecutionSupervisor  # PR-2

    sid = str(uuid.uuid4())
    sup: ExecutionSupervisor = agent_service_with_redis._supervisor
    await sup.admit(session_id=sid, user_id=sample_user.id, execution_mode="foreground")
    fresh = await session_repo.get_by_id(sid)
    assert fresh.execution_mode == "foreground"
    assert fresh.execution_phase == "running"


# -- C-FSM-2: T3 promote FG → BG sets background_reason='auto_degrade' ---------
@pytest.mark.xfail(strict=False, reason="PR-2: ExecutionSupervisor.promote not yet implemented")
async def test_C_FSM_2_promote_t3_fg_to_bg(
    agent_service_with_redis, sample_user, session_repo,
):
    from app.domain.services.execution_supervisor import ExecutionSupervisor

    sid = str(uuid.uuid4())
    sup: ExecutionSupervisor = agent_service_with_redis._supervisor
    await sup.admit(session_id=sid, user_id=sample_user.id, execution_mode="foreground")
    expires = datetime.now(timezone.utc) + timedelta(hours=2)
    await sup.promote(session_id=sid, user_id=sample_user.id, expires_at=expires)
    fresh = await session_repo.get_by_id(sid)
    assert fresh.execution_mode == "background"
    assert fresh.background_reason == "auto_degrade"
    assert fresh.was_background is True


# -- C-FSM-3: T6 / T8 suspend transitions --------------------------------------
@pytest.mark.xfail(strict=False, reason="PR-2: suspend transitions not yet implemented")
async def test_C_FSM_3_suspend_idle_marks_phase_suspended(
    agent_service_with_redis, sample_user, session_repo,
):
    from app.domain.services.execution_supervisor import ExecutionSupervisor

    sid = str(uuid.uuid4())
    sup: ExecutionSupervisor = agent_service_with_redis._supervisor
    expires = datetime.now(timezone.utc) + timedelta(hours=2)
    await sup.admit(
        session_id=sid, user_id=sample_user.id, execution_mode="background",
        background_reason="explicit", expires_at=expires,
    )
    await sup.suspend_idle(session_id=sid, user_id=sample_user.id)
    fresh = await session_repo.get_by_id(sid)
    assert fresh.execution_phase == "suspended"
    assert fresh.suspended_reason == "bg_idle_timeout"


# -- C-Admission-1: 4-key Lua admission enforces system+user budgets -----------
@pytest.mark.xfail(strict=False, reason="PR-2: LUA_ADMIT not yet shipped")
async def test_C_Admission_1_lua_admit_4_key_returns_correct_codes(redis_client):
    from app.domain.services._lua_scripts import (
        LUA_ADMIT_SHA, LUA_ADMIT_SOURCE, run_lua_with_fallback,
    )

    user_id = "u-admission-1"
    sys_key = "supervisor:system:bg_count"
    user_key = f"supervisor:user:{user_id}"
    hot_key_template = "supervisor:hot:{}"
    bg_key = f"supervisor:bg:{user_id}"

    # First admit success
    sid1 = "sess-1"
    expires_at = "1999999999"
    rc = await run_lua_with_fallback(
        redis_client, source=LUA_ADMIT_SOURCE, sha=LUA_ADMIT_SHA,
        keys=[sys_key, user_key, hot_key_template.format(sid1), bg_key],
        args=[sid1, expires_at, "100", "5"],
    )
    assert rc == 0, f"first admit should succeed; got rc={rc}"

    # Round-3 audit P1#6 fix: assert KEYS[4] supervisor:bg ZSET membership +
    # score (per spec v3 §5.1 LUA_ADMIT must ZADD on success).  The prior
    # version only checked rc==0, missing the multi-key invariant.
    score = await redis_client.zscore(bg_key, sid1)
    assert score is not None, (
        f"LUA_ADMIT must ZADD session_id to supervisor:bg ZSET on success "
        f"(spec §5.1 KEYS[4] + ZADD line 477)"
    )
    assert int(float(score)) == int(expires_at), (
        f"ZADD score should equal expires_at_unix; got {score}, expected {expires_at}"
    )

    # Re-admit same session → 3 (already_bg)
    rc = await run_lua_with_fallback(
        redis_client, source=LUA_ADMIT_SOURCE, sha=LUA_ADMIT_SHA,
        keys=[sys_key, user_key, hot_key_template.format(sid1), bg_key],
        args=[sid1, expires_at, "100", "5"],
    )
    assert rc == 3, f"re-admit must return 3 (already_bg); got rc={rc}"

    # Also cover system_full (rc=1): set max_sys=1 and admit a different session
    # — second admit should hit the system cap before the user cap.
    sid2 = "sess-2"
    rc_sys_full = await run_lua_with_fallback(
        redis_client, source=LUA_ADMIT_SOURCE, sha=LUA_ADMIT_SHA,
        keys=[sys_key, user_key, hot_key_template.format(sid2), bg_key],
        args=[sid2, expires_at, "1", "5"],  # max_sys=1, already 1 used → system_full
    )
    assert rc_sys_full == 1, f"max_sys=1 with 1 used must return 1 (system_full); got {rc_sys_full}"


# -- C-Admission-2: User slot exhausted returns 2 (user_full) ------------------
@pytest.mark.xfail(strict=False, reason="PR-2: LUA_ADMIT user_full path not yet shipped")
async def test_C_Admission_2_user_slot_exhausted(redis_client):
    from app.domain.services._lua_scripts import (
        LUA_ADMIT_SHA, LUA_ADMIT_SOURCE, run_lua_with_fallback,
    )

    user_id = "u-admission-2"

    def keys_for(sid: str) -> list[str]:
        return [
            "supervisor:system:bg_count",
            f"supervisor:user:{user_id}",
            f"supervisor:hot:{sid}",
            f"supervisor:bg:{user_id}",
        ]

    # Fill 5 slots
    for i in range(5):
        rc = await run_lua_with_fallback(
            redis_client, source=LUA_ADMIT_SOURCE, sha=LUA_ADMIT_SHA,
            keys=keys_for(f"sess-{i}"),
            args=[f"sess-{i}", "1999999999", "100", "5"],
        )
        assert rc == 0
    # 6th must reject with user_full (rc=2)
    rc = await run_lua_with_fallback(
        redis_client, source=LUA_ADMIT_SOURCE, sha=LUA_ADMIT_SHA,
        keys=keys_for("sess-overflow"),
        args=["sess-overflow", "1999999999", "100", "5"],
    )
    assert rc == 2


# -- C-Lua-Revoke-Idempotent: Double revoke returns 0 the second time ----------
@pytest.mark.xfail(strict=False, reason="PR-2: LUA_REVOKE not yet shipped")
async def test_C_Lua_Revoke_Idempotent(redis_client):
    from app.domain.services._lua_scripts import (
        LUA_ADMIT_SHA, LUA_ADMIT_SOURCE, LUA_REVOKE_SHA, LUA_REVOKE_SOURCE,
        run_lua_with_fallback,
    )

    user_id = "u-revoke"
    sid = "sess-rev"
    hot_key = f"supervisor:hot:{sid}"
    keys_admit = [
        "supervisor:system:bg_count", f"supervisor:user:{user_id}",
        hot_key, f"supervisor:bg:{user_id}",
    ]
    await run_lua_with_fallback(
        redis_client, source=LUA_ADMIT_SOURCE, sha=LUA_ADMIT_SHA,
        keys=keys_admit, args=[sid, "1999999999", "100", "5"],
    )

    # Round-3 audit P1#6 fix: pre-populate hot Hash with FG-mode fields so we
    # can assert LUA_REVOKE leaves them intact (spec v3 §5.2 — LUA_REVOKE must
    # NOT touch supervisor:hot, since T7 BG→FG reconnect still needs it for
    # subscriber_count / cancellation_pending / inflight_*_count).
    await redis_client.hset(hot_key, mapping={
        "subscriber_count": "1",
        "cancellation_pending": "0",
        "inflight_llm_count": "0",
        "inflight_tool_count": "0",
        "last_activity_at": "1700000000.0",
    })

    keys_revoke = [
        "supervisor:system:bg_count", f"supervisor:user:{user_id}",
        f"supervisor:bg:{user_id}",  # NOT hot — round-2 P0-3
    ]
    rc1 = await run_lua_with_fallback(
        redis_client, source=LUA_REVOKE_SOURCE, sha=LUA_REVOKE_SHA,
        keys=keys_revoke, args=[sid],
    )
    assert rc1 == 1, f"first revoke of admitted session should return 1; got {rc1}"

    # Round-3 audit P1#6 fix + round-2 P2-B refinement: assert hot Hash is
    # PRESERVED after revoke (per spec v3 §5.2 line 488-499 — LUA_REVOKE only
    # touches sys + user + bg).  Verify ALL pre-populated fields survive (not
    # just subscriber_count) — comment claimed "subscriber_count and inflight
    # counts survive" but only subscriber_count was checked.
    hot_fields = await redis_client.hgetall(hot_key)
    assert hot_fields, (
        f"LUA_REVOKE must NOT delete supervisor:hot:{{sid}} (spec v3 §5.2 P0-3 fix)"
    )

    def _hget(field: str) -> str | None:
        # redis-py default returns bytes; normalize to str for assertion clarity.
        raw = hot_fields.get(field.encode()) or hot_fields.get(field)
        return raw.decode() if isinstance(raw, bytes) else raw

    for field in (
        "subscriber_count",
        "cancellation_pending",
        "inflight_llm_count",
        "inflight_tool_count",
        "last_activity_at",
    ):
        assert _hget(field) is not None, (
            f"hot Hash field {field!r} must survive LUA_REVOKE; "
            f"hot_fields keys: {sorted(hot_fields.keys())}"
        )

    rc2 = await run_lua_with_fallback(
        redis_client, source=LUA_REVOKE_SOURCE, sha=LUA_REVOKE_SHA,
        keys=keys_revoke, args=[sid],
    )
    assert rc2 == 0, f"idempotent: second revoke of removed session should return 0; got {rc2}"


# -- C-Lua-NoScript: EVALSHA cache miss falls back to SCRIPT LOAD + EVAL -------
@pytest.mark.xfail(strict=False, reason="PR-2: NOSCRIPT fallback not yet shipped")
async def test_C_Lua_NoScript_fallback_after_script_flush(redis_client):
    from app.domain.services._lua_scripts import (
        LUA_ADMIT_SHA, LUA_ADMIT_SOURCE, run_lua_with_fallback,
    )

    user_id = "u-noscript"
    keys = [
        "supervisor:system:bg_count", f"supervisor:user:{user_id}",
        "supervisor:hot:sess-ns", f"supervisor:bg:{user_id}",
    ]
    # Flush server script cache to force NOSCRIPT
    await redis_client.script_flush()
    # First call: EVALSHA miss → fallback → success
    rc = await run_lua_with_fallback(
        redis_client, source=LUA_ADMIT_SOURCE, sha=LUA_ADMIT_SHA,
        keys=keys, args=["sess-ns", "1999999999", "100", "5"],
    )
    assert rc == 0
    # Second call: cache warm, EVALSHA succeeds
    rc = await run_lua_with_fallback(
        redis_client, source=LUA_ADMIT_SOURCE, sha=LUA_ADMIT_SHA,
        keys=keys, args=["sess-ns-2", "1999999999", "100", "5"],
    )
    assert rc == 0


# -- C-Restart-1: Reconciler FINISHING → terminal + LUA_REVOKE + bg_terminal_server_restart
# Round-5 audit P2 fix: prior body created a fresh UUID against a live DB but
# never inserted the FINISHING row + never triggered the lifespan/reconciler.
# `session_repo.get_by_id(sid)` would return None → AttributeError on
# `fresh.terminal_reason`, an erratic xfail signal.  Converted to explicit
# placeholder matching C-Wire-2 / C-Notif-Reuse pattern.
@pytest.mark.xfail(strict=False, reason="PR-2: reconciler not yet wired")
async def test_C_Restart_1_finishing_reconciles_to_terminal(
    agent_service_with_redis, sample_user, session_repo, notification_repo,
):
    pytest.fail(
        "placeholder — flip when PR-2 ships lifespan reconciler. "
        "Implementation must:\n"
        "  (1) pre-seed a FINISHING BG session row (status=FINISHING, "
        "execution_mode=background, expires_at=NOW+2h)\n"
        "  (2) restart the app process OR explicitly invoke "
        "`reconcile_supervisor_state_quiescent` to simulate boot\n"
        "  (3) assert post-reconcile: status terminal, terminal_reason='server_restart'\n"
        "  (4) assert LUA_REVOKE was called (Redis sys/user/bg keys cleared)\n"
        "  (5) assert notification row emitted with "
        "event_type='bg_terminal_server_restart'\n"
        "Per spec v3 §5.5 + decision 5 (round-2 P0-2 fix)."
    )


# -- C-Restart-2: Reconciler running BG → suspended (no LUA_REVOKE; slot stays)
# Round-5 audit P2 fix: same hollow-body issue as C-Restart-1.  Converted to
# explicit placeholder.
@pytest.mark.xfail(strict=False, reason="PR-2: reconciler running-BG path not yet wired")
async def test_C_Restart_2_running_bg_reconciles_to_suspended(
    agent_service_with_redis, sample_user, session_repo, notification_repo, redis_client,
):
    pytest.fail(
        "placeholder — flip when PR-2 ships lifespan reconciler running-BG path. "
        "Implementation must:\n"
        "  (1) pre-seed a running BG session row (status=running, "
        "execution_mode=background, execution_phase=running)\n"
        "  (2) seed Redis: HSET supervisor:user:{uid} {sid} {expires_at_unix}\n"
        "  (3) restart app OR invoke `reconcile_supervisor_state_quiescent`\n"
        "  (4) assert post-reconcile: execution_phase='suspended', "
        "suspended_reason='server_restart'\n"
        "  (5) assert slot stays in supervisor:user (no LUA_REVOKE — slot held "
        "for T9 resume per spec v3 §5.5 + round-2 P0-2 distinction)\n"
        "  (6) assert notification row with event_type='bg_suspended_server_restart'\n"
        "Per spec v3 §5.5."
    )


# -- C-Restart-NEW: New FINISHING transitions correctly mid-flight -------------
@pytest.mark.xfail(strict=False, reason="PR-2: reconciler new-FINISHING path not yet wired")
async def test_C_Restart_NEW_new_finishing_path(agent_service_with_redis, sample_user):
    # Placeholder — exact contract per spec v3 §5.5; flesh out as PR-2 lands
    pytest.fail("placeholder — flip when PR-2 ships C-Restart-NEW contract test")


# -- C-Repo-Atomic-Terminal: update_to_terminal writes status+reason+phase atomically
@pytest.mark.xfail(strict=False, reason="PR-2: SessionRepository.update_to_terminal not yet implemented")
async def test_C_Repo_Atomic_Terminal(session_repo, sample_session):
    from app.domain.models.session import SessionStatus

    await session_repo.update_to_terminal(
        sample_session.id,
        SessionStatus.COMPLETED,  # uppercase per round-2 P1-5
        terminal_reason="user_cancel",
    )
    fresh = await session_repo.get_by_id(sample_session.id)
    assert fresh.status == SessionStatus.COMPLETED
    assert fresh.terminal_reason == "user_cancel"
    assert fresh.execution_phase == "terminated"


# -- C-Repo-Find-NamedTuple: find_running_background returns BgSessionRow(4 fields)
# Round-6 audit P1 fix: prior body had `if rows:` guard — when
# `find_running_background()` returns [] (PR-0 baseline: no BG sessions seeded),
# every assertion is skipped, producing silent XPASS.  Converted to explicit
# placeholder pattern matching C-Restart-1/2 / C-Notif-Reuse.
@pytest.mark.xfail(strict=False, reason="PR-2: find_running_background not yet implemented")
async def test_C_Repo_Find_NamedTuple(session_repo, sample_user):
    pytest.fail(
        "placeholder — flip when PR-2 ships SessionRepository.find_running_background. "
        "Implementation must:\n"
        "  (1) pre-seed at least 1 BG running session row for sample_user "
        "(execution_mode='background', execution_phase='running')\n"
        "  (2) call `rows = await session_repo.find_running_background()` "
        "(NO user_id param per spec v3 §7.1)\n"
        "  (3) assert rows is non-empty (len >= 1) — guards against silent XPASS\n"
        "  (4) assert `isinstance(row, BgSessionRow)` for each row\n"
        "  (5) assert each row has 4 fields: session_id (str), task_id "
        "(str | None), user_id (str), status (SessionStatus)\n"
        "Per spec v3 §7.1 + round-2 P1#5 (4-field NamedTuple with task_id)."
    )


# -- C-FINISHING-1: cancel_reason='supervisor_suspend' bypasses _set_terminal_status
@pytest.mark.xfail(strict=False, reason="PR-3a: bypass tuple extension not yet shipped")
async def test_C_FINISHING_1_supervisor_suspend_bypasses_terminal(
    runner_factory, session_repo, sample_user,
):
    from app.domain.models.session import SessionStatus

    sid = str(uuid.uuid4())
    runner = runner_factory(session_id=sid, user_id=sample_user.id)
    runner.cancel_reason = "supervisor_suspend"

    called = []
    original_set_terminal = runner._set_terminal_status

    async def tracked(*a, **kw):
        called.append((a, kw))
        return await original_set_terminal(*a, **kw)
    runner._set_terminal_status = tracked

    await runner._handle_cancel()  # exact method name per actual runner
    assert not called, f"_set_terminal_status was called for supervisor_suspend: {called}"
    fresh = await session_repo.get_by_id(sid)
    assert fresh.status == SessionStatus.RUNNING


# -- C-Callback-Compose: try/finally — supervisor cleanup runs even when original raises
@pytest.mark.xfail(strict=False, reason="PR-3a: composed callback not yet shipped")
async def test_C_Callback_Compose_supervisor_cleanup_on_original_raise(
    agent_service_with_redis, sample_user, redis_client,
):
    sid = str(uuid.uuid4())
    user_id = sample_user.id
    # Pre-admit
    expires = datetime.now(timezone.utc) + timedelta(hours=2)
    sup = agent_service_with_redis._supervisor
    await sup.admit(
        session_id=sid, user_id=user_id, execution_mode="background",
        background_reason="explicit", expires_at=expires,
    )

    raised: list[str] = []

    async def original(passed_sid: str):
        raised.append(passed_sid)
        raise RuntimeError("simulated original failure")

    composed = agent_service_with_redis._compose_completion_callbacks(
        original=original, session_id=sid, user_id=user_id,
    )
    with pytest.raises(RuntimeError, match="simulated"):
        await composed(sid)

    # original ran
    assert raised == [sid]
    # supervisor cleanup ran (slot revoked even on raise)
    exists = await redis_client.hexists(f"supervisor:user:{user_id}", sid)
    assert exists == 0


# -- C-Inflight-1: on_llm_end decrements supervisor:hot.inflight_llm_count -----
@pytest.mark.xfail(strict=False, reason="PR-3b: SupervisorAwareCallbackHandler not yet shipped")
async def test_C_Inflight_1_on_llm_end_decrements_counter(redis_client):
    from unittest.mock import MagicMock

    from app.domain.services.cost_callback_handler import SupervisorAwareCallbackHandler

    sid = "sess-inflight-1"
    handler = SupervisorAwareCallbackHandler(
        session_id=sid, user_id="u-1",
        persister=MagicMock(),
        supervisor=_make_test_supervisor(redis_client),
    )
    await handler.on_llm_start(serialized={}, prompts=["test"], run_id="r1")
    val_after_start = int(await redis_client.hget(f"supervisor:hot:{sid}", "inflight_llm_count"))
    assert val_after_start == 1
    await handler.on_llm_end(response={}, run_id="r1")
    val_after_end = int(await redis_client.hget(f"supervisor:hot:{sid}", "inflight_llm_count") or 0)
    assert val_after_end == 0


# -- C-Inflight-2: ainvoke wrap covers Pydantic validation errors --------------
@pytest.mark.xfail(strict=False, reason="PR-3b: SupervisorAwareToolWrapper not yet shipped")
async def test_C_Inflight_2_ainvoke_wrap_covers_validation_errors(redis_client):
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel

    from app.domain.services.tools._supervisor_tool_wrapper import SupervisorAwareToolWrapper

    class Args(BaseModel):
        x: int  # validation will fail if non-int passed

    inner = StructuredTool(
        name="add_one",
        description="adds one",
        args_schema=Args,
        func=lambda x: x + 1,
        coroutine=lambda x: _async_add_one(x),
    )
    wrapped = SupervisorAwareToolWrapper(
        inner=inner, supervisor=_make_test_supervisor(redis_client),
    )

    sid = "sess-inflight-2"
    config = {"configurable": {"session_id": sid}}
    # Fail validation by passing non-int
    with pytest.raises(Exception):
        await wrapped.ainvoke({"x": "not-an-int"}, config=config)

    # Even on validation failure, counter must return to 0 (ainvoke instrumented before validation)
    val = int(await redis_client.hget(f"supervisor:hot:{sid}", "inflight_tool_count") or 0)
    assert val == 0


async def _async_add_one(x):
    return x + 1


def _make_test_supervisor(redis_client):
    """Lightweight supervisor stub for inflight tests — only exposes inflight_inc/dec.

    PR-2 REPLACEMENT MARKER: When PR-2 ships ExecutionSupervisor, replace
    this ``__new__`` bypass with a real ctor call:
    ``ExecutionSupervisor(redis=redis_client)``.  Search for
    ``_make_test_supervisor`` to find all call sites at PR-2 boundary.
    """
    from app.domain.services.execution_supervisor import ExecutionSupervisor

    sup = ExecutionSupervisor.__new__(ExecutionSupervisor)
    sup._redis = redis_client
    return sup


# -- C-Cancel-1: POST /cancel routes through agent_service.stop_session --------
@pytest.mark.xfail(strict=False, reason="PR-3c: cancel endpoint not yet shipped")
async def test_C_Cancel_1_routes_through_stop_session(
    asgi_client, sample_session, sample_user_token,
):
    from unittest.mock import AsyncMock, patch

    sid = sample_session.id
    with patch(
        "app.application.services.agent_service.AgentService.stop_session",
        new_callable=AsyncMock,
    ) as mock_stop:
        resp = await asgi_client.post(
            f"/api/sessions/{sid}/cancel",
            json={"reason": "user_cancel"},
            headers={"Authorization": f"Bearer {sample_user_token}"},
        )
        assert resp.status_code == 200
        # Verify signature: stop_session(session_id, user_id) — round-2 fix P0-5
        mock_stop.assert_awaited_once()
        # Round-3 audit P1-NEW-3 fix: real route at session_routes.py:538-542
        # calls `stop_session` with **kwargs**, not positional args.  Anchor must
        # accept either calling style (positional, kwargs, or mixed) so a
        # correct kwargs implementation isn't false-failed.
        call_args = mock_stop.await_args
        # Resolve session_id and user_id regardless of positional vs keyword form
        session_id_arg = (
            call_args.args[0] if len(call_args.args) >= 1
            else call_args.kwargs.get("session_id")
        )
        user_id_arg = (
            call_args.args[1] if len(call_args.args) >= 2
            else call_args.kwargs.get("user_id")
        )
        assert session_id_arg == sid, (
            f"stop_session session_id must equal {sid}; got args={call_args.args}, "
            f"kwargs={call_args.kwargs}"
        )
        # Round-2 audit P2#1: assert user_id is passed (positional or kwarg) per
        # spec v3 §6.5 — `stop_session(session_id, user_id, is_admin=False)`.
        # Without this, PR-3c could regress to `stop_session(sid, reason='...')`
        # silently and the anchor would still pass.
        assert user_id_arg is not None and isinstance(user_id_arg, str) and user_id_arg, (
            f"stop_session must receive user_id (str); "
            f"got args={call_args.args}, kwargs={call_args.kwargs}"
        )
        # user_id is the second positional arg


# -- C-Auth-1: Cross-user cancel returns 403 -----------------------------------
@pytest.mark.xfail(strict=False, reason="PR-3c: auth check not yet shipped")
async def test_C_Auth_1_cross_user_cancel_403(
    asgi_client, sample_session, other_user_token,
):
    sid = sample_session.id
    resp = await asgi_client.post(
        f"/api/sessions/{sid}/cancel",
        json={"reason": "user_cancel"},
        headers={"Authorization": f"Bearer {other_user_token}"},
    )
    assert resp.status_code == 403


# -- C-MultiTab-1: Tab2 receives OwnerConflictEvent on SSE ---------------------
@pytest.mark.xfail(strict=False, reason="PR-3c: subscriber_scope CAS not yet shipped")
async def test_C_MultiTab_1_owner_conflict_event(
    asgi_client, sample_session, sample_user_token,
):
    # Tab1 holds CAS lease via subscriber_scope; Tab2 receives OwnerConflictEvent
    pytest.fail("placeholder — flip when PR-3c subscriber_scope lands")


# -- C-Notif-Types: 8 typed event_type values valid via existing repo ----------
@pytest.mark.xfail(strict=False, reason="PR-4: 8 new event_type values not yet emitted")
async def test_C_Notif_Types_eight_supervisor_event_types(notification_repo, sample_user):
    # Round-6 audit P2 fix: prior body read `find_unread` on a fresh
    # `notification_repo` (empty) without first triggering any PR-4 emitter.
    # `expected_types.issubset(actual_types)` where actual_types is empty fails
    # — but for the WRONG reason (no producer ran), not because the contract
    # was tested.  Converted to explicit placeholder pattern matching
    # C-Restart-1/2 / C-Notif-Reconcile.
    pytest.fail(
        "placeholder — flip when PR-4 ships notification emitter integration. "
        "Implementation must:\n"
        "  (1) trigger 8 distinct B3-supervisor events through their actual "
        "emitter paths (NOT direct repo writes):\n"
        "      - bg_completed (BG runner natural completion)\n"
        "      - bg_cancelled (BG runner via stop_session(reason='user_cancel'))\n"
        "      - bg_failed_resume (reconciler resume failure)\n"
        "      - bg_failed_watchdog (runner watchdog timeout)\n"
        "      - bg_terminal_server_restart (reconciler FINISHING path)\n"
        "      - bg_suspended_timeout (idle watchdog)\n"
        "      - bg_suspended_server_restart (reconciler running-BG path)\n"
        "      - bg_retry_exhausted (retry budget exhausted)\n"
        "  (2) read `notification_repo.find_unread(sample_user.id)` AFTER each\n"
        "  (3) assert `expected_types.issubset(actual_types)` — all 8 emitted\n"
        "  (4) assert no event_type values OUTSIDE the 8 (catch typos)\n"
        "Per spec v3 §6.8 + decision 6 (notification reuse via memory_system_notifications)."
    )
    # Round-7 audit P3 fix: reference body kept as comments to avoid live
    # unreachable code that linters / static analyzers flag.  PR-4 author
    # uncomments + adapts:
    # expected_types = {
    #     "bg_completed", "bg_cancelled", "bg_failed_resume", "bg_failed_watchdog",
    #     "bg_terminal_server_restart", "bg_suspended_timeout",
    #     "bg_suspended_server_restart", "bg_retry_exhausted",
    # }
    # notifs = await notification_repo.find_unread(sample_user.id)
    # actual_types = {n.event_type for n in notifs}
    # assert expected_types.issubset(actual_types)


# -- C-Notif-Reuse: spec v3 §6.8 reuse — no new table, no /api/v3/notifications route
# Round-4 audit P1 fix: prior body was hollow — both assertions ("no
# session_notifications table" + "/api/v2/notifications/unread exists") are
# ALREADY TRUE in PR-0 (no migration adds the table; the v2 route exists in
# current code at notification_routes.py:45). Test would XPASS in a live env,
# breaking the xfail stability guarantee.  The negative invariant the anchor
# really wants — "PR-4 emits 8 new event_types via the EXISTING emitter, not
# via a new table/route" — needs a positive PR-4 assertion that doesn't
# trivially pass in PR-0.
# Converted to explicit placeholder (matches C-Restart-NEW / C-MultiTab-1 pattern).
@pytest.mark.xfail(strict=False, reason="PR-4: reuse-decision contract verification not yet shipped")
async def test_C_Notif_Reuse_no_new_table(asgi_client):
    pytest.fail(
        "placeholder — flip when PR-4 ships the 8 new event_type values "
        "(bg_completed, bg_cancelled, bg_failed_resume, bg_failed_watchdog, "
        "bg_terminal_server_restart, bg_suspended_timeout, "
        "bg_suspended_server_restart, bg_retry_exhausted) via the EXISTING "
        "MemoryNotificationEmitter + /api/v2/notifications route. "
        "Implementation must:\n"
        "  (1) verify no `session_notifications` table is added by PR-4\n"
        "  (2) verify `/api/v2/notifications/unread` endpoint still serves\n"
        "  (3) emit one of each 8 event_type values + assert the existing\n"
        "      memory_system_notifications repo round-trips them\n"
        "  (4) assert `/api/v3/notifications` does NOT exist (no new route)\n"
        "Per spec v3 §6.8 + decision 1 (notification reuse)."
    )
    # Round-7 audit P3 fix: reference body kept as `#`-prefixed comments to
    # avoid live unreachable code that linters / static analyzers flag.
    # PR-4 author uncomments + adapts:
    #
    # import subprocess
    # from pathlib import Path
    #
    # # Resolve repo root deterministically regardless of pytest invocation cwd
    # repo_root = Path(__file__).resolve()
    # while repo_root.parent != repo_root and not (repo_root / "CLAUDE.md").exists():
    #     repo_root = repo_root.parent
    # api_dir = repo_root / "api"
    # assert api_dir.exists(), f"could not locate api/ from {Path(__file__)}"
    #
    # # Fail anchor: assert no migration created session_notifications table
    # cmd = ["uv", "run", "alembic", "show", "head"]
    # result = subprocess.run(cmd, cwd=str(api_dir), capture_output=True, text=True)
    # output = result.stdout + result.stderr
    # assert "session_notifications" not in output, (
    #     "PR-4 must NOT add session_notifications table — reuse memory_system_notifications"
    # )
    # # Also assert /api/v2/notifications endpoint exists (reuse target)
    # resp = await asgi_client.get("/api/v2/notifications/unread")
    # assert resp.status_code in (200, 401)  # exists (401 if unauthenticated; 200 if seeded)


# -- C-Notif-Reconcile: Reconciler emits FINISHING + running-BG notifications ---
# Round-5 audit P2 fix: prior body had `# Pre-seed ...` and `# Trigger reconciler`
# as TODO comments — body never actually seeded sessions or triggered the
# reconciler.  `find_unread` on an empty notification table returns [] and
# the assertions fail trivially (xfail), but for the WRONG reason: the
# fixture wasn't exercised, not because the contract was tested.
# Converted to explicit placeholder pattern.
@pytest.mark.xfail(strict=False, reason="PR-4: reconciler emit hooks not yet shipped")
async def test_C_Notif_Reconcile_emits_two_paths(
    notification_repo, sample_user, session_repo,
):
    pytest.fail(
        "placeholder — flip when PR-4 ships reconciler notification emit. "
        "Implementation must:\n"
        "  (1) pre-seed two BG session rows for sample_user: one FINISHING, "
        "one running BG\n"
        "  (2) restart app OR invoke `reconcile_supervisor_state_quiescent`\n"
        "  (3) assert notification rows for sample_user contain BOTH:\n"
        "      - event_type='bg_terminal_server_restart' (from FINISHING path)\n"
        "      - event_type='bg_suspended_server_restart' (from running-BG path)\n"
        "  (4) assert no notifications were dropped or duplicated\n"
        "Per spec v3 §5.5 reconciler + §6.8 notifications + decision 5."
    )


# -- C-Notif-Watchdog: Runner emits bg_failed_watchdog after TIMED_OUT ---------
@pytest.mark.xfail(strict=False, reason="PR-4: runner notification emit not yet shipped")
async def test_C_Notif_Watchdog_emitted_by_runner(
    runner_factory, notification_repo, sample_user, session_repo,
):
    sid = str(uuid.uuid4())
    runner = runner_factory(session_id=sid, user_id=sample_user.id)
    # Trigger watchdog timeout path (PR-4 wires this)
    await runner.handle_watchdog_timeout()

    notifs = await notification_repo.find_unread(sample_user.id)
    assert any(
        n.event_type == "bg_failed_watchdog" and str(n.session_id) == sid
        for n in notifs
    )
