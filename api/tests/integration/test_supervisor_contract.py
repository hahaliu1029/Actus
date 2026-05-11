"""B3-core supervisor contract anchors.

These tests assert the contracts defined in spec v3 §3-§7. They fail at
PR-0 shipped them as xfail; PR-1..PR-4 flip anchors to PASS as their
respective features land.

Anchor groups (per spec v3 §8.1):

- C-FSM-1..3 (3): FSM transitions T1/T2 admit, T3 promote, T8 suspend
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

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


# -- C-FSM-1: T1 admit foreground sets execution_mode='foreground' atomically --
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


# -- C-FSM-3: T8 suspend transition -------------------------------------------
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


class _RepoNotificationEmitter:
    def __init__(self, repo) -> None:
        self._repo = repo

    async def emit(self, *, user_id: str, event_type: str, payload: dict) -> None:
        await self._repo.create(
            notification_id=str(uuid.uuid4()),
            user_id=user_id,
            event_type=event_type,
            payload=payload,
        )


async def _seed_bg_slot(
    redis_client,
    *,
    user_id: str,
    session_id: str,
    expires_at: datetime,
) -> None:
    expires_unix = expires_at.timestamp()
    await redis_client.hset(f"supervisor:user:{user_id}", session_id, f"{expires_unix:.6f}")
    await redis_client.incr("supervisor:system:bg_count")
    await redis_client.zadd(f"supervisor:bg:{user_id}", {session_id: expires_unix})


# -- C-Restart-1: Reconciler FINISHING → terminal + LUA_REVOKE + bg_terminal_server_restart
async def test_C_Restart_1_finishing_reconciles_to_terminal(
    agent_service_with_redis,
    sample_user,
    session_repo,
    notification_repo,
    make_session,
    redis_client,
):
    from app.domain.models.session import SessionStatus

    expires = datetime.now(timezone.utc) + timedelta(hours=2)
    session = await make_session(
        status=SessionStatus.FINISHING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires,
        execution_phase="running",
    )
    await _seed_bg_slot(
        redis_client,
        user_id=sample_user.id,
        session_id=session.id,
        expires_at=expires,
    )

    summary = await agent_service_with_redis._supervisor.reconcile_running_background_at_boot(
        notification_emitter=_RepoNotificationEmitter(notification_repo)
    )

    fresh = await session_repo.get_by_id(session.id)
    assert summary == {"finishing": 1, "suspended": 0, "total": 1}
    assert fresh.status == SessionStatus.TIMED_OUT
    assert fresh.terminal_reason == "server_restart"
    assert fresh.execution_phase == "terminated"
    assert await redis_client.hexists(f"supervisor:user:{sample_user.id}", session.id) == 0
    assert await redis_client.zscore(f"supervisor:bg:{sample_user.id}", session.id) is None
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 0
    notifications = await notification_repo.list_unread(sample_user.id)
    assert any(
        n.event_type == "bg_terminal_server_restart"
        and n.payload.get("session_id") == session.id
        for n in notifications
    )


# -- C-Restart-2: Reconciler running BG → suspended (no LUA_REVOKE; slot stays)
async def test_C_Restart_2_running_bg_reconciles_to_suspended(
    agent_service_with_redis,
    sample_user,
    session_repo,
    notification_repo,
    redis_client,
    make_session,
):
    from app.domain.models.session import SessionStatus

    expires = datetime.now(timezone.utc) + timedelta(hours=2)
    session = await make_session(
        status=SessionStatus.RUNNING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires,
        execution_phase="running",
    )
    await _seed_bg_slot(
        redis_client,
        user_id=sample_user.id,
        session_id=session.id,
        expires_at=expires,
    )

    summary = await agent_service_with_redis._supervisor.reconcile_running_background_at_boot(
        notification_emitter=_RepoNotificationEmitter(notification_repo)
    )

    fresh = await session_repo.get_by_id(session.id)
    assert summary == {"finishing": 0, "suspended": 1, "total": 1}
    assert fresh.status == SessionStatus.RUNNING
    assert fresh.execution_phase == "suspended"
    assert fresh.suspended_reason == "server_restart"
    assert await redis_client.hexists(f"supervisor:user:{sample_user.id}", session.id) == 1
    assert await redis_client.zscore(f"supervisor:bg:{sample_user.id}", session.id) is not None
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 1
    notifications = await notification_repo.list_unread(sample_user.id)
    assert any(
        n.event_type == "bg_suspended_server_restart"
        and n.payload.get("session_id") == session.id
        for n in notifications
    )


# -- C-Restart-NEW: New FINISHING transitions correctly mid-flight -------------
async def test_C_Restart_NEW_new_finishing_path(
    agent_service_with_redis,
    sample_user,
    session_repo,
    make_session,
    redis_client,
):
    from app.domain.models.session import SessionStatus

    expires = datetime.now(timezone.utc) + timedelta(hours=2)
    session = await make_session(
        status=SessionStatus.FINISHING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires,
        execution_phase="running",
    )
    await _seed_bg_slot(
        redis_client,
        user_id=sample_user.id,
        session_id=session.id,
        expires_at=expires,
    )

    first = await agent_service_with_redis._supervisor.reconcile_running_background_at_boot()
    second = await agent_service_with_redis._supervisor.reconcile_running_background_at_boot()

    fresh = await session_repo.get_by_id(session.id)
    assert first == {"finishing": 1, "suspended": 0, "total": 1}
    assert second == {"finishing": 0, "suspended": 0, "total": 0}
    assert fresh.status == SessionStatus.TIMED_OUT
    assert fresh.terminal_reason == "server_restart"
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 0


# -- C-Repo-Atomic-Terminal: update_to_terminal writes status+reason+phase atomically
async def test_C_Repo_Atomic_Terminal(session_repo, sample_session):
    from app.domain.models.session import SessionStatus

    transitioned = await session_repo.update_to_terminal(
        sample_session.id,
        SessionStatus.COMPLETED,  # uppercase per round-2 P1-5
        terminal_reason="user_cancel",
    )
    fresh = await session_repo.get_by_id(sample_session.id)
    assert transitioned is True
    assert fresh.status == SessionStatus.COMPLETED
    assert fresh.terminal_reason == "user_cancel"
    assert fresh.execution_phase == "terminated"

    second_transition = await session_repo.update_to_terminal(
        sample_session.id,
        SessionStatus.COMPLETED,
        terminal_reason="natural",
    )
    fresh_again = await session_repo.get_by_id(sample_session.id)
    assert second_transition is False
    assert fresh_again.terminal_reason == "user_cancel"


# -- C-Repo-Find-NamedTuple: find_running_background returns BgSessionRow(4 fields)
async def test_C_Repo_Find_NamedTuple(session_repo, make_session):
    from datetime import datetime, timedelta, timezone

    from app.domain.models.session import SessionStatus
    from app.domain.repositories.session_repository import BgSessionRow

    session = await make_session(
        execution_mode="background",
        background_reason="explicit",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
        execution_phase="running",
        task_id=None,
    )
    rows = await session_repo.find_running_background()
    assert rows
    row = next(row for row in rows if row.session_id == session.id)
    assert isinstance(row, BgSessionRow)
    assert row.session_id == session.id
    assert row.task_id is None
    assert row.user_id == session.user_id
    assert row.status == SessionStatus.RUNNING
    sid, task_id, uid, status = row
    assert (sid, task_id, uid, status) == (
        session.id,
        None,
        session.user_id,
        SessionStatus.RUNNING,
    )


class _CancelInputStream:
    def __init__(self) -> None:
        self._calls = 0

    async def is_empty(self) -> bool:
        self._calls += 1
        return self._calls > 1

    async def pop(self):
        return None, None


class _CancelOutputStream:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def put(self, event_json: str) -> str:
        self.events.append(event_json)
        return f"event-{len(self.events)}"


class _CancelTask:
    def __init__(self, cancel_reason: str) -> None:
        self.cancel_reason = cancel_reason
        self.input_stream = _CancelInputStream()
        self.output_stream = _CancelOutputStream()


async def _cancel_flow(_message):
    raise asyncio.CancelledError
    if False:
        yield None


def _prime_runner_for_cancel(runner) -> None:
    from app.domain.models.event import MessageEvent
    from app.domain.models.user_tool_enablement import ToolType

    async def _empty_preferences(_tool_type: ToolType):
        return {}

    runner._pop_event = AsyncMock(return_value=MessageEvent(message="hello"))
    runner._run_flow = _cancel_flow
    runner._load_user_preferences_map = AsyncMock(side_effect=_empty_preferences)
    runner._load_enabled_skills = AsyncMock(return_value=[])
    runner._apply_preselected_skills = AsyncMock()
    runner._skill_bundle_sync.prepare_startup_sync = AsyncMock()
    runner._skill_bundle_sync.await_initial_sync = AsyncMock()
    runner._skill_bundle_sync.start_background_sync = MagicMock()
    runner._select_skills_from_pool = MagicMock(return_value=[])
    runner._select_skills_for_message = AsyncMock(return_value=([], None))


# -- C-FINISHING-1: cancel_reason='supervisor_suspend' bypasses _set_terminal_status
async def test_C_FINISHING_1_supervisor_suspend_bypasses_terminal(
    runner_factory, session_repo, sample_user, make_session,
):
    from app.domain.models.session import SessionStatus

    sid = str(uuid.uuid4())
    await make_session(id=sid, status=SessionStatus.RUNNING.value)
    runner = runner_factory(session_id=sid, user_id=sample_user.id)
    task = _CancelTask(cancel_reason="supervisor_suspend")
    _prime_runner_for_cancel(runner)

    called = []
    original_set_terminal = runner._set_terminal_status

    async def tracked(*a, **kw):
        called.append((a, kw))
        return await original_set_terminal(*a, **kw)
    runner._set_terminal_status = tracked

    with pytest.raises(asyncio.CancelledError):
        await runner.invoke(task)
    await asyncio.sleep(0)

    assert not called, f"_set_terminal_status was called for supervisor_suspend: {called}"
    assert task.output_stream.events == []
    fresh = await session_repo.get_by_id(sid)
    assert fresh.status == SessionStatus.RUNNING


# -- C-Callback-Compose: try/finally — supervisor cleanup runs even when original raises
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
async def test_C_Cancel_1_routes_through_stop_session(
    asgi_client, sample_session, sample_user_token, agent_service_with_redis,
    app, redis_client,
):
    from unittest.mock import AsyncMock, patch
    from app.infrastructure.storage.redis import get_redis
    from app.interfaces.dependencies import rate_limit_write

    sid = sample_session.id

    async def _noop_rate_limit() -> None:
        return None

    app.dependency_overrides[get_redis] = lambda: redis_client
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    with patch(
        "app.application.services.agent_service.AgentService.stop_session",
        new_callable=AsyncMock,
    ) as mock_stop:
        try:
            resp = await asgi_client.post(
                f"/api/sessions/{sid}/cancel",
                json={"reason": "user_cancel"},
                headers={"Authorization": f"Bearer {sample_user_token}"},
            )
        finally:
            app.dependency_overrides.pop(get_redis, None)
            app.dependency_overrides.pop(rate_limit_write, None)
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
async def test_C_Auth_1_cross_user_cancel_403(
    asgi_client, sample_session, other_user_token, agent_service_with_redis,
    app, redis_client,
):
    from app.infrastructure.storage.redis import get_redis
    from app.interfaces.dependencies import rate_limit_write

    async def _noop_rate_limit() -> None:
        return None

    sid = sample_session.id
    app.dependency_overrides[get_redis] = lambda: redis_client
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    try:
        resp = await asgi_client.post(
            f"/api/sessions/{sid}/cancel",
            json={"reason": "user_cancel"},
            headers={"Authorization": f"Bearer {other_user_token}"},
        )
    finally:
        app.dependency_overrides.pop(get_redis, None)
        app.dependency_overrides.pop(rate_limit_write, None)
    assert resp.status_code == 403


# -- C-MultiTab-1: Tab2 receives OwnerConflictEvent on SSE ---------------------
async def test_C_MultiTab_1_owner_conflict_event(
    asgi_client, sample_session, sample_user_token, agent_service_with_redis,
    app, redis_client,
):
    from app.infrastructure.storage.redis import get_redis
    from app.interfaces.dependencies import rate_limit_chat

    async def _noop_rate_limit() -> None:
        return None

    sid = sample_session.id
    owner = f"{sample_session.user_id}:tab1"
    tab2 = "tab2"
    await redis_client.set(f"supervisor:owner:{sid}", owner, ex=10)

    app.dependency_overrides[get_redis] = lambda: redis_client
    app.dependency_overrides[rate_limit_chat] = _noop_rate_limit
    try:
        resp = await asgi_client.post(
            f"/api/sessions/{sid}/chat",
            json={"message": "hello"},
            headers={
                "Authorization": f"Bearer {sample_user_token}",
                "X-Connection-Id": tab2,
            },
        )
    finally:
        app.dependency_overrides.pop(get_redis, None)
        app.dependency_overrides.pop(rate_limit_chat, None)

    assert resp.status_code == 200
    assert "owner_conflict" in resp.text
    assert owner in resp.text
    assert f"{sample_session.user_id}:{tab2}" in resp.text


# -- C-Notif-Types: 8 typed event_type values valid via existing repo ----------
async def test_C_Notif_Types_eight_supervisor_event_types(notification_repo, sample_user):
    from app.application.services.memory_notification_emitter import (
        ALL_VALID_EVENT_TYPES,
        _B3_CORE_EVENT_TYPES,
    )

    expected_types = {
        "bg_completed",
        "bg_cancelled",
        "bg_failed_resume",
        "bg_failed_watchdog",
        "bg_terminal_server_restart",
        "bg_suspended_timeout",
        "bg_suspended_server_restart",
        "bg_retry_exhausted",
    }
    assert _B3_CORE_EVENT_TYPES == expected_types
    assert expected_types.issubset(ALL_VALID_EVENT_TYPES)

    emitter = _RepoNotificationEmitter(notification_repo)
    for event_type in sorted(expected_types):
        await emitter.emit(
            user_id=sample_user.id,
            event_type=event_type,
            payload={"anchor": "C-Notif-Types", "event_type": event_type},
        )

    notifs = await notification_repo.list_unread(sample_user.id, limit=20)
    actual_types = {
        n.event_type
        for n in notifs
        if n.payload.get("anchor") == "C-Notif-Types"
    }
    assert actual_types == expected_types


# -- C-Notif-Reuse: spec v3 §6.8 reuse — no new table, no /api/v3/notifications route
async def test_C_Notif_Reuse_no_new_table(app, asgi_client):
    api_dir = Path(__file__).resolve().parents[2]
    migration_text = "\n".join(
        path.read_text()
        for path in (api_dir / "alembic" / "versions").glob("*.py")
    )
    assert "session_notifications" not in migration_text

    routes = {getattr(route, "path", "") for route in app.routes}
    assert "/api/v2/notifications/unread" in routes
    assert not any(route.startswith("/api/v3/notifications") for route in routes)

    resp = await asgi_client.get("/api/v2/notifications/unread")
    assert resp.status_code in (200, 401)


# -- C-Notif-Reconcile: Reconciler emits FINISHING + running-BG notifications ---
# Round-5 audit P2 fix: prior body had `# Pre-seed ...` and `# Trigger reconciler`
# as TODO comments — body never actually seeded sessions or triggered the
# reconciler.  `find_unread` on an empty notification table returns [] and
# the assertions fail trivially (xfail), but for the WRONG reason: the
# fixture wasn't exercised, not because the contract was tested.
async def test_C_Notif_Reconcile_emits_two_paths(
    agent_service_with_redis,
    notification_repo,
    sample_user,
    make_session,
    redis_client,
):
    from app.domain.models.session import SessionStatus

    expires = datetime.now(timezone.utc) + timedelta(hours=2)
    finishing = await make_session(
        status=SessionStatus.FINISHING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires,
        execution_phase="running",
        was_background=True,
    )
    running = await make_session(
        status=SessionStatus.RUNNING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires,
        execution_phase="running",
        was_background=True,
    )
    for session in (finishing, running):
        await _seed_bg_slot(
            redis_client,
            user_id=sample_user.id,
            session_id=session.id,
            expires_at=expires,
        )

    summary = await agent_service_with_redis._supervisor.reconcile_running_background_at_boot(
        notification_emitter=_RepoNotificationEmitter(notification_repo)
    )

    assert summary == {"finishing": 1, "suspended": 1, "total": 2}
    notifs = await notification_repo.list_unread(sample_user.id, limit=20)
    actual = {(n.event_type, n.payload.get("session_id")) for n in notifs}
    assert ("bg_terminal_server_restart", finishing.id) in actual
    assert ("bg_suspended_server_restart", running.id) in actual


# -- C-Notif-Watchdog: Runner emits bg_failed_watchdog after TIMED_OUT ---------
async def test_C_Notif_Watchdog_emitted_by_runner(
    runner_factory, notification_repo, sample_user,
):
    from app.domain.models.session import SessionStatus

    sid = str(uuid.uuid4())
    runner = runner_factory(
        session_id=sid,
        user_id=sample_user.id,
        memory_notification_emitter=_RepoNotificationEmitter(notification_repo),
        was_background=True,
    )
    terminal_calls: list[tuple[SessionStatus, str | None]] = []

    async def _fake_terminal_write(
        status: SessionStatus,
        terminal_reason: str | None = None,
    ) -> None:
        terminal_calls.append((status, terminal_reason))

    runner._set_terminal_status = _fake_terminal_write

    await runner._set_terminal_status_with_notifications(SessionStatus.TIMED_OUT)

    assert terminal_calls == [(SessionStatus.TIMED_OUT, None)]
    notifs = await notification_repo.list_unread(sample_user.id)
    assert any(
        n.event_type == "bg_failed_watchdog"
        and n.payload.get("session_id") == sid
        for n in notifs
    )
