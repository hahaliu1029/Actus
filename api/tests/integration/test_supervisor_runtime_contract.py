from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


class _CancelableTask:
    def __init__(self) -> None:
        self.cancel_calls: list[str] = []

    def cancel(self, reason: str = "stop") -> bool:
        self.cancel_calls.append(reason)
        return True


async def test_update_status_rejects_terminal_status(session_repo, sample_session):
    from app.domain.models.session import SessionStatus

    with pytest.raises(ValueError, match="update_to_terminal"):
        await session_repo.update_status(sample_session.id, SessionStatus.COMPLETED)


async def test_boot_reconciler_ignores_legacy_terminal_background_rows(
    agent_service_with_redis,
    make_session,
    session_repo,
):
    from app.domain.models.session import SessionStatus

    session = await make_session(
        status=SessionStatus.COMPLETED.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
        execution_phase="running",
        terminal_reason=None,
    )

    summary = await agent_service_with_redis._supervisor.reconcile_running_background_at_boot()

    fresh = await session_repo.get_by_id(session.id)
    assert summary == {"finishing": 0, "suspended": 0, "total": 0}
    assert fresh.status == SessionStatus.COMPLETED
    assert fresh.execution_phase == "running"
    assert fresh.suspended_reason is None


async def test_idle_watchdog_scan_suspends_stale_background_session(
    agent_service_with_redis,
    make_session,
    session_repo,
    redis_client,
    sample_user,
):
    from app.domain.models.session import SessionStatus

    expires_at = datetime.now(timezone.utc) + timedelta(hours=2)
    session = await make_session(
        status=SessionStatus.RUNNING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires_at,
        execution_phase="running",
        was_background=True,
    )
    supervisor = agent_service_with_redis._supervisor
    await supervisor._run_lua_admit(
        session_id=session.id,
        user_id=sample_user.id,
        expires_at=expires_at,
        generation=session.execution_revision,
    )
    task = _CancelableTask()
    supervisor._register_runner(session.id, task)
    await redis_client.hset(
        f"supervisor:hot:{session.id}",
        mapping={
            "last_activity_at": (
                datetime.now(timezone.utc) - timedelta(seconds=181)
            ).timestamp(),
        },
    )

    await agent_service_with_redis._idle_watchdog._scan_once()

    fresh = await session_repo.get_by_id(session.id)
    assert fresh.status == SessionStatus.RUNNING
    assert fresh.execution_phase == "suspended"
    assert fresh.suspended_reason == "bg_idle_timeout"
    assert task.cancel_calls == ["supervisor_suspend"]
    assert session.id not in supervisor._runners
    assert await redis_client.zscore(f"supervisor:bg:{sample_user.id}", session.id)
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 1


async def test_idle_watchdog_expired_sweep_terminates_and_revokes_slots(
    agent_service_with_redis,
    sample_user,
    make_session,
    session_repo,
    redis_client,
):
    from app.domain.models.session import SessionStatus

    expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    session = await make_session(
        status=SessionStatus.RUNNING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires_at,
        execution_phase="running",
    )
    expires_score = expires_at.timestamp()
    await redis_client.hset(
        f"supervisor:user:{sample_user.id}",
        session.id,
        f"{expires_score:.6f}",
    )
    await redis_client.incr("supervisor:system:bg_count")
    await redis_client.zadd(
        f"supervisor:bg:{sample_user.id}",
        {session.id: expires_score},
    )

    await agent_service_with_redis._idle_watchdog._sweep_expired_once()

    fresh = await session_repo.get_by_id(session.id)
    assert fresh.status == SessionStatus.TIMED_OUT
    assert fresh.execution_phase == "terminated"
    assert fresh.terminal_reason == "watchdog_timeout"
    assert (
        await redis_client.hexists(f"supervisor:user:{sample_user.id}", session.id)
    ) == 0
    assert await redis_client.zscore(f"supervisor:bg:{sample_user.id}", session.id) is None
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 0


async def test_idle_watchdog_expired_sweep_terminates_suspended_only_background(
    agent_service_with_redis,
    sample_user,
    make_session,
    session_repo,
    redis_client,
):
    from app.domain.models.session import SessionStatus

    expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    session = await make_session(
        status=SessionStatus.RUNNING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires_at,
        execution_phase="suspended",
        suspended_reason="bg_idle_timeout",
    )
    expires_score = expires_at.timestamp()
    await redis_client.hset(
        f"supervisor:user:{sample_user.id}",
        session.id,
        f"{expires_score:.6f}",
    )
    await redis_client.incr("supervisor:system:bg_count")
    await redis_client.zadd(
        f"supervisor:bg:{sample_user.id}",
        {session.id: expires_score},
    )

    await agent_service_with_redis._idle_watchdog._sweep_expired_once()

    fresh = await session_repo.get_by_id(session.id)
    assert fresh.status == SessionStatus.TIMED_OUT
    assert fresh.execution_phase == "terminated"
    assert fresh.terminal_reason == "watchdog_timeout"
    assert (
        await redis_client.zscore(f"supervisor:bg:{sample_user.id}", session.id)
    ) is None
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 0


async def test_background_resume_refreshes_existing_redis_slot_score(
    agent_service_with_redis,
    sample_user,
    make_session,
    session_repo,
    redis_client,
):
    from app.domain.models.session import SessionStatus

    old_expires_at = datetime.now(timezone.utc) + timedelta(seconds=60)
    new_expires_at = datetime.now(timezone.utc) + timedelta(hours=2)
    session = await make_session(
        status=SessionStatus.RUNNING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=old_expires_at,
        execution_phase="suspended",
        suspended_reason="bg_idle_timeout",
        sandbox_state="active",
    )
    old_score = old_expires_at.timestamp()
    await redis_client.hset(
        f"supervisor:user:{sample_user.id}",
        session.id,
        f"{old_score:.6f}",
    )
    await redis_client.incr("supervisor:system:bg_count")
    await redis_client.zadd(
        f"supervisor:bg:{sample_user.id}",
        {session.id: old_score},
    )

    claim = await session_repo.claim_background_retry_from_suspend(
        session.id,
        expires_at=new_expires_at,
    )
    assert claim is not None
    claimed_retry_budget, claimed_execution_revision = claim

    admission_rc = await agent_service_with_redis._supervisor.resume(
        session_id=session.id,
        user_id=sample_user.id,
        execution_mode="background",
        expires_at=new_expires_at,
        retry_budget_remaining=claimed_retry_budget,
        expected_execution_revision=claimed_execution_revision,
    )

    fresh = await session_repo.get_by_id(session.id)
    refreshed_score = await redis_client.zscore(
        f"supervisor:bg:{sample_user.id}",
        session.id,
    )
    user_hash_value = await redis_client.hget(
        f"supervisor:user:{sample_user.id}",
        session.id,
    )
    assert claimed_retry_budget == 2
    assert admission_rc == 3
    assert fresh.execution_phase == "running"
    assert fresh.suspended_reason is None
    assert fresh.retry_budget_remaining == 2
    assert int(float(refreshed_score)) == int(new_expires_at.timestamp())
    assert int(float(user_hash_value)) == int(new_expires_at.timestamp())
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 1


async def test_terminate_does_not_overwrite_existing_terminal_reason(
    agent_service_with_redis,
    sample_user,
    make_session,
    session_repo,
    redis_client,
):
    from app.domain.models.session import SessionStatus

    expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    completed_at = datetime.now() - timedelta(seconds=1)
    session = await make_session(
        status=SessionStatus.COMPLETED.value,
        completed_at=completed_at,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires_at,
        execution_phase="terminated",
        terminal_reason="natural",
        was_background=True,
    )
    expires_score = expires_at.timestamp()
    await redis_client.hset(
        f"supervisor:user:{sample_user.id}",
        session.id,
        f"{expires_score:.6f}",
    )
    await redis_client.incr("supervisor:system:bg_count")
    await redis_client.zadd(
        f"supervisor:bg:{sample_user.id}",
        {session.id: expires_score},
    )

    await agent_service_with_redis._supervisor.terminate(
        session_id=session.id,
        user_id=sample_user.id,
        terminal_reason="watchdog_timeout",
        status=SessionStatus.TIMED_OUT,
    )

    fresh = await session_repo.get_by_id(session.id)
    assert fresh.status == SessionStatus.COMPLETED
    assert fresh.terminal_reason == "natural"
    assert fresh.execution_phase == "terminated"
    assert fresh.completed_at == completed_at
    assert (
        await redis_client.zscore(f"supervisor:bg:{sample_user.id}", session.id)
    ) is None
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 0


async def test_expired_sweep_revokes_foreground_residual_slot_without_timeout(
    agent_service_with_redis,
    sample_user,
    make_session,
    session_repo,
    redis_client,
):
    from app.domain.models.session import SessionStatus

    expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    session = await make_session(
        status=SessionStatus.RUNNING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires_at,
        execution_phase="running",
        was_background=True,
    )
    expires_score = expires_at.timestamp()
    await redis_client.hset(
        f"supervisor:user:{sample_user.id}",
        session.id,
        f"{expires_score:.6f}",
    )
    await redis_client.incr("supervisor:system:bg_count")
    await redis_client.zadd(
        f"supervisor:bg:{sample_user.id}",
        {session.id: expires_score},
    )
    await session_repo.update_supervisor_fields(
        session.id,
        execution_mode="foreground",
        background_reason=None,
        expires_at=None,
        execution_phase="running",
        suspended_reason=None,
    )

    await agent_service_with_redis._idle_watchdog._sweep_expired_once()

    fresh = await session_repo.get_by_id(session.id)
    assert fresh.status == SessionStatus.RUNNING
    assert fresh.execution_mode == "foreground"
    assert fresh.execution_phase == "running"
    assert fresh.terminal_reason is None
    assert (
        await redis_client.zscore(f"supervisor:bg:{sample_user.id}", session.id)
    ) is None
    assert (
        await redis_client.hexists(f"supervisor:user:{sample_user.id}", session.id)
    ) == 0
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 0


async def test_expired_sweep_retries_revoke_after_terminal_success(
    agent_service_with_redis,
    sample_user,
    make_session,
    session_repo,
    redis_client,
    monkeypatch,
):
    from app.domain.models.session import SessionStatus

    expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    session = await make_session(
        status=SessionStatus.RUNNING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires_at,
        execution_phase="running",
    )
    expires_score = expires_at.timestamp()
    await redis_client.hset(
        f"supervisor:user:{sample_user.id}",
        session.id,
        f"{expires_score:.6f}",
    )
    await redis_client.incr("supervisor:system:bg_count")
    await redis_client.zadd(
        f"supervisor:bg:{sample_user.id}",
        {session.id: expires_score},
    )

    supervisor = agent_service_with_redis._supervisor
    original_revoke = supervisor._lua_revoke
    revoke_calls = 0

    async def fail_revoke_once(**kwargs):
        nonlocal revoke_calls
        revoke_calls += 1
        if revoke_calls == 1:
            raise RuntimeError("simulated redis revoke failure")
        return await original_revoke(**kwargs)

    monkeypatch.setattr(supervisor, "_lua_revoke", fail_revoke_once)

    await agent_service_with_redis._idle_watchdog._sweep_expired_once()

    terminal = await session_repo.get_by_id(session.id)
    assert terminal.status == SessionStatus.TIMED_OUT
    assert terminal.terminal_reason == "watchdog_timeout"
    assert await redis_client.hexists(f"supervisor:user:{sample_user.id}", session.id)
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 1

    await agent_service_with_redis._idle_watchdog._sweep_expired_once()

    fresh = await session_repo.get_by_id(session.id)
    assert fresh.status == SessionStatus.TIMED_OUT
    assert fresh.terminal_reason == "watchdog_timeout"
    assert (
        await redis_client.zscore(f"supervisor:bg:{sample_user.id}", session.id)
    ) is None
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 0


async def test_expired_sweep_revokes_orphan_redis_slot_without_pg_row(
    agent_service_with_redis,
    sample_user,
    redis_client,
):
    expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    expires_score = expires_at.timestamp()
    orphan_session_id = "orphan-b3-bg-slot"
    await redis_client.hset(
        f"supervisor:user:{sample_user.id}",
        orphan_session_id,
        f"{expires_score:.6f}",
    )
    await redis_client.incr("supervisor:system:bg_count")
    await redis_client.zadd(
        f"supervisor:bg:{sample_user.id}",
        {orphan_session_id: expires_score},
    )

    await agent_service_with_redis._idle_watchdog._sweep_expired_once()

    assert (
        await redis_client.zscore(
            f"supervisor:bg:{sample_user.id}",
            orphan_session_id,
        )
    ) is None
    assert (
        await redis_client.hexists(
            f"supervisor:user:{sample_user.id}",
            orphan_session_id,
        )
    ) == 0
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 0


async def test_expired_sweep_terminal_failure_keeps_zset_retry_entry(
    agent_service_with_redis,
    sample_user,
    make_session,
    redis_client,
    monkeypatch,
):
    from app.domain.models.session import SessionStatus

    expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    session = await make_session(
        status=SessionStatus.RUNNING.value,
        execution_mode="background",
        background_reason="explicit",
        expires_at=expires_at,
        execution_phase="running",
    )
    expires_score = expires_at.timestamp()
    await redis_client.hset(
        f"supervisor:user:{sample_user.id}",
        session.id,
        f"{expires_score:.6f}",
    )
    await redis_client.incr("supervisor:system:bg_count")
    await redis_client.zadd(
        f"supervisor:bg:{sample_user.id}",
        {session.id: expires_score},
    )

    async def fail_terminate(**_kwargs):
        raise RuntimeError("simulated terminal write failure")

    monkeypatch.setattr(
        agent_service_with_redis._supervisor,
        "terminate_expired_background",
        fail_terminate,
    )

    await agent_service_with_redis._idle_watchdog._sweep_expired_once()

    assert await redis_client.zscore(f"supervisor:bg:{sample_user.id}", session.id) is not None
    assert await redis_client.hexists(f"supervisor:user:{sample_user.id}", session.id) == 1
    assert int(await redis_client.get("supervisor:system:bg_count") or 0) == 1
