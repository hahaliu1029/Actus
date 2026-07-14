from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.domain.errors.supervisor import SupervisorContractError
from app.domain.models.session import Session, SessionStatus
from app.domain.services.execution_supervisor import ExecutionSupervisor
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _RecordingRedis:
    def __init__(
        self,
        *,
        hincrby_value: int = 1,
        fail_hset: bool = False,
        generation: int | None = None,
    ) -> None:
        self.hincrby_value = hincrby_value
        self.fail_hset = fail_hset
        self.generation = generation
        self.hincrby_calls: list[tuple[str, str, int]] = []
        self.expire_calls: list[tuple[str, int]] = []
        self.hset_calls: list[dict[str, object]] = []
        self.hget_calls: list[tuple[str, str]] = []
        self.zadd_calls: list[tuple[str, dict[str, float]]] = []

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        self.hincrby_calls.append((key, field, amount))
        return self.hincrby_value

    async def expire(self, key: str, seconds: int) -> None:
        self.expire_calls.append((key, seconds))

    async def hset(
        self,
        key: str,
        field: str | None = None,
        value: int | None = None,
        mapping: dict[str, int] | None = None,
    ) -> None:
        if self.fail_hset:
            raise RuntimeError("redis hset failed")
        self.hset_calls.append(
            {"key": key, "field": field, "value": value, "mapping": mapping}
        )

    async def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self.zadd_calls.append((key, mapping))

    async def hget(self, key: str, field: str) -> str | None:
        self.hget_calls.append((key, field))
        return None if self.generation is None else str(self.generation)


class _BrokenRedis:
    async def hincrby(self, key: str, field: str, amount: int) -> int:
        raise RuntimeError("redis bounced")


class _Repo:
    def __init__(self, session: Session | None = None) -> None:
        self.updates: list[tuple[str, dict[str, str]]] = []
        self.promote_calls: list[tuple[str, dict[str, object]]] = []
        self.terminal_updates: list[tuple[str, SessionStatus, str]] = []
        self._session = session
        self.fail_update = False
        self.promote_result: int | None = 3
        self.suspend_idle_calls: list[str] = []

    async def update_supervisor_fields(self, session_id: str, **fields) -> None:
        if self.fail_update:
            raise RuntimeError("pg write failed")
        self.updates.append((session_id, fields))
        if self._session and self._session.id == session_id:
            for key, value in fields.items():
                setattr(self._session, key, value)

    async def promote_foreground_to_background(
        self,
        session_id: str,
        **fields,
    ) -> int | None:
        if self.fail_update:
            raise RuntimeError("pg write failed")
        self.promote_calls.append((session_id, fields))
        return self.promote_result

    async def suspend_running_background_if_active(self, session_id: str) -> bool:
        if self.fail_update:
            raise RuntimeError("pg write failed")
        session = self._session
        if session is None or session.id != session_id:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")
        if (
            session.status != SessionStatus.RUNNING
            or session.execution_mode != "background"
            or session.execution_phase not in ("running", "recovering")
        ):
            return False
        self.suspend_idle_calls.append(session_id)
        session.execution_phase = "suspended"
        session.suspended_reason = "bg_idle_timeout"
        return True

    async def get_by_id(self, session_id: str) -> Session | None:
        if self._session is None or self._session.id != session_id:
            return None
        return self._session

    async def update_to_terminal(
        self,
        session_id: str,
        status: SessionStatus,
        terminal_reason: str,
    ) -> None:
        self.terminal_updates.append((session_id, status, terminal_reason))
        if self._session and self._session.id == session_id:
            self._session.status = status
            self._session.terminal_reason = terminal_reason


class _CancelableTask:
    def __init__(self) -> None:
        self.cancel_calls: list[str] = []

    def cancel(self, reason: str = "stop") -> bool:
        self.cancel_calls.append(reason)
        return True


async def test_inflight_inc_swallow_redis_error() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=_BrokenRedis(), session_repository=object()
    )

    value = await supervisor.inflight_inc(session_id="session-1", kind="llm")

    assert value == 0


async def test_inflight_dec_swallow_redis_error() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=_BrokenRedis(), session_repository=object()
    )

    value = await supervisor.inflight_dec(session_id="session-1", kind="llm")

    assert value == 0


async def test_inflight_dec_does_not_refresh_hot_hash_ttl() -> None:
    redis = _RecordingRedis(hincrby_value=2)
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=object())

    value = await supervisor.inflight_dec(session_id="session-1", kind="llm")

    assert value == 2
    assert redis.hincrby_calls == [
        ("supervisor:hot:session-1", "inflight_llm_count", -1)
    ]
    assert redis.expire_calls == []


async def test_inflight_dec_preserves_negative_value_for_drift_observation() -> None:
    redis = _RecordingRedis(hincrby_value=-1)
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=object())

    value = await supervisor.inflight_dec(session_id="session-1", kind="llm")

    assert value == -1
    assert redis.hset_calls == []
    assert redis.expire_calls == []


async def test_lua_admit_receives_activity_timestamp(monkeypatch) -> None:
    from app.domain.services import execution_supervisor as supervisor_module

    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=object())
    calls: list[dict[str, object]] = []

    async def fake_run_lua_with_fallback(*args, **kwargs) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(
        supervisor_module,
        "run_lua_with_fallback",
        fake_run_lua_with_fallback,
    )
    before = datetime.now(timezone.utc).timestamp()

    await supervisor._run_lua_admit(
        session_id="session-1",
        user_id="user-1",
        expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        generation=7,
    )

    after = datetime.now(timezone.utc).timestamp()
    admit_args = calls[0]["args"]
    assert len(admit_args) == 9
    activity_at = float(admit_args[4])
    assert before <= activity_at <= after
    assert admit_args[5:] == [7, 0, "v1|7|user-1", 86400]


async def test_suspend_idle_cancels_live_task_with_supervisor_suspend(
    monkeypatch,
) -> None:
    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="running",
            was_background=True,
        )
    )
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)
    task = _CancelableTask()
    supervisor._register_runner("session-3", task)
    revoke_calls: list[dict[str, str]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor.suspend_idle(session_id="session-3", user_id="user-1")

    assert repo.suspend_idle_calls == ["session-3"]
    assert repo._session is not None
    assert repo._session.execution_phase == "suspended"
    assert repo._session.suspended_reason == "bg_idle_timeout"
    assert task.cancel_calls == ["supervisor_suspend"]
    assert revoke_calls == []
    assert "session-3" not in supervisor._runners


async def test_suspend_idle_noops_when_session_already_terminal() -> None:
    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            status=SessionStatus.COMPLETED,
            execution_mode="background",
            execution_phase="terminated",
            was_background=True,
        )
    )
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)
    task = _CancelableTask()
    supervisor._register_runner("session-3", task)

    await supervisor.suspend_idle(session_id="session-3", user_id="user-1")

    assert repo.suspend_idle_calls == []
    assert repo.updates == []
    assert repo._session is not None
    assert repo._session.status == SessionStatus.COMPLETED
    assert repo._session.execution_phase == "terminated"
    assert task.cancel_calls == []
    assert "session-3" in supervisor._runners


async def test_resume_background_existing_slot_resets_inflight_counts(
    monkeypatch,
) -> None:
    from datetime import datetime, timezone

    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="running",
            execution_revision=7,
        )
    )
    redis = _RecordingRedis()
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=repo)
    admit_calls: list[dict[str, object]] = []

    async def fake_admit(**kwargs) -> int:
        admit_calls.append(kwargs)
        return 3

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)

    async def fake_sync(**kwargs) -> int:
        return 1

    monkeypatch.setattr(supervisor, "_sync_background_expiry_to_redis", fake_sync)

    rc = await supervisor.resume(
        session_id="session-3",
        user_id="user-1",
        execution_mode="background",
        expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        retry_budget_remaining=1,
        expected_execution_revision=7,
    )

    assert rc == 3
    assert admit_calls[0]["generation"] == 7
    assert redis.hset_calls == [
        {
            "key": "supervisor:hot:session-3",
            "field": None,
            "value": None,
            "mapping": {
                "inflight_llm_count": 0,
                "inflight_tool_count": 0,
            },
        }
    ]


async def test_resume_background_existing_slot_returns_rc_when_inflight_reset_fails(
    monkeypatch,
) -> None:
    from datetime import datetime, timezone

    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="running",
            execution_revision=7,
        )
    )
    redis = _RecordingRedis(fail_hset=True)
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=repo)

    async def fake_admit(**kwargs) -> int:
        return 3

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)

    async def fake_sync(**kwargs) -> int:
        return 1

    monkeypatch.setattr(supervisor, "_sync_background_expiry_to_redis", fake_sync)

    rc = await supervisor.resume(
        session_id="session-3",
        user_id="user-1",
        execution_mode="background",
        expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        retry_budget_remaining=1,
        expected_execution_revision=7,
    )

    assert rc == 3


async def test_resume_background_does_not_patch_pg_after_retry_claim(
    monkeypatch,
) -> None:
    from datetime import datetime, timezone

    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            status=SessionStatus.COMPLETED,
            execution_mode="background",
            execution_phase="terminated",
        )
    )
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)

    async def fake_admit(**kwargs) -> int:
        return 0

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)

    with pytest.raises(SupervisorContractError, match="stale retry claim"):
        await supervisor.resume(
            session_id="session-3",
            user_id="user-1",
            execution_mode="background",
            expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
            retry_budget_remaining=1,
            expected_execution_revision=0,
        )

    assert repo.updates == []
    assert repo._session is not None
    assert repo._session.status == SessionStatus.COMPLETED
    assert repo._session.execution_phase == "terminated"


async def test_resume_background_rejects_stale_claim_before_redis_admission(
    monkeypatch,
) -> None:
    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="suspended",
            execution_revision=7,
        )
    )
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)
    admit_calls: list[dict[str, object]] = []

    async def fake_admit(**kwargs) -> int:
        admit_calls.append(kwargs)
        return 0

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)

    with pytest.raises(SupervisorContractError, match="stale retry claim"):
        await supervisor.resume(
            session_id="session-3",
            user_id="user-1",
            execution_mode="background",
            expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
            retry_budget_remaining=1,
            expected_execution_revision=7,
        )

    assert admit_calls == []


async def test_resume_background_post_admission_recheck_revokes_stale_generation(
    monkeypatch,
) -> None:
    session = Session(
        id="session-3",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        execution_revision=7,
    )
    repo = _Repo(session)
    supervisor = ExecutionSupervisor(
        redis_client=_RecordingRedis(), session_repository=repo
    )
    revoke_calls: list[dict[str, object]] = []

    async def fake_admit(**kwargs) -> int:
        session.execution_mode = "foreground"
        session.execution_phase = "running"
        session.execution_revision = 8
        return 0

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)
    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    with pytest.raises(SupervisorContractError, match="stale retry claim"):
        await supervisor.resume(
            session_id="session-3",
            user_id="user-1",
            execution_mode="background",
            expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
            retry_budget_remaining=1,
            expected_execution_revision=7,
        )

    assert revoke_calls == [
        {
            "session_id": "session-3",
            "user_id": "user-1",
            "reason": "resume_retry_stale",
            "expected_generation": 7,
            "allow_legacy": True,
        }
    ]


class _PostAdmissionReadFailureRepo(_Repo):
    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self.read_count = 0

    async def get_by_id(self, session_id: str) -> Session | None:
        self.read_count += 1
        if self.read_count == 2:
            raise RuntimeError("post-admission PG read failed")
        return await super().get_by_id(session_id)


async def test_resume_background_revokes_new_slot_when_post_read_raises(
    monkeypatch,
) -> None:
    session = Session(
        id="session-3",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        execution_revision=7,
    )
    supervisor = ExecutionSupervisor(
        redis_client=_RecordingRedis(),
        session_repository=_PostAdmissionReadFailureRepo(session),
    )
    revoke_calls: list[dict[str, object]] = []

    async def fake_admit(**kwargs) -> int:
        return 0

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)
    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    with pytest.raises(RuntimeError, match="post-admission PG read failed"):
        await supervisor.resume(
            session_id="session-3",
            user_id="user-1",
            execution_mode="background",
            expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
            previous_expires_at=datetime(2026, 5, 11, tzinfo=timezone.utc),
            retry_budget_remaining=1,
            expected_execution_revision=7,
        )

    assert revoke_calls == [
        {
            "session_id": "session-3",
            "user_id": "user-1",
            "reason": "resume_retry_rollback",
            "expected_generation": 7,
            "allow_legacy": True,
        }
    ]


async def test_resume_background_restores_existing_slot_when_sync_raises(
    monkeypatch,
) -> None:
    session = Session(
        id="session-3",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        execution_revision=7,
    )
    supervisor = ExecutionSupervisor(
        redis_client=_RecordingRedis(),
        session_repository=_Repo(session),
    )
    previous_expires_at = datetime(2026, 5, 11, tzinfo=timezone.utc)
    requested_expires_at = datetime(2026, 5, 12, tzinfo=timezone.utc)
    sync_calls: list[dict[str, object]] = []

    async def fake_admit(**kwargs) -> int:
        return 3

    async def fake_sync(**kwargs) -> int:
        sync_calls.append(kwargs)
        if len(sync_calls) == 1:
            raise RuntimeError("expiry sync failed")
        return 1

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)
    monkeypatch.setattr(supervisor, "_sync_background_expiry_to_redis", fake_sync)

    with pytest.raises(RuntimeError, match="expiry sync failed"):
        await supervisor.resume(
            session_id="session-3",
            user_id="user-1",
            execution_mode="background",
            expires_at=requested_expires_at,
            previous_expires_at=previous_expires_at,
            retry_budget_remaining=1,
            expected_execution_revision=7,
        )

    assert sync_calls == [
        {
            "session_id": "session-3",
            "user_id": "user-1",
            "expires_at": requested_expires_at,
            "execution_revision": 7,
        },
        {
            "session_id": "session-3",
            "user_id": "user-1",
            "expires_at": previous_expires_at,
            "execution_revision": 7,
        },
    ]


async def test_resume_background_reconciles_ambiguous_lua_cancel(
    monkeypatch,
) -> None:
    session = Session(
        id="session-3",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        execution_revision=7,
    )
    supervisor = ExecutionSupervisor(
        redis_client=_RecordingRedis(),
        session_repository=_Repo(session),
    )
    previous_expires_at = datetime(2026, 5, 11, tzinfo=timezone.utc)
    sync_calls: list[dict[str, object]] = []

    async def fake_admit(**kwargs) -> int:
        raise asyncio.CancelledError("ambiguous Lua result")

    async def fake_sync(**kwargs) -> int:
        sync_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)
    monkeypatch.setattr(supervisor, "_sync_background_expiry_to_redis", fake_sync)

    with pytest.raises(asyncio.CancelledError, match="ambiguous Lua result"):
        await supervisor.resume(
            session_id="session-3",
            user_id="user-1",
            execution_mode="background",
            expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
            previous_expires_at=previous_expires_at,
            retry_budget_remaining=1,
            expected_execution_revision=7,
        )

    assert sync_calls == [
        {
            "session_id": "session-3",
            "user_id": "user-1",
            "expires_at": previous_expires_at,
            "execution_revision": 7,
        }
    ]


@pytest.mark.parametrize("admission_rc", [0, 5])
async def test_resume_background_accepts_new_or_matching_generation_slot(
    monkeypatch,
    admission_rc: int,
) -> None:
    session = Session(
        id="session-3",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        execution_revision=7,
    )
    supervisor = ExecutionSupervisor(
        redis_client=_RecordingRedis(),
        session_repository=_Repo(session),
    )

    async def fake_admit(**kwargs) -> int:
        return admission_rc

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)

    assert await supervisor.resume(
        session_id="session-3",
        user_id="user-1",
        execution_mode="background",
        expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        retry_budget_remaining=1,
        expected_execution_revision=7,
    ) == admission_rc


@pytest.mark.parametrize(
    ("admission_rc", "expected_code", "error_type"),
    [
        (1, "R1", SupervisorContractError),
        (2, "R2", SupervisorContractError),
        (4, "R3", SupervisorContractError),
        (8, None, RuntimeError),
    ],
)
async def test_resume_background_classifies_admission_rejections(
    monkeypatch,
    admission_rc: int,
    expected_code: str | None,
    error_type: type[Exception],
) -> None:
    session = Session(
        id="session-3",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        execution_revision=7,
    )
    supervisor = ExecutionSupervisor(
        redis_client=_RecordingRedis(),
        session_repository=_Repo(session),
    )

    async def fake_admit(**kwargs) -> int:
        return admission_rc

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)

    with pytest.raises(error_type) as error:
        await supervisor.resume(
            session_id="session-3",
            user_id="user-1",
            execution_mode="background",
            expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
            retry_budget_remaining=1,
            expected_execution_revision=7,
        )

    if expected_code is not None:
        assert isinstance(error.value, SupervisorContractError)
        assert error.value.rejection_code == expected_code


@pytest.mark.parametrize("initial_rc", [6, 7])
async def test_resume_background_authoritatively_repairs_legacy_admission_rc(
    monkeypatch,
    initial_rc: int,
) -> None:
    session = Session(
        id="session-3",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        execution_revision=7,
    )
    supervisor = ExecutionSupervisor(
        redis_client=_RecordingRedis(),
        session_repository=_Repo(session),
    )
    admit_calls: list[dict[str, object]] = []
    admission_results = iter((initial_rc, 3))
    reset_calls: list[str] = []

    async def fake_admit(**kwargs) -> int:
        admit_calls.append(kwargs)
        return next(admission_results)

    async def fake_sync(**kwargs) -> int:
        return 1

    async def fake_reset(*, session_id: str) -> None:
        reset_calls.append(session_id)

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)
    monkeypatch.setattr(supervisor, "_sync_background_expiry_to_redis", fake_sync)
    monkeypatch.setattr(supervisor, "_reset_inflight_counts", fake_reset)

    assert await supervisor.resume(
        session_id="session-3",
        user_id="user-1",
        execution_mode="background",
        expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        retry_budget_remaining=1,
        expected_execution_revision=7,
    ) == 3

    assert len(admit_calls) == 2
    assert admit_calls[1]["authoritative_repair"] is True
    assert reset_calls == ["session-3"]


async def test_rollback_background_resume_admission_revokes_new_slot(
    monkeypatch,
) -> None:
    from datetime import datetime, timezone

    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="suspended",
            execution_revision=7,
        )
    )
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)
    revoke_calls: list[dict[str, object]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor.rollback_background_resume_admission(
        session_id="session-3",
        user_id="user-1",
        admission_rc=0,
        previous_expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        expected_execution_revision=7,
    )

    assert revoke_calls == [
        {
            "session_id": "session-3",
            "user_id": "user-1",
            "reason": "resume_retry_rollback",
            "expected_generation": 7,
            "allow_legacy": True,
        }
    ]


async def test_rollback_background_resume_admission_restores_existing_slot_expiry(
    monkeypatch,
) -> None:
    from datetime import datetime, timezone

    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=object())
    revoke_calls: list[dict[str, str]] = []
    sync_calls: list[dict[str, object]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    async def fake_sync(**kwargs) -> int:
        sync_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)
    monkeypatch.setattr(supervisor, "_sync_background_expiry_to_redis", fake_sync)
    previous_expires_at = datetime(2026, 5, 12, tzinfo=timezone.utc)

    await supervisor.rollback_background_resume_admission(
        session_id="session-3",
        user_id="user-1",
        admission_rc=3,
        previous_expires_at=previous_expires_at,
        expected_execution_revision=7,
    )

    assert revoke_calls == []
    assert sync_calls == [
        {
            "session_id": "session-3",
            "user_id": "user-1",
            "expires_at": previous_expires_at,
            "execution_revision": 7,
        }
    ]


async def test_revoke_background_resume_admission_revokes_existing_slot(
    monkeypatch,
) -> None:
    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            status=SessionStatus.COMPLETED,
            execution_mode="background",
            execution_phase="terminated",
            execution_revision=7,
        )
    )
    redis = _RecordingRedis(generation=7)
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=repo)
    revoke_calls: list[dict[str, object]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor.revoke_background_resume_admission(
        session_id="session-3",
        user_id="user-1",
        admission_rc=3,
        expected_execution_revision=7,
    )

    assert revoke_calls == [
        {
            "session_id": "session-3",
            "user_id": "user-1",
            "reason": "resume_retry_terminal",
            "expected_generation": 7,
            "allow_legacy": True,
        }
    ]
    assert redis.hget_calls == []


async def test_promote_revokes_slot_when_pg_guard_does_not_match(monkeypatch) -> None:
    from datetime import datetime, timezone

    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            status=SessionStatus.RUNNING,
            execution_mode="foreground",
            execution_phase="running",
            execution_revision=7,
        )
    )
    repo.promote_result = None
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)
    admit_calls: list[dict[str, object]] = []
    revoke_calls: list[dict[str, object]] = []

    async def fake_admit(**kwargs) -> int:
        admit_calls.append(kwargs)
        return 0

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_run_lua_admit", fake_admit)
    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    retry_budget_remaining = await supervisor.promote(
        session_id="session-3",
        user_id="user-1",
        expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
    )

    assert retry_budget_remaining is None
    assert admit_calls[0]["generation"] == 8
    assert len(repo.promote_calls) == 1
    promoted_session_id, promote_fields = repo.promote_calls[0]
    assert promoted_session_id == "session-3"
    assert promote_fields["expires_at"] == datetime(
        2026, 5, 12, tzinfo=timezone.utc
    )
    assert promote_fields["retry_budget_remaining"] == 3
    assert promote_fields["expected_execution_revision"] == 7
    assert promote_fields["background_reason"] == "auto_degrade"
    pending_event = promote_fields["pending_event"]
    assert pending_event.payload.execution_revision == 8
    assert revoke_calls == [
        {
            "session_id": "session-3",
            "user_id": "user-1",
            "reason": "promote_stale",
            "expected_generation": 8,
            "allow_legacy": True,
        }
    ]


async def test_terminate_watchdog_timeout_emits_bg_failed_watchdog(
    monkeypatch,
) -> None:
    repo = _Repo(
        Session(
            id="session-expired",
            user_id="user-1",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="running",
            was_background=True,
            execution_revision=7,
        )
    )
    supervisor = ExecutionSupervisor(
        redis_client=_RecordingRedis(generation=7),
        session_repository=repo,
        session_state_machine=DefaultSessionStateMachine(uow_factory=lambda: None),
    )
    emitter = AsyncMock()
    revoke_calls: list[dict[str, object]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor.terminate(
        session_id="session-expired",
        user_id="user-1",
        terminal_reason="watchdog_timeout",
        status=SessionStatus.TIMED_OUT,
        notification_emitter=emitter,
    )

    assert repo.terminal_updates == [
        ("session-expired", SessionStatus.TIMED_OUT, "watchdog_timeout")
    ]
    emitter.emit.assert_awaited_once_with(
        user_id="user-1",
        event_type="bg_failed_watchdog",
        payload={"session_id": "session-expired"},
    )
    assert revoke_calls == [
        {
            "session_id": "session-expired",
            "user_id": "user-1",
            "reason": "watchdog_timeout",
            "expected_generation": 7,
            "allow_legacy": True,
        }
    ]


async def test_runner_session_complete_skips_revoke_for_supervisor_suspend(
    monkeypatch,
) -> None:
    supervisor = ExecutionSupervisor(
        redis_client=object(), session_repository=object()
    )
    runner = object()
    supervisor._register_runner("session-1", runner)
    revoke_calls: list[dict[str, str]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor._on_runner_session_complete(
        session_id="session-1",
        user_id="user-1",
        cancel_reason="supervisor_suspend",
    )

    assert revoke_calls == []
    assert "session-1" not in supervisor._runners


async def test_runner_session_complete_revokes_for_terminal_reason(
    monkeypatch,
) -> None:
    repo = _Repo(
        Session(
            id="session-2",
            user_id="user-1",
            status=SessionStatus.COMPLETED,
            execution_mode="background",
            execution_phase="terminated",
            execution_revision=7,
        )
    )
    redis = _RecordingRedis(generation=7)
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=repo)
    supervisor._register_runner("session-2", object())
    revoke_calls: list[dict[str, object]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor._on_runner_session_complete(
        session_id="session-2",
        user_id="user-1",
        cancel_reason="user_cancel",
    )

    assert revoke_calls == [
        {
            "session_id": "session-2",
            "user_id": "user-1",
            "reason": "user_cancel",
            "expected_generation": 7,
            "allow_legacy": True,
        }
    ]
    assert redis.hget_calls == [
        ("supervisor:bg-generation:user-1", "session-2")
    ]
    assert "session-2" not in supervisor._runners
