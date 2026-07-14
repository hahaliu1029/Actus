from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from app.application.errors.exceptions import ConflictError
from app.application.services.agent_service import AgentService
from app.domain.errors.sandbox_lifecycle import (
    SandboxLifecycleError,
    SessionFinalizedError,
)
from app.domain.errors.supervisor import SupervisorContractError
from app.domain.models.session import (
    SandboxBinding,
    SandboxBindingState,
    Session,
    SessionStatus,
)

from tests.app.application.services.conftest import default_snapshot as _default_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SessionRepo:
    def __init__(self, claim_result: tuple[int, int] | None = (1, 7)) -> None:
        self.claim_result = claim_result
        self.claim_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []
        self.rollback_calls: list[dict[str, object]] = []
        self.rollback_result = True
        self.terminal_calls: list[dict[str, object]] = []

    async def claim_background_retry_from_suspend(
        self,
        session_id: str,
        *,
        expires_at: datetime,
    ) -> tuple[int, int] | None:
        self.claim_calls.append({"session_id": session_id, "expires_at": expires_at})
        return self.claim_result

    async def update_supervisor_fields(self, session_id: str, **fields: object) -> None:
        self.update_calls.append({"session_id": session_id, **fields})

    async def rollback_background_retry_claim_if_active(
        self,
        session_id: str,
        *,
        expected_execution_revision: int,
        retry_budget_remaining: int,
        expires_at: datetime | None,
        suspended_reason: str | None,
    ) -> bool:
        self.rollback_calls.append(
            {
                "session_id": session_id,
                "expected_execution_revision": expected_execution_revision,
                "retry_budget_remaining": retry_budget_remaining,
                "expires_at": expires_at,
                "suspended_reason": suspended_reason,
            }
        )
        return self.rollback_result

    async def update_to_terminal(
        self,
        session_id: str,
        status: SessionStatus,
        terminal_reason: str,
    ) -> bool:
        self.terminal_calls.append(
            {
                "session_id": session_id,
                "status": status,
                "terminal_reason": terminal_reason,
            }
        )
        return True


class _Uow:
    def __init__(self, repo: _SessionRepo) -> None:
        self.session = repo

    async def __aenter__(self) -> "_Uow":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


class _Lifecycle:
    def __init__(self) -> None:
        self.resume_calls: list[str] = []
        self.suspend_calls: list[str] = []

    async def resume(self, session_id: str) -> object:
        self.resume_calls.append(session_id)
        return object()

    async def suspend(self, session_id: str) -> None:
        self.suspend_calls.append(session_id)


class _FinalizingLifecycle(_Lifecycle):
    async def resume(self, session_id: str) -> object:
        self.resume_calls.append(session_id)
        raise SessionFinalizedError(
            session_id,
            destroyed_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
            detail="sandbox lost",
        )


class _FailingLifecycle(_Lifecycle):
    async def resume(self, session_id: str) -> object:
        self.resume_calls.append(session_id)
        raise SandboxLifecycleError("sandbox resume failed")


class _CancelledLifecycle(_Lifecycle):
    async def resume(self, session_id: str) -> object:
        self.resume_calls.append(session_id)
        raise asyncio.CancelledError("cancelled during sandbox resume")


class _Supervisor:
    def __init__(
        self,
        *,
        fail_resume: bool = False,
        fail_unexpected: bool = False,
        fail_terminate: bool = False,
        admission_rc: int = 0,
    ) -> None:
        self.resume_calls: list[dict[str, object]] = []
        self.terminate_calls: list[dict[str, object]] = []
        self.rollback_admission_calls: list[dict[str, object]] = []
        self.revoke_admission_calls: list[dict[str, object]] = []
        self.fail_resume = fail_resume
        self.fail_unexpected = fail_unexpected
        self.fail_terminate = fail_terminate
        self.admission_rc = admission_rc
        self.cleanup_expiry_calls = 0
        self.cleanup_expiry = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
        self.fence_events: list[tuple[str, str]] = []
        self.fence_active = False
        self.resume_inside_fence = False

    @asynccontextmanager
    async def mode_transition_fence(self, *, session_id: str):
        self.fence_active = True
        self.fence_events.append(("enter", session_id))
        try:
            yield
        finally:
            self.fence_events.append(("exit", session_id))
            self.fence_active = False

    def new_auto_degrade_cleanup_expiry(self) -> datetime:
        self.cleanup_expiry_calls += 1
        return self.cleanup_expiry

    async def resume(self, **kwargs: object) -> int:
        self.resume_inside_fence = self.fence_active
        self.resume_calls.append(kwargs)
        if self.fail_resume:
            raise SupervisorContractError("R1", "suspended", "background", "full")
        if self.fail_unexpected:
            raise RuntimeError("pg write failed")
        return self.admission_rc

    async def terminate(self, **kwargs: object) -> None:
        self.terminate_calls.append(kwargs)
        if self.fail_terminate:
            raise RuntimeError("terminal write failed")

    async def rollback_background_resume_admission(self, **kwargs: object) -> None:
        self.rollback_admission_calls.append(kwargs)

    async def revoke_background_resume_admission(self, **kwargs: object) -> None:
        self.revoke_admission_calls.append(kwargs)


def _make_service(
    lifecycle: _Lifecycle,
    repo: _SessionRepo | None = None,
    supervisor: _Supervisor | None = None,
) -> AgentService:
    repo = repo or _SessionRepo()
    service = AgentService(
        uow_factory=lambda: _Uow(repo),
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=object,
        search_engine=object(),
        file_storage=object(),
        sandbox_lifecycle_service=lifecycle,
    )
    service._supervisor = supervisor or _Supervisor()
    return service


def _background_suspended_session(
    *,
    retry_budget_remaining: int = 2,
    binding_state: SandboxBindingState = SandboxBindingState.SUSPENDED,
    expires_at: datetime | None = None,
    background_reason: str = "explicit",
) -> Session:
    return Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="suspended",
        background_reason=background_reason,
        retry_budget_remaining=retry_budget_remaining,
        expires_at=expires_at,
        sandbox_binding=SandboxBinding(id="sandbox-1", state=binding_state),
    )


async def test_retry_from_suspend_resumes_sandbox_and_restarts_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    service = _make_service(lifecycle)
    session = _background_suspended_session()
    resumed: list[tuple[Session, str]] = []

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def _resume_task_with_handoff(
        session_arg: Session, text: str, *, retry_lifecycle_context=None,
    ) -> object:
        resumed.append((session_arg, text))
        return object()

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)
    monkeypatch.setattr(service, "_resume_task_with_handoff", _resume_task_with_handoff)

    result = await service.retry_from_suspend(
        "s1",
        "u1",
        is_admin=False,
        user_role="user",
    )

    assert result["status"] == SessionStatus.RUNNING
    assert result["request_status"] == "resumed"
    assert result["retry_budget_remaining"] == 1
    assert lifecycle.resume_calls == ["s1"]
    assert service._supervisor.resume_calls
    supervisor_call = service._supervisor.resume_calls[0]
    assert supervisor_call["session_id"] == "s1"
    assert supervisor_call["user_id"] == "u1"
    assert supervisor_call["execution_mode"] == "background"
    assert supervisor_call["retry_budget_remaining"] == 1
    assert supervisor_call["expected_execution_revision"] == 7
    assert service._supervisor.resume_inside_fence is True
    assert service._supervisor.fence_events == [
        ("enter", "s1"),
        ("exit", "s1"),
    ]
    assert isinstance(supervisor_call["expires_at"], datetime)
    assert supervisor_call["expires_at"].tzinfo == timezone.utc
    assert resumed[0][0].execution_phase == "running"
    assert resumed[0][0].retry_budget_remaining == 1
    assert "后台任务" in resumed[0][1]


async def test_retry_auto_degrade_uses_shared_cleanup_window_then_rolls_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    supervisor = _Supervisor()
    service = _make_service(lifecycle, supervisor=supervisor)
    session = _background_suspended_session(background_reason="auto_degrade")

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def _resume_task_with_handoff(*args, **kwargs) -> object:
        return object()

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)
    monkeypatch.setattr(service, "_resume_task_with_handoff", _resume_task_with_handoff)

    result = await service.retry_from_suspend("s1", "u1")

    assert supervisor.cleanup_expiry_calls == 1
    assert supervisor.resume_calls[0]["expires_at"] == supervisor.cleanup_expiry
    assert result["expires_at"] == service._to_unix_seconds(supervisor.cleanup_expiry)


async def test_retry_explicit_background_preserves_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    supervisor = _Supervisor()
    service = _make_service(lifecycle, supervisor=supervisor)
    explicit_deadline = datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc)
    session = _background_suspended_session(expires_at=explicit_deadline)

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def _resume_task_with_handoff(*args, **kwargs) -> object:
        return object()

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)
    monkeypatch.setattr(service, "_resume_task_with_handoff", _resume_task_with_handoff)

    result = await service.retry_from_suspend("s1", "u1")

    assert supervisor.cleanup_expiry_calls == 0
    assert supervisor.resume_calls[0]["expires_at"] == explicit_deadline
    assert result["expires_at"] == service._to_unix_seconds(explicit_deadline)


async def test_retry_from_suspend_rejects_lost_atomic_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=None)
    service = _make_service(lifecycle, repo=repo)
    session = _background_suspended_session()

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(ConflictError):
        await service.retry_from_suspend("s1", "u1")

    assert repo.claim_calls
    assert repo.update_calls == []
    assert lifecycle.resume_calls == []
    assert service._supervisor.resume_calls == []


async def test_retry_from_suspend_rolls_back_claim_when_supervisor_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor(fail_resume=True)
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    session = _background_suspended_session()

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(ConflictError):
        await service.retry_from_suspend("s1", "u1")

    assert lifecycle.resume_calls == []
    assert repo.rollback_calls == [
        {
            "session_id": "s1",
            "expected_execution_revision": 7,
            "retry_budget_remaining": 2,
            "expires_at": None,
            "suspended_reason": None,
        }
    ]
    assert repo.update_calls == []


async def test_retry_from_suspend_rolls_back_claim_when_supervisor_fails_unexpectedly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor(fail_unexpected=True)
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    session = _background_suspended_session()

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(RuntimeError):
        await service.retry_from_suspend("s1", "u1")

    assert lifecycle.resume_calls == []
    assert repo.rollback_calls == [
        {
            "session_id": "s1",
            "expected_execution_revision": 7,
            "retry_budget_remaining": 2,
            "expires_at": None,
            "suspended_reason": None,
        }
    ]
    assert repo.update_calls == []


async def test_retry_from_suspend_revokes_new_redis_slot_when_lifecycle_resume_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _FailingLifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor(admission_rc=0)
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    original_expires_at = datetime(2026, 5, 12, tzinfo=timezone.utc)
    session = _background_suspended_session(expires_at=original_expires_at)

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(ConflictError):
        await service.retry_from_suspend("s1", "u1")

    assert repo.rollback_calls == [
        {
            "session_id": "s1",
            "expected_execution_revision": 7,
            "retry_budget_remaining": 2,
            "expires_at": original_expires_at,
            "suspended_reason": None,
        }
    ]
    assert supervisor.rollback_admission_calls == [
        {
            "session_id": "s1",
            "user_id": "u1",
            "admission_rc": 0,
            "previous_expires_at": original_expires_at,
            "expected_execution_revision": 7,
        }
    ]


async def test_retry_from_suspend_restores_existing_redis_slot_when_handoff_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor(admission_rc=3)
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    original_expires_at = datetime(2026, 5, 12, tzinfo=timezone.utc)
    session = _background_suspended_session(expires_at=original_expires_at)

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def _resume_task_with_handoff(*args, **kwargs) -> object:
        raise RuntimeError("handoff failed")

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)
    monkeypatch.setattr(service, "_resume_task_with_handoff", _resume_task_with_handoff)

    with pytest.raises(RuntimeError):
        await service.retry_from_suspend("s1", "u1")

    assert repo.rollback_calls == [
        {
            "session_id": "s1",
            "expected_execution_revision": 7,
            "retry_budget_remaining": 2,
            "expires_at": original_expires_at,
            "suspended_reason": None,
        }
    ]
    assert supervisor.rollback_admission_calls == [
        {
            "session_id": "s1",
            "user_id": "u1",
            "admission_rc": 3,
            "previous_expires_at": original_expires_at,
            "expected_execution_revision": 7,
        }
    ]
    assert lifecycle.suspend_calls == ["s1"]


async def test_retry_from_suspend_skips_redis_rollback_when_pg_cas_loses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    repo.rollback_result = False
    supervisor = _Supervisor(admission_rc=3)
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    session = _background_suspended_session(
        expires_at=datetime(2026, 5, 12, tzinfo=timezone.utc)
    )

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def _resume_task_with_handoff(*args, **kwargs) -> object:
        raise RuntimeError("handoff failed after reconnect won")

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)
    monkeypatch.setattr(service, "_resume_task_with_handoff", _resume_task_with_handoff)

    with pytest.raises(RuntimeError, match="handoff failed after reconnect won"):
        await service.retry_from_suspend("s1", "u1")

    assert repo.rollback_calls
    assert supervisor.rollback_admission_calls == []
    assert lifecycle.suspend_calls == []


async def test_retry_from_suspend_rejects_exhausted_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    service = _make_service(lifecycle)
    session = _background_suspended_session(retry_budget_remaining=0)

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(ConflictError):
        await service.retry_from_suspend("s1", "u1")

    assert lifecycle.resume_calls == []
    assert service._supervisor.resume_calls == []


async def test_retry_from_suspend_rejects_phase_that_already_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    service = _make_service(lifecycle)
    session = _background_suspended_session()
    session.execution_phase = "running"

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(ConflictError):
        await service.retry_from_suspend("s1", "u1")

    assert lifecycle.resume_calls == []
    assert service._supervisor.resume_calls == []


async def test_retry_from_suspend_rejects_destroyed_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    service = _make_service(lifecycle)
    session = _background_suspended_session(
        binding_state=SandboxBindingState.DESTROYED
    )

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(ConflictError):
        await service.retry_from_suspend("s1", "u1")

    assert lifecycle.resume_calls == []
    assert service._supervisor.resume_calls == []


async def test_retry_from_suspend_finalizes_when_resume_discovers_lost_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _FinalizingLifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor()
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    session = _background_suspended_session()

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(ConflictError):
        await service.retry_from_suspend("s1", "u1")

    assert lifecycle.resume_calls == ["s1"]
    assert repo.update_calls == []
    assert supervisor.terminate_calls == [
        {
            "session_id": "s1",
            "user_id": "u1",
            "terminal_reason": "resume_state_lost",
            "status": SessionStatus.TIMED_OUT,
        }
    ]


async def test_retry_from_suspend_fallback_terminalizes_when_supervisor_terminate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _FinalizingLifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor(fail_terminate=True, admission_rc=3)
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    original_expires_at = datetime(2026, 5, 12, tzinfo=timezone.utc)
    session = _background_suspended_session(expires_at=original_expires_at)

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(ConflictError):
        await service.retry_from_suspend("s1", "u1")

    assert repo.terminal_calls == [
        {
            "session_id": "s1",
            "status": SessionStatus.TIMED_OUT,
            "terminal_reason": "resume_state_lost",
        }
    ]
    assert repo.update_calls == [
        {
            "session_id": "s1",
            "retry_budget_remaining": 0,
            "expires_at": None,
            "suspended_reason": None,
        }
    ]
    assert supervisor.revoke_admission_calls == [
        {
            "session_id": "s1",
            "user_id": "u1",
            "admission_rc": 3,
            "expected_execution_revision": 7,
        }
    ]
    assert supervisor.rollback_admission_calls == []


class _CommitFailUow(_Uow):
    def __init__(self, repo: _SessionRepo) -> None:
        super().__init__(repo)
        self.db_session = self

    async def commit(self) -> None:
        raise RuntimeError("claim commit failed")


async def test_retry_from_suspend_claim_commit_failure_stops_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor()
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    service._uow_factory = lambda: _CommitFailUow(repo)
    session = _background_suspended_session()

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(RuntimeError, match="claim commit failed"):
        await service.retry_from_suspend("s1", "u1")

    assert supervisor.resume_calls == []
    assert lifecycle.resume_calls == []
    assert supervisor.fence_events == [("enter", "s1"), ("exit", "s1")]


class _CancelCommitOnceUow(_Uow):
    def __init__(self, repo: _SessionRepo, factory: "_CancelCommitOnceFactory") -> None:
        super().__init__(repo)
        self.db_session = self
        self.factory = factory

    async def commit(self) -> None:
        if self.factory.cancel_next_commit:
            self.factory.cancel_next_commit = False
            raise asyncio.CancelledError("ambiguous claim commit")


class _CancelCommitOnceFactory:
    def __init__(self, repo: _SessionRepo) -> None:
        self.repo = repo
        self.cancel_next_commit = True

    def __call__(self) -> _CancelCommitOnceUow:
        return _CancelCommitOnceUow(self.repo, self)


async def test_retry_claim_cancelled_commit_compensates_ambiguous_pg_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor()
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    service._uow_factory = _CancelCommitOnceFactory(repo)
    session = _background_suspended_session()

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(asyncio.CancelledError, match="ambiguous claim commit"):
        await service.retry_from_suspend("s1", "u1")

    assert repo.rollback_calls == [
        {
            "session_id": "s1",
            "expected_execution_revision": 7,
            "retry_budget_remaining": 2,
            "expires_at": None,
            "suspended_reason": None,
        }
    ]
    assert supervisor.resume_calls == []
    assert lifecycle.resume_calls == []


class _CancelledResumeSupervisor(_Supervisor):
    async def resume(self, **kwargs: object) -> int:
        self.resume_inside_fence = self.fence_active
        self.resume_calls.append(kwargs)
        raise asyncio.CancelledError("cancelled during Redis admission")


class _CancelledFenceExitSupervisor(_Supervisor):
    @asynccontextmanager
    async def mode_transition_fence(self, *, session_id: str):
        self.fence_active = True
        self.fence_events.append(("enter", session_id))
        try:
            yield
        finally:
            self.fence_events.append(("exit", session_id))
            self.fence_active = False
        raise asyncio.CancelledError("cancelled during fence exit")


async def test_retry_resume_cancelled_after_claim_commit_rolls_back_pg_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _CancelledResumeSupervisor()
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    session = _background_suspended_session()

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(asyncio.CancelledError, match="Redis admission"):
        await service.retry_from_suspend("s1", "u1")

    assert repo.rollback_calls == [
        {
            "session_id": "s1",
            "expected_execution_revision": 7,
            "retry_budget_remaining": 2,
            "expires_at": None,
            "suspended_reason": None,
        }
    ]
    assert lifecycle.resume_calls == []


async def test_retry_fence_exit_cancelled_after_admission_rolls_back_claim_and_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _CancelledFenceExitSupervisor(admission_rc=0)
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    original_expires_at = datetime(2026, 5, 12, tzinfo=timezone.utc)
    session = _background_suspended_session(expires_at=original_expires_at)

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(asyncio.CancelledError, match="fence exit"):
        await service.retry_from_suspend("s1", "u1")

    assert repo.rollback_calls == [
        {
            "session_id": "s1",
            "expected_execution_revision": 7,
            "retry_budget_remaining": 2,
            "expires_at": original_expires_at,
            "suspended_reason": None,
        }
    ]
    assert supervisor.rollback_admission_calls == [
        {
            "session_id": "s1",
            "user_id": "u1",
            "admission_rc": 0,
            "previous_expires_at": original_expires_at,
            "expected_execution_revision": 7,
        }
    ]
    assert lifecycle.resume_calls == []


async def test_retry_sandbox_resume_cancelled_rolls_back_claim_slot_and_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _CancelledLifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor(admission_rc=0)
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    original_expires_at = datetime(2026, 5, 12, tzinfo=timezone.utc)
    session = _background_suspended_session(expires_at=original_expires_at)

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)

    with pytest.raises(asyncio.CancelledError, match="sandbox resume"):
        await service.retry_from_suspend("s1", "u1")

    assert len(repo.rollback_calls) == 1
    assert repo.rollback_calls[0]["expected_execution_revision"] == 7
    assert supervisor.rollback_admission_calls == [
        {
            "session_id": "s1",
            "user_id": "u1",
            "admission_rc": 0,
            "previous_expires_at": original_expires_at,
            "expected_execution_revision": 7,
        }
    ]
    assert lifecycle.suspend_calls == ["s1"]


async def test_retry_handoff_cancelled_after_sandbox_resume_rolls_back_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _Lifecycle()
    repo = _SessionRepo(claim_result=(1, 7))
    supervisor = _Supervisor(admission_rc=3)
    service = _make_service(lifecycle, repo=repo, supervisor=supervisor)
    original_expires_at = datetime(2026, 5, 12, tzinfo=timezone.utc)
    session = _background_suspended_session(expires_at=original_expires_at)

    async def _get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def _resume_task_with_handoff(*args, **kwargs) -> object:
        raise asyncio.CancelledError("cancelled during handoff")

    monkeypatch.setattr(service, "_get_accessible_session", _get_accessible_session)
    monkeypatch.setattr(service, "_resume_task_with_handoff", _resume_task_with_handoff)

    with pytest.raises(asyncio.CancelledError, match="handoff"):
        await service.retry_from_suspend("s1", "u1")

    assert lifecycle.resume_calls == ["s1"]
    assert len(repo.rollback_calls) == 1
    assert repo.rollback_calls[0]["expected_execution_revision"] == 7
    assert supervisor.rollback_admission_calls == [
        {
            "session_id": "s1",
            "user_id": "u1",
            "admission_rc": 3,
            "previous_expires_at": original_expires_at,
            "expected_execution_revision": 7,
        }
    ]
    assert lifecycle.suspend_calls == ["s1"]
