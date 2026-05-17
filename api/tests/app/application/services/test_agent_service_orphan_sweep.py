"""P2#1 (round-23): orphan resume sweep must NOT immediately cleanup the Redis
confirmation entry after submitting task.resume().

Task.resume() is async-fire-and-forget (enqueues the resume command; actual
LangGraph checkpoint advancement happens in background).  If we cleanup the
queue entry immediately and the background resume fails, the interrupt is
stuck forever — the queue entry is gone so no subsequent sweep or /resume
can retry.

Correct behavior: submit the resume, log it, but do NOT call
confirmation_manager.cleanup().  The cleanup is deferred to commit_resume
(called from within the graph when the interrupt is actually consumed).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.agent_service import AgentService
from app.domain.models.session import Session, SessionStatus

from tests.app.application.services.conftest import default_snapshot as _default_snapshot
from tests.app.application.services.test_agent_service import (
    _DummyTaskClass,
    _uow_factory,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _OrphanDetail:
    """Minimal stand-in for an orphaned ConfirmationDetail."""

    session_id: str = "s-orphan"
    tool_call_id: str = "tc-orphan"
    user_id: str = "u-orphan"
    tool_name: str = "shell_execute"
    tool_args: dict = field(default_factory=lambda: {"command": "ls /"})
    risk_level: str = "high"
    arg_digest: str = "d-orph"
    primary_arg: str = "ls *"
    dir_arg: Optional[str] = None
    matched_patterns: list = field(default_factory=list)
    # deadline already passed → qualifies for expired orphan path
    deadline_ts: float = field(default_factory=lambda: time.time() - 100)
    status: str = "processing"
    # PE-0 claim nonce (may be None for legacy orphans without a PE claim)
    claim_nonce: Optional[str] = None


class _FakeConfirmationManager:
    """Records calls to cleanup, mark_pending, find_orphaned_processing."""

    def __init__(self, orphans: list) -> None:
        self._orphans = orphans
        self.cleanup_calls: list[tuple[str, str]] = []
        self.mark_pending_calls: list[tuple[str, str]] = []
        self.find_expired_calls: int = 0
        self.find_orphaned_calls: int = 0

    async def acquire_sweep_lock(self, worker_id: str) -> bool:
        return True

    async def find_expired(self) -> list:
        self.find_expired_calls += 1
        return []  # No regular expired items — only orphans

    async def find_orphaned_processing(
        self, processing_age_threshold_seconds: int = 300
    ) -> list:
        self.find_orphaned_calls += 1
        return list(self._orphans)

    async def mark_processing(self, session_id: str, tool_call_id: str) -> None:
        pass

    async def cleanup(self, session_id: str, tool_call_id: str) -> None:
        self.cleanup_calls.append((session_id, tool_call_id))

    async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
        self.mark_pending_calls.append((session_id, tool_call_id))


class _ResumeTrackingTask:
    """A task stub that records resume() calls."""

    def __init__(self) -> None:
        self.resume_calls: list = []
        self.done_flag = False

    @property
    def done(self) -> bool:
        return self.done_flag

    async def resume(self, command) -> None:
        self.resume_calls.append(command)

    async def invoke(self) -> None:
        return None


class _UoWWithSession:
    """UoW stub that returns a fake session."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self.session = MagicMock()
        self.session.get_by_id = AsyncMock(return_value=self._session)
        self.approval_grants = MagicMock()
        self.approval_grants.find_by_confirmation_id = AsyncMock(return_value=None)

    async def __aenter__(self) -> "_UoWWithSession":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSweepExpiredOrphanDoesNotCleanupImmediately:
    """P2#1: task.resume is async-submit; cleanup must NOT be called immediately."""

    async def test_sweep_expired_orphan_does_not_cleanup_when_resume_is_async(
        self, monkeypatch
    ) -> None:
        """Expired orphan: task.resume is called but cleanup is NOT called immediately.

        The queue entry must remain so that if the background graph advancement
        fails, the next sweep cycle can retry.  Final cleanup is the responsibility
        of commit_resume (called from within the graph on interrupt consumption).
        """
        orphan = _OrphanDetail()
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        fake_session = Session(
            id=orphan.session_id,
            user_id=orphan.user_id,
            status=SessionStatus.RUNNING,
        )
        fake_uow = _UoWWithSession(fake_session)
        fake_task = _ResumeTrackingTask()

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        # Patch _get_task to return our fake_task
        async def _fake_get_task(_session: Session) -> _ResumeTrackingTask:
            return fake_task

        monkeypatch.setattr(service, "_get_task", _fake_get_task)

        # Run one sweep iteration directly by calling the inner logic:
        # We simulate a single sweep cycle by invoking _confirmation_sweep_loop's
        # body once (we can't await the loop directly, so we replicate the
        # critical orphan-processing block inline).
        #
        # Approach: monkeypatch asyncio.sleep to a no-op and run one loop tick.
        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError  # stop after one full iteration

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        # task.resume MUST have been called (orphan → submit timeout_fallback)
        assert len(fake_task.resume_calls) == 1, (
            "Expected exactly one task.resume() call for the expired orphan, "
            f"got {len(fake_task.resume_calls)}"
        )

        # cleanup MUST NOT have been called (defer to commit_resume or next sweep)
        assert len(fake_cm.cleanup_calls) == 0, (
            "cleanup() must NOT be called immediately after task.resume() — "
            "Task.resume is async-submit; actual graph advancement is async. "
            f"Unexpected cleanup calls: {fake_cm.cleanup_calls}"
        )

    async def test_sweep_expired_orphan_does_not_cleanup_when_resume_raises(
        self, monkeypatch
    ) -> None:
        """If task.resume raises, cleanup must still NOT be called."""
        orphan = _OrphanDetail()
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        fake_session = Session(
            id=orphan.session_id,
            user_id=orphan.user_id,
            status=SessionStatus.RUNNING,
        )
        fake_uow = _UoWWithSession(fake_session)

        class _RaisingTask:
            done = False

            async def resume(self, command) -> None:
                raise RuntimeError("checkpoint error — background resume failed")

            async def invoke(self) -> None:
                return None

        raising_task = _RaisingTask()

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        async def _fake_get_task(_session: Session):
            return raising_task

        monkeypatch.setattr(service, "_get_task", _fake_get_task)

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        # cleanup MUST NOT be called even when resume raises
        assert len(fake_cm.cleanup_calls) == 0, (
            "cleanup() must not be called when task.resume() raises — "
            "the queue entry must remain for the next sweep to retry. "
            f"Unexpected cleanup calls: {fake_cm.cleanup_calls}"
        )


# ---------------------------------------------------------------------------
# P2#1 (round-24): resume payload must include claim_nonce + action="deny"
# ---------------------------------------------------------------------------


class TestSweepExpiredOrphanResumePayload:
    """P2#1 (round-24): sweeper must forward claim_nonce in the resume payload.

    Without claim_nonce, interrupt_helper falls back to the legacy path
    (claim_nonce is None check at react_graph.py line ~2862) and
    commit_resume cleanup never runs, leaving Redis processing entries
    permanently.

    Also verifies that action="deny" is used (not "timeout_fallback" which
    is not a valid ResumeSignal.action literal and would raise ValueError
    in the PE path).
    """

    async def test_sweep_expired_orphan_resume_includes_claim_nonce(
        self, monkeypatch
    ) -> None:
        """Expired orphan resume payload must carry claim_nonce and action='deny'."""
        orphan = _OrphanDetail(claim_nonce="deadbeefcafe0123deadbeefcafe0123")
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        fake_session = Session(
            id=orphan.session_id,
            user_id=orphan.user_id,
            status=SessionStatus.RUNNING,
        )
        fake_uow = _UoWWithSession(fake_session)
        fake_task = _ResumeTrackingTask()

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        async def _fake_get_task(_session: Session) -> _ResumeTrackingTask:
            return fake_task

        monkeypatch.setattr(service, "_get_task", _fake_get_task)

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        assert len(fake_task.resume_calls) == 1, (
            f"Expected one task.resume() call, got {len(fake_task.resume_calls)}"
        )

        cmd = fake_task.resume_calls[0]
        payload = cmd.resume  # langgraph Command.resume dict
        assert payload.get("claim_nonce") == orphan.claim_nonce, (
            "Resume payload must include claim_nonce so interrupt_helper routes to "
            "the PE commit_resume path (which performs queue cleanup). "
            f"Got payload: {payload}"
        )
        assert payload.get("action") == "deny", (
            "action must be 'deny' (not 'timeout_fallback') — ResumeSignal.action "
            "is Literal['approve', 'deny']; 'timeout_fallback' raises ValueError in "
            f"PE path. Got payload: {payload}"
        )
        assert payload.get("tool_call_id") == orphan.tool_call_id, (
            "Resume payload must include tool_call_id for interrupt_helper routing. "
            f"Got payload: {payload}"
        )

    async def test_sweep_expired_orphan_resume_claim_nonce_none_still_works(
        self, monkeypatch
    ) -> None:
        """Legacy orphan with no claim_nonce: resume is still submitted (legacy fallback).

        Ensures we don't break older orphans that were created before PE-0
        and have claim_nonce=None stored in Redis.
        """
        orphan = _OrphanDetail(claim_nonce=None)  # legacy — no PE claim
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        fake_session = Session(
            id=orphan.session_id,
            user_id=orphan.user_id,
            status=SessionStatus.RUNNING,
        )
        fake_uow = _UoWWithSession(fake_session)
        fake_task = _ResumeTrackingTask()

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        async def _fake_get_task(_session: Session) -> _ResumeTrackingTask:
            return fake_task

        monkeypatch.setattr(service, "_get_task", _fake_get_task)

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        # resume is still submitted even with claim_nonce=None (legacy fallback path)
        assert len(fake_task.resume_calls) == 1, (
            "task.resume() must still be called even for legacy orphans with "
            f"claim_nonce=None. Got {len(fake_task.resume_calls)} calls."
        )
        # cleanup must NOT be called immediately (same invariant as the nonce case)
        assert len(fake_cm.cleanup_calls) == 0, (
            f"Unexpected cleanup calls for legacy orphan: {fake_cm.cleanup_calls}"
        )


# ---------------------------------------------------------------------------
# P2#2 (round-24): deleted session → immediate cleanup
# ---------------------------------------------------------------------------


class TestSweepOrphanWithDeletedSession:
    """P2#2 (round-24): when session row is gone, orphan cleanup must run immediately.

    If the session was deleted, _get_task / _create_task can never succeed,
    so deferring to commit_resume is impossible.  The sweeper must call
    cleanup() right away to prevent permanent Redis resource leak and repeated
    warning noise every sweep cycle.
    """

    async def test_sweep_orphan_with_deleted_session_cleans_up(
        self, monkeypatch
    ) -> None:
        """session.get_by_id returns None → cleanup() is called immediately."""
        orphan = _OrphanDetail(claim_nonce="aabbccddeeff00112233445566778899")
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        # UoW returns None for the session (simulates deleted session row)
        fake_uow = _UoWWithSession(None)  # type: ignore[arg-type]

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        # cleanup MUST be called exactly once for the unrecoverable orphan
        assert len(fake_cm.cleanup_calls) == 1, (
            "cleanup() must be called immediately when the session row is deleted — "
            "the orphan is unrecoverable and must not leak Redis state. "
            f"Got {len(fake_cm.cleanup_calls)} cleanup calls."
        )
        assert fake_cm.cleanup_calls[0] == (orphan.session_id, orphan.tool_call_id), (
            f"cleanup() called with wrong args: {fake_cm.cleanup_calls[0]}"
        )

    async def test_sweep_orphan_with_deleted_session_cleanup_exception_does_not_raise(
        self, monkeypatch
    ) -> None:
        """cleanup() raising must not propagate — sweeper must stay alive."""
        orphan = _OrphanDetail(claim_nonce=None)
        fake_uow = _UoWWithSession(None)  # type: ignore[arg-type]

        class _FailingCleanupCM(_FakeConfirmationManager):
            async def cleanup(self, session_id: str, tool_call_id: str) -> None:
                raise RuntimeError("Redis down during cleanup")

        fake_cm = _FailingCleanupCM(orphans=[orphan])

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        # Must not raise even though cleanup() throws
        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass
        # If we reach here without an unhandled exception the test passes


# ---------------------------------------------------------------------------
# P2#2 (Round-25): PE orphan on legacy task → direct cleanup, no resume
# ---------------------------------------------------------------------------


class _FakeFlowNoPermissionEngine:
    """Stub _flow with no _permission_engine (legacy task)."""
    _permission_engine = None


class _LegacyTaskWithFlow(_ResumeTrackingTask):
    """Task stub that reports _flow._permission_engine = None (legacy task)."""

    def __init__(self) -> None:
        super().__init__()
        self._flow = _FakeFlowNoPermissionEngine()


class TestSweepPeOrphanOnLegacyTask:
    """P2#2 (round-25): PE orphan (has claim_nonce) on a legacy task (no PE).

    When the task config was hot-switched after the orphan was written or
    build_permission_engine failed, the task has _flow._permission_engine=None.
    Submitting a PE-style resume (with claim_nonce) would cause interrupt_helper
    to call commit_resume on a None PE (AttributeError), and cleanup would
    never happen, permanently leaking the Redis 'processing' entry.

    Fix: detect the mismatch and cleanup directly without submitting a resume.
    """

    async def test_sweep_pe_orphan_on_legacy_task_resumes_and_cleans_up(
        self, monkeypatch
    ) -> None:
        """PE orphan + legacy task → legacy deny resume submitted AND cleanup IS called.

        Codex round-36 P2#1 (correcting round-35): legacy ``interrupt_helper``
        NEVER calls ``commit_resume`` (no PE in flow), so the queue-cleanup
        path that the PE branch defers to does not exist for legacy tasks.
        Round 35 deferred cleanup to "the next sweep cycle", but the next
        sweep cycle observes the same orphan, resubmits the legacy resume,
        and again defers — the entry stays in 'processing' forever and the
        sweeper resubmits the same deny every 30 s indefinitely.

        Correct behavior: after submitting the legacy resume successfully,
        cleanup the queue entry synchronously.  This trades a small race
        window (background resume failing AFTER resume() returns) for the
        guaranteed cleanup that the legacy path requires.  The failure-mode
        test below covers the resume-raises branch where cleanup is skipped.
        """
        orphan = _OrphanDetail(claim_nonce="feedcafe" * 4)  # has claim_nonce → PE orphan
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        fake_session = Session(
            id=orphan.session_id,
            user_id=orphan.user_id,
            status=SessionStatus.RUNNING,
        )
        fake_uow = _UoWWithSession(fake_session)
        legacy_task = _LegacyTaskWithFlow()  # _flow._permission_engine is None

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        async def _fake_get_task(_session: Session) -> _LegacyTaskWithFlow:
            return legacy_task

        monkeypatch.setattr(service, "_get_task", _fake_get_task)

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        # task.resume MUST still be called (legacy deny to advance graph past interrupt)
        assert len(legacy_task.resume_calls) == 1, (
            "P2 round-36 FAIL: task.resume() must be called for PE orphan on "
            "legacy task — to attempt to advance the LangGraph checkpoint past "
            f"interrupt_helper.  Got {len(legacy_task.resume_calls)} resume calls."
        )
        # Resume command must NOT contain claim_nonce (forces legacy interrupt_helper path)
        resume_cmd = legacy_task.resume_calls[0]
        assert resume_cmd.resume.get("claim_nonce") is None, (
            f"P2 round-36 FAIL: legacy resume must NOT include claim_nonce "
            f"(would route to PE commit_resume and fail). "
            f"Got resume payload: {resume_cmd.resume!r}"
        )

        # Round 36 P2#1: cleanup MUST be called after the legacy resume
        # succeeds.  The legacy path never invokes commit_resume — without
        # a cleanup here the queue entry leaks forever.
        assert len(fake_cm.cleanup_calls) == 1, (
            "P2 round-36 FAIL: cleanup() MUST be called synchronously after "
            "successful legacy resume — the legacy path has no commit_resume "
            "to defer cleanup to.  "
            f"Got {len(fake_cm.cleanup_calls)} cleanup calls."
        )
        assert fake_cm.cleanup_calls[0] == (
            orphan.session_id,
            orphan.tool_call_id,
        ), (
            "P2 round-36 FAIL: cleanup must target the orphan's "
            f"(session_id, tool_call_id).  Got {fake_cm.cleanup_calls[0]!r}."
        )

    async def test_sweep_pe_orphan_on_legacy_task_no_cleanup_when_resume_fails(
        self, monkeypatch
    ) -> None:
        """PE orphan + legacy task → cleanup NOT called when legacy resume fails.

        Codex round-36 P2#1 (refined): cleanup only fires after a *successful*
        legacy resume.  When ``Task.resume()`` raises (background checkpoint
        error), we leave the queue entry in 'processing' so the next sweep
        cycle can retry the resume submission itself — only after a successful
        submission do we cleanup synchronously.  This avoids stranding the
        LangGraph checkpoint in interrupt state without any retry signal.
        """
        orphan = _OrphanDetail(claim_nonce="badf00d0" * 4)  # has claim_nonce → PE orphan
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        fake_session = Session(
            id=orphan.session_id,
            user_id=orphan.user_id,
            status=SessionStatus.RUNNING,
        )
        fake_uow = _UoWWithSession(fake_session)

        class _FailingLegacyTaskWithFlow(_LegacyTaskWithFlow):
            """Legacy task whose resume() raises — simulates checkpoint error."""

            async def resume(self, command) -> None:
                self.resume_calls.append(command)
                raise RuntimeError("checkpoint backend unavailable")

        failing_task = _FailingLegacyTaskWithFlow()

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        async def _fake_get_task(_session: Session) -> _FailingLegacyTaskWithFlow:
            return failing_task

        monkeypatch.setattr(service, "_get_task", _fake_get_task)

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        # resume was attempted
        assert len(failing_task.resume_calls) == 1, (
            "P2 round-35 FAIL: task.resume() must be attempted even for failing task. "
            f"Got {len(failing_task.resume_calls)} resume calls."
        )
        # cleanup must NOT be called when resume failed — leave entry for retry
        assert len(fake_cm.cleanup_calls) == 0, (
            "P2 round-35 FAIL: cleanup() must NOT be called when legacy resume fails "
            "(leave entry for next sweep cycle to retry). "
            f"Got {len(fake_cm.cleanup_calls)} cleanup calls."
        )

    async def test_sweep_pe_orphan_on_pe_task_still_submits_resume(
        self, monkeypatch
    ) -> None:
        """Regression guard: PE orphan on a PE-active task still submits resume (no cleanup)."""
        orphan = _OrphanDetail(claim_nonce="aabbccdd" * 4)  # has claim_nonce
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        fake_session = Session(
            id=orphan.session_id,
            user_id=orphan.user_id,
            status=SessionStatus.RUNNING,
        )
        fake_uow = _UoWWithSession(fake_session)

        # Task with a non-None _flow._permission_engine (PE-active)
        class _PeFlow:
            _permission_engine = object()  # non-None → PE-active

        class _PeTask(_ResumeTrackingTask):
            def __init__(self) -> None:
                super().__init__()
                self._flow = _PeFlow()

        pe_task = _PeTask()

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        async def _fake_get_task(_session: Session) -> _PeTask:
            return pe_task

        monkeypatch.setattr(service, "_get_task", _fake_get_task)

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        # PE-active task → resume must be submitted (cleanup deferred to commit_resume)
        assert len(pe_task.resume_calls) == 1, (
            "P2#2 round-25 regression FAIL: PE orphan on PE-active task must submit resume. "
            f"Got {len(pe_task.resume_calls)} resume calls."
        )
        # cleanup must NOT be called immediately (deferred to commit_resume)
        assert len(fake_cm.cleanup_calls) == 0, (
            "P2#2 round-25 regression FAIL: cleanup must not be called immediately "
            f"for PE orphan on PE-active task. Got {len(fake_cm.cleanup_calls)} cleanup calls."
        )


# ---------------------------------------------------------------------------
# P2#1 (Round-26): _flow lives on _task_runner, not directly on the Task
# ---------------------------------------------------------------------------
#
# RedisStreamTask (production) exposes _flow via _task_runner._flow.
# The previous code did getattr(task, "_flow", None) which always returned
# None for production tasks, so _is_legacy_task was always False and every
# PE orphan fell through to task.resume() — causing AttributeError when the
# task had no PE wired (_task_runner._flow._permission_engine is None).
#
# Fix: resolve _flow via _task_runner first, then fall back to direct _flow
# (test-stub compatibility).


class _FakeFlowLegacy:
    """_flow stub with no _permission_engine (legacy / PE-disabled)."""

    _permission_engine = None


class _FakeFlowPeActive:
    """_flow stub with a live _permission_engine object (PE-active)."""

    _permission_engine = object()  # non-None → PE-active


class _FakeTaskRunner:
    """Minimal TaskRunner stub carrying a _flow attribute."""

    def __init__(self, flow: object) -> None:
        self._flow = flow


class _RedisStreamTaskShapeLegacy(_ResumeTrackingTask):
    """Mimics production RedisStreamTask layout with a legacy _task_runner._flow."""

    def __init__(self) -> None:
        super().__init__()
        self._task_runner = _FakeTaskRunner(_FakeFlowLegacy())
        # NOTE: no self._flow — mirrors the real RedisStreamTask


class _RedisStreamTaskShapePeActive(_ResumeTrackingTask):
    """Mimics production RedisStreamTask layout with a PE-active _task_runner._flow."""

    def __init__(self) -> None:
        super().__init__()
        self._task_runner = _FakeTaskRunner(_FakeFlowPeActive())
        # NOTE: no self._flow — mirrors the real RedisStreamTask


class TestSweepPeOrphanRedisStreamTaskShape:
    """P2#1 (round-26): _flow is resolved via _task_runner._flow on production tasks.

    RedisStreamTask does NOT expose _flow directly; it lives on _task_runner.
    The old code (getattr(task, '_flow', None)) always returned None for
    production tasks, making _is_legacy_task always False.  Result: every PE
    orphan fell through to task.resume() even when the task had no PE wired,
    causing AttributeError inside interrupt_helper and permanent Redis leaks.
    """

    async def test_redis_stream_task_shape_legacy_pe_orphan_resumes_and_cleans_up(
        self, monkeypatch
    ) -> None:
        """PE orphan + RedisStreamTask-shaped task with no PE → legacy resume + cleanup.

        Codex round-36 P2#1 (correcting round-35): the legacy interrupt_helper
        path never invokes ``commit_resume`` (no PE in flow), so cleanup must
        happen synchronously here after a successful resume submission.
        Deferring to "next sweep cycle" produces a permanent leak because the
        next sweep observes the same orphan, resubmits the same legacy resume,
        and again defers — forever.

        This reproduces the production layout: task has _task_runner._flow with
        _permission_engine=None (legacy / PE-disabled).  The sweeper detects this
        via the _task_runner chain and must:
        1. Call task.resume() with NO claim_nonce (legacy path)
        2. After successful submission, call cleanup() synchronously
        """
        orphan = _OrphanDetail(claim_nonce="deadbeef" * 4)  # has claim_nonce → PE orphan
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        fake_session = Session(
            id=orphan.session_id,
            user_id=orphan.user_id,
            status=SessionStatus.RUNNING,
        )
        fake_uow = _UoWWithSession(fake_session)
        legacy_task = _RedisStreamTaskShapeLegacy()

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        async def _fake_get_task(_session: Session) -> _RedisStreamTaskShapeLegacy:
            return legacy_task

        monkeypatch.setattr(service, "_get_task", _fake_get_task)

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        # task.resume MUST be called — legacy deny to advance checkpoint
        assert len(legacy_task.resume_calls) == 1, (
            "P2 round-36 FAIL: task.resume() must be called for PE orphan on "
            "RedisStreamTask-shaped legacy task (to attempt checkpoint advance). "
            f"Got {len(legacy_task.resume_calls)} resume calls."
        )
        # Resume must NOT have claim_nonce (forces legacy interrupt_helper path)
        resume_cmd = legacy_task.resume_calls[0]
        assert resume_cmd.resume.get("claim_nonce") is None, (
            f"P2 round-36 FAIL: legacy resume must NOT include claim_nonce. "
            f"Got resume payload: {resume_cmd.resume!r}"
        )

        # Round 36 P2#1: cleanup MUST be called after successful legacy resume.
        # Legacy path has no commit_resume to defer cleanup to → synchronous
        # cleanup here is the only way to drain the queue entry.
        assert len(fake_cm.cleanup_calls) == 1, (
            "P2 round-36 FAIL: cleanup() MUST be called synchronously after "
            "successful legacy resume on RedisStreamTask-shaped legacy task. "
            f"Got {len(fake_cm.cleanup_calls)} cleanup calls."
        )
        assert fake_cm.cleanup_calls[0] == (
            orphan.session_id,
            orphan.tool_call_id,
        )

    async def test_redis_stream_task_shape_pe_active_pe_orphan_submits_resume(
        self, monkeypatch
    ) -> None:
        """Regression guard: PE orphan + RedisStreamTask-shaped PE-active task → resume submitted.

        When _task_runner._flow._permission_engine is non-None (PE-active), the
        sweeper must still submit task.resume() (cleanup deferred to commit_resume).
        """
        orphan = _OrphanDetail(claim_nonce="cafebabe" * 4)  # has claim_nonce → PE orphan
        fake_cm = _FakeConfirmationManager(orphans=[orphan])

        fake_session = Session(
            id=orphan.session_id,
            user_id=orphan.user_id,
            status=SessionStatus.RUNNING,
        )
        fake_uow = _UoWWithSession(fake_session)
        pe_task = _RedisStreamTaskShapePeActive()

        service = AgentService(
            uow_factory=lambda: fake_uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=object,
            task_cls=_DummyTaskClass,
            search_engine=object(),
            file_storage=object(),
        )
        service._confirmation_manager = fake_cm

        async def _fake_get_task(_session: Session) -> _RedisStreamTaskShapePeActive:
            return pe_task

        monkeypatch.setattr(service, "_get_task", _fake_get_task)

        sleep_calls: list[float] = []

        async def _fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            if len(sleep_calls) > 1:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        try:
            await service._confirmation_sweep_loop()
        except asyncio.CancelledError:
            pass

        # PE-active task → resume must be submitted (cleanup deferred to commit_resume)
        assert len(pe_task.resume_calls) == 1, (
            "P2#1 round-26 regression FAIL: PE orphan on RedisStreamTask-shaped PE-active "
            "task must submit resume (cleanup deferred to commit_resume). "
            f"Got {len(pe_task.resume_calls)} resume calls."
        )
        # cleanup must NOT be called immediately
        assert len(fake_cm.cleanup_calls) == 0, (
            "P2#1 round-26 regression FAIL: cleanup must not be called immediately for "
            "PE orphan on RedisStreamTask-shaped PE-active task. "
            f"Got {len(fake_cm.cleanup_calls)} cleanup calls."
        )
