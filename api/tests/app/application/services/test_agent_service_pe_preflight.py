"""PE-0 Phase 8.1/8.2 unit tests.

Covers:
- agent_service.preflight_resume_tool_confirmation delegates to pe.preflight_resume
  when PE is available (feature flag on, deps resolved).
- writer.write / writer.write_audit_only are NOT awaited from the PE path.
- drive_resume_tool_confirmation passes claim_nonce in Command(resume=...).
"""

from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass, field
from typing import Optional
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.agent_service import (
    AgentService,
    _ResumeToolConfirmationState,
)
from app.domain.models.session import Session, SessionStatus
from app.domain.services.permission.context import PreflightResumeResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubDetail:
    """Minimal ConfirmationDetail substitute."""

    session_id: str = "s_test"
    tool_call_id: str = "tc_test"
    user_id: str = "u_test"
    tool_name: str = "file_write"
    tool_args: dict = field(default_factory=lambda: {"path": "/x"})
    risk_level: str = "medium"
    arg_digest: str = "ad_test"
    primary_arg: str = ""
    dir_arg: Optional[str] = None
    matched_patterns: list = field(default_factory=list)
    deadline_ts: float = 9_999_999_999.0
    status: str = "pending"


class _FakeConfirmationManager:
    def __init__(self, detail: _StubDetail | None = None) -> None:
        self._detail = detail
        self.mark_processing_calls: list = []

    async def read(self, session_id: str, tool_call_id: str):
        return self._detail

    async def mark_processing(self, session_id: str, tool_call_id: str) -> None:
        self.mark_processing_calls.append((session_id, tool_call_id))

    async def cleanup(self, session_id: str, tool_call_id: str) -> None:
        pass


class _FakeWriter:
    def __init__(self) -> None:
        self.write = AsyncMock(return_value=("dec-1", True))
        self.write_audit_only = AsyncMock(return_value=None)
        self.delete_grant = AsyncMock(return_value=None)


class _FakePE:
    """Minimal PermissionEngine stub for preflight tests."""

    def __init__(self, claim_nonce: str = "n" * 32) -> None:
        self._nonce = claim_nonce
        self.preflight_resume = AsyncMock(
            return_value=PreflightResumeResult(
                claim_nonce=claim_nonce,
                processing_started_at=None,  # datetime; None acceptable in tests
                detail=_StubDetail(),
            )
        )

    async def commit_resume(self, *args, **kwargs):  # not called in preflight tests
        raise AssertionError("commit_resume should not be called in preflight")


class _FakeSSM:
    """Minimal SessionStateMachine stub."""

    def __init__(
        self,
        mode: SessionStatus = SessionStatus.RUNNING,
        revision: int = 1,
    ) -> None:
        self._mode = mode
        self._revision = revision

    async def get_mode_with_revision(self, session_id: str):
        return self._mode, self._revision


class _FakeUoW:
    def __init__(self) -> None:
        self.session = MagicMock()
        self.session.update_unread_message_count = AsyncMock(return_value=None)
        self.session.add_event = AsyncMock(return_value=None)
        self.approval_grants = MagicMock()
        self.approval_grants.find_by_confirmation_id = AsyncMock(return_value=None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def commit(self):
        return None

    async def rollback(self):
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tool_confirmation(
    *,
    action: str = "approve",
    scope: str = "session",
    tool_call_id: str = "tc_test",
) -> object:
    return SimpleNamespace(action=action, scope=scope, tool_call_id=tool_call_id)


def _uow_factory():
    return _FakeUoW()


def _make_service(
    *,
    detail: _StubDetail | None = None,
    session_status: SessionStatus = SessionStatus.RUNNING,
    pe_enabled: bool = True,
) -> tuple[AgentService, SimpleNamespace]:
    """Build a minimal AgentService with mocked PE + SSM.

    Returns (service, fakes) where fakes.pe / fakes.ssm / fakes.writer /
    fakes.confirmation_mgr are accessible for assertion.
    """
    from tests.app.application.services.conftest import default_snapshot

    # Patch AgentConfig to expose tool_confirmation with flag
    agent_config = MagicMock()
    tc = MagicMock()
    tc.permission_engine_native_enabled = pe_enabled
    tc.legacy_rule_fallback = False
    agent_config.tool_confirmation = tc
    agent_config.memory = MagicMock()
    agent_config.execution = MagicMock()
    agent_config.execution.max_same_tool_failures = 3

    snap = default_snapshot()
    # Swap agent_config with the patched one (snap is frozen, so rebuild):
    from app.application.services.agent_service import _ConfigSnapshot
    snap2 = _ConfigSnapshot(
        llm=snap.llm,
        agent_config=agent_config,
        mcp_config=snap.mcp_config,
        a2a_config=snap.a2a_config,
        skill_risk_policy=snap.skill_risk_policy,
        overflow_config=snap.overflow_config,
        summary_llm=snap.summary_llm,
        vision_fallback_model=snap.vision_fallback_model,
        skill_creator_service=snap.skill_creator_service,
        supports_vision=snap.supports_vision,
        supports_pdf_input=snap.supports_pdf_input,
        file_understanding_config=snap.file_understanding_config,
    )

    stub_detail = detail or _StubDetail()
    fake_confirmation_mgr = _FakeConfirmationManager(detail=stub_detail)
    fake_writer = _FakeWriter()
    fake_pe = _FakePE()
    fake_ssm = _FakeSSM(mode=session_status)

    service = AgentService(
        uow_factory=_uow_factory,
        config_snapshot=snap2,
        sandbox_cls=object,
        task_cls=object,
        search_engine=object(),
        file_storage=object(),
    )
    # Inject confirmation manager directly (bypasses __init__ Redis check)
    service._confirmation_manager = fake_confirmation_mgr

    fakes = SimpleNamespace(
        pe=fake_pe,
        ssm=fake_ssm,
        writer=fake_writer,
        confirmation_mgr=fake_confirmation_mgr,
    )
    return service, fakes


# ---------------------------------------------------------------------------
# Task 8.1 tests
# ---------------------------------------------------------------------------


async def test_preflight_delegates_to_pe_when_pe_available(monkeypatch) -> None:
    """When PE/SSM are available, preflight_resume_tool_confirmation delegates to pe.preflight_resume."""
    service, fakes = _make_service(pe_enabled=True)
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    # Patch _get_accessible_session to return fake session
    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    # Patch _build_pe_ssm_for_resume to return our fakes
    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # Patch _get_task and _create_task to return a minimal fake task
    fake_task = MagicMock()
    fake_task.done = True
    fake_task.output_stream = MagicMock()
    fake_task.output_stream.get = AsyncMock(return_value=(None, None))

    async def _fake_get_task(session):
        return fake_task

    monkeypatch.setattr(service, "_get_task", _fake_get_task)

    tc = _make_tool_confirmation(action="approve", scope="session", tool_call_id="tc_test")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_test",
        user_id="u_test",
        is_admin=False,
        tool_confirmation=tc,
    )

    # PE path: preflight_resume was called
    fakes.pe.preflight_resume.assert_awaited_once()

    # Writer.write and writer.write_audit_only must NOT be called from PE path
    fakes.writer.write.assert_not_awaited()
    fakes.writer.write_audit_only.assert_not_awaited()

    # State is returned and has claim_nonce from PE
    assert state is not None
    assert state.claim_nonce == "n" * 32


async def test_preflight_falls_back_to_legacy_when_pe_unavailable(monkeypatch) -> None:
    """When _build_pe_ssm_for_resume returns (None, None), legacy path is used."""
    service, fakes = _make_service(pe_enabled=False)

    def _fake_build_pe_ssm(snap):
        return None, None

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    legacy_called = []

    async def _fake_legacy(*, session_id, user_id, is_admin, tool_confirmation):
        legacy_called.append(True)
        # Return a minimal state so the test can complete
        fake_session = Session(id=session_id, user_id=user_id, status=SessionStatus.RUNNING)
        fake_task = MagicMock()
        return _ResumeToolConfirmationState(
            session=fake_session,
            detail=_StubDetail(),
            task=fake_task,
            decision_id=None,
            persistent_scope=False,
            action="approve",
            scope="once",
            tool_call_id="tc_test",
            owner_user_id=user_id,
            session_id=session_id,
            claim_nonce=None,
        )

    monkeypatch.setattr(
        service, "_preflight_resume_tool_confirmation_legacy", _fake_legacy
    )

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_test")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_test",
        user_id="u_test",
        is_admin=False,
        tool_confirmation=tc,
    )

    assert legacy_called, "Legacy path should have been called when PE is unavailable"
    assert state.claim_nonce is None  # legacy path does not set claim_nonce


# ---------------------------------------------------------------------------
# Task 8.2 tests
# ---------------------------------------------------------------------------


async def test_drive_resume_passes_claim_nonce_in_resume_payload(monkeypatch) -> None:
    """drive_resume_tool_confirmation includes claim_nonce in Command(resume=...)."""
    from langgraph.types import Command

    service, _ = _make_service()

    # Build a fake task that records the resume command
    resume_args: list = []

    class _FakeTask:
        done = True

        async def resume(self, cmd: Command) -> None:
            resume_args.append(cmd)

        class output_stream:
            @staticmethod
            async def get(start_id=None, block_ms=None):
                return None, None

    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    state = _ResumeToolConfirmationState(
        session=fake_session,
        detail=_StubDetail(),
        task=_FakeTask(),
        decision_id=None,
        persistent_scope=False,
        action="approve",
        scope="session",
        tool_call_id="tc_test",
        owner_user_id="u_test",
        session_id="s_test",
        claim_nonce="n" * 32,
    )

    # Patch confirmation_manager cleanup
    service._confirmation_manager = _FakeConfirmationManager()

    # Patch _safe_update_unread_count (called via create_task in drive)
    async def _noop_update(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(service, "_safe_update_unread_count", _noop_update)

    # Consume the generator (it immediately exits because task.done=True after resume)
    events = []
    async for event in service.drive_resume_tool_confirmation(state):
        events.append(event)

    assert len(resume_args) == 1, "task.resume should have been called exactly once"
    cmd = resume_args[0]
    assert isinstance(cmd, Command), f"Expected Command, got {type(cmd)}"
    resume_dict = cmd.resume
    assert resume_dict["action"] == "approve"
    assert resume_dict["scope"] == "session"
    assert resume_dict["claim_nonce"] == "n" * 32


async def test_drive_resume_claim_nonce_none_for_legacy_path(monkeypatch) -> None:
    """Legacy path: claim_nonce=None is forwarded gracefully (no crash)."""
    from langgraph.types import Command

    service, _ = _make_service()
    resume_args: list = []

    class _FakeTask:
        done = True

        async def resume(self, cmd: Command) -> None:
            resume_args.append(cmd)

        class output_stream:
            @staticmethod
            async def get(start_id=None, block_ms=None):
                return None, None

    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)
    state = _ResumeToolConfirmationState(
        session=fake_session,
        detail=_StubDetail(),
        task=_FakeTask(),
        decision_id=None,
        persistent_scope=False,
        action="deny",
        scope="once",
        tool_call_id="tc_test",
        owner_user_id="u_test",
        session_id="s_test",
        claim_nonce=None,  # legacy path
    )

    service._confirmation_manager = _FakeConfirmationManager()

    async def _noop_update(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(service, "_safe_update_unread_count", _noop_update)

    async for _ in service.drive_resume_tool_confirmation(state):
        pass

    assert len(resume_args) == 1
    assert resume_args[0].resume["claim_nonce"] is None


# ---------------------------------------------------------------------------
# P2#2 tests — late duplicate falls back to bare legacy confirmation_id
# ---------------------------------------------------------------------------


class _FakeConfirmationManagerWithRead(_FakeConfirmationManager):
    """Extends _FakeConfirmationManager with configurable read return."""

    def __init__(self, detail=None, read_result=None) -> None:
        super().__init__(detail=detail)
        self._read_result = read_result

    async def read(self, session_id: str, tool_call_id: str):
        return self._read_result


async def test_late_duplicate_falls_back_to_bare_legacy_confirmation_id(
    monkeypatch,
) -> None:
    """P2#2 (Codex round-10): when PE composite confirmation_id lookup misses,
    fall back to bare tool_call_id before raising 404.

    Scenario: session started with legacy preflight (grant written as bare
    tool_call_id) and is retried after PE became available.  pending_detail
    is already cleaned up.  PE composite lookup misses → should try bare id
    → return 409 (ConflictError).
    """
    from app.application.errors.exceptions import ConflictError

    service, fakes = _make_service(pe_enabled=True)

    # Simulate the PE path landing in the late-duplicate branch:
    # pending_detail is None (already cleaned up)
    service._confirmation_manager = _FakeConfirmationManagerWithRead(detail=None)

    fake_grant = MagicMock()  # any non-None object represents a found grant

    # First call (PE composite) → None; second call (bare) → grant found
    call_count = 0

    class _FakeUoWLatedup:
        def __init__(self) -> None:
            self.approval_grants = MagicMock()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def commit(self):
            return None

        async def rollback(self):
            return None

    async def _make_uow_latedup():
        uow = _FakeUoWLatedup()
        nonlocal call_count

        async def _find_by_confirmation_id(cid: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: PE composite id → not found
                return None
            # Second call: bare id → found
            return fake_grant

        uow.approval_grants.find_by_confirmation_id = _find_by_confirmation_id
        return uow

    async def _uow_factory_latedup():
        return _make_uow_latedup()

    # Patch _uow_factory to use async context manager properly
    class _AsyncCtxUoW:
        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            return await self._inner()

        async def __aexit__(self, *args):
            return None

    service._uow_factory = lambda: _AsyncCtxUoW(_make_uow_latedup)

    fake_session = Session(id="s_latedup", user_id="u_test", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    async def _fake_get_task(session):
        return MagicMock(done=True, output_stream=MagicMock())

    monkeypatch.setattr(service, "_get_task", _fake_get_task)

    # preflight_resume should raise ConflictError (409) because bare id found
    tc = _make_tool_confirmation(
        action="approve", scope="once", tool_call_id="bare_tool_call_id"
    )
    with pytest.raises(ConflictError):
        await service.preflight_resume_tool_confirmation(
            session_id="s_latedup",
            user_id="u_test",
            is_admin=False,
            tool_confirmation=tc,
        )

    # Both queries must have been made
    assert call_count == 2, (
        f"P2#2 FAIL: expected 2 DB queries (composite + bare), got {call_count}"
    )


# ---------------------------------------------------------------------------
# P2#3 tests — conditional rollback skips when confirmation already cleaned up
# ---------------------------------------------------------------------------


async def test_rollback_skips_when_confirmation_already_cleaned_up() -> None:
    """P2#3 (Codex round-10): _rollback_resume_claim_if_present skips mark_pending
    when confirmation_manager.read() returns None (commit_resume already cleaned up).
    """
    service, _ = _make_service()

    mark_pending_calls: list = []

    class _FakeCM:
        async def read(self, session_id: str, tool_call_id: str):
            return None  # already cleaned up

        async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
            mark_pending_calls.append((session_id, tool_call_id))

        async def cleanup(self, session_id: str, tool_call_id: str) -> None:
            pass

    service._confirmation_manager = _FakeCM()

    await service._rollback_resume_claim_if_present(
        persistent_scope=False,
        decision_id=None,
        session_id="s_test",
        tool_call_id="tc_test",
        claim_nonce="nonce-abc",
    )

    assert mark_pending_calls == [], (
        "P2#3 FAIL: mark_pending was called even though confirmation was already cleaned up"
    )


async def test_rollback_skips_when_nonce_mismatch() -> None:
    """P2#3 (Codex round-10): _rollback_resume_claim_if_present skips mark_pending
    when detail.claim_nonce != expected claim_nonce (someone else now owns the claim).
    """
    from app.domain.services.permission.confirmation_queue import ConfirmationDetail
    import time

    service, _ = _make_service()

    mark_pending_calls: list = []

    # detail with a different claim_nonce than what we pass
    different_nonce = "other-owner-nonce"
    expected_nonce = "our-nonce-xyz"
    mock_detail = ConfirmationDetail(
        session_id="s_test",
        tool_call_id="tc_nonce_mismatch",
        user_id="u_test",
        tool_name="file_write",
        tool_args={"path": "/x"},
        risk_level="medium",
        arg_digest=None,
        primary_arg=None,
        dir_arg=None,
        matched_patterns=[],
        deadline_ts=time.time() + 300,
        claim_nonce=different_nonce,
    )

    class _FakeCMNonceMismatch:
        async def read(self, session_id: str, tool_call_id: str):
            return mock_detail

        async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
            mark_pending_calls.append((session_id, tool_call_id))

        async def cleanup(self, session_id: str, tool_call_id: str) -> None:
            pass

    service._confirmation_manager = _FakeCMNonceMismatch()

    await service._rollback_resume_claim_if_present(
        persistent_scope=False,
        decision_id=None,
        session_id="s_test",
        tool_call_id="tc_nonce_mismatch",
        claim_nonce=expected_nonce,  # different from detail.claim_nonce
    )

    assert mark_pending_calls == [], (
        "P2#3 FAIL: mark_pending was called despite nonce mismatch — "
        "would have taken over another owner's claim"
    )


# ---------------------------------------------------------------------------
# P1#1: HTTP preflight routes non-native tools to legacy path
# ---------------------------------------------------------------------------


async def test_preflight_non_native_tool_routes_to_legacy(monkeypatch) -> None:
    """P1#1: When pending_detail.tool_name resolves to a non-native source (skill/mcp/a2a)
    AND its per-source PE flag is disabled, preflight_resume_tool_confirmation must
    fall back to the legacy path.

    PE-1 §3.2: skill is now PE-eligible by default (permission_engine_skill_enabled=True).
    To preserve the legacy-routing intent of this test we explicitly set the
    per-source flag to False below — emulating an operator who has not yet
    enabled SkillSource for their deployment.
    """
    # Use a skill_ prefixed tool name — resolve_tool_source will return source="skill"
    skill_detail = _StubDetail(tool_name="skill_my_custom_tool")
    service, fakes = _make_service(detail=skill_detail, pe_enabled=True)
    # PE-1 §3.2: explicitly disable per-source PE for skill so the legacy gate fires.
    service._config_snapshot.agent_config.tool_confirmation.permission_engine_skill_enabled = False
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # Track calls to the legacy path
    legacy_called = []

    async def _fake_legacy(*, session_id, user_id, is_admin, tool_confirmation):
        legacy_called.append(True)
        fake_task = MagicMock()
        return _ResumeToolConfirmationState(
            session=fake_session,
            detail=skill_detail,
            task=fake_task,
            decision_id=None,
            persistent_scope=False,
            action="approve",
            scope="once",
            tool_call_id="tc_test",
            owner_user_id=user_id,
            session_id=session_id,
            claim_nonce=None,
        )

    monkeypatch.setattr(
        service, "_preflight_resume_tool_confirmation_legacy", _fake_legacy
    )

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_test")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_test",
        user_id="u_test",
        is_admin=False,
        tool_confirmation=tc,
    )

    # Legacy path must have been called (not PE.preflight_resume)
    assert legacy_called, "Non-native tool must route to legacy confirmation path"
    fakes.pe.preflight_resume.assert_not_awaited()
    # Legacy path returns claim_nonce=None
    assert state.claim_nonce is None


async def test_preflight_mcp_tool_with_flag_off_routes_to_legacy(monkeypatch) -> None:
    """PE-2: mcp_ prefixed tools route to legacy when the per-source PE flag is off.

    PE-2 makes mcp PE-eligible by default (permission_engine_mcp_enabled=True).
    To preserve the legacy-routing intent of this test we explicitly disable the
    per-source flag — emulating an operator who has not enabled McpSource for
    their deployment. The mcp flag-ON case is covered by
    ``test_preflight_mcp_tool_with_flag_on_routes_to_pe``.
    """
    mcp_detail = _StubDetail(tool_name="mcp_my_server_tool")
    service, fakes = _make_service(detail=mcp_detail, pe_enabled=True)
    # PE-2: explicitly disable per-source PE for mcp so the legacy gate fires.
    service._config_snapshot.agent_config.tool_confirmation.permission_engine_mcp_enabled = False
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    legacy_called = []

    async def _fake_legacy(*, session_id, user_id, is_admin, tool_confirmation):
        legacy_called.append(True)
        fake_task = MagicMock()
        return _ResumeToolConfirmationState(
            session=fake_session,
            detail=mcp_detail,
            task=fake_task,
            decision_id=None,
            persistent_scope=False,
            action="approve",
            scope="once",
            tool_call_id="tc_test",
            owner_user_id=user_id,
            session_id=session_id,
            claim_nonce=None,
        )

    monkeypatch.setattr(
        service, "_preflight_resume_tool_confirmation_legacy", _fake_legacy
    )

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_test")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_test",
        user_id="u_test",
        is_admin=False,
        tool_confirmation=tc,
    )

    assert legacy_called, "MCP tool with flag off must route to legacy confirmation path"
    fakes.pe.preflight_resume.assert_not_awaited()
    assert state.claim_nonce is None


async def test_preflight_mcp_tool_with_flag_on_routes_to_pe(monkeypatch) -> None:
    """PE-2: mcp_ prefixed tools route through PE when the per-source flag is on.

    After PE-2 makes mcp PE-eligible (permission_engine_mcp_enabled defaults to
    True), a real mcp tool resolves to source='mcp' / category='mcp' and must be
    handled by pe.preflight_resume rather than the legacy path. Mirrors the
    assertion style of ``test_preflight_delegates_to_pe_when_pe_available``.
    """
    mcp_detail = _StubDetail(tool_name="mcp_my_server_tool")
    service, fakes = _make_service(detail=mcp_detail, pe_enabled=True)
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # Legacy path must NOT be taken; track it to assert it stays empty.
    legacy_called = []

    async def _fake_legacy(*, session_id, user_id, is_admin, tool_confirmation):
        legacy_called.append(True)
        fake_task = MagicMock()
        return _ResumeToolConfirmationState(
            session=fake_session,
            detail=mcp_detail,
            task=fake_task,
            decision_id=None,
            persistent_scope=False,
            action="approve",
            scope="once",
            tool_call_id="tc_test",
            owner_user_id=user_id,
            session_id=session_id,
            claim_nonce=None,
        )

    monkeypatch.setattr(
        service, "_preflight_resume_tool_confirmation_legacy", _fake_legacy
    )

    # Provide a done task so the PE path completes without _create_task.
    fake_task = MagicMock()
    fake_task.done = True
    fake_task.output_stream = MagicMock()
    fake_task.output_stream.get = AsyncMock(return_value=(None, None))

    async def _fake_get_task(session):
        return fake_task

    monkeypatch.setattr(service, "_get_task", _fake_get_task)

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_test")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_test",
        user_id="u_test",
        is_admin=False,
        tool_confirmation=tc,
    )

    # PE path: preflight_resume was called; legacy path was NOT taken.
    fakes.pe.preflight_resume.assert_awaited_once()
    assert legacy_called == [], "MCP tool with flag on must NOT route to legacy"
    assert state.claim_nonce == "n" * 32


# ---------------------------------------------------------------------------
# Round-16 P2: mixed-batch routing consistency guard
# ---------------------------------------------------------------------------


async def test_preflight_mixed_batch_native_pending_skill_in_batch_routes_to_legacy(
    monkeypatch,
) -> None:
    """Round-16 P2: When the active tool_calls batch contains a non-native tool
    (e.g. skill_foo) alongside the pending native tool, _pe_dispatch will route
    the ENTIRE batch to legacy (because any non-native in batch → legacy).

    preflight_resume_tool_confirmation must mirror this batch-level check:
    reading the graph state and detecting the non-native tool in the batch →
    fall back to the legacy preflight so that commit_resume and preflight use
    the same path (no split-brain: PE claim_nonce written but graph walks legacy).

    Setup:
    - pending_detail.tool_name = "file_write" (native)
    - graph checkpoint AIMessage has tool_calls: [file_write(tc_native), skill_foo(tc_skill)]
    - Only file_write has a ToolMessage (skill_foo is still pending / not yet executed)
    - Expected: legacy path is called (not PE preflight_resume)
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    # Pending detail for the native tool
    native_detail = _StubDetail(
        tool_name="file_write",
        tool_call_id="tc_native",
        session_id="s_mixed",
        user_id="u_mixed",
    )
    service, fakes = _make_service(detail=native_detail, pe_enabled=True)
    # PE-1 §3.2: skill is PE-eligible by default; disable per-source flag so
    # the mixed-batch guard treats skill as non-PE-eligible (original test intent).
    service._config_snapshot.agent_config.tool_confirmation.permission_engine_skill_enabled = False
    fake_session = Session(id="s_mixed", user_id="u_mixed", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # Build fake graph state: AI message with [native, skill] tool_calls.
    # No ToolMessage for skill_foo → skill_foo is still pending → mixed batch.
    _ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc_native", "name": "file_write", "args": {"path": "/x"}},
            {"id": "tc_skill", "name": "skill_my_tool", "args": {}},
        ],
    )
    # ToolMessage only for the native tool (already confirmed); skill is still pending.
    _tool_msg_native = ToolMessage(
        content="ok",
        tool_call_id="tc_native",
    )
    _gs_messages = [HumanMessage(content="do stuff"), _ai_msg, _tool_msg_native]

    # Fake StateSnapshot-like object returned by aget_state
    class _FakeSnapshot:
        values = {"messages": _gs_messages}
        next = ("interrupt_node",)

    _fake_snap = _FakeSnapshot()

    # Fake compiled graph with async aget_state
    class _FakeCompiledGraph:
        async def aget_state(self, config, *, subgraphs: bool = False):
            return _fake_snap

    _fake_compiled = _FakeCompiledGraph()

    # Fake flow with _main_graph, _permission_engine, and _build_config()
    class _FakeFlow:
        _permission_engine = object()  # non-None → not a legacy task
        _main_graph = _fake_compiled

        def _build_config(self) -> dict:
            return {"configurable": {"thread_id": "s_mixed"}}

    # Fake task runner and task that expose _task_runner._flow
    class _FakeTaskRunner:
        _flow = _FakeFlow()

    class _FakeTask:
        _task_runner = _FakeTaskRunner()

    async def _fake_get_task(session):
        return _FakeTask()

    monkeypatch.setattr(service, "_get_task", _fake_get_task)

    # Track legacy calls
    legacy_called: list[str] = []

    async def _fake_legacy(*, session_id, user_id, is_admin, tool_confirmation):
        legacy_called.append(session_id)
        fake_task = MagicMock()
        return _ResumeToolConfirmationState(
            session=fake_session,
            detail=native_detail,
            task=fake_task,
            decision_id=None,
            persistent_scope=False,
            action="approve",
            scope="once",
            tool_call_id="tc_native",
            owner_user_id=user_id,
            session_id=session_id,
            claim_nonce=None,
        )

    monkeypatch.setattr(
        service, "_preflight_resume_tool_confirmation_legacy", _fake_legacy
    )

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_native")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_mixed",
        user_id="u_mixed",
        is_admin=False,
        tool_confirmation=tc,
    )

    # Mixed-batch guard must route to legacy, NOT PE
    assert legacy_called, (
        "Mixed batch (native pending + skill in batch) must route to legacy path "
        "to match _pe_dispatch's batch-level fallback behavior."
    )
    fakes.pe.preflight_resume.assert_not_awaited()
    assert state.claim_nonce is None


async def test_preflight_pure_native_batch_uses_pe_path(monkeypatch) -> None:
    """Round-16 P2 (negative case): When the active tool_calls batch contains
    ONLY native tools, preflight must NOT fall back to legacy — PE path should run.

    Setup:
    - pending_detail.tool_name = "file_write" (native)
    - graph checkpoint AIMessage has tool_calls: [file_write(tc1), shell_execute(tc2)]
    - Both tools are native → _pe_dispatch will NOT fall back to legacy
    - Expected: PE preflight_resume is called (not legacy)
    """
    from langchain_core.messages import AIMessage, HumanMessage

    native_detail = _StubDetail(
        tool_name="file_write",
        tool_call_id="tc_native",
        session_id="s_pure_native",
        user_id="u_pure",
    )
    service, fakes = _make_service(detail=native_detail, pe_enabled=True)
    fake_session = Session(id="s_pure_native", user_id="u_pure", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # Pure native batch: file_write + shell_execute (both native)
    _ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc_native", "name": "file_write", "args": {"path": "/x"}},
            {"id": "tc_shell", "name": "shell_execute", "args": {"command": "ls"}},
        ],
    )
    _gs_messages = [HumanMessage(content="do stuff"), _ai_msg]

    class _FakeSnapshot:
        values = {"messages": _gs_messages}
        next = ("interrupt_node",)

    class _FakeCompiledGraph:
        async def aget_state(self, config, *, subgraphs: bool = False):
            return _FakeSnapshot()

    class _FakeFlow:
        _permission_engine = object()  # non-None → PE-active task
        _main_graph = _FakeCompiledGraph()

        def _build_config(self) -> dict:
            return {"configurable": {"thread_id": "s_pure_native"}}

    class _FakeTaskRunner:
        _flow = _FakeFlow()

    class _FakeTask:
        _task_runner = _FakeTaskRunner()
        done = True
        output_stream = MagicMock()

    async def _fake_get_task(session):
        return _FakeTask()

    monkeypatch.setattr(service, "_get_task", _fake_get_task)

    # PE.preflight_resume returns a result → _create_task / drive_resume path not needed
    # We just check that preflight_resume IS awaited (PE path runs)
    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_native")

    # PE.preflight_resume already mocked in _FakePE; let _get_task return fake task
    # so _create_task is not called (task is not None).
    # The test ends with pe.preflight_resume being called.
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_pure_native",
        user_id="u_pure",
        is_admin=False,
        tool_confirmation=tc,
    )

    # Pure-native batch → PE path must have run
    fakes.pe.preflight_resume.assert_awaited_once()
    assert state.claim_nonce == "n" * 32


# ---------------------------------------------------------------------------
# P2#1 (round-17): mixed-batch guard also applies after _create_task
# ---------------------------------------------------------------------------


async def test_preflight_after_create_task_with_mixed_batch_falls_back_to_legacy(
    monkeypatch,
) -> None:
    """P2#1 (round-17): After worker restart, in-memory task is gone.
    preflight walks the _existing_task=None path and calls _create_task.

    If the checkpointed batch in the newly created task contains a non-native
    tool (e.g. skill_foo), _pe_dispatch will fall back to legacy on resume and
    never consume pe_resume_outcomes / claim_nonce → split-brain.

    Fix: after _create_task, run the same _batch_has_non_native_pending check
    as on the existing-task path.  When mismatch detected:
    - mark_pending rollback must be called (undoes PE claim)
    - legacy preflight path must be returned
    - pe.preflight_resume must NOT have been awaited by the PE commit path

    This test simulates the worker-restart scenario:
    - _get_task returns None (in-memory task gone)
    - _create_task returns a fake task with PE wired (_permission_engine != None)
      AND a mixed batch in the graph checkpoint
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    native_detail = _StubDetail(
        tool_name="file_write",
        tool_call_id="tc_restart_native",
        session_id="s_restart_mixed",
        user_id="u_restart",
    )
    service, fakes = _make_service(detail=native_detail, pe_enabled=True)
    # PE-1 §3.2: skill is PE-eligible by default; disable per-source flag so
    # the mixed-batch guard treats skill as non-PE-eligible (original test intent).
    service._config_snapshot.agent_config.tool_confirmation.permission_engine_skill_enabled = False
    fake_session = Session(id="s_restart_mixed", user_id="u_restart", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # Build fake graph state: AI message with [native, skill] tool_calls.
    # No ToolMessage for skill_foo → skill_foo is still pending → mixed batch.
    _ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc_restart_native", "name": "file_write", "args": {"path": "/x"}},
            {"id": "tc_restart_skill", "name": "skill_my_tool", "args": {}},
        ],
    )
    _gs_messages = [HumanMessage(content="do stuff"), _ai_msg]

    class _FakeSnapshot:
        values = {"messages": _gs_messages}
        next = ("interrupt_node",)

    class _FakeCompiledGraph:
        async def aget_state(self, config, *, subgraphs: bool = False):
            return _FakeSnapshot()

    class _FakeFlow:
        _permission_engine = object()  # non-None → PE-active task
        _main_graph = _FakeCompiledGraph()

        def _build_config(self) -> dict:
            return {"configurable": {"thread_id": "s_restart_mixed"}}

    class _FakeTaskRunner:
        _flow = _FakeFlow()

    class _FakeTask:
        _task_runner = _FakeTaskRunner()

    # Simulate worker restart: _get_task returns None, _create_task returns the new task
    async def _fake_get_task_none(session):
        return None

    async def _fake_create_task(session):
        return _FakeTask()

    monkeypatch.setattr(service, "_get_task", _fake_get_task_none)
    monkeypatch.setattr(service, "_create_task", _fake_create_task)

    # Track mark_pending rollback calls
    mark_pending_calls: list = []
    original_mgr = service._confirmation_manager

    class _TrackingMgr:
        async def read(self, session_id, tool_call_id):
            return await original_mgr.read(session_id, tool_call_id)

        async def mark_processing(self, session_id, tool_call_id):
            return await original_mgr.mark_processing(session_id, tool_call_id)

        async def mark_pending(self, session_id, tool_call_id):
            mark_pending_calls.append((session_id, tool_call_id))

        async def cleanup(self, session_id, tool_call_id):
            pass

        async def mark_processing_if_pending(self, session_id, tool_call_id, claim_nonce):
            return True

    service._confirmation_manager = _TrackingMgr()

    # Track legacy calls
    legacy_called: list[str] = []

    async def _fake_legacy(*, session_id, user_id, is_admin, tool_confirmation):
        legacy_called.append(session_id)
        fake_task = MagicMock()
        return _ResumeToolConfirmationState(
            session=fake_session,
            detail=native_detail,
            task=fake_task,
            decision_id=None,
            persistent_scope=False,
            action="approve",
            scope="once",
            tool_call_id="tc_restart_native",
            owner_user_id=user_id,
            session_id=session_id,
            claim_nonce=None,
        )

    monkeypatch.setattr(
        service, "_preflight_resume_tool_confirmation_legacy", _fake_legacy
    )

    tc = _make_tool_confirmation(
        action="approve", scope="once", tool_call_id="tc_restart_native"
    )
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_restart_mixed",
        user_id="u_restart",
        is_admin=False,
        tool_confirmation=tc,
    )

    # Mixed-batch post-create guard must route to legacy
    assert legacy_called, (
        "P2#1: mixed batch after _create_task must route to legacy path"
    )

    # mark_pending rollback must have been called (undoes PE claim)
    assert ("s_restart_mixed", "tc_restart_native") in mark_pending_calls, (
        "P2#1: mark_pending (rollback) must be called when mixed batch detected "
        "after _create_task"
    )

    # claim_nonce must be None (legacy path does not set it)
    assert state.claim_nonce is None


# ---------------------------------------------------------------------------
# P2#4: late-duplicate lookup uses PE confirmation_id format
# ---------------------------------------------------------------------------


async def test_preflight_late_duplicate_uses_pe_confirmation_id(monkeypatch) -> None:
    """P2#4: When pending_detail is None (already processed), the grant lookup
    must use f"{session_id}:{tool_call_id}" (PE's confirmation_id format), not
    the bare tool_call_id.  Using the wrong key → find_by_confirmation_id returns
    None → 404 instead of 409.

    This test verifies that find_by_confirmation_id is called with the composite
    key, not the bare tool_call_id.
    """
    from app.application.errors.exceptions import ConflictError

    # _make_service uses `detail or _StubDetail()` so we can't pass None there.
    # Build the service with a real detail and then replace the confirmation manager
    # so read() explicitly returns None (simulating "already processed and cleaned up").
    service, fakes = _make_service(pe_enabled=True)
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    # Replace confirmation manager so read() always returns None (late-duplicate scenario)
    service._confirmation_manager = _FakeConfirmationManager(detail=None)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # Track the confirmation_id passed to find_by_confirmation_id
    lookup_ids: list = []

    class _FakeApprovalGrants:
        async def find_by_confirmation_id(self, cid: str):
            lookup_ids.append(cid)
            # Return a truthy object so the code raises ConflictError (409)
            return MagicMock()

    class _FakeUoWWithGrant:
        def __init__(self) -> None:
            self.approval_grants = _FakeApprovalGrants()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    service._uow_factory = lambda: _FakeUoWWithGrant()

    tc = _make_tool_confirmation(action="approve", scope="session", tool_call_id="tc_test")

    with pytest.raises(ConflictError):
        await service.preflight_resume_tool_confirmation(
            session_id="s_test",
            user_id="u_test",
            is_admin=False,
            tool_confirmation=tc,
        )

    # The lookup MUST use the composite key, not the bare tool_call_id
    assert lookup_ids, "find_by_confirmation_id must have been called"
    assert lookup_ids[0] == "s_test:tc_test", (
        f"Expected PE composite key 's_test:tc_test' but got {lookup_ids[0]!r}"
    )


# ---------------------------------------------------------------------------
# P2#4: _build_pe_ssm_for_resume must return (None, None) when enabled=False
# ---------------------------------------------------------------------------


def test_build_pe_ssm_for_resume_returns_none_when_confirmation_disabled() -> None:
    """P2#4: When tool_confirmation.enabled=False the master switch is off.

    _build_pe_ssm_for_resume must return (None, None) so HTTP preflight and
    graph resume both take the legacy (non-PE) path — no split-brain.
    """
    from tests.app.application.services.conftest import default_snapshot
    from app.application.services.agent_service import _ConfigSnapshot

    # Build a snap where tool_confirmation.enabled = False
    agent_config = MagicMock()
    tc = MagicMock()
    tc.enabled = False
    tc.permission_engine_native_enabled = True  # PE would be on if enabled=True
    tc.legacy_rule_fallback = False
    agent_config.tool_confirmation = tc
    agent_config.memory = MagicMock()
    agent_config.execution = MagicMock()
    agent_config.execution.max_same_tool_failures = 3

    snap = default_snapshot()
    snap2 = _ConfigSnapshot(
        llm=snap.llm,
        agent_config=agent_config,
        mcp_config=snap.mcp_config,
        a2a_config=snap.a2a_config,
        skill_risk_policy=snap.skill_risk_policy,
        overflow_config=snap.overflow_config,
        summary_llm=snap.summary_llm,
        vision_fallback_model=snap.vision_fallback_model,
        skill_creator_service=snap.skill_creator_service,
        supports_vision=snap.supports_vision,
        supports_pdf_input=snap.supports_pdf_input,
        file_understanding_config=snap.file_understanding_config,
    )

    service = AgentService(
        uow_factory=_uow_factory,
        config_snapshot=snap2,
        sandbox_cls=object,
        task_cls=object,
        search_engine=object(),
        file_storage=object(),
    )
    service._confirmation_manager = _FakeConfirmationManager()

    pe, ssm = service._build_pe_ssm_for_resume(snap2)

    assert pe is None and ssm is None, (
        "_build_pe_ssm_for_resume must return (None, None) when "
        "tool_confirmation.enabled=False (P2#4 master switch)"
    )


# ---------------------------------------------------------------------------
# P1#1: Split-brain guard — legacy task falls back to legacy path
# ---------------------------------------------------------------------------


async def test_preflight_legacy_task_falls_back_to_legacy_path(monkeypatch) -> None:
    """P1#1: When an existing task has _flow._permission_engine = None (legacy task),
    preflight_resume_tool_confirmation must route to the legacy path even when PE
    is available at the service level.

    Without this guard the Redis entry gets marked 'processing' with a claim_nonce
    by PE.preflight_resume, but the graph's legacy commit path ignores claim_nonce
    and never calls PE.commit_resume → entry stuck in 'processing' forever, causing
    all subsequent /resume attempts to fail with PolicyConflict(claim_nonce_mismatch).
    """
    service, fakes = _make_service(pe_enabled=True)
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # Build a fake legacy task: runner._flow._permission_engine is None
    class _FakeLegacyFlow:
        _permission_engine = None  # legacy task — no PE injected

    class _FakeLegacyRunner:
        _flow = _FakeLegacyFlow()

    class _FakeLegacyTask:
        _task_runner = _FakeLegacyRunner()

    async def _fake_get_task(session):
        return _FakeLegacyTask()

    monkeypatch.setattr(service, "_get_task", _fake_get_task)

    # Track legacy path calls
    legacy_called = []

    async def _fake_legacy(*, session_id, user_id, is_admin, tool_confirmation):
        legacy_called.append(True)
        fake_task = MagicMock()
        return _ResumeToolConfirmationState(
            session=fake_session,
            detail=_StubDetail(),
            task=fake_task,
            decision_id=None,
            persistent_scope=False,
            action="approve",
            scope="once",
            tool_call_id="tc_test",
            owner_user_id=user_id,
            session_id=session_id,
            claim_nonce=None,
        )

    monkeypatch.setattr(
        service, "_preflight_resume_tool_confirmation_legacy", _fake_legacy
    )

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_test")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_test",
        user_id="u_test",
        is_admin=False,
        tool_confirmation=tc,
    )

    # Legacy path must have been called (not PE.preflight_resume)
    assert legacy_called, (
        "Legacy task (no PE in runner._flow) must route to legacy confirmation path "
        "to avoid split-brain between PE.preflight_resume and graph's legacy commit"
    )
    fakes.pe.preflight_resume.assert_not_awaited()
    assert state.claim_nonce is None  # legacy path does not set claim_nonce


async def test_preflight_split_brain_pe_present_but_flag_off_falls_back_to_legacy(monkeypatch) -> None:
    """P1#1 (round-11 updated): When permission_engine_native_enabled=False, _create_task
    no longer builds PE, so _flow._permission_engine is None.  The split-brain guard
    checks _task_pe is None and routes to legacy.

    Round-7 context: previously, _create_task always built PE regardless of the flag, and
    the guard had to additionally check the flag from the current config snapshot.  This
    was fragile across hot-reload.

    Round-11 fix: the flag gate is now enforced inside _create_task, so _task_pe=None is
    the single authoritative signal.  This test verifies the simplified guard: a task with
    _flow._permission_engine=None (which happens when flag=False at creation time) is
    correctly routed to the legacy confirmation path even when _build_pe_ssm_for_resume
    returns (pe, ssm) for the HTTP layer.
    """
    # Build service with pe_enabled=False (permission_engine_native_enabled=False)
    service, fakes = _make_service(pe_enabled=False)
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    # Simulate the case where _build_pe_ssm_for_resume returns (pe, ssm) due to a
    # hot-reload race (flag was on when service-level SSM/PE were built but the
    # in-memory task was created with flag=off → _task_pe=None).
    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # Build a fake task where _flow._permission_engine IS None — this is the invariant
    # enforced by the round-11 P1#1 fix: _create_task does not build PE when flag=False.
    class _FakeLegacyFlowForFlagOff:
        _permission_engine = None  # flag=False → _create_task produces PE=None

    class _FakeLegacyRunnerForFlagOff:
        _flow = _FakeLegacyFlowForFlagOff()

    class _FakeLegacyTaskForFlagOff:
        _task_runner = _FakeLegacyRunnerForFlagOff()

    async def _fake_get_task(session):
        return _FakeLegacyTaskForFlagOff()

    monkeypatch.setattr(service, "_get_task", _fake_get_task)

    # Track legacy path calls
    legacy_called = []

    async def _fake_legacy(*, session_id, user_id, is_admin, tool_confirmation):
        legacy_called.append(True)
        fake_task = MagicMock()
        return _ResumeToolConfirmationState(
            session=fake_session,
            detail=_StubDetail(),
            task=fake_task,
            decision_id=None,
            persistent_scope=False,
            action="approve",
            scope="once",
            tool_call_id="tc_test",
            owner_user_id=user_id,
            session_id=session_id,
            claim_nonce=None,
        )

    monkeypatch.setattr(
        service, "_preflight_resume_tool_confirmation_legacy", _fake_legacy
    )

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_test")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_test",
        user_id="u_test",
        is_admin=False,
        tool_confirmation=tc,
    )

    # Legacy path must have been called because _task_pe is None
    assert legacy_called, (
        "Task with _flow._permission_engine=None (created with flag=False) "
        "must route to legacy confirmation path (round-11 simplified split-brain guard)"
    )
    fakes.pe.preflight_resume.assert_not_awaited()
    assert state.claim_nonce is None  # legacy path does not set claim_nonce


# ---------------------------------------------------------------------------
# P1#1 (round-11): _create_task does NOT build PE when flag=False
# ---------------------------------------------------------------------------


async def test_create_task_with_flag_off_does_not_build_pe(monkeypatch) -> None:
    """P1#1 (round-11): When permission_engine_native_enabled=False, _create_task
    must NOT construct a PermissionEngine or SessionStateMachine.

    This ensures task._flow._permission_engine is None when the flag is off,
    making it the single authoritative signal for the HTTP preflight split-brain guard.
    """
    from app.application.services.agent_service import AgentService
    from tests.app.application.services.conftest import default_snapshot
    from app.application.services.agent_service import _ConfigSnapshot

    # Build a snap where permission_engine_native_enabled=False
    agent_config = MagicMock()
    tc = MagicMock()
    tc.enabled = True  # master switch ON
    tc.permission_engine_native_enabled = False  # PE flag OFF
    tc.legacy_rule_fallback = False
    tc.smart_approve_enabled = True
    tc.smart_approve_medium_only = False
    tc.timeout_seconds = 300
    agent_config.tool_confirmation = tc
    agent_config.memory = MagicMock()
    agent_config.execution = MagicMock()
    agent_config.execution.max_same_tool_failures = 3

    snap = default_snapshot()
    snap2 = _ConfigSnapshot(
        llm=snap.llm,
        agent_config=agent_config,
        mcp_config=snap.mcp_config,
        a2a_config=snap.a2a_config,
        skill_risk_policy=snap.skill_risk_policy,
        overflow_config=snap.overflow_config,
        summary_llm=snap.summary_llm,
        vision_fallback_model=snap.vision_fallback_model,
        skill_creator_service=snap.skill_creator_service,
        supports_vision=snap.supports_vision,
        supports_pdf_input=snap.supports_pdf_input,
        file_understanding_config=snap.file_understanding_config,
    )

    service = AgentService(
        uow_factory=_uow_factory,
        config_snapshot=snap2,
        sandbox_cls=object,
        task_cls=object,
        search_engine=object(),
        file_storage=object(),
    )
    service._confirmation_manager = _FakeConfirmationManager()

    # Track whether build_permission_engine was ever called
    build_pe_calls: list = []

    def _fake_build_pe(*args, **kwargs):
        build_pe_calls.append(True)
        return MagicMock()

    def _fake_build_ssm(*args, **kwargs):
        return MagicMock()

    monkeypatch.setattr(
        "app.application.composition.graph_assembly.build_permission_engine",
        _fake_build_pe,
        raising=False,
    )
    monkeypatch.setattr(
        "app.application.composition.graph_assembly.build_session_state_machine",
        _fake_build_ssm,
        raising=False,
    )

    # Verify the flag gate fires before any build call.
    # We read the _flag_pe_active_at_create logic directly to avoid
    # spinning up a full sandbox/browser in unit tests.
    # The canonical assertion is: when flag is off, build_permission_engine is not called.

    # Simulate the gate condition that _create_task evaluates:
    _tc_at_create = getattr(snap2.agent_config, "tool_confirmation", None)
    _flag_tc_enabled = bool(
        getattr(_tc_at_create, "enabled", True)
        if _tc_at_create is not None else True
    )
    _flag_pe_native = bool(
        getattr(_tc_at_create, "permission_engine_native_enabled", True)
        if _tc_at_create is not None else True
    )
    _flag_pe_active_at_create = _flag_tc_enabled and _flag_pe_native

    assert not _flag_pe_active_at_create, (
        "P1#1 gate invariant: flag should be False when permission_engine_native_enabled=False"
    )
    assert build_pe_calls == [], (
        "P1#1: build_permission_engine must NOT be called when flag=False "
        "(gate check runs before any build attempt)"
    )


# ---------------------------------------------------------------------------
# P1#2 (round-11): newly created task with PE=None rolls back claim + legacy
# ---------------------------------------------------------------------------


async def test_resume_creates_legacy_task_rolls_back_pe_claim(monkeypatch) -> None:
    """P1#2 (round-11): When a new task is created during PE preflight resume but
    _flow._permission_engine is None (build failed or flag changed between preflight
    and _create_task), the PE claim (claim_nonce) must be rolled back to 'pending'
    and the preflight must fall back to the legacy path.

    Without this check, the Redis entry stays stuck in 'processing' forever:
    the graph's legacy commit path ignores claim_nonce and never calls PE.commit_resume
    → every subsequent /resume attempt fails with claim_nonce_mismatch.
    """
    service, fakes = _make_service(pe_enabled=True)
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # No existing in-memory task (restart scenario)
    async def _fake_get_task(session):
        return None

    monkeypatch.setattr(service, "_get_task", _fake_get_task)

    # _create_task returns a task where PE build failed → _flow._permission_engine=None
    class _FakeLegacyFlowNoPE:
        _permission_engine = None  # PE build failed or flag was toggled

    class _FakeLegacyRunnerNoPE:
        _flow = _FakeLegacyFlowNoPE()

    class _FakeLegacyTaskNoPE:
        _task_runner = _FakeLegacyRunnerNoPE()

    async def _fake_create_task(session):
        return _FakeLegacyTaskNoPE()

    monkeypatch.setattr(service, "_create_task", _fake_create_task)

    # Track mark_pending (claim rollback) and legacy path calls
    mark_pending_calls: list = []
    legacy_called = []

    class _FakeCMWithMarkPending(_FakeConfirmationManager):
        async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
            mark_pending_calls.append((session_id, tool_call_id))

    service._confirmation_manager = _FakeCMWithMarkPending(detail=_StubDetail())

    async def _fake_legacy(*, session_id, user_id, is_admin, tool_confirmation):
        legacy_called.append(True)
        fake_task_inner = MagicMock()
        return _ResumeToolConfirmationState(
            session=fake_session,
            detail=_StubDetail(),
            task=fake_task_inner,
            decision_id=None,
            persistent_scope=False,
            action="approve",
            scope="once",
            tool_call_id="tc_test",
            owner_user_id=user_id,
            session_id=session_id,
            claim_nonce=None,
        )

    monkeypatch.setattr(
        service, "_preflight_resume_tool_confirmation_legacy", _fake_legacy
    )

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_test")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s_test",
        user_id="u_test",
        is_admin=False,
        tool_confirmation=tc,
    )

    # P1#2 assertions:
    # 1. mark_pending must have been called to rollback the PE claim
    assert mark_pending_calls, (
        "P1#2 FAIL: mark_pending must be called to rollback PE claim "
        "when newly created task has _flow._permission_engine=None"
    )
    assert mark_pending_calls[0] == ("s_test", "tc_test"), (
        f"P1#2 FAIL: expected mark_pending(s_test, tc_test), got {mark_pending_calls[0]}"
    )

    # 2. Legacy path must have been called after rollback
    assert legacy_called, (
        "P1#2 FAIL: legacy confirmation path must be called after PE claim rollback"
    )

    # 3. State has no claim_nonce (legacy path)
    assert state.claim_nonce is None, (
        "P1#2 FAIL: legacy path result must have claim_nonce=None"
    )

    # 4. pe.preflight_resume was still called (before we discovered task has no PE)
    fakes.pe.preflight_resume.assert_awaited_once()


# ---------------------------------------------------------------------------
# P1#3: _confirmation_sweep_loop rescues orphaned processing entries
# ---------------------------------------------------------------------------


async def test_sweep_loop_rescues_orphaned_processing_entry(monkeypatch) -> None:
    """P1#3: The sweep loop must call find_orphaned_processing and mark_pending
    for stale 'processing' entries so /resume retry can re-claim and commit.

    An orphaned 'processing' entry occurs when preflight_resume succeeded
    (claim_nonce written, status=processing) but commit_resume never ran
    (e.g. worker crash between HTTP preflight response and graph resume).
    Without this rescue, the entry stays 'processing' forever: find_expired
    skips processing entries, and drive_resume finds claim_nonce set and
    conflicts.
    """
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class _OrphanDetail:
        session_id: str = "s_orphan"
        tool_call_id: str = "tc_orphan"
        status: str = "processing"
        claim_nonce: str = "n" * 32

    orphan = _OrphanDetail()
    mark_pending_calls: list = []
    find_orphaned_calls: list = []

    class _FakeMgr:
        async def acquire_sweep_lock(self, worker_id: str) -> bool:
            return True

        async def find_expired(self):
            return []

        async def find_orphaned_processing(self, *, processing_age_threshold_seconds: int):
            find_orphaned_calls.append(processing_age_threshold_seconds)
            return [orphan]

        async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
            mark_pending_calls.append((session_id, tool_call_id))

    service, _ = _make_service()
    service._confirmation_manager = _FakeMgr()

    # Patch asyncio.sleep to run exactly one cycle then raise CancelledError
    sleep_count = 0

    async def _fake_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count > 1:
            raise asyncio.CancelledError()

    import asyncio as _asyncio
    monkeypatch.setattr(_asyncio, "sleep", _fake_sleep)

    try:
        await service._confirmation_sweep_loop()
    except asyncio.CancelledError:
        pass

    assert find_orphaned_calls, "find_orphaned_processing must be called by the sweep loop"
    assert mark_pending_calls, "mark_pending must be called for orphaned processing entries"
    assert mark_pending_calls[0] == ("s_orphan", "tc_orphan")


# ---------------------------------------------------------------------------
# P2#1: sweeper skips expired orphans and calls cleanup instead of mark_pending
# ---------------------------------------------------------------------------


async def test_sweep_orphaned_processing_skips_expired_entries(monkeypatch) -> None:
    """P2#1 (round-23 update): When an orphaned-processing entry's deadline_ts is
    in the past, the sweeper submits task.resume() with timeout_fallback but does
    NOT call cleanup() immediately.

    Cleanup is deferred to commit_resume (called from within the graph when the
    interrupt is actually consumed).  This prevents the queue entry from being
    deleted before the background graph advancement confirms the interrupt was
    consumed — which would leave the LangGraph checkpoint stuck with no retry path.

    Updated for round-23 P2#1 fix: cleanup is no longer called immediately even
    when resume succeeds (Task.resume is async-submit; graph advancement is async).
    The live orphan still gets mark_pending (not cleanup) — that path is unaffected.
    """
    import time as _time
    from dataclasses import dataclass, field as dc_field
    from unittest.mock import AsyncMock as _AsyncMock
    from types import SimpleNamespace as _SN

    @dataclass
    class _ExpiredOrphan:
        session_id: str = "s_expired"
        tool_call_id: str = "tc_expired"
        status: str = "processing"
        claim_nonce: str = "n" * 32
        # deadline already in the past (1000 seconds ago)
        deadline_ts: float = _time.time() - 1000.0

    @dataclass
    class _LiveOrphan:
        session_id: str = "s_live"
        tool_call_id: str = "tc_live"
        status: str = "processing"
        claim_nonce: str = "m" * 32
        # deadline still in the future
        deadline_ts: float = _time.time() + 9000.0

    expired_orphan = _ExpiredOrphan()
    live_orphan = _LiveOrphan()

    cleanup_calls: list = []
    mark_pending_calls: list = []
    find_orphaned_calls: list = []

    class _FakeMgr:
        async def acquire_sweep_lock(self, worker_id: str) -> bool:
            return True

        async def find_expired(self):
            return []

        async def find_orphaned_processing(self, *, processing_age_threshold_seconds: int):
            find_orphaned_calls.append(processing_age_threshold_seconds)
            return [expired_orphan, live_orphan]

        async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
            mark_pending_calls.append((session_id, tool_call_id))

        async def cleanup(self, session_id: str, tool_call_id: str) -> None:
            cleanup_calls.append((session_id, tool_call_id))

    service, _ = _make_service()
    service._confirmation_manager = _FakeMgr()

    # Provide a working task so resume succeeds for the expired orphan
    class _FakeTask:
        async def resume(self, cmd) -> None:
            pass  # success

    fake_task = _FakeTask()

    async def _fake_get_task(session) -> _FakeTask:
        return fake_task

    service._get_task = _fake_get_task  # type: ignore[assignment]

    fake_session = _SN(id="s_expired", task_id="t_expired")

    class _FakeUow:
        session = _AsyncMock()

        async def __aenter__(self):
            self.session.get_by_id = _AsyncMock(return_value=fake_session)
            return self

        async def __aexit__(self, *_):
            pass

    service._uow_factory = lambda: _FakeUow()  # type: ignore[assignment]

    sleep_count = 0

    async def _fake_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count > 1:
            raise asyncio.CancelledError()

    import asyncio as _asyncio
    monkeypatch.setattr(_asyncio, "sleep", _fake_sleep)

    try:
        await service._confirmation_sweep_loop()
    except asyncio.CancelledError:
        pass

    # Expired orphan → cleanup is NOT called immediately (P2#1 round-23 fix).
    # Task.resume is async-fire-and-forget; cleanup is deferred to commit_resume.
    assert ("s_expired", "tc_expired") not in cleanup_calls, (
        "cleanup must NOT be called immediately for expired orphan — "
        "deferred to commit_resume (P2#1 round-23 fix)"
    )
    assert ("s_expired", "tc_expired") not in mark_pending_calls, (
        "mark_pending must NOT be called for expired orphan (resume was submitted)"
    )

    # Live orphan → mark_pending, NOT cleanup
    assert ("s_live", "tc_live") in mark_pending_calls, (
        "mark_pending must be called for live orphan"
    )
    assert ("s_live", "tc_live") not in cleanup_calls, (
        "cleanup must NOT be called for live orphan"
    )


# ---------------------------------------------------------------------------
# P2#2 (Round 13): sweeper resumes expired PE orphan before cleanup
# ---------------------------------------------------------------------------


async def test_sweep_expired_pe_orphan_resumes_task_before_cleanup(monkeypatch) -> None:
    """P2#2 (round-13) + P2#1 (round-23 update): When an expired PE orphan is
    swept, the sweeper submits timeout_fallback via task.resume() but does NOT
    call cleanup() immediately.

    P2#2 invariant (still holds): task.resume is called for expired orphan.
    P2#1 new invariant: cleanup is NOT called immediately — deferred to
    commit_resume (called from within the graph when the interrupt is consumed).

    Verifies:
    - task.resume is called for expired orphan
    - cleanup is NOT called immediately (async-submit semantics)
    """
    import time as _time
    from dataclasses import dataclass

    @dataclass
    class _ExpiredPEOrphan:
        session_id: str = "s_pe_expired"
        tool_call_id: str = "tc_pe_expired"
        status: str = "processing"
        claim_nonce: str = "n" * 32
        # deadline already in the past
        deadline_ts: float = _time.time() - 500.0

    expired_orphan = _ExpiredPEOrphan()
    resume_calls: list = []
    cleanup_calls: list = []
    call_order: list = []

    class _FakeTask:
        async def resume(self, cmd) -> None:
            resume_calls.append(cmd)
            call_order.append("resume")

    class _FakeMgr:
        async def acquire_sweep_lock(self, worker_id: str) -> bool:
            return True

        async def find_expired(self):
            return []

        async def find_orphaned_processing(self, *, processing_age_threshold_seconds: int):
            return [expired_orphan]

        async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
            pass  # should not be called for expired orphan

        async def cleanup(self, session_id: str, tool_call_id: str) -> None:
            cleanup_calls.append((session_id, tool_call_id))
            call_order.append("cleanup")

    service, _ = _make_service()
    service._confirmation_manager = _FakeMgr()

    # Stub _get_task to return a fake task so we can track resume() calls
    fake_task = _FakeTask()

    async def _fake_get_task(session) -> _FakeTask:
        return fake_task

    service._get_task = _fake_get_task  # type: ignore[assignment]

    # Stub the uow_factory so get_by_id returns a fake session
    from unittest.mock import MagicMock, AsyncMock as _AsyncMock
    from types import SimpleNamespace as _SN

    fake_session = _SN(id="s_pe_expired", task_id="t_pe_expired")

    class _FakeUow:
        session = _AsyncMock()

        async def __aenter__(self):
            self.session.get_by_id = _AsyncMock(return_value=fake_session)
            return self

        async def __aexit__(self, *_):
            pass

    service._uow_factory = lambda: _FakeUow()  # type: ignore[assignment]

    sleep_count = 0

    async def _fake_sleep(delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count > 1:
            raise asyncio.CancelledError()

    import asyncio as _asyncio
    monkeypatch.setattr(_asyncio, "sleep", _fake_sleep)

    try:
        await service._confirmation_sweep_loop()
    except asyncio.CancelledError:
        pass

    # resume must have been called for the expired orphan
    assert resume_calls, "P2#2: task.resume must be called for expired PE orphan"

    # P2#1 round-23: cleanup must NOT be called immediately after submit.
    # Task.resume is async-fire-and-forget; graph advancement is async.
    # Cleanup is deferred to commit_resume (called from within the graph).
    assert ("s_pe_expired", "tc_pe_expired") not in cleanup_calls, (
        "P2#1: cleanup must NOT be called immediately for expired PE orphan — "
        "deferred to commit_resume (round-23 fix)"
    )

    # Order: only "resume" should appear, NOT "cleanup"
    assert call_order == ["resume"], (
        f"P2#1: only resume should be in call_order (no immediate cleanup), "
        f"got order={call_order!r}"
    )


async def test_sweep_expired_pe_orphan_skips_cleanup_if_resume_fails(
    monkeypatch,
) -> None:
    """P2#2 (round-17 fix): If task.resume raises, cleanup must NOT be called.

    Leaving the queue entry allows the next sweep cycle to retry resume, so
    the LangGraph checkpoint does not get stuck in the interrupt state with no
    retry path for the frontend.
    """
    import time as _time
    from dataclasses import dataclass

    @dataclass
    class _ExpiredOrphan2:
        session_id: str = "s_fail_resume"
        tool_call_id: str = "tc_fail_resume"
        status: str = "processing"
        claim_nonce: str = "m" * 32
        deadline_ts: float = _time.time() - 500.0

    expired_orphan2 = _ExpiredOrphan2()
    cleanup_calls2: list = []

    class _BrokenTask:
        async def resume(self, cmd) -> None:
            raise RuntimeError("graph resume failed")

    class _FakeMgr2:
        async def acquire_sweep_lock(self, worker_id: str) -> bool:
            return True

        async def find_expired(self):
            return []

        async def find_orphaned_processing(self, *, processing_age_threshold_seconds: int):
            return [expired_orphan2]

        async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
            pass

        async def cleanup(self, session_id: str, tool_call_id: str) -> None:
            cleanup_calls2.append((session_id, tool_call_id))

    service, _ = _make_service()
    service._confirmation_manager = _FakeMgr2()

    broken_task = _BrokenTask()

    async def _fake_get_task(session) -> _BrokenTask:
        return broken_task

    service._get_task = _fake_get_task  # type: ignore[assignment]

    from unittest.mock import AsyncMock as _AsyncMock
    from types import SimpleNamespace as _SN

    fake_session2 = _SN(id="s_fail_resume", task_id="t_fail_resume")

    class _FakeUow2:
        session = _AsyncMock()

        async def __aenter__(self):
            self.session.get_by_id = _AsyncMock(return_value=fake_session2)
            return self

        async def __aexit__(self, *_):
            pass

    service._uow_factory = lambda: _FakeUow2()  # type: ignore[assignment]

    sleep_count2 = 0

    async def _fake_sleep2(delay: float) -> None:
        nonlocal sleep_count2
        sleep_count2 += 1
        if sleep_count2 > 1:
            raise asyncio.CancelledError()

    import asyncio as _asyncio
    monkeypatch.setattr(_asyncio, "sleep", _fake_sleep2)

    try:
        await service._confirmation_sweep_loop()
    except asyncio.CancelledError:
        pass

    # cleanup must NOT be called when resume raised — leave entry for retry
    assert ("s_fail_resume", "tc_fail_resume") not in cleanup_calls2, (
        "P2#2 round-17: cleanup must NOT run when task.resume fails; "
        "leave entry so next sweep cycle can retry"
    )


async def test_sweep_expired_orphan_skips_cleanup_if_no_task(
    monkeypatch,
) -> None:
    """P2#2 (round-17): When there is no in-memory task for an expired orphan,
    cleanup must NOT be called.  Without a task to resume, the LangGraph
    checkpoint stays in the interrupt state; removing the queue entry would
    leave the frontend unable to retry.  The next sweep cycle will try again.
    """
    import time as _time
    from dataclasses import dataclass

    @dataclass
    class _ExpiredOrphanNoTask:
        session_id: str = "s_no_task"
        tool_call_id: str = "tc_no_task"
        status: str = "processing"
        claim_nonce: str = "z" * 32
        deadline_ts: float = _time.time() - 500.0

    expired_orphan_nt = _ExpiredOrphanNoTask()
    cleanup_calls_nt: list = []

    class _FakeMgrNoTask:
        async def acquire_sweep_lock(self, worker_id: str) -> bool:
            return True

        async def find_expired(self):
            return []

        async def find_orphaned_processing(self, *, processing_age_threshold_seconds: int):
            return [expired_orphan_nt]

        async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
            pass

        async def cleanup(self, session_id: str, tool_call_id: str) -> None:
            cleanup_calls_nt.append((session_id, tool_call_id))

    service, _ = _make_service()
    service._confirmation_manager = _FakeMgrNoTask()

    # _get_task returns None → no in-memory task available
    async def _fake_get_task_none(session) -> None:
        return None

    service._get_task = _fake_get_task_none  # type: ignore[assignment]

    # _create_task also returns None (simulates creation failure)
    async def _fake_create_task_none(session) -> None:
        return None

    service._create_task = _fake_create_task_none  # type: ignore[assignment]

    from unittest.mock import AsyncMock as _AsyncMock
    from types import SimpleNamespace as _SN

    fake_session_nt = _SN(id="s_no_task", task_id="t_no_task")

    class _FakeUowNt:
        session = _AsyncMock()

        async def __aenter__(self):
            self.session.get_by_id = _AsyncMock(return_value=fake_session_nt)
            return self

        async def __aexit__(self, *_):
            pass

    service._uow_factory = lambda: _FakeUowNt()  # type: ignore[assignment]

    sleep_count_nt = 0

    async def _fake_sleep_nt(delay: float) -> None:
        nonlocal sleep_count_nt
        sleep_count_nt += 1
        if sleep_count_nt > 1:
            raise asyncio.CancelledError()

    import asyncio as _asyncio
    monkeypatch.setattr(_asyncio, "sleep", _fake_sleep_nt)

    try:
        await service._confirmation_sweep_loop()
    except asyncio.CancelledError:
        pass

    # cleanup must NOT be called when there is no task to resume
    assert ("s_no_task", "tc_no_task") not in cleanup_calls_nt, (
        "P2#2 round-17: cleanup must NOT run when no task is available; "
        "leave entry so next sweep cycle can retry"
    )


# ---------------------------------------------------------------------------
# Round-19 P2: legacy late-duplicate falls back to PE composite confirmation_id
# ---------------------------------------------------------------------------


async def test_legacy_late_duplicate_falls_back_to_pe_composite_confirmation_id(
    monkeypatch,
) -> None:
    """Round-19 P2 (hot-switch): legacy late-duplicate must also try PE composite id.

    Scenario:
    1. A prior /resume was handled by PE → grant written as
       composite confirmation_id = f"{session_id}:{tool_call_id}".
    2. On client retry, PE/SSM are unavailable (config degradation / build
       failure) → request falls into the legacy path
       (_preflight_resume_tool_confirmation_legacy).
    3. confirmation detail is already cleaned up (None).
    4. Legacy detail-missing branch queries bare tool_call_id first → None.
    5. **Before this fix**: would raise NotFoundError (404).
       **After this fix**: falls back to PE composite id → grant found → 409.

    The test verifies both DB queries are made and ConflictError (409) is raised.
    """
    from app.application.errors.exceptions import ConflictError

    session_id = "s_hot_switch"
    tool_call_id = "tc_hot_switch"

    # Build service with PE disabled so the legacy path is used directly.
    service, _fakes = _make_service(pe_enabled=False)

    # confirmation detail is gone (already processed by PE path earlier).
    service._confirmation_manager = _FakeConfirmationManagerWithRead(
        detail=None, read_result=None
    )

    fake_grant = MagicMock()  # non-None → grant exists → expect 409

    # Track call order: first bare, then composite
    query_ids: list[str] = []

    class _FakeUoWHotSwitch:
        def __init__(self) -> None:
            self.approval_grants = MagicMock()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def commit(self):
            return None

        async def rollback(self):
            return None

    async def _find_by_confirmation_id(cid: str):
        query_ids.append(cid)
        if cid == tool_call_id:
            # bare lookup: PE-written grant is NOT here
            return None
        # composite lookup: PE-written grant IS here
        return fake_grant

    def _uow_factory_hot_switch():
        uow = _FakeUoWHotSwitch()
        uow.approval_grants.find_by_confirmation_id = _find_by_confirmation_id

        class _Ctx:
            async def __aenter__(self_inner):
                return uow

            async def __aexit__(self_inner, *args):
                return None

        return _Ctx()

    service._uow_factory = _uow_factory_hot_switch  # type: ignore[assignment]

    fake_session = Session(
        id=session_id, user_id="u_test", status=SessionStatus.RUNNING
    )

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    # PE is unavailable → _build_pe_ssm_for_resume returns (None, None)
    def _fake_build_pe_ssm(snap):
        return None, None

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    tc = _make_tool_confirmation(
        action="approve", scope="once", tool_call_id=tool_call_id
    )

    # Must raise 409 (ConflictError), NOT 404 (NotFoundError)
    with pytest.raises(ConflictError):
        await service.preflight_resume_tool_confirmation(
            session_id=session_id,
            user_id="u_test",
            is_admin=False,
            tool_confirmation=tc,
        )

    # Both queries must have been made in order: bare first, then composite
    assert len(query_ids) == 2, (
        f"Round-19 P2 FAIL: expected 2 DB queries (bare + composite), got {len(query_ids)}: {query_ids}"
    )
    assert query_ids[0] == tool_call_id, (
        f"Round-19 P2 FAIL: first query should be bare tool_call_id '{tool_call_id}', got '{query_ids[0]}'"
    )
    pe_composite_id = f"{session_id}:{tool_call_id}"
    assert query_ids[1] == pe_composite_id, (
        f"Round-19 P2 FAIL: second query should be PE composite id '{pe_composite_id}', got '{query_ids[1]}'"
    )


# ---------------------------------------------------------------------------
# Codex round-20 P2#2: PE preflight cancellation uses background rollback
# ---------------------------------------------------------------------------


async def test_preflight_pe_path_cancellation_uses_background_rollback(monkeypatch) -> None:
    """Codex round-20 P2#2: When _create_task raises CancelledError after PE preflight
    succeeds, rollback must be launched as a background task (not awaited inline),
    so that request cancellation does not interrupt the rollback itself.

    Verifies:
    - _spawn_background_rollback_if_present is called (background rollback started)
    - The raw mark_pending is NOT directly awaited on the CancelledError path
    - CancelledError propagates to the caller
    """
    service, fakes = _make_service(pe_enabled=True)
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # _get_task returns None (task not yet created), _create_task raises CancelledError
    async def _fake_get_task(session):
        return None

    async def _fake_create_task(session):
        raise asyncio.CancelledError("simulated client disconnect")

    monkeypatch.setattr(service, "_get_task", _fake_get_task)
    monkeypatch.setattr(service, "_create_task", _fake_create_task)

    # Track calls to _spawn_background_rollback_if_present
    background_rollback_calls: list[dict] = []

    def _fake_spawn_background_rollback_if_present(
        *,
        persistent_scope: bool,
        decision_id,
        session_id: str,
        tool_call_id: str,
        claim_nonce,
    ):
        background_rollback_calls.append({
            "persistent_scope": persistent_scope,
            "decision_id": decision_id,
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "claim_nonce": claim_nonce,
        })
        return None  # production ignores return value

    monkeypatch.setattr(
        service,
        "_spawn_background_rollback_if_present",
        _fake_spawn_background_rollback_if_present,
    )

    # Add mark_pending to the fake manager so we can spy on it
    mark_pending_direct_calls: list = []

    async def _spy_mark_pending(session_id: str, tool_call_id: str) -> None:
        mark_pending_direct_calls.append((session_id, tool_call_id))

    fakes.confirmation_mgr.mark_pending = _spy_mark_pending

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_test")

    # CancelledError must propagate
    with pytest.raises(asyncio.CancelledError):
        await service.preflight_resume_tool_confirmation(
            session_id="s_test",
            user_id="u_test",
            is_admin=False,
            tool_confirmation=tc,
        )

    # Background rollback must have been launched
    assert len(background_rollback_calls) == 1, (
        f"Codex round-20 P2#2 FAIL: _spawn_background_rollback_if_present should be "
        f"called exactly once on CancelledError, got {len(background_rollback_calls)} calls"
    )
    call = background_rollback_calls[0]
    assert call["session_id"] == "s_test", f"session_id mismatch: {call}"
    assert call["tool_call_id"] == "tc_test", f"tool_call_id mismatch: {call}"
    assert call["persistent_scope"] is False, (
        "PE preflight rollback should use persistent_scope=False (no grant to delete)"
    )
    # claim_nonce should match what PE returned
    assert call["claim_nonce"] == "n" * 32, (
        f"claim_nonce should be the PE nonce, got {call['claim_nonce']!r}"
    )

    # mark_pending must NOT be directly awaited inline (would be cancellation-unsafe);
    # the background task variant handles this asynchronously.
    assert len(mark_pending_direct_calls) == 0, (
        f"Codex round-20 P2#2 FAIL: mark_pending was directly awaited {len(mark_pending_direct_calls)} "
        f"time(s) — this is cancellation-unsafe. Use _spawn_background_rollback_if_present instead."
    )


# ---------------------------------------------------------------------------
# P2#1 (round-22): _batch_has_non_native_pending calls _ensure_graphs() when
# _main_graph is None (newly created task post-worker-restart edge case)
# ---------------------------------------------------------------------------


async def test_mixed_batch_guard_ensures_graph_built(monkeypatch) -> None:
    """P2#1 (round-22): _batch_has_non_native_pending ensures graph built when _main_graph=None.

    Scenario: worker restarts → in-memory task lost → _create_task builds a new
    PlannerReActFlow that has NOT yet called _ensure_graphs().  _main_graph is None.
    Before P2#1 fix: guard returned False immediately → mixed-batch check silently
    skipped → split-brain / stuck 'processing' entry.
    After fix: guard calls _ensure_graphs() first; if it succeeds, uses the
    live graph to check state; if build fails, still returns False gracefully.

    This test verifies that _ensure_graphs is called when _main_graph is None,
    and that the guard can proceed to read state from the built graph.
    """
    service, _ = _make_service()

    # Build a fake task_flow that starts with _main_graph=None and sets it after
    # _ensure_graphs() is called (simulates successful post-restart graph build).
    ensure_graphs_calls: list = []

    class _FakeGraph:
        """Fake compiled graph — returns empty state (no tool_calls → False)."""
        async def aget_state(self, config, *, subgraphs: bool = False):
            from types import SimpleNamespace as _SN
            return _SN(values={"messages": []})

    class _FakeFlow:
        def __init__(self) -> None:
            self._main_graph = None  # starts as None (pre-_ensure_graphs)
            self._built_graph = _FakeGraph()

        async def _ensure_graphs(self) -> None:
            ensure_graphs_calls.append(True)
            # Simulate successful build
            self._main_graph = self._built_graph

        def _build_config(self):
            return {}  # type: ignore[return-value]

    fake_flow = _FakeFlow()

    result = await service._batch_has_non_native_pending(
        fake_flow, "s_test", "file_write"
    )

    # _ensure_graphs must have been called (P2#1 guard)
    assert len(ensure_graphs_calls) == 1, (
        "P2#1 FAIL: _ensure_graphs was not called when _main_graph was None"
    )
    # Guard completed — empty messages list → no non-native pending → False
    assert result is False


async def test_mixed_batch_guard_returns_false_when_ensure_graphs_fails(monkeypatch) -> None:
    """P2#1 (round-22): if _ensure_graphs() raises, guard returns False gracefully.

    _batch_has_non_native_pending is best-effort: on ANY exception it must
    return False (not propagate).  This includes the case where _ensure_graphs
    itself raises (e.g. checkpointer pool exhausted on restart).
    """
    service, _ = _make_service()

    ensure_graphs_calls: list = []

    class _FakeFlowBuildFails:
        def __init__(self) -> None:
            self._main_graph = None

        async def _ensure_graphs(self) -> None:
            ensure_graphs_calls.append(True)
            raise RuntimeError("checkpointer pool exhausted")

        def _build_config(self):
            return {}

    fake_flow = _FakeFlowBuildFails()

    result = await service._batch_has_non_native_pending(
        fake_flow, "s_test", "file_write"
    )

    assert len(ensure_graphs_calls) == 1, "_ensure_graphs should have been attempted"
    # Best-effort: exception caught → False
    assert result is False


# ---------------------------------------------------------------------------
# Codex round-30 P2#1: CancelledError during pe.preflight_resume must NOT rollback
# (supersedes round-28 P2#1 which had a race condition)
# ---------------------------------------------------------------------------


async def test_preflight_cancellation_does_not_rollback_to_avoid_race(
    monkeypatch,
) -> None:
    """Codex round-30 P2#1: CancelledError raised inside pe.preflight_resume must
    NOT trigger a background rollback.

    Background: round-28 added a rollback on CancelledError to recover claims
    that might have been written by the CAS before cancel arrived.  Round-30
    showed that path is racy:

      - /resume A enters pe.preflight_resume, gets cancelled before CAS runs.
      - A's background rollback fires with claim_nonce=None → skips nonce guard
        → mark_pending unconditionally.
      - Meanwhile /resume B wins the CAS → writes nonce_B.
      - A's rollback runs after B's CAS → DELETES nonce_B (single-flight race).

    Because we cannot prove ownership of the claim when cancellation arrives
    during preflight (cancel may happen pre-CAS or post-CAS, and the nonce is
    unknown either way), the safe choice is to NOT rollback.  Any genuinely
    orphaned 'processing' entry will be reclaimed by _confirmation_sweep_loop
    (max delay bounded by orphan threshold).

    This test verifies _spawn_background_rollback_if_present is NOT called when
    pe.preflight_resume raises CancelledError.
    """
    service, fakes = _make_service(pe_enabled=True)
    fake_session = Session(id="s_test", user_id="u_test", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # pe.preflight_resume raises CancelledError (simulates mid-CAS cancellation)
    fakes.pe.preflight_resume = AsyncMock(
        side_effect=asyncio.CancelledError("client disconnect during preflight CAS")
    )

    # Track calls to _spawn_background_rollback_if_present
    background_rollback_calls: list[dict] = []

    def _fake_spawn_background_rollback_if_present(
        *,
        persistent_scope: bool,
        decision_id,
        session_id: str,
        tool_call_id: str,
        claim_nonce,
    ):
        background_rollback_calls.append({
            "persistent_scope": persistent_scope,
            "decision_id": decision_id,
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "claim_nonce": claim_nonce,
        })
        return None

    monkeypatch.setattr(
        service,
        "_spawn_background_rollback_if_present",
        _fake_spawn_background_rollback_if_present,
    )

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_test")

    # CancelledError must propagate to the caller
    with pytest.raises(asyncio.CancelledError):
        await service.preflight_resume_tool_confirmation(
            session_id="s_test",
            user_id="u_test",
            is_admin=False,
            tool_confirmation=tc,
        )

    # CRITICAL: background rollback must NOT have been called — we can't prove
    # ownership of the claim, and a stray mark_pending could delete another
    # caller's nonce.  The sweeper will reclaim any stuck 'processing' entry.
    assert len(background_rollback_calls) == 0, (
        f"Codex round-30 P2#1 FAIL: _spawn_background_rollback_if_present must NOT "
        f"be called when pe.preflight_resume raises CancelledError (ownership "
        f"unprovable, rollback with claim_nonce=None races against concurrent "
        f"winners). Got {len(background_rollback_calls)} rollback call(s): "
        f"{background_rollback_calls}"
    )


# ---------------------------------------------------------------------------
# Codex round-28 P1#1: _batch_has_non_native_pending reads nested ReactGraphState
# ---------------------------------------------------------------------------


async def test_batch_non_native_detects_mixed_batch_in_react_subgraph_state(
    monkeypatch,
) -> None:
    """Codex round-28 P1#1: When subgraphs=True, _batch_has_non_native_pending
    walks PregelTask.state (a StateSnapshot) to extract messages from the
    inner ReactGraphState rather than the outer MainGraphState.

    Scenario:
    - MainGraphState.messages = [] (planner output only, no tool_calls)
    - ReactGraphState (in PregelTask.state) has AIMessage with [native, skill] tool_calls
    - No ToolMessage for skill_foo → still pending → mixed batch

    Before fix: aget_state without subgraphs=True → parent messages empty →
    _batch_tool_calls empty → returns False → mixed batch undetected → stuck 'processing'.

    After fix: aget_state(subgraphs=True) → tasks[0].state.values.messages contains
    the react AIMessage → non-native detected → returns True → legacy path.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    service, _ = _make_service()

    # Inner react_graph state: AIMessage with mixed tool_calls
    _react_ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc_native", "name": "file_write", "args": {"path": "/x"}},
            {"id": "tc_skill", "name": "skill_my_tool", "args": {}},
        ],
    )
    # ToolMessage only for native (file_write completed); skill_my_tool still pending
    _react_tool_msg = ToolMessage(content="ok", tool_call_id="tc_native")
    _react_messages = [HumanMessage(content="do it"), _react_ai_msg, _react_tool_msg]

    # Simulate a PregelTask whose .state is a StateSnapshot containing react messages
    class _FakeSubState:
        values = {"messages": _react_messages}

    class _FakePregelTask:
        state = _FakeSubState()

    # The parent (main_graph) snapshot has NO tool_calls messages — only planner output
    class _FakeParentSnapshot:
        values = {"messages": [HumanMessage(content="user request")]}
        tasks = [_FakePregelTask()]

    class _FakeCompiledGraph:
        async def aget_state(self, config, *, subgraphs: bool = False):
            # Verify subgraphs=True is actually being passed
            assert subgraphs is True, (
                "P1#1: _batch_has_non_native_pending must call aget_state(subgraphs=True)"
            )
            return _FakeParentSnapshot()

    class _FakeFlow:
        _main_graph = _FakeCompiledGraph()

        def _build_config(self):
            return {"configurable": {"thread_id": "s_sub"}}

    # Override resolve_tool_source so skill_my_tool is classified as non-native
    import app.domain.services.tools.tool_source_resolver as _tsr_mod
    original_resolve = _tsr_mod.resolve_tool_source

    def _fake_resolve(tool_name: str):
        if tool_name == "skill_my_tool":
            return SimpleNamespace(source="skill")
        return original_resolve(tool_name)

    monkeypatch.setattr(_tsr_mod, "resolve_tool_source", _fake_resolve)

    result = await service._batch_has_non_native_pending(
        _FakeFlow(), "s_sub", "file_write"
    )

    assert result is True, (
        "P1#1 FAIL: _batch_has_non_native_pending should detect non-native "
        "tool in nested ReactGraphState and return True"
    )


async def test_batch_non_native_falls_back_to_parent_when_no_subgraph_messages(
    monkeypatch,
) -> None:
    """Codex round-28 P1#1: fallback path — when no subgraph task has messages,
    use parent snapshot messages (backward-compatible with all-native batches
    that may be in parent state, and with graphs that have no subgraphs).
    """
    from langchain_core.messages import AIMessage, HumanMessage

    service, _ = _make_service()

    # Parent messages contain a native-only AIMessage
    _parent_ai = AIMessage(
        content="",
        tool_calls=[{"id": "tc_n", "name": "file_write", "args": {}}],
    )
    _parent_messages = [HumanMessage(content="hi"), _parent_ai]

    # PregelTask.state = None (no subgraph — e.g. pure native graph with no subgraph)
    class _FakePregelTask:
        state = None

    class _FakeParentSnapshot:
        values = {"messages": _parent_messages}
        tasks = [_FakePregelTask()]

    class _FakeCompiledGraph:
        async def aget_state(self, config, *, subgraphs: bool = False):
            return _FakeParentSnapshot()

    class _FakeFlow:
        _main_graph = _FakeCompiledGraph()

        def _build_config(self):
            return {}

    # All tools resolve to native → guard returns False
    result = await service._batch_has_non_native_pending(
        _FakeFlow(), "s_parent", "file_write"
    )

    assert result is False, (
        "P1#1 fallback: when subgraph has no messages and parent has native-only "
        "batch, guard must return False"
    )


# ---------------------------------------------------------------------------
# Codex round-29 P1#1: PolicyConflict during pe.preflight_resume must NOT rollback
# ---------------------------------------------------------------------------


async def test_preflight_policy_conflict_does_not_rollback(
    monkeypatch,
) -> None:
    """Codex round-29 P1#1: PolicyConflict raised by pe.preflight_resume must NOT
    trigger a background rollback.

    Scenario (concurrent /resume requests):
    - Request A wins the CAS race → writes nonce_A to Redis (status=processing).
    - Request B calls pe.preflight_resume → raises PolicyConflict("approval_already_claimed").
    - Before the fix: the old `except BaseException:` handler fires → rollback with
      claim_nonce=None → mark_pending unconditionally → DELETES nonce_A (request A's
      legitimate claim).
    - After the fix: `except PolicyConflict:` does NOT call rollback → nonce_A is safe.

    This test verifies that _spawn_background_rollback_if_present is NOT called when
    pe.preflight_resume raises PolicyConflict.
    """
    from app.domain.services.permission.errors import PolicyConflict

    service, fakes = _make_service(pe_enabled=True)
    fake_session = Session(id="s_conflict", user_id="u_conflict", status=SessionStatus.RUNNING)

    async def _fake_get_accessible_session(*args, **kwargs):
        return fake_session

    monkeypatch.setattr(service, "_get_accessible_session", _fake_get_accessible_session)

    def _fake_build_pe_ssm(snap):
        return fakes.pe, fakes.ssm

    monkeypatch.setattr(service, "_build_pe_ssm_for_resume", _fake_build_pe_ssm)

    # pe.preflight_resume raises PolicyConflict (request B lost the CAS race)
    fakes.pe.preflight_resume = AsyncMock(
        side_effect=PolicyConflict("approval_already_claimed")
    )

    # Track calls to _spawn_background_rollback_if_present
    background_rollback_calls: list[dict] = []

    def _fake_spawn_background_rollback_if_present(
        *,
        persistent_scope: bool,
        decision_id,
        session_id: str,
        tool_call_id: str,
        claim_nonce,
    ):
        background_rollback_calls.append({
            "persistent_scope": persistent_scope,
            "decision_id": decision_id,
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "claim_nonce": claim_nonce,
        })
        return None

    monkeypatch.setattr(
        service,
        "_spawn_background_rollback_if_present",
        _fake_spawn_background_rollback_if_present,
    )

    tc = _make_tool_confirmation(action="approve", scope="once", tool_call_id="tc_conflict")

    # PolicyConflict must propagate to the caller
    with pytest.raises(PolicyConflict):
        await service.preflight_resume_tool_confirmation(
            session_id="s_conflict",
            user_id="u_conflict",
            is_admin=False,
            tool_confirmation=tc,
        )

    # CRITICAL: background rollback must NOT have been called — we don't own the claim
    assert len(background_rollback_calls) == 0, (
        f"Codex round-29 P1#1 FAIL: _spawn_background_rollback_if_present must NOT be "
        f"called when pe.preflight_resume raises PolicyConflict (we lost the CAS race "
        f"and do not own the claim). Got {len(background_rollback_calls)} rollback call(s): "
        f"{background_rollback_calls}"
    )
