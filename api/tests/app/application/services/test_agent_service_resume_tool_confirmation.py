"""R5b-3 I1-I4 invariant 单元测试：``_resume_tool_confirmation`` 的 Writer 路径。

覆盖：
- **I1/I4 late-duplicate**：``writer.write`` 返 ``(existing_id, False)`` → raise
  ``ConflictError`` (HTTP 409)，``task.resume`` 不调
- **I2 kickoff 回滚**：``task.resume`` 抛异常 → ``writer.delete_grant(decision_id)``
  被调 + ``mark_pending`` 被调，异常由 outer handler 吞成 ErrorEvent
- **I3 幂等 retry**：sequential 同 ``confirmation_id`` 第二次调用返 ``(id, False)``
  → 第二次 409，``task.resume`` 仍然只调 1 次
- **Happy path（session/always scope）**：Writer.write + mark_processing +
  task.resume + cleanup 顺序正确
- **Happy path（once scope）**：不走 Writer（Phase 1 已知 race）
- **ValueError claim collision**：raise ``BadRequestError``（400）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.errors.exceptions import BadRequestError, ConflictError
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


@dataclass
class _StubDetail:
    """Minimal stand-in for ``ConfirmationDetail``（Writer-facing fields only）。"""

    session_id: str = "s1"
    tool_call_id: str = "tc-1"
    user_id: str = "u1"
    tool_name: str = "shell_execute"
    tool_args: dict = field(default_factory=lambda: {"command": "ls /"})
    risk_level: str = "medium"
    arg_digest: str = "d1"
    primary_arg: str = "ls *"
    dir_arg: Optional[str] = ""
    matched_patterns: list = field(default_factory=list)
    deadline_ts: float = 9999999999.0
    status: str = "pending"


class _FakeWriter:
    """mock ``ApprovalStateWriter``; records .write / .delete_grant / .write_audit_only invocations."""

    def __init__(
        self,
        *,
        write_return: tuple[str, bool] = ("dec-1", True),
        write_exc: Exception | None = None,
        write_audit_only_exc: BaseException | None = None,
    ) -> None:
        self.write_return = write_return
        self.write_exc = write_exc
        self.write_audit_only_exc = write_audit_only_exc
        self.write_calls: list = []
        self.delete_grant_calls: list[str] = []
        self.write_audit_only_calls: list[dict] = []

    async def write(self, decision) -> tuple[str, bool]:
        self.write_calls.append(decision)
        if self.write_exc is not None:
            raise self.write_exc
        return self.write_return

    async def delete_grant(self, decision_id: str) -> None:
        self.delete_grant_calls.append(decision_id)

    async def write_audit_only(self, **kwargs) -> None:
        self.write_audit_only_calls.append(kwargs)
        if self.write_audit_only_exc is not None:
            raise self.write_audit_only_exc


class _ResumeTask:
    """Task with controllable ``.resume`` outcome + no-op output stream。"""

    def __init__(self, *, resume_exc: Exception | None = None) -> None:
        self._resume_exc = resume_exc
        self.resume_calls: list = []
        self.done_flag = True  # make yield-loop exit immediately
        self.output_stream = MagicMock()
        self.output_stream.get = AsyncMock(return_value=(None, None))
        self.input_stream = MagicMock()

    @property
    def done(self) -> bool:
        return self.done_flag

    async def resume(self, command) -> None:
        self.resume_calls.append(command)
        if self._resume_exc is not None:
            raise self._resume_exc


def _make_service() -> AgentService:
    return AgentService(
        uow_factory=_uow_factory,
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        search_engine=object(),
        file_storage=object(),
    )


def _patch_writer(monkeypatch, fake: _FakeWriter) -> None:
    """Force ``ApprovalStateWriter(uow_factory=...)`` inside the function to
    return ``fake``. The import lives inside ``_resume_tool_confirmation`` so
    patch the module attribute."""

    monkeypatch.setattr(
        "app.application.services.approval_state_writer.ApprovalStateWriter",
        lambda uow_factory: fake,
    )


class _StubConfirmationManager:
    """Minimal ConfirmationManager with controllable detail + CAS flags."""

    def __init__(
        self,
        detail: _StubDetail | None = None,
        *,
        cas_outcomes: list[bool] | None = None,
    ) -> None:
        self._detail = detail
        self._cas_outcomes = list(cas_outcomes) if cas_outcomes else []
        self.mark_processing_calls: list = []
        self.mark_processing_if_pending_calls: list = []
        self.mark_pending_calls: list = []
        self.cleanup_calls: list = []
        # Records the claim_nonce PE passes on the winning claim, so tests can
        # assert the PE-path rollback reuses the SAME nonce (end-to-end thread).
        self.last_claim_nonce: str | None = None

    async def read(self, session_id: str, tool_call_id: str):
        return self._detail

    async def mark_processing(self, session_id: str, tool_call_id: str) -> None:
        self.mark_processing_calls.append((session_id, tool_call_id))

    async def mark_processing_if_pending(
        self,
        session_id: str,
        tool_call_id: str,
        *,
        claim_nonce: str | None = None,
        processing_started_at=None,
    ) -> bool:
        # Signature mirrors the real ConfirmationQueue.mark_processing_if_pending
        # (keyword-only claim_nonce / processing_started_at, added by PE-0
        # ConfirmationQueue); default_engine.preflight passes both. Record the
        # nonce (don't otherwise act on it) so the PE-path rollback test can
        # assert the same nonce is threaded into the conditional rollback.
        self.mark_processing_if_pending_calls.append((session_id, tool_call_id))
        self.last_claim_nonce = claim_nonce
        if not self._cas_outcomes:
            return True  # 默认赢得 claim
        return self._cas_outcomes.pop(0)

    async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
        self.mark_pending_calls.append((session_id, tool_call_id))

    async def cleanup(self, session_id: str, tool_call_id: str) -> None:
        self.cleanup_calls.append((session_id, tool_call_id))


@dataclass
class _Confirmation:
    action: str
    scope: str
    tool_call_id: str = "tc-1"


async def _install_common_patches(
    service: AgentService,
    monkeypatch,
    *,
    detail: _StubDetail,
    task: _ResumeTask | None = None,
    cas_outcomes: list[bool] | None = None,
    session_owner_id: str = "u1",
) -> _StubConfirmationManager:
    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="s1", user_id=session_owner_id, status=SessionStatus.RUNNING)

    mgr = _StubConfirmationManager(detail=detail, cas_outcomes=cas_outcomes)
    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    service._confirmation_manager = mgr

    bound_task = task if task is not None else _ResumeTask()

    async def fake_get_task(_session):
        return bound_task

    async def fake_create_task(_session):
        return bound_task

    async def fake_safe_update(_sid: str) -> None:
        return None

    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_create_task", fake_create_task)
    monkeypatch.setattr(service, "_safe_update_unread_count", fake_safe_update)
    return mgr


async def _consume(gen) -> list:
    events: list = []
    async for ev in gen:
        events.append(ev)
    return events


# ---------------- Happy path: session scope ----------------


async def test_happy_path_session_scope_writer_then_resume(monkeypatch) -> None:
    """session scope: Writer.write → (id, True) → mark_processing → task.resume → cleanup。"""
    service = _make_service()
    task = _ResumeTask()
    mgr = await _install_common_patches(service, monkeypatch, detail=_StubDetail(), task=task)
    fake_writer = _FakeWriter(write_return=("dec-happy", True))
    _patch_writer(monkeypatch, fake_writer)

    conf = _Confirmation(action="approve", scope="session")
    await _consume(
        service._resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf
        )
    )

    assert len(fake_writer.write_calls) == 1
    decision = fake_writer.write_calls[0]
    assert decision.scope == "session"
    assert decision.effect == "approve"
    assert decision.source_type == "user_click"
    assert decision.confirmation_id == "tc-1"
    assert decision.expires_at is not None  # session scope 必须有 TTL
    assert decision.user_id == "u1"  # grant owner == detail.user_id
    assert len(task.resume_calls) == 1
    assert mgr.mark_processing_calls == [("s1", "tc-1")]
    assert mgr.cleanup_calls == [("s1", "tc-1")]
    assert fake_writer.delete_grant_calls == []


# ---------------- Happy path: always scope ----------------


async def test_happy_path_always_scope_writes_with_no_expiry(monkeypatch) -> None:
    """always scope: session_id=None + expires_at=None（R5a domain invariant）。"""
    service = _make_service()
    task = _ResumeTask()
    await _install_common_patches(service, monkeypatch, detail=_StubDetail(), task=task)
    fake_writer = _FakeWriter(write_return=("dec-always", True))
    _patch_writer(monkeypatch, fake_writer)

    conf = _Confirmation(action="approve", scope="always")
    await _consume(
        service._resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf
        )
    )

    decision = fake_writer.write_calls[0]
    assert decision.scope == "always"
    assert decision.session_id is None  # always scope domain invariant
    assert decision.expires_at is None  # always scope 永不过期


# ---------------- I1/I4 late-duplicate ----------------


async def test_late_duplicate_returns_409_without_resume(monkeypatch) -> None:
    """I1 losing-path / I4 late duplicate: writer 返 (id, False) → ConflictError (409)。"""
    service = _make_service()
    task = _ResumeTask()
    mgr = await _install_common_patches(service, monkeypatch, detail=_StubDetail(), task=task)
    fake_writer = _FakeWriter(write_return=("dec-existing", False))
    _patch_writer(monkeypatch, fake_writer)

    conf = _Confirmation(action="approve", scope="session")
    with pytest.raises(ConflictError) as exc_info:
        await _consume(
            service._resume_tool_confirmation(
                session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf
            )
        )
    assert exc_info.value.status_code == 409
    assert "tc-1" in exc_info.value.msg
    # 赢家已经执行过 → 本次不能再触发 resume / cleanup
    assert task.resume_calls == []
    assert mgr.mark_processing_calls == []
    assert mgr.cleanup_calls == []
    assert fake_writer.delete_grant_calls == []


# ---------------- I2 kickoff failure ----------------


async def test_kickoff_failure_rolls_back_grant_and_marks_pending(monkeypatch) -> None:
    """I2: task.resume 抛异常 → writer.delete_grant + mark_pending 都触发。

    Codex round-4: rollback 走独立 asyncio.Task（``_spawn_background_rollback``），
    测试 hook 成同步记录并 spawn，再 ``asyncio.sleep(0)`` 让 loop 跑完。
    """
    import asyncio as _aio

    service = _make_service()
    task = _ResumeTask(resume_exc=RuntimeError("task dead"))
    mgr = await _install_common_patches(service, monkeypatch, detail=_StubDetail(), task=task)
    fake_writer = _FakeWriter(write_return=("dec-kickoff", True))
    _patch_writer(monkeypatch, fake_writer)

    spawn_calls: list[dict] = []

    def _record_spawn(**kwargs):
        spawn_calls.append(kwargs)
        return _aio.create_task(service._rollback_resume_claim(**kwargs))

    monkeypatch.setattr(service, "_spawn_background_rollback", _record_spawn)

    conf = _Confirmation(action="approve", scope="session")
    events = await _consume(
        service._resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf
        )
    )
    # 让 spawned rollback task 完成
    await _aio.sleep(0)
    await _aio.sleep(0)

    assert len(fake_writer.write_calls) == 1
    assert spawn_calls and spawn_calls[0]["decision_id"] == "dec-kickoff"
    assert fake_writer.delete_grant_calls == ["dec-kickoff"]
    assert mgr.mark_pending_calls == [("s1", "tc-1")]
    assert mgr.cleanup_calls == []
    assert len(events) == 1  # outer handler yield ErrorEvent


# ---------------- I3 sequential retry ----------------


async def test_sequential_retry_second_call_returns_409(monkeypatch) -> None:
    """I3: 第一次 (True), 第二次 (False) → 第二次 409 且 resume 计数仍为 1。"""
    service = _make_service()
    task = _ResumeTask()
    await _install_common_patches(service, monkeypatch, detail=_StubDetail(), task=task)

    class _ToggleWriter(_FakeWriter):
        def __init__(self) -> None:
            super().__init__()
            self._n = 0

        async def write(self, decision):
            self.write_calls.append(decision)
            self._n += 1
            return ("dec-seq", self._n == 1)

    toggle = _ToggleWriter()
    monkeypatch.setattr(
        "app.application.services.approval_state_writer.ApprovalStateWriter",
        lambda uow_factory: toggle,
    )

    conf = _Confirmation(action="approve", scope="session")
    await _consume(
        service._resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf
        )
    )
    with pytest.raises(ConflictError):
        await _consume(
            service._resume_tool_confirmation(
                session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf
            )
        )
    assert len(task.resume_calls) == 1  # tool 只执行一次（I3）
    assert len(toggle.write_calls) == 2


# ---------------- ValueError claim collision ----------------


async def test_claim_collision_raises_bad_request(monkeypatch) -> None:
    """不同 (scope, effect) 同 confirmation_id → Writer raise ValueError → BadRequestError。"""
    service = _make_service()
    task = _ResumeTask()
    await _install_common_patches(service, monkeypatch, detail=_StubDetail(), task=task)
    fake_writer = _FakeWriter(write_exc=ValueError("claim collision: scope=always vs session"))
    _patch_writer(monkeypatch, fake_writer)

    conf = _Confirmation(action="approve", scope="session")
    with pytest.raises(BadRequestError) as exc_info:
        await _consume(
            service._resume_tool_confirmation(
                session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf
            )
        )
    assert exc_info.value.status_code == 400
    assert task.resume_calls == []


# ---------------- scope="once": CAS single-flight + audit ----------------
#
# Codex CRITICAL 锁死：once 不准无脑 bypass——并发 /resume 必须通过
# ``mark_processing_if_pending`` Lua CAS 去重，且必须保留 audit 证据。


class _RecordingUoW:
    """监控 tool_approval_log.create 调用；once audit 覆盖必备。"""

    def __init__(self) -> None:
        self.tool_approval_log = MagicMock()
        self.tool_approval_log.create = AsyncMock(return_value=None)
        self.session = MagicMock()
        self.session.update_unread_message_count = AsyncMock(return_value=None)
        self.session.add_event = AsyncMock(return_value=None)

    async def __aenter__(self) -> "_RecordingUoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


def _recording_uow_factory(recorder: _RecordingUoW):
    def _factory() -> _RecordingUoW:
        return recorder

    return _factory


def _make_service_with_uow(recorder: _RecordingUoW) -> AgentService:
    return AgentService(
        uow_factory=_recording_uow_factory(recorder),
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        search_engine=object(),
        file_storage=object(),
    )


class _OnceWriterGuard:
    """2026-04-21 CS4 合同：once scope must NOT call Writer.write/delete_grant，
    but MUST call ``write_audit_only`` (single-writer audit 入口)。

    替代旧版 ``_BoomWriter`` —— 旧版 boom-on-any-call 是在 once audit 还直写
    ``tool_approval_log.create`` 时的守卫；合同收口后 once audit 必须经
    ``write_audit_only``，所以此 guard 允许并记录 audit 调用、同时仍对
    grant 写入强回归守卫。
    """

    def __init__(self) -> None:
        self.write_audit_only_calls: list[dict] = []

    async def write(self, decision):
        raise AssertionError("scope='once' must NOT call Writer.write")

    async def delete_grant(self, _id):
        raise AssertionError("scope='once' must NOT call delete_grant")

    async def write_audit_only(self, **kwargs) -> None:
        self.write_audit_only_calls.append(kwargs)


async def test_once_scope_winning_resume_writes_audit(monkeypatch) -> None:
    """once + CAS 赢家：task.resume 执行 + audit log 走 write_audit_only
    （owner = detail.user_id）。"""
    recorder = _RecordingUoW()
    service = _make_service_with_uow(recorder)
    task = _ResumeTask()
    mgr = await _install_common_patches(
        service, monkeypatch, detail=_StubDetail(), task=task,
        cas_outcomes=[True],
    )

    once_guard = _OnceWriterGuard()
    monkeypatch.setattr(
        "app.application.services.approval_state_writer.ApprovalStateWriter",
        lambda uow_factory: once_guard,
    )

    conf = _Confirmation(action="deny", scope="once")  # deny once 也必须留 audit
    await _consume(
        service._resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf
        )
    )

    # CAS 被调、resume 被触发、cleanup 执行
    assert mgr.mark_processing_if_pending_calls == [("s1", "tc-1")]
    assert mgr.mark_processing_calls == []  # once 路径不再调老 mark_processing
    assert len(task.resume_calls) == 1
    assert mgr.cleanup_calls == [("s1", "tc-1")]
    # audit 必须经 writer.write_audit_only 写入（once 也留证据——Codex CRITICAL 回归）
    # CS4 合同：不再直写 tool_approval_log.create（recorder 的 create mock 不被调）
    assert len(once_guard.write_audit_only_calls) == 1
    kwargs = once_guard.write_audit_only_calls[0]
    assert kwargs["user_id"] == "u1"  # owner = detail.user_id
    assert kwargs["tool_name"] == "shell_execute"
    assert kwargs["action"] == "deny"
    assert kwargs["scope"] == "once"
    assert kwargs["approved_by"] == "user"
    # 合同面：agent_service 不再绕开 writer 直写底层 repo
    recorder.tool_approval_log.create.assert_not_awaited()


async def test_once_scope_losing_concurrent_claim_returns_409(monkeypatch) -> None:
    """once + CAS 失败者（并发 /resume）：返 ConflictError 409；task.resume 不调；audit 不写。"""
    recorder = _RecordingUoW()
    service = _make_service_with_uow(recorder)
    task = _ResumeTask()
    mgr = await _install_common_patches(
        service, monkeypatch, detail=_StubDetail(), task=task,
        cas_outcomes=[False],  # 模拟对手赢了 CAS
    )

    once_guard = _OnceWriterGuard()
    monkeypatch.setattr(
        "app.application.services.approval_state_writer.ApprovalStateWriter",
        lambda uow_factory: once_guard,
    )

    conf = _Confirmation(action="approve", scope="once")
    with pytest.raises(ConflictError) as exc_info:
        await _consume(
            service._resume_tool_confirmation(
                session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf
            )
        )
    assert exc_info.value.status_code == 409
    # 败者不得触发 resume / cleanup / audit（tool 不执行、无证据）
    assert task.resume_calls == []
    assert mgr.cleanup_calls == []
    # CS4 合同：audit 必须经 writer；败者连 writer 都不调，底层 repo 也必然未被调
    assert once_guard.write_audit_only_calls == []
    recorder.tool_approval_log.create.assert_not_awaited()


# ---------------- HIGH: admin 代理审批时 grant owner 必须是 session owner ----------------


async def test_admin_impersonation_grant_uses_detail_user_id(monkeypatch) -> None:
    """Codex HIGH 回归：admin 用自己账号调 /resume 处理他人 session 的确认时，
    Writer 构造的 ApprovalDecision.user_id 必须是 detail.user_id（= session owner），
    不是 HTTP 请求者（admin）；否则 grant 被写到 admin 账号下污染数据。"""
    service = _make_service()
    # ConfirmationDetail.user_id 是真实会话 owner "alice"
    detail = _StubDetail(user_id="alice", session_id="s1")
    task = _ResumeTask()
    await _install_common_patches(
        service, monkeypatch, detail=detail, task=task,
        session_owner_id="alice",  # session.user_id 也是 alice
    )
    fake_writer = _FakeWriter(write_return=("dec-admin", True))
    _patch_writer(monkeypatch, fake_writer)

    # admin "bob" 作 HTTP 请求者：is_admin=True 让 _get_accessible_session 放行
    conf = _Confirmation(action="approve", scope="always")
    await _consume(
        service._resume_tool_confirmation(
            session_id="s1",
            user_id="bob-admin",  # 请求者 != owner
            is_admin=True,
            tool_confirmation=conf,
        )
    )

    assert len(fake_writer.write_calls) == 1
    decision = fake_writer.write_calls[0]
    assert decision.user_id == "alice", (
        f"grant owner 应 = detail.user_id='alice'，实际={decision.user_id}；"
        "admin 代理导致的 owner 污染没有被拦住"
    )


# ---------------- Codex round-2 MEDIUM: status=processing 归一 ConflictError ----------------


async def test_preflight_raises_conflict_on_processing_status(monkeypatch) -> None:
    """detail.status == 'processing'（对手已赢 CAS）→ ConflictError(409)，
    **不**再按时序分叉成 BadRequestError(400)。"""
    service = _make_service()
    detail = _StubDetail(status="processing")  # 已被他请求推进状态
    task = _ResumeTask()
    await _install_common_patches(service, monkeypatch, detail=detail, task=task)
    # Writer 不应被调
    fake_writer = _FakeWriter()
    _patch_writer(monkeypatch, fake_writer)

    conf = _Confirmation(action="approve", scope="session")
    with pytest.raises(ConflictError) as exc_info:
        await service.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf,
        )
    assert exc_info.value.status_code == 409
    assert "tc-1" in exc_info.value.msg
    assert fake_writer.write_calls == []  # 未进入 claim 路径
    assert task.resume_calls == []


async def test_preflight_detail_missing_and_no_grant_returns_404(monkeypatch) -> None:
    """真正不存在 / 已过期：detail=None 且 grants 表查不到 confirmation_id →
    NotFoundError(404)。与 late-duplicate（grant 存在）的 409 严格区分。"""
    service = _make_service()
    task = _ResumeTask()
    await _install_common_patches(service, monkeypatch, detail=None, task=task)

    from app.application.errors.exceptions import NotFoundError as _NFE
    conf = _Confirmation(action="approve", scope="session")
    with pytest.raises(_NFE) as exc_info:
        await service.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf,
        )
    assert exc_info.value.status_code == 404


# ---------------- Codex round-6 HIGH: I4 late-duplicate 合同 ----------------


async def test_preflight_late_duplicate_after_cleanup_returns_409(monkeypatch) -> None:
    """Codex round-6 HIGH: winner 完成并 cleanup confirmation_detail 后，再次
    /resume 同 confirmation_id → 查 grants 表命中 → ConflictError(409)，
    而**不是**把 winner 已处理完的 confirmation 误报为 "不存在/已过期" 404。

    前端凭 409 重连 /events?since=<last_event_id> 复播已完成的 tool event。"""
    import asyncio as _aio

    class _GrantFoundUoW:
        """UoW stub：find_by_confirmation_id 返非空 grant（模拟 winner 已持久）。"""

        def __init__(self) -> None:
            self.approval_grants = MagicMock()
            existing = MagicMock()
            existing.decision_id = "dec-winner-historic"
            existing.confirmation_id = "tc-1"
            self.approval_grants.find_by_confirmation_id = AsyncMock(
                return_value=existing
            )
            self.session = MagicMock()
            self.session.update_unread_message_count = AsyncMock(return_value=None)
            self.session.add_event = AsyncMock(return_value=None)
            self.tool_approval_log = MagicMock()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

    def _uow_factory():
        return _GrantFoundUoW()

    service = AgentService(
        uow_factory=_uow_factory,
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        search_engine=object(),
        file_storage=object(),
    )

    task = _ResumeTask()
    # detail=None 模拟 winner 已 cleanup
    await _install_common_patches(service, monkeypatch, detail=None, task=task)

    conf = _Confirmation(action="approve", scope="session")
    with pytest.raises(ConflictError) as exc_info:
        await service.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf,
        )
    assert exc_info.value.status_code == 409
    assert "tc-1" in exc_info.value.msg
    # msg 应包含引导客户端走 /events reconnect 的提示
    assert "/events" in exc_info.value.msg or "重连" in exc_info.value.msg


async def test_preflight_once_no_detail_no_grant_still_404(monkeypatch) -> None:
    """once scope 不持久 grant：winner cleanup 后再 /resume 仍 404（once 本就没
    late-duplicate 证据；和 persistent 的 409 严格区分）。"""
    service = _make_service()
    task = _ResumeTask()
    # detail=None + default _NoopApprovalGrantsRepo 返 None
    await _install_common_patches(service, monkeypatch, detail=None, task=task)

    from app.application.errors.exceptions import NotFoundError as _NFE
    conf = _Confirmation(action="approve", scope="once")
    with pytest.raises(_NFE) as exc_info:
        await service.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf,
        )
    assert exc_info.value.status_code == 404


async def test_preflight_late_duplicate_lookup_failure_returns_503(monkeypatch) -> None:
    """Codex round-7 MEDIUM: detail=None 时 find_by_confirmation_id 抛异常
    （DB 瞬时故障）→ ServiceUnavailableError(503)，**不**伪装成 404。

    真实 winner-cleanup duplicate 但基础设施故障，客户端必须知道是可重试的
    暂态错误（不是 confirmation 永久丢失），才能走标准的 503 retry-after 策略。"""
    from app.application.errors.exceptions import ServiceUnavailableError

    class _LookupFailingUoW:
        """UoW stub：find_by_confirmation_id 必抛异常（模拟 DB 瞬时故障）。"""

        def __init__(self) -> None:
            self.approval_grants = MagicMock()
            self.approval_grants.find_by_confirmation_id = AsyncMock(
                side_effect=RuntimeError("DB pool exhausted")
            )
            self.session = MagicMock()
            self.session.update_unread_message_count = AsyncMock(return_value=None)
            self.session.add_event = AsyncMock(return_value=None)
            self.tool_approval_log = MagicMock()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

    def _uow_factory():
        return _LookupFailingUoW()

    service = AgentService(
        uow_factory=_uow_factory,
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        search_engine=object(),
        file_storage=object(),
    )

    task = _ResumeTask()
    await _install_common_patches(service, monkeypatch, detail=None, task=task)

    conf = _Confirmation(action="approve", scope="session")
    with pytest.raises(ServiceUnavailableError) as exc_info:
        await service.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf,
        )
    assert exc_info.value.status_code == 503
    assert "tc-1" in exc_info.value.msg


# ---------------- Codex round-2 HIGH: preflight 同步抛，不经 SSE iterate ----------------


async def test_preflight_is_plain_coroutine_not_async_generator(monkeypatch) -> None:
    """证明 ``preflight_resume_tool_confirmation`` 返 coroutine（不是 async
    generator），异常在 await 点同步抛出——这就是 session_routes 在
    EventSourceResponse 创建**之前** catch 异常并映射到 HTTP 的关键。
    """
    import inspect

    service = _make_service()
    # async def method → inspect.iscoroutinefunction 返 True
    # async def + yield（async generator）→ inspect.isasyncgenfunction 返 True
    assert inspect.iscoroutinefunction(service.preflight_resume_tool_confirmation)
    assert not inspect.isasyncgenfunction(service.preflight_resume_tool_confirmation)

    # 相反，drive_resume_tool_confirmation 是 async generator（进入 SSE 后产事件）
    assert inspect.isasyncgenfunction(service.drive_resume_tool_confirmation)


async def test_preflight_conflict_raises_before_any_drive_iteration(monkeypatch) -> None:
    """preflight 抛 ConflictError 时 drive 从未被触发——模拟 session_routes 的
    调用顺序：``await preflight(...)`` 先抛出 → HTTP handler 返 409 → 根本不会
    创建 ``EventSourceResponse``。
    """
    service = _make_service()
    task = _ResumeTask()
    await _install_common_patches(service, monkeypatch, detail=_StubDetail(), task=task)
    fake_writer = _FakeWriter(write_return=("dec-x", False))  # losing claim
    _patch_writer(monkeypatch, fake_writer)

    conf = _Confirmation(action="approve", scope="session")
    with pytest.raises(ConflictError):
        # 模拟生产路径：session_routes 的 chat endpoint 在创建 EventSourceResponse 前 await
        await service.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf,
        )
    # drive 永不被 session_routes 调用——task.resume 必须是 0
    assert task.resume_calls == []
    assert len(fake_writer.write_calls) == 1  # preflight 确实问了 Writer


# ---------------- Codex round-3 CRITICAL: drive CancelledError 必须回滚 claim ----------------


async def test_drive_cancelled_error_before_task_resume_rolls_back_claim(monkeypatch) -> None:
    """SSE 客户端断连 → drive 里 task.resume 抛 ``asyncio.CancelledError`` →
    回滚必须走 **独立 asyncio.Task** (``_spawn_background_rollback``)，
    不能在被取消的父 task 里直接 await——否则 delete_grant/mark_pending 会
    立刻再被取消，orphan claim 永久卡死 processing（Codex round-4 CRITICAL）。
    """
    import asyncio

    service = _make_service()
    task = _ResumeTask(resume_exc=asyncio.CancelledError())
    mgr = await _install_common_patches(
        service, monkeypatch, detail=_StubDetail(), task=task,
    )
    fake_writer = _FakeWriter(write_return=("dec-cancel", True))
    _patch_writer(monkeypatch, fake_writer)

    # 监视 _spawn_background_rollback：不真启新 task（测试环境难控），直接
    # 同步 await 原始 _rollback_resume_claim 确保回滚逻辑被触达。
    spawn_calls: list[dict] = []

    def _record_spawn(**kwargs):
        spawn_calls.append(kwargs)
        # 模拟 spawn 后立即执行（测试场景）——生产是独立 task
        import asyncio as _aio
        return _aio.create_task(
            service._rollback_resume_claim(**kwargs)
        )

    monkeypatch.setattr(service, "_spawn_background_rollback", _record_spawn)

    conf = _Confirmation(action="approve", scope="session")
    state = await service.preflight_resume_tool_confirmation(
        session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf,
    )
    assert fake_writer.write_calls

    gen = service.drive_resume_tool_confirmation(state)
    with pytest.raises(asyncio.CancelledError):
        async for _ in gen:
            pass

    # 让 spawned rollback task 跑完（独立 task 不会被父 cancel 影响，但事件循环
    # 需要一次 turn 让它调度）
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Codex round-4 锁死：rollback 必须通过 spawn 机制调度，不是同 task 里 await
    assert spawn_calls, (
        "CancelledError 路径没有调用 _spawn_background_rollback——rollback 会和"
        "父 task 一起被取消，orphan claim 永久卡死"
    )
    assert spawn_calls[0]["decision_id"] == "dec-cancel"
    # 独立 task 跑完后 grant 实际被删
    assert fake_writer.delete_grant_calls == ["dec-cancel"]
    assert mgr.mark_pending_calls == [("s1", "tc-1")]


async def test_claim_envelope_post_cancel_spawns_rollback(monkeypatch) -> None:
    """Codex round-4 CRITICAL：``_claim_with_post_cancel_rollback`` envelope
    保证 writer.write 成功但父 task 被 cancel 时，done_callback 仍 spawn rollback。

    场景：claim task 赢了 UNIQUE（newly_created=True），但主 preflight 在
    ``await asyncio.shield(...)`` 被 cancel。shield 传 CancelledError 给主路径，
    但 claim task 继续跑完；done_callback 检测到 newly=True → 调 spawn rollback。
    """
    import asyncio

    service = _make_service()

    spawn_calls: list[dict] = []

    def _record_spawn(**kwargs):
        spawn_calls.append(kwargs)
        return None

    monkeypatch.setattr(service, "_spawn_background_rollback", _record_spawn)

    async def _slow_claim() -> tuple[Optional[str], bool]:
        # 模拟 writer.write 慢一点但会完成
        await asyncio.sleep(0.01)
        return ("dec-post-cancel", True)

    # 启动 envelope
    envelope_task = asyncio.create_task(
        service._claim_with_post_cancel_rollback(
            _slow_claim,
            session_id="s1",
            tool_call_id="tc-envelope",
            persistent_scope=True,
        )
    )
    # 让它进 shield
    await asyncio.sleep(0)
    # 外部 cancel（模拟父 task 被 sse_starlette 取消）
    envelope_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await envelope_task

    # 等 inner claim task + done_callback 跑完
    for _ in range(10):
        await asyncio.sleep(0.01)
        if spawn_calls:
            break

    assert spawn_calls, (
        "claim 成功后父 task 被 cancel，done_callback 未 spawn rollback——"
        "grant 已写入但无清理路径，orphan 卡死"
    )
    assert spawn_calls[0]["decision_id"] == "dec-post-cancel"
    assert spawn_calls[0]["persistent_scope"] is True


async def test_once_audit_cancelled_spawns_rollback(monkeypatch) -> None:
    """Codex round-5 CRITICAL: scope='once' 的 audit 写入被 CancelledError 打断时，
    preflight 必须 spawn 后台 rollback 把 CAS 推到 processing 的 confirmation
    回滚到 pending——否则 drive 未启动但状态卡死 processing，sweeper 不管，
    后续 /resume 持续 409。
    """
    import asyncio as _aio

    # 独立 AgentService + 能让 audit 写入抛 CancelledError 的 UoW
    class _CancellingAuditUoW:
        def __init__(self) -> None:
            self.tool_approval_log = MagicMock()
            # audit 写入阶段抛 CancelledError（模拟父 task 被 SSE 断连取消）
            self.tool_approval_log.create = AsyncMock(
                side_effect=_aio.CancelledError()
            )
            self.session = MagicMock()
            self.session.update_unread_message_count = AsyncMock(return_value=None)
            self.session.add_event = AsyncMock(return_value=None)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

    uow_instance = _CancellingAuditUoW()

    def _uow_factory():
        return uow_instance

    service = AgentService(
        uow_factory=_uow_factory,
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        search_engine=object(),
        file_storage=object(),
    )

    detail = _StubDetail()
    task = _ResumeTask()
    mgr = await _install_common_patches(
        service, monkeypatch, detail=detail, task=task,
        cas_outcomes=[True],  # once CAS 赢得 claim
    )

    spawn_calls: list[dict] = []

    def _record_spawn(**kwargs):
        spawn_calls.append(kwargs)
        return _aio.create_task(service._rollback_resume_claim(**kwargs))

    monkeypatch.setattr(service, "_spawn_background_rollback", _record_spawn)

    # scope="once" 走 CAS 分支；audit 抛 CancelledError 应走 rollback + re-raise
    conf = _Confirmation(action="deny", scope="once")
    with pytest.raises(_aio.CancelledError):
        await service.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf,
        )

    # 让 spawned rollback task 完成
    await _aio.sleep(0)
    await _aio.sleep(0)

    # Codex round-5 锁死：audit CancelledError 必须触发 spawn rollback
    assert spawn_calls, (
        "once audit 的 CancelledError 未触发 _spawn_background_rollback——"
        "confirmation 卡死 processing，sweeper 跳过，前端持续 409"
    )
    assert spawn_calls[0]["persistent_scope"] is False
    assert spawn_calls[0]["decision_id"] is None
    assert spawn_calls[0]["session_id"] == "s1"
    assert spawn_calls[0]["tool_call_id"] == "tc-1"
    # spawned rollback 实际调 mark_pending
    assert mgr.mark_pending_calls == [("s1", "tc-1")]


async def test_claim_envelope_loser_does_not_spawn_rollback(monkeypatch) -> None:
    """loser 分支（newly_created=False）在 cancel 路径下**不**触发 rollback——
    没自己写的 claim 不应该被误删。"""
    import asyncio

    service = _make_service()
    spawn_calls: list = []
    monkeypatch.setattr(
        service,
        "_spawn_background_rollback",
        lambda **kwargs: spawn_calls.append(kwargs),
    )

    async def _loser_claim() -> tuple[Optional[str], bool]:
        await asyncio.sleep(0.01)
        return ("dec-winner", False)  # 对手赢了

    envelope_task = asyncio.create_task(
        service._claim_with_post_cancel_rollback(
            _loser_claim,
            session_id="s1",
            tool_call_id="tc-loser",
            persistent_scope=True,
        )
    )
    await asyncio.sleep(0)
    envelope_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await envelope_task

    for _ in range(10):
        await asyncio.sleep(0.01)

    # loser 路径没写 claim → 不 spawn rollback
    assert spawn_calls == []


async def test_preflight_task_creation_failure_raises_service_unavailable(monkeypatch) -> None:
    """task 创建失败必须抛 ``ServiceUnavailableError`` (503)；**PE 路径**下回滚走
    ``_spawn_background_rollback_if_present`` —— 一个独立 asyncio task（父 task 被
    cancel 时回滚不被连带取消），用 PE 在 claim 时写入的**同一个 claim_nonce** 做
    *条件* 回滚（nonce 不匹配 / 已被 commit_resume 清理 → 跳过，避免 resurrect 脏
    Redis hash）。PE 回滚的是 *claim*（Redis processing→pending），``decision_id=None``、
    ``persistent_scope=False``，不删持久 grant（区别于 legacy 路径的
    ``_spawn_background_rollback`` + grant 删除）。

    (A4-2 follow-up: 本测试原为 legacy preflight 写，代码已默认走 PE 路径
    [agent_service.py:1686]，故迁移到 PE 回滚合同。)"""
    from app.application.errors.exceptions import ServiceUnavailableError

    service = _make_service()
    detail = _StubDetail()

    async def fake_get_accessible_session(*args, **kwargs):
        return Session(id="s1", user_id="u1", status=SessionStatus.RUNNING)

    mgr = _StubConfirmationManager(detail=detail)
    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    service._confirmation_manager = mgr

    async def fake_get_task(_session):
        return None

    async def fake_create_task(_session):
        return None

    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_create_task", fake_create_task)

    # PE preflight writes the grant during approve-evaluate before the claim;
    # the fake writer lets preflight reach the claim + task-creation step.
    fake_writer = _FakeWriter(write_return=("dec-no-task", True))
    _patch_writer(monkeypatch, fake_writer)

    # PE path rolls the CLAIM back via the independent _if_present task. Record
    # its kwargs (don't run the real conditional rollback — the nonce-matching /
    # cleanup-skip logic of _rollback_resume_claim_if_present is covered by its
    # own tests; here we lock the spawn *contract* on task-creation failure).
    rollback_calls: list[dict] = []

    def _record_spawn_if_present(**kwargs):
        rollback_calls.append(kwargs)
        return None

    monkeypatch.setattr(
        service, "_spawn_background_rollback_if_present", _record_spawn_if_present
    )

    conf = _Confirmation(action="approve", scope="session")
    with pytest.raises(ServiceUnavailableError) as exc_info:
        await service.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False, tool_confirmation=conf,
        )
    assert exc_info.value.status_code == 503

    # PE rollback contract on task-creation failure: exactly one independent
    # conditional-rollback spawned, rolling back the CLAIM — decision_id=None,
    # non-persistent — with the SAME claim_nonce PE wrote into the queue, so the
    # conditional rollback can nonce-match the still-pending entry.
    assert len(rollback_calls) == 1, "task 创建失败路径未 spawn 条件回滚任务"
    rb = rollback_calls[0]
    assert rb["decision_id"] is None
    assert rb["persistent_scope"] is False
    assert rb["session_id"] == "s1"
    assert rb["tool_call_id"] == "tc-1"
    assert mgr.last_claim_nonce is not None, "PE 未通过 claim 路径写 claim_nonce"
    assert rb["claim_nonce"] == mgr.last_claim_nonce  # 同一 nonce 端到端线程化
