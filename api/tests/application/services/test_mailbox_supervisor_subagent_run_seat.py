"""C4.1a PR-3 — ResultReadyHandler subagent_run record PROLOGUE 测试（spec §5.1 + §7 PR-3）。

第三独立 prologue：coordinator_step child + repo 注入 → project_local_result →
repo.record；best-effort（失败 swallow，destroy 照常）；隔离 cost_rollup /
envelope_store=None 以证明「本 seat 零 get_by_id」（INV-C4.1-1，R4#P3）。

Mock 策略同 test_mailbox_supervisor_persist_terminal.py：直接 handle + side_effect()。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.mailbox_supervisor import (
    ResultReadyHandler,
    SupervisorContext,
)
from app.domain.models.mailbox_envelope import (
    CostAggregate,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.models.needs_authorization_details import NeedsAuthorizationDetails
from app.domain.models.session import DestroyReason, SandboxBinding, Session
from app.domain.models.subagent_worker import (
    WorkerRuntimeType,
    WorkerTerminalOutcome,
)


pytestmark = pytest.mark.anyio


def _payload_success() -> dict[str, Any]:
    return ResultReadyPayload(
        summary="done",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(
            total_input_tokens=100, total_output_tokens=50,
            total_usd=0.001, tool_call_count=2,
        ),
    ).model_dump(mode="json")


def _payload_needs_auth() -> dict[str, Any]:
    return ResultReadyPayload(
        summary="s",
        outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
        needs_authorization_details=NeedsAuthorizationDetails(
            reason="out_of_tool_allowlist", requested_tool="shell_execute",
        ),
    ).model_dump(mode="json")


def _make_envelope(*, payload: Optional[dict[str, Any]] = None) -> MailboxEnvelope:
    return MailboxEnvelope(
        envelope_id="01HSPYU0CR0000000000000001",
        type=MailboxEnvelopeType.RESULT_READY,
        parent_session_id="root-1",
        child_session_id="child-1",
        correlation_id="01HSPYU0CR0000000000000002",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=payload if payload is not None else _payload_success(),
        reclaim_count=0,
    )


def _make_session(
    *,
    parent_session_id: Optional[str] = "root-1",
    tool_filter_preset: Optional[str] = "coordinator_step",
    coordinator_run_id: Optional[str] = "run-1",
    work_unit_id: Optional[str] = "wu-1",
) -> Session:
    return Session(
        id="child-1",
        parent_session_id=parent_session_id,
        worker_type="subagent" if parent_session_id is not None else "root",
        tool_filter_preset=tool_filter_preset,  # type: ignore[arg-type]
        sandbox_binding=SandboxBinding(),
        coordinator_run_id=coordinator_run_id,
        work_unit_id=work_unit_id,
    )


class _AuditRepoStub:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def get_processed(self, parent_session_id: str, envelope_id: str) -> bool:
        row = self.rows.get((parent_session_id, envelope_id))
        return row is not None and row.get("processed_at") is not None

    async def upsert_processing(
        self, envelope: MailboxEnvelope, *, processing_at: datetime
    ) -> None:
        self.rows.setdefault(
            (envelope.parent_session_id, envelope.envelope_id), {}
        )["processing_at"] = processing_at

    async def mark_processed(
        self, parent_session_id: str, envelope_id: str, *, processed_at: datetime
    ) -> None:
        self.rows.setdefault((parent_session_id, envelope_id), {})[
            "processed_at"
        ] = processed_at


def _build_ctx(
    *,
    subagent_run_repo: Optional[AsyncMock] = None,
    session_repo: Optional[MagicMock] = None,
) -> SupervisorContext:
    """本 seat 隔离：cost_rollup_service / coordinator_envelope_store 恒 None，
    故任何 get_by_id 只可能来自本 seat 的 prologue（INV-C4.1-1，R4#P3）。"""
    sandbox_lifecycle = AsyncMock()
    sandbox_lifecycle.destroy = AsyncMock(return_value=None)
    telemetry = AsyncMock()
    telemetry.emit = AsyncMock(return_value=None)
    return SupervisorContext(
        root_session_id="root-1",
        pod_id="pod-a",
        instance_id="i1",
        redis=MagicMock(),
        audit_repo=_AuditRepoStub(),
        publisher=MagicMock(),
        sandbox_lifecycle=sandbox_lifecycle,
        agent_service_callback=AsyncMock(return_value=None),
        telemetry=telemetry,
        session_repo=session_repo,
        cost_rollup_service=None,
        coordinator_envelope_store=None,
        subagent_run_repo=subagent_run_repo,
    )


class TestResultReadySubagentRunSeat:
    async def test_coordinator_child_records_projected_run(self) -> None:
        repo = AsyncMock()
        repo.record = AsyncMock(return_value=None)
        session_repo = MagicMock()
        session_repo.get_by_id = AsyncMock(return_value=_make_session())
        ctx = _build_ctx(subagent_run_repo=repo, session_repo=session_repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        repo.record.assert_awaited_once()
        run = repo.record.await_args.args[0]
        assert run.worker_runtime_type == WorkerRuntimeType.LOCAL
        assert run.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert run.parent_session_id == "root-1"
        assert run.child_session_id == "child-1"
        assert run.source_ref == "wu-1"
        assert run.cost_authoritative is True
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_flag_off_repo_none_no_record_no_get_by_id(self) -> None:
        # INV-C4.1-1：repo=None（flag OFF）→ 本 seat 零 record + 零 get_by_id
        # （cost_rollup/envelope_store 已 None，隔离本 seat）。destroy 照常。
        session_repo = MagicMock()
        session_repo.get_by_id = AsyncMock(return_value=_make_session())
        ctx = _build_ctx(subagent_run_repo=None, session_repo=session_repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        session_repo.get_by_id.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_record_failure_does_not_block_destroy(self) -> None:
        # INV-C4.1-2：record raise → swallow；destroy + mark_processed 照常。
        repo = AsyncMock()
        repo.record = AsyncMock(side_effect=RuntimeError("db down"))
        session_repo = MagicMock()
        session_repo.get_by_id = AsyncMock(return_value=_make_session())
        ctx = _build_ctx(subagent_run_repo=repo, session_repo=session_repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()  # 不得抛

        repo.record.assert_awaited_once()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()
        assert await ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    async def test_non_coordinator_child_not_recorded(self) -> None:
        # research child（preset=subagent_research）→ 本 seat 跳过（互斥，§5.0.1）。
        repo = AsyncMock()
        repo.record = AsyncMock(return_value=None)
        session_repo = MagicMock()
        session_repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="subagent_research",
                coordinator_run_id=None,
                work_unit_id=None,
            )
        )
        ctx = _build_ctx(subagent_run_repo=repo, session_repo=session_repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        repo.record.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_parent_session_id_none_skipped(self) -> None:
        repo = AsyncMock()
        repo.record = AsyncMock(return_value=None)
        session_repo = MagicMock()
        session_repo.get_by_id = AsyncMock(
            return_value=_make_session(parent_session_id=None)
        )
        ctx = _build_ctx(subagent_run_repo=repo, session_repo=session_repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        repo.record.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_malformed_payload_swallowed(self) -> None:
        # 非-dict payload → {} → ResultReadyPayload.model_validate({}) raise → swallow。
        repo = AsyncMock()
        repo.record = AsyncMock(return_value=None)
        session_repo = MagicMock()
        session_repo.get_by_id = AsyncMock(return_value=_make_session())
        ctx = _build_ctx(subagent_run_repo=repo, session_repo=session_repo)

        env = MailboxEnvelope.model_construct(
            envelope_id="01HSPYU0CR0000000000000099",
            type=MailboxEnvelopeType.RESULT_READY,
            parent_session_id="root-1",
            child_session_id="child-1",
            correlation_id="01HSPYU0CR0000000000000098",
            emitted_at=datetime.now(tz=timezone.utc),
            producer_role=ProducerRole.CHILD_AGENT,
            payload=None,  # 非 dict → prologue 内 {} → model_validate raise → swallow
            reclaim_count=0,
        )
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()  # 不得抛

        repo.record.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_malformed_dict_payload_swallowed(self) -> None:
        # 畸形 dict（present 但缺 required summary）→ prologue 内 ResultReadyPayload
        # .model_validate raise → swallow（与 non-dict→{} 是两条不同路径，spec §7
        # PR-3 均要求）。**必须用 model_construct**：MailboxEnvelope 的 after-validator
        # `_validate_payload_matches_type` 会按 type 对 payload 跑 ResultReadyPayload
        # .model_validate，普通构造会在 setup 就 raise（校验发生在 envelope 层，不在
        # prologue）——model_construct 绕过 envelope 校验，让畸形 dict 抵达 prologue。
        repo = AsyncMock()
        repo.record = AsyncMock(return_value=None)
        session_repo = MagicMock()
        session_repo.get_by_id = AsyncMock(return_value=_make_session())
        ctx = _build_ctx(subagent_run_repo=repo, session_repo=session_repo)

        env = MailboxEnvelope.model_construct(
            envelope_id="01HSPYU0CR0000000000000097",
            type=MailboxEnvelopeType.RESULT_READY,
            parent_session_id="root-1",
            child_session_id="child-1",
            correlation_id="01HSPYU0CR0000000000000096",
            emitted_at=datetime.now(tz=timezone.utc),
            producer_role=ProducerRole.CHILD_AGENT,
            payload={"outcome": "success"},  # dict 但缺 required summary → prologue model_validate raise
            reclaim_count=0,
        )
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()  # 不得抛

        repo.record.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_needs_authorization_recorded_with_error_summary(self) -> None:
        repo = AsyncMock()
        repo.record = AsyncMock(return_value=None)
        session_repo = MagicMock()
        session_repo.get_by_id = AsyncMock(return_value=_make_session())
        ctx = _build_ctx(subagent_run_repo=repo, session_repo=session_repo)

        env = _make_envelope(payload=_payload_needs_auth())
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        repo.record.assert_awaited_once()
        run = repo.record.await_args.args[0]
        assert run.terminal_outcome == WorkerTerminalOutcome.NEEDS_AUTHORIZATION
        assert run.error_summary == (
            "needs_authorization: out_of_tool_allowlist (tool=shell_execute)"
        )
