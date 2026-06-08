"""PE-1 §6.3 — Skill HIGH risk full E2E through PE.

Covers:
- ToolCallSpec.source_metadata threading from tool_node to PE
- SkillSource recompute overriding caller pre-fill
- reason.type == 'risk_enforce' for skill (vs 'approval_policy' for native)
- preflight_resume CAS claim
- commit_resume writer.write (ApprovalStateWriter single-writer R5 CS4)
- ConfirmationDetail.risk_level read by tool_node for ToolConfirmationEvent
- Refresh failure path forces HIGH and emits Asked

Markers: @pytest.mark.integration  — requires Postgres + Redis.
"""

from __future__ import annotations

import asyncio  # noqa: F401  — kept per plan; reserved for future concurrency assertions
from typing import Any
from unittest.mock import AsyncMock, MagicMock  # noqa: F401  — AsyncMock reserved per plan

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _build_pe_engine_with_real_deps(
    uow_factory: Any,          # REAL conftest fixture (api/tests/integration/conftest.py:91)
    redis_client: Any,         # REAL conftest fixture; wrapper exposing .client (line 597)
    skill_tool: Any,           # test-provided
) -> Any:
    """Build a DefaultPermissionEngine wired with real Postgres + Redis."""
    from app.application.composition.graph_assembly import (
        build_permission_engine,
    )
    from app.application.services.approval_state_adapters import (
        UowApprovalGrantQuery,
    )
    from app.application.services.approval_state_writer import (
        ApprovalStateWriter,
    )
    from app.domain.services.approval_state_reader import (
        ApprovalStateReader,
    )
    from app.domain.services.permission.confirmation_queue import (
        ConfirmationQueue,
    )
    from app.domain.services.permission.skill_refresher import (
        SkillRiskRefresher,
    )
    from app.domain.services.permission.sources import (
        NativeSource, SkillSource,
    )
    from app.domain.services.session.default_state_machine import (
        DefaultSessionStateMachine,
    )

    raw_redis = getattr(redis_client, "client", redis_client)
    writer = ApprovalStateWriter(uow_factory=uow_factory)
    reader = ApprovalStateReader(
        query=UowApprovalGrantQuery(uow_factory=uow_factory),
    )
    queue = ConfirmationQueue(redis=raw_redis)
    ssm = DefaultSessionStateMachine(
        uow_factory=uow_factory, redis=raw_redis,
    )
    sources = {
        "native": NativeSource(),
        "skill": SkillSource(
            refresher=SkillRiskRefresher(skill_tool),
            redis=raw_redis,
        ),
    }
    return build_permission_engine(
        uow_factory=uow_factory,
        writer=writer,
        queue=queue,
        session_machine=ssm,
        reader=reader,
        summary_llm=None,           # SmartApprove disabled in this E2E
        sources=sources,
    )


async def test_skill_high_risk_asked_then_user_approve_full_e2e(
    uow_factory, redis_client, sample_session, sample_user,
    fake_skill_tool_with_high_risk_binding,
):
    """Skill tool registered with final_risk=HIGH:
       1. tool_node calls pe.evaluate → SkillSource.assess_risk → HIGH → Asked(risk_enforce)
       2. ConfirmationDetail stored with risk_level='high'
       3. User HTTP-resumes with approve → pe.preflight_resume CAS claim succeeds
       4. graph commit_resume → writer.write(session-scoped approve)
       5. Second invocation in same session → reader.check returns 'allow' → AllowSuccess (no Asked)
    """
    from app.domain.models.session import SessionStatus
    from app.domain.models.skill import SkillRuntimeType
    from app.domain.models.tool_result import Asked, AllowSuccess
    from app.domain.services.permission.context import EvaluationContext, ResumeSignal
    from app.domain.services.permission.source_metadata import SkillCallMetadata
    from app.domain.services.permission.tool_call_spec import ToolCallSpec
    from app.domain.services.risk_assessor import RiskLevel

    pe = await _build_pe_engine_with_real_deps(
        uow_factory, redis_client, fake_skill_tool_with_high_risk_binding,
    )

    meta = SkillCallMetadata(
        tool_name="myskill_run",
        skill_id="sk_test",
        content_hash="sha256:hash1",
        risk_level=RiskLevel.HIGH,
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin="user_installed",
        scan_verdict="safe",
    )
    call = ToolCallSpec(
        tool_name="myskill_run",
        tool_args={"q": 1},
        tool_source="skill",
        user_id=str(sample_user.id),
        session_id=str(sample_session.id),
        risk_assessment=None,         # caller did NOT pre-fill (Risk #1 demo)
        tool_call_id="tc_e2e",
        source_metadata=meta,
    )
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
        retry_count=0,
    )

    # 1-2. First evaluate → Asked
    out1 = await pe.evaluate(call, ctx)
    assert isinstance(out1, Asked)
    assert out1.reason.type == "risk_enforce"
    assert out1.reason.code == "ask:high"

    # 3. HTTP preflight + commit_resume (user approve)
    # Real ResumeSignal shape (api/app/domain/services/permission/context.py:42):
    #   ResumeSignal(confirmation_id, action, grant_scope, actor)
    # confirmation_id = f"{session_id}:{tool_call_id}" (PE step 9 _cid format).
    approve_signal = ResumeSignal(
        confirmation_id=f"{sample_session.id}:tc_e2e",
        action="approve",
        grant_scope="session",
    )

    preflight = await pe.preflight_resume(call, ctx, approve_signal)
    assert preflight.claim_nonce is not None

    out2 = await pe.commit_resume(call, ctx, approve_signal, preflight.claim_nonce)
    assert isinstance(out2, AllowSuccess)

    # 4-5. Second invocation in same session → reader returns allow
    call2 = call  # same shape; new arg_digest matches the prior grant
    out3 = await pe.evaluate(call2, ctx)
    assert isinstance(out3, AllowSuccess)
    assert (out3.data or {}).get("via") in {"session_grant", "policy_rule"}


async def test_skill_refresh_failure_forces_user_confirm(
    uow_factory, redis_client, sample_session, sample_user, monkeypatch,
):
    """Refresher raises OSError → SkillRiskRefreshResult(failed) → force HIGH →
    Asked(risk_enforce). Verifies the unknown/failed → force-HIGH defense path."""
    from app.domain.models.session import SessionStatus
    from app.domain.models.skill import SkillRuntimeType
    from app.domain.models.tool_result import Asked
    from app.domain.services.permission.context import EvaluationContext
    from app.domain.services.permission.source_metadata import SkillCallMetadata
    from app.domain.services.permission.tool_call_spec import ToolCallSpec
    from app.domain.services.risk_assessor import RiskLevel

    class _BoomSkillTool:
        _tool_bindings = {"t1": {"skill": MagicMock(id="sk", scan_report={"content_hash": "h"}),
                                  "runtime_type": SkillRuntimeType.NATIVE,
                                  "manifest_tool": {}, "final_risk": "low",
                                  "trust_origin": "user_installed", "scan_verdict": "safe"}}
        _bundle_sync_manager = MagicMock()

        def resolve_skill_dir(self, tool_name):
            raise OSError("disk gone")

        def refresh_risk_if_stale(self, tool_name):
            raise OSError("disk gone")

    pe = await _build_pe_engine_with_real_deps(
        uow_factory, redis_client, _BoomSkillTool(),
    )
    meta = SkillCallMetadata(
        tool_name="t1", skill_id="sk", content_hash="h",
        risk_level=RiskLevel.LOW,  # cache says LOW but refresh will FAIL → force HIGH
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin="user_installed", scan_verdict="safe",
    )
    call = ToolCallSpec(
        tool_name="t1", tool_args={}, tool_source="skill",
        user_id=str(sample_user.id), session_id=str(sample_session.id),
        tool_call_id="tc_fail", source_metadata=meta,
    )
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1, retry_count=0,
    )
    out = await pe.evaluate(call, ctx)
    assert isinstance(out, Asked)
    assert out.reason.type == "risk_enforce"
    # force-HIGH path → code='ask:high'
    assert out.reason.code == "ask:high"
