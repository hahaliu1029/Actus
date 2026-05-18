"""PE-1 §5.1 / §5.2 — agent_service preflight must allow UnsupportedSource
and PEInfrastructureUnavailable to propagate to the FastAPI handler. The
handler-side mapping (422 / 503 / correlation_id body) is verified by
test_pe_exception_handlers.py (T16).

This test is a unit-level guard against future regressions where someone
adds a broad ``except Exception:`` around pe.preflight_resume and
accidentally rewrites the two PE-1 infra exceptions into a 500.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.services.permission.errors import (
    PEInfrastructureUnavailable,
    UnsupportedSource,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_fake_ssm():
    """Real preflight calls ``await ssm.get_mode_with_revision(...)`` to build
    the EvaluationContext BEFORE pe.preflight_resume."""
    from app.domain.models.session import SessionStatus

    ssm = MagicMock()
    ssm.get_mode_with_revision = AsyncMock(
        return_value=(SessionStatus.RUNNING, 1),
    )
    return ssm


def _make_pending_detail():
    """pending_detail is read by preflight to build the PE call_spec.

    Real attributes consumed before ``pe.preflight_resume``:
    - ``tool_name`` — passed through ``resolve_tool_source``. Round 2 P1#1
      changed the unknown-name fallback: instead of pretending the call is
      ``source="native"``, it now treats unknown as non-PE-eligible and
      routes to legacy. So this test uses a real native tool name
      (``file_write``) to make ``is_pe_eligible_tool_source`` return True
      and reach ``pe.preflight_resume`` — which is the line under test.
    - ``tool_args`` — wrapped via ``dict(...)`` so it must be a real Mapping
    - ``user_id`` — copied into ``ToolCallSpec.user_id`` (frozen dataclass)
    - ``primary_arg`` / ``dir_arg`` / ``arg_digest`` — read via ``getattr``
    """
    return MagicMock(
        tool_name="file_write",
        tool_call_id="tc_x",
        tool_args={},
        user_id="u1",
        risk_level="high",
        arg_digest="d",
        primary_arg="",
        dir_arg=None,
        matched_patterns=[],
        deadline_ts=0.0,
    )


def _wire_agent_service_for_preflight(svc, fake_pe, fake_ssm, monkeypatch):
    """Shared monkeypatch wiring so each test reaches ``pe.preflight_resume``.

    The path through ``preflight_resume_tool_confirmation`` before the target
    line touches:
    1. ``self._config_snapshot`` (read; ``agent_config.tool_confirmation``
       needs ``enabled`` + ``permission_engine_native_enabled`` truthy — a
       plain MagicMock satisfies the gate because attribute access yields
       truthy Mocks).
    2. ``self._build_pe_ssm_for_resume(snap)`` → ``(pe, ssm)``.
    3. ``self._get_accessible_session(...)`` → real session-like object.
    4. ``self._confirmation_manager.read(...)`` → pending_detail.
    5. ``self._get_task(session)`` returning ``None`` so we skip the
       split-brain + batch-mixed guards.
    6. ``ssm.get_mode_with_revision(...)`` → ``(RUNNING, rev)`` so the
       SessionMode check passes.
    7. ``pe.preflight_resume(...)`` ← target raise site.
    """
    monkeypatch.setattr(
        svc, "_build_pe_ssm_for_resume",
        MagicMock(return_value=(fake_pe, fake_ssm)),
        raising=False,
    )
    svc._confirmation_manager = MagicMock()
    svc._confirmation_manager.read = AsyncMock(return_value=_make_pending_detail())
    svc._get_accessible_session = AsyncMock(return_value=MagicMock())
    svc._get_task = AsyncMock(return_value=None)
    svc._config_snapshot = MagicMock()


async def test_preflight_lets_unsupported_source_propagate(monkeypatch):
    """If PE.preflight_resume raises UnsupportedSource (caller-bug surface),
    agent_service must NOT catch it — exception_handlers maps to 422."""
    from app.application.services.agent_service import AgentService

    svc = AgentService.__new__(AgentService)
    fake_pe = MagicMock()
    fake_pe.preflight_resume = AsyncMock(side_effect=UnsupportedSource("mystery"))
    fake_ssm = _make_fake_ssm()

    _wire_agent_service_for_preflight(svc, fake_pe, fake_ssm, monkeypatch)

    with pytest.raises(UnsupportedSource):
        await svc.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False,
            tool_confirmation=MagicMock(
                action="approve", scope="session", tool_call_id="tc_x",
            ),
        )


async def test_preflight_lets_pe_infra_unavailable_propagate(monkeypatch):
    """If PE.preflight_resume raises PEInfrastructureUnavailable (Redis
    outage during preflight), agent_service must NOT catch it —
    exception_handlers maps to 503 with correlation_id."""
    from app.application.services.agent_service import AgentService

    svc = AgentService.__new__(AgentService)
    fake_pe = MagicMock()
    fake_pe.preflight_resume = AsyncMock(
        side_effect=PEInfrastructureUnavailable("redis_get_fail_key: timeout"),
    )
    fake_ssm = _make_fake_ssm()

    _wire_agent_service_for_preflight(svc, fake_pe, fake_ssm, monkeypatch)

    with pytest.raises(PEInfrastructureUnavailable):
        await svc.preflight_resume_tool_confirmation(
            session_id="s1", user_id="u1", is_admin=False,
            tool_confirmation=MagicMock(
                action="approve", scope="session", tool_call_id="tc_x",
            ),
        )
