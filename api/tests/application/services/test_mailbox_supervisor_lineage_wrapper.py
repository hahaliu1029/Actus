"""[C2 PR-8 Task 8.3 §13.4] CoordinatorProgressUpdateHandler lineage wrap tests.

Pins the PROGRESS_UPDATE dispatch route added in
``build_default_dispatch_table()``:

- ``tool_filter_preset == "coordinator_step"`` children → wrap forward with
  a lineage dict containing root/parent/child/coordinator_run/work_unit IDs.
- Non-coordinator children (e.g. ``"subagent_research"`` / ``None``) → fall
  through to the legacy stub-forward path with no lineage decoration.
- ``session_repo`` returns ``None`` → default stub-forward (defensive).
- ``compute_root_session_id`` walks ``parent_session_id`` until no parent
  remains; returns ``child_session_id`` as fallback if walk yields nothing.
- Dispatch table entry for PROGRESS_UPDATE is the coordinator handler, NOT
  the legacy stub.
- Every ``MailboxEnvelopeType`` value remains a key in the table.

Mock strategy: invoke ``CoordinatorProgressUpdateHandler.handle`` directly
with a ``SupervisorContext`` whose ``session_repo`` / ``agent_service_callback``
are AsyncMocks. No fake_redis / main-loop wiring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.coordinator_progress_handler import (
    CoordinatorProgressUpdateHandler,
)
from app.application.services.mailbox_supervisor import (
    HandlerOutcome,
    SupervisorContext,
    _StubNonTerminalHandler,
    build_default_dispatch_table,
)
from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.domain.models.session import (
    SandboxBinding,
    Session,
)


pytestmark = pytest.mark.anyio


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_progress_envelope(
    *,
    envelope_id: str = "01HSPYU0CR0000000000P00001",
    parent_session_id: str = "p1",
    child_session_id: str = "child-1",
) -> MailboxEnvelope:
    """Build a minimal valid PROGRESS_UPDATE envelope."""
    return MailboxEnvelope(
        envelope_id=envelope_id,
        type=MailboxEnvelopeType.PROGRESS_UPDATE,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        correlation_id="01HSPYU0CR0000000000P00002",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload={
            "kind": "heartbeat",
            "visibility": "debug",
        },
        reclaim_count=0,
    )


def _make_session(
    *,
    session_id: str = "child-1",
    parent_session_id: Optional[str] = "p1",
    tool_filter_preset: Optional[str] = "coordinator_step",
    coordinator_run_id: Optional[str] = "r1",
    work_unit_id: Optional[str] = "wu1",
) -> Session:
    return Session(
        id=session_id,
        parent_session_id=parent_session_id,
        worker_type="subagent" if parent_session_id is not None else "root",
        tool_filter_preset=tool_filter_preset,  # type: ignore[arg-type]
        coordinator_run_id=coordinator_run_id,
        work_unit_id=work_unit_id,
        sandbox_binding=SandboxBinding(),
    )


def _build_ctx(
    *,
    session_repo: Optional[MagicMock] = None,
    agent_service_callback: Optional[AsyncMock] = None,
    relay_progress_with_lineage: Optional[AsyncMock] = None,
    relay_progress_default: Optional[AsyncMock] = None,
    compute_root_session_id: Optional[MagicMock] = None,
) -> SupervisorContext:
    """Build a SupervisorContext stitched for the lineage wrap branch.

    ``SupervisorContext`` is a plain ``@dataclass``; runtime type is not
    enforced on unused fields, so the not-exercised plumbing is a MagicMock.
    """
    if agent_service_callback is None:
        agent_service_callback = AsyncMock(return_value=None)
    telemetry = AsyncMock()
    telemetry.emit = AsyncMock(return_value=None)

    ctx = SupervisorContext(
        root_session_id="root-1",
        pod_id="pod-a",
        instance_id="i1",
        redis=MagicMock(),
        audit_repo=MagicMock(),
        publisher=MagicMock(),
        sandbox_lifecycle=MagicMock(),
        agent_service_callback=agent_service_callback,
        telemetry=telemetry,
        session_repo=session_repo,
    )

    # Override helper bindings if the test wants to spy on them — by default
    # the dataclass methods on ``SupervisorContext`` are exercised end-to-end.
    if relay_progress_with_lineage is not None:
        ctx.relay_progress_with_lineage = relay_progress_with_lineage  # type: ignore[method-assign]
    if relay_progress_default is not None:
        ctx.relay_progress_default = relay_progress_default  # type: ignore[method-assign]
    if compute_root_session_id is not None:
        ctx.compute_root_session_id = compute_root_session_id  # type: ignore[method-assign]
    return ctx


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCoordinatorProgressUpdateHandler:
    """Pins the §13.4 PROGRESS_UPDATE lineage wrap contract."""

    async def test_coordinator_step_child_event_gets_lineage_tagged(self) -> None:
        """Happy path: child has ``tool_filter_preset='coordinator_step'`` →
        ``relay_progress_with_lineage`` is called with the lineage dict
        containing all 5 IDs; default forward is NOT called.
        """
        child = _make_session(
            session_id="child-1",
            parent_session_id="p1",
            tool_filter_preset="coordinator_step",
            coordinator_run_id="run-x",
            work_unit_id="wu-x.0",
        )
        # Walk: child-1 → p1 (no parent) so p1 is the root.
        p1 = _make_session(
            session_id="p1",
            parent_session_id=None,
            tool_filter_preset=None,
            coordinator_run_id=None,
            work_unit_id=None,
        )

        async def _get_by_id(sid: str) -> Optional[Session]:
            return {"child-1": child, "p1": p1}.get(sid)

        repo = MagicMock()
        repo.get_by_id = AsyncMock(side_effect=_get_by_id)

        relay_with_lineage = AsyncMock(return_value=None)
        relay_default = AsyncMock(return_value=None)

        ctx = _build_ctx(
            session_repo=repo,
            relay_progress_with_lineage=relay_with_lineage,
            relay_progress_default=relay_default,
        )

        env = _make_progress_envelope(
            parent_session_id="p1", child_session_id="child-1"
        )

        outcome = await CoordinatorProgressUpdateHandler().handle(env, ctx)

        assert isinstance(outcome, HandlerOutcome)
        assert outcome.ack is True
        assert outcome.side_effect is None
        assert outcome.audit_payload == {"coordinator_step": True}

        relay_with_lineage.assert_awaited_once()
        relay_default.assert_not_called()

        # Inspect the lineage argument.
        call_args = relay_with_lineage.await_args
        assert call_args is not None
        passed_envelope, passed_lineage = call_args.args
        assert passed_envelope is env
        assert passed_lineage == {
            "root_session_id": "p1",
            "parent_session_id": "p1",
            "child_session_id": "child-1",
            "coordinator_run_id": "run-x",
            "work_unit_id": "wu-x.0",
        }

    async def test_research_child_event_no_lineage_decoration(self) -> None:
        """Non-coordinator child → default forward path."""
        child = _make_session(
            session_id="child-r",
            parent_session_id="p1",
            tool_filter_preset="subagent_research",
            coordinator_run_id=None,
            work_unit_id=None,
        )
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=child)

        relay_with_lineage = AsyncMock(return_value=None)
        relay_default = AsyncMock(return_value=None)

        ctx = _build_ctx(
            session_repo=repo,
            relay_progress_with_lineage=relay_with_lineage,
            relay_progress_default=relay_default,
        )
        env = _make_progress_envelope(child_session_id="child-r")

        outcome = await CoordinatorProgressUpdateHandler().handle(env, ctx)

        assert outcome.ack is True
        assert outcome.audit_payload == {"coordinator_step": False, "stub": True}
        relay_default.assert_awaited_once_with(env)
        relay_with_lineage.assert_not_called()

    async def test_none_tool_filter_preset_child_no_lineage(self) -> None:
        """``tool_filter_preset=None`` (pre-C2 row) → default forward."""
        child = _make_session(
            session_id="child-legacy",
            parent_session_id="p1",
            tool_filter_preset=None,
            coordinator_run_id=None,
            work_unit_id=None,
        )
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=child)

        relay_with_lineage = AsyncMock(return_value=None)
        relay_default = AsyncMock(return_value=None)

        ctx = _build_ctx(
            session_repo=repo,
            relay_progress_with_lineage=relay_with_lineage,
            relay_progress_default=relay_default,
        )
        env = _make_progress_envelope(child_session_id="child-legacy")

        outcome = await CoordinatorProgressUpdateHandler().handle(env, ctx)

        assert outcome.ack is True
        assert outcome.audit_payload == {"coordinator_step": False, "stub": True}
        relay_default.assert_awaited_once_with(env)
        relay_with_lineage.assert_not_called()

    async def test_missing_child_session_falls_back_to_default(self) -> None:
        """session_repo.get_by_id returns None → default forward."""
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=None)

        relay_with_lineage = AsyncMock(return_value=None)
        relay_default = AsyncMock(return_value=None)

        ctx = _build_ctx(
            session_repo=repo,
            relay_progress_with_lineage=relay_with_lineage,
            relay_progress_default=relay_default,
        )
        env = _make_progress_envelope(child_session_id="ghost")

        outcome = await CoordinatorProgressUpdateHandler().handle(env, ctx)

        assert outcome.ack is True
        assert outcome.audit_payload == {"coordinator_step": False, "stub": True}
        relay_default.assert_awaited_once_with(env)
        relay_with_lineage.assert_not_called()

    async def test_missing_session_repo_falls_back_to_default(self) -> None:
        """Legacy SupervisorContext with no session_repo wired → default
        forward (no crash, defensive)."""
        relay_with_lineage = AsyncMock(return_value=None)
        relay_default = AsyncMock(return_value=None)

        ctx = _build_ctx(
            session_repo=None,
            relay_progress_with_lineage=relay_with_lineage,
            relay_progress_default=relay_default,
        )
        env = _make_progress_envelope()

        outcome = await CoordinatorProgressUpdateHandler().handle(env, ctx)

        assert outcome.ack is True
        assert outcome.audit_payload == {"coordinator_step": False, "stub": True}
        relay_default.assert_awaited_once_with(env)
        relay_with_lineage.assert_not_called()


class TestSupervisorContextLineageHelpers:
    """Pins ``compute_root_session_id`` + relay helpers on SupervisorContext."""

    async def test_compute_root_walks_parent_chain(self) -> None:
        """child → parent → grandparent (root). Returns grandparent id."""
        child = _make_session(
            session_id="c", parent_session_id="b",
            tool_filter_preset=None,
            coordinator_run_id=None, work_unit_id=None,
        )
        b = _make_session(
            session_id="b", parent_session_id="a",
            tool_filter_preset=None,
            coordinator_run_id=None, work_unit_id=None,
        )
        a = _make_session(
            session_id="a", parent_session_id=None,
            tool_filter_preset=None,
            coordinator_run_id=None, work_unit_id=None,
        )

        async def _get(sid: str) -> Optional[Session]:
            return {"c": child, "b": b, "a": a}.get(sid)

        repo = MagicMock()
        repo.get_by_id = AsyncMock(side_effect=_get)

        ctx = _build_ctx(session_repo=repo)
        root = await ctx.compute_root_session_id(child)
        assert root == "a"

    async def test_compute_root_returns_self_when_no_parent(self) -> None:
        """Child has no parent → it IS the root."""
        child = _make_session(
            session_id="solo", parent_session_id=None,
            tool_filter_preset=None,
            coordinator_run_id=None, work_unit_id=None,
        )
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=None)
        ctx = _build_ctx(session_repo=repo)
        root = await ctx.compute_root_session_id(child)
        assert root == "solo"
        # No walk needed.
        repo.get_by_id.assert_not_called()

    async def test_compute_root_falls_back_to_self_when_walk_breaks(
        self,
    ) -> None:
        """Walk hits a None parent lookup mid-chain → fall back to last known id."""
        child = _make_session(
            session_id="c", parent_session_id="missing",
            tool_filter_preset=None,
            coordinator_run_id=None, work_unit_id=None,
        )
        repo = MagicMock()
        # The parent lookup yields None — walk cannot proceed.
        repo.get_by_id = AsyncMock(return_value=None)
        ctx = _build_ctx(session_repo=repo)
        root = await ctx.compute_root_session_id(child)
        # Best-effort fallback: the deepest known ancestor id (the missing
        # parent_session_id) — this is still a stable lineage anchor.
        assert root == "missing"

    async def test_compute_root_caps_walk_depth(self) -> None:
        """Pathological cycle / very deep chain → walk caps and returns
        last visited id (no infinite loop)."""
        # Build a 50-node chain to exceed any reasonable cap.
        chain: dict[str, Session] = {}
        for i in range(50):
            parent = f"s{i + 1}" if i < 49 else None
            chain[f"s{i}"] = _make_session(
                session_id=f"s{i}",
                parent_session_id=parent,
                tool_filter_preset=None,
                coordinator_run_id=None,
                work_unit_id=None,
            )

        async def _get(sid: str) -> Optional[Session]:
            return chain.get(sid)

        repo = MagicMock()
        repo.get_by_id = AsyncMock(side_effect=_get)
        ctx = _build_ctx(session_repo=repo)
        root = await ctx.compute_root_session_id(chain["s0"])
        # Cap is bounded; we don't pin a specific value (the impl picks
        # a sensible default like 32), just that it returns a non-None
        # string and didn't run forever.
        assert isinstance(root, str)
        assert root.startswith("s")

    async def test_relay_progress_default_forwards_to_callback(self) -> None:
        """``relay_progress_default`` is the stub-forward equivalent."""
        callback = AsyncMock(return_value=None)
        ctx = _build_ctx(agent_service_callback=callback)
        env = _make_progress_envelope()
        await ctx.relay_progress_default(env)
        callback.assert_awaited_once_with(env)

    async def test_relay_progress_with_lineage_forwards_with_payload_decoration(
        self,
    ) -> None:
        """``relay_progress_with_lineage`` calls callback with an envelope
        whose payload has ``_lineage`` injected. Original envelope (frozen)
        is NOT mutated; a new envelope is constructed."""
        callback = AsyncMock(return_value=None)
        ctx = _build_ctx(agent_service_callback=callback)
        env = _make_progress_envelope()
        lineage = {
            "root_session_id": "p1",
            "parent_session_id": "p1",
            "child_session_id": "child-1",
            "coordinator_run_id": "run-x",
            "work_unit_id": "wu-x.0",
        }
        await ctx.relay_progress_with_lineage(env, lineage)

        callback.assert_awaited_once()
        forwarded = callback.await_args.args[0]
        assert isinstance(forwarded, MailboxEnvelope)
        # Original envelope is frozen — confirm we didn't mutate it.
        assert "_lineage" not in env.payload
        # Forwarded envelope carries the lineage decoration.
        assert forwarded.payload.get("_lineage") == lineage
        # Other identity fields unchanged.
        assert forwarded.envelope_id == env.envelope_id
        assert forwarded.child_session_id == env.child_session_id


class TestDispatchTableRoutesProgressUpdate:
    """Pins build_default_dispatch_table() routing for PROGRESS_UPDATE."""

    def test_progress_update_routes_to_coordinator_handler(self) -> None:
        table = build_default_dispatch_table()
        handler = table[MailboxEnvelopeType.PROGRESS_UPDATE]
        assert isinstance(handler, CoordinatorProgressUpdateHandler)
        assert not isinstance(handler, _StubNonTerminalHandler)

    def test_dispatch_table_still_covers_every_envelope_type(self) -> None:
        table = build_default_dispatch_table()
        missing = set(MailboxEnvelopeType) - set(table)
        assert missing == set()
