"""[C2 PR-8 Task 8.3 §13.4] CoordinatorProgressUpdateHandler.

Replaces ``_StubNonTerminalHandler`` for ``MailboxEnvelopeType.PROGRESS_UPDATE``
in :func:`app.application.services.mailbox_supervisor.build_default_dispatch_table`.
Decorates the forwarded envelope with the coordinator lineage tuple
``(root_session_id, parent_session_id, child_session_id, coordinator_run_id,
work_unit_id)`` whenever the child session was spawned by the C2 coordinator
subgraph (``Session.tool_filter_preset == "coordinator_step"``). For every
other child (research / general subagents, pre-C2 legacy rows) the handler
falls through to the same stub-forward path the legacy ``_StubNonTerminalHandler``
took — backward compatible.

Layer note: this module lives in ``application/`` because it consumes
``MailboxEnvelope`` (domain) + ``SupervisorContext`` / ``HandlerOutcome``
(application). No FastAPI / SQLAlchemy imports.

The actual ``coordinator_progress_update`` SSE event emission (with the lineage
fields populated) is wired by PR-8 Task 8.4 in the subgraph / orchestrator —
this handler's job is the *inbound mailbox decoration* only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-only to avoid circular import
    from app.application.services.mailbox_supervisor import (
        HandlerOutcome,
        SupervisorContext,
    )

from app.domain.models.mailbox_envelope import MailboxEnvelope
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET


logger = logging.getLogger(__name__)


class CoordinatorProgressUpdateHandler:
    """[C2 PR-8 §13.4] PROGRESS_UPDATE → lineage-decorated forward.

    Flow:
        1. Look the child session up via ``ctx.session_repo``.
        2. If ``tool_filter_preset == "coordinator_step"`` → build the lineage
           dict and forward via ``ctx.relay_progress_with_lineage``.
        3. Else (research/general/None preset, or missing session_repo, or
           missing child row) → forward via ``ctx.relay_progress_default``
           (the legacy stub-forward path).
        4. Always ``ack=True`` with no ``side_effect`` — PROGRESS_UPDATE is a
           non-terminal envelope and matches the live ``_StubNonTerminalHandler``
           ACK shape.

    [r5 P0-4] non-terminal handler contract: ``HandlerOutcome(ack=True,
    side_effect=None)`` — see ``HandlerOutcome`` docstring (spec §6.x).
    """

    # [codex PR-8 R3 P1 -- deferred to PR-9] CoordinatorProgressUpdateHandler
    # injects ``_lineage`` into the envelope payload via
    # ``ctx.relay_progress_with_lineage``, but the LIVE
    # ``agent_service_callback`` for PROGRESS_UPDATE in
    # ``_pr4_5_agent_service_callback`` (service_dependencies wiring) currently
    # only forwards CANCEL_REQUEST envelopes — PROGRESS_UPDATE envelopes are
    # dropped before reaching the parent session's SSE event_queue. The
    # lineage tagging in this PR is therefore "installed but unreached" until
    # PR-9 wires the PROGRESS_UPDATE forward path (parent SSE projection
    # from coordinator-step children with lineage-aware kind/visibility
    # fields so heartbeat noise can be filtered). See PR-9 ramp for the
    # end-to-end frontend timeline.
    async def handle(
        self, envelope: MailboxEnvelope, ctx: "SupervisorContext"
    ) -> "HandlerOutcome":
        # Late import — avoids a top-level circular dependency between this
        # module and ``mailbox_supervisor`` (which imports this handler from
        # ``build_default_dispatch_table``).
        from app.application.services.mailbox_supervisor import HandlerOutcome

        # Defensive: legacy SupervisorContext / pre-C2 wiring may not have
        # ``session_repo`` populated. Fall through to the stub path so the
        # envelope is still ACKed and forwarded — never silently drop.
        #
        # Audit payload semantics:
        # - Positive coordinator_step lineage path → ``{"coordinator_step": True}``
        # - Fall-through (no session_repo, missing child, non-coordinator
        #   preset) → ``{"coordinator_step": False, "stub": True}``. The
        #   ``"stub": True`` mirrors the prior ``_StubNonTerminalHandler``
        #   shape so audit consumers see no behavioral drift on
        #   non-coordinator children.
        if ctx.session_repo is None:
            await ctx.relay_progress_default(envelope)
            return HandlerOutcome(
                ack=True,
                audit_payload={"coordinator_step": False, "stub": True},
            )

        child_session = await ctx.session_repo.get_by_id(envelope.child_session_id)
        if child_session is None or child_session.tool_filter_preset != COORDINATOR_STEP_PRESET:
            await ctx.relay_progress_default(envelope)
            return HandlerOutcome(
                ack=True,
                audit_payload={"coordinator_step": False, "stub": True},
            )

        lineage = {
            "root_session_id": await ctx.compute_root_session_id(child_session),
            "parent_session_id": child_session.parent_session_id,
            "child_session_id": envelope.child_session_id,
            "coordinator_run_id": child_session.coordinator_run_id,
            "work_unit_id": child_session.work_unit_id,
        }
        await ctx.relay_progress_with_lineage(envelope, lineage)
        return HandlerOutcome(ack=True, audit_payload={"coordinator_step": True})
