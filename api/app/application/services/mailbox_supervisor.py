"""C3 MailboxSupervisor (spec §6.1/§6.3/§6.4/§6.5).

Per-root supervisor task. Single owner of terminal sandbox transitions for
this root session (M1 invariant). PR-3a ships the skeleton + dispatch table +
main loop with stub handlers; PR-3b adds reliability (XAUTOCLAIM/poison/
crash-recovery); PR-3c integrates into agent_task_runner lifecycle; PR-4
swaps stub handlers for real destroy hooks.

Layer note: this module lives in ``application/`` (not ``domain/``) because
it directly depends on ``redis.asyncio`` and the infrastructure consumer
wrapper. Domain models (``MailboxEnvelope``, audit Protocol) are still imported
from ``domain/`` — only the orchestration is application-layer.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Protocol

from redis.asyncio import Redis

from app.domain.external.mailbox_publisher import MailboxPublisher
from app.domain.models.mailbox_envelope import (
    MAILBOX_XREADGROUP_BLOCK_MS,
    MAILBOX_XREADGROUP_COUNT,
    MailboxEnvelope,
    MailboxEnvelopeType,
)
from app.domain.repositories.mailbox_envelope_audit_repository import (
    MailboxEnvelopeAuditRepository,
)
from app.infrastructure.external.mailbox.redis_mailbox_consumer import (
    RedisMailboxConsumer,
)


logger = logging.getLogger(__name__)


class _SandboxLifecycleProtocol(Protocol):
    async def destroy(self, session_id: str, reason) -> None: ...


class _TelemetryProtocol(Protocol):
    async def emit(self, name: str, data: dict) -> None: ...


@dataclass
class SupervisorContext:
    """Handler injection bag — supervisor passes this to every handler.

    Handlers MUST treat ``ctx`` as read-only — there is no per-call ``ctx``
    cloning. State that must be observed across handler invocations belongs
    on ``audit_repo`` (durable) or on the per-supervisor instance via a
    dedicated dependency, NOT mutated on ``ctx``.
    """

    root_session_id: str
    pod_id: str
    instance_id: str
    redis: Redis
    audit_repo: MailboxEnvelopeAuditRepository
    publisher: MailboxPublisher
    sandbox_lifecycle: _SandboxLifecycleProtocol
    agent_service_callback: Callable[[MailboxEnvelope], Awaitable[None]]
    telemetry: _TelemetryProtocol
    clock: Callable[[], float] = time.monotonic

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)


@dataclass
class HandlerOutcome:
    """Handler return shape — drives ACK / PEL retention + post-ACK hooks.

    Per spec §6.x (and codex r2 [P1] fix — prior wording inverted the spec):

    - ``ack=True`` + ``side_effect=None`` → ACK immediately. Canonical for
      non-terminal handlers (PROGRESS_UPDATE/SPAWN_ACK/...) that forward to
      ``ctx.agent_service_callback`` and have no destructive follow-up work.
    - ``ack=False`` + ``side_effect`` set → run side_effect first.
        - side_effect completes normally → ACK (terminal handlers in PR-4
          return this shape: destroy() runs, success → ACK).
        - side_effect raises → leave in PEL (no ACK); PR-3b XAUTOCLAIM
          retries, reclaim_count > MAILBOX_POISON_MAX_RECLAIM → poison drop.
    - ``ack=False`` + ``side_effect=None`` → leave in PEL deliberately
      (explicit defer; PR-3b reliability layer handles).
    - ``ack=True`` + ``side_effect`` set → ACK after side_effect normal
      completion; same PEL-retain on raise as the canonical ``ack=False``
      shape (functionally equivalent, but encourages callers to express
      intent via ``ack=False`` when destruction is the gating event).

    The ACK-or-not branch is in :meth:`MailboxSupervisor._handle_envelope`.
    """

    ack: bool
    side_effect: Optional[Callable[[], Awaitable[None]]] = None
    audit_payload: dict = field(default_factory=dict)


class EnvelopeHandler(Protocol):
    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome: ...


# ─── PR-3a stub handlers (PR-4 replaces with real destroy hooks) ──────────────


class _StubNonTerminalHandler:
    """PR-3a placeholder for non-terminal envelopes (PROGRESS_UPDATE,
    SPAWN_REQUEST, ...). Forwards to the in-process agent_service callback
    and ACKs. PR-4 will diverge per-type with real business logic."""

    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        await ctx.agent_service_callback(envelope)
        return HandlerOutcome(ack=True, audit_payload={"stub": True})


class _StubTerminalHandler:
    """PR-3a placeholder for terminal envelopes (RESULT_READY, CANCEL_ACK).

    Emits telemetry and ACKs. PR-4 will swap for
    ResultReadyHandler/CancelAckHandler that actually call destroy().
    """

    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        await ctx.telemetry.emit(
            "mailbox.terminal_envelope_dispatched_stub",
            {
                "envelope_id": envelope.envelope_id,
                "type": envelope.type.value,
                "child_session_id": envelope.child_session_id,
            },
        )
        return HandlerOutcome(ack=True, audit_payload={"stub_terminal": True})


# ─── Dispatch table (spec §6.3) ───────────────────────────────────────────────


# Sentinel values for ``sig.bind()`` arity probing — never compared, only used
# to assert the handler signature can accept two positional args.
_HANDLER_VALIDATION_ENV_SENTINEL = object()
_HANDLER_VALIDATION_CTX_SENTINEL = object()


def _validate_dispatch_table(
    table: dict[MailboxEnvelopeType, EnvelopeHandler],
) -> None:
    """Hard-fail at construction if ``table`` would let the supervisor drop
    or crash on a legitimate envelope.

    Checks (in order):
    1. Every ``MailboxEnvelopeType`` value is a key — otherwise the
       unknown-type branch in :meth:`MailboxSupervisor._handle_envelope`
       would ACK-drop the missing type.
    2. Every value has a callable ``handle`` attribute — ``None`` slips
       past (1) but ``.get(type)`` would return ``None`` at dispatch time.
    3. ``handle`` is ``async def`` — a sync ``def handle`` is ``callable``
       but ``await handler.handle(...)`` would crash with ``TypeError``.
    4. ``handle``'s signature accepts ``(envelope, ctx)`` positionally —
       wrong arity (codex r5 [P2]) passes (3) but crashes at call site.

    Called from :meth:`MailboxSupervisor.__init__` for BOTH the user-supplied
    ``dispatch_table`` AND the default returned by
    :func:`build_default_dispatch_table` — codex r5 [P2] wanted the default
    path under the same guard so a future regression that drops a stub
    handler trips the validator instead of silently ACK-dropping envelopes.
    """
    missing = set(MailboxEnvelopeType) - set(table)
    if missing:
        raise ValueError(
            "MailboxSupervisor dispatch_table must cover every "
            f"MailboxEnvelopeType — missing: "
            f"{sorted(t.value for t in missing)}. Start from "
            "build_default_dispatch_table() and mutate entries "
            "rather than passing a partial table."
        )

    invalid: list[tuple[str, str]] = []
    for t, h in table.items():
        handle = getattr(h, "handle", None)
        if not callable(handle):
            invalid.append((t.value, "no callable .handle"))
            continue
        if not inspect.iscoroutinefunction(handle):
            invalid.append((t.value, "handle is not async def"))
            continue
        try:
            inspect.signature(handle).bind(
                _HANDLER_VALIDATION_ENV_SENTINEL,
                _HANDLER_VALIDATION_CTX_SENTINEL,
            )
        except TypeError as e:
            invalid.append((t.value, f"signature won't accept (envelope, ctx): {e}"))

    if invalid:
        raise ValueError(
            "MailboxSupervisor dispatch_table values must expose an "
            "``async def handle(envelope, ctx) -> HandlerOutcome`` method — "
            f"invalid: {invalid!r}"
        )


def build_default_dispatch_table() -> dict[MailboxEnvelopeType, EnvelopeHandler]:
    """PR-3a default: every type routed to a stub. PR-4 overrides terminal +
    cascade entries with real handlers.

    INV: every value of ``MailboxEnvelopeType`` MUST be a key in the returned
    table — otherwise the supervisor would fall into the unknown-type branch
    and ACK-drop legitimate envelopes. CI enforcement: see the unit test that
    counts entries.
    """
    stub_terminal = _StubTerminalHandler()
    stub_nonterminal = _StubNonTerminalHandler()
    return {
        MailboxEnvelopeType.RESULT_READY: stub_terminal,
        MailboxEnvelopeType.CANCEL_ACK: stub_terminal,
        MailboxEnvelopeType.SPAWN_REQUEST: stub_nonterminal,
        MailboxEnvelopeType.SPAWN_ACK: stub_nonterminal,
        MailboxEnvelopeType.PROGRESS_UPDATE: stub_nonterminal,
        MailboxEnvelopeType.APPROVAL_REQUEST: stub_nonterminal,
        MailboxEnvelopeType.APPROVAL_RESPONSE: stub_nonterminal,
        MailboxEnvelopeType.CANCEL_REQUEST: stub_nonterminal,
        MailboxEnvelopeType.DEPENDENCY_BLOCKED: stub_nonterminal,
        MailboxEnvelopeType.HANDOFF_REQUEST: stub_nonterminal,
    }


# ─── Supervisor main loop ─────────────────────────────────────────────────────


class MailboxSupervisor:
    """Per-root supervisor task (M1). Single asyncio.Task lifecycle.

    The supervisor runs ``ensure_group()`` once, then loops on XREADGROUP +
    handler dispatch. The loop is robust against handler exceptions (the
    envelope stays in PEL for XAUTOCLAIM retry in PR-3b) and against
    transient Redis errors (logged + 0.5s back-off).

    ``CancelledError`` is the ONLY exception that escapes the inner loop —
    the parent task must be cancellable for clean asyncio shutdown.
    """

    def __init__(
        self,
        ctx: SupervisorContext,
        *,
        dispatch_table: Optional[dict[MailboxEnvelopeType, EnvelopeHandler]] = None,
        block_ms: int = MAILBOX_XREADGROUP_BLOCK_MS,
        count: int = MAILBOX_XREADGROUP_COUNT,
        idle_poll_sleep_s: float = 0.0,
    ) -> None:
        self._ctx = ctx
        # INV (codex r1 [P1] fix): an explicit dispatch_table MUST cover every
        # MailboxEnvelopeType. The unknown-type branch in _handle_envelope is a
        # defense against schema drift (new wire type added without dispatch
        # entry), NOT a soft fallback — falling through to ACK-drop a legitimate
        # type because the caller passed a partial override would silently
        # discard envelopes. PR-4 swaps specific terminal handlers by starting
        # from build_default_dispatch_table() and mutating entries; we validate
        # here so a typo or forgotten entry trips at construction, not in
        # production after a real envelope arrives.
        # INV (codex r1/r2/r4/r5): validation always runs on BOTH the user-
        # supplied dispatch_table AND the default — see _validate_dispatch_table
        # for the full rationale. The default path is validated too so a
        # future regression that drops an enum value or replaces a stub with
        # a sync def trips at construction, not in production after a real
        # envelope arrives.
        if dispatch_table is None:
            dispatch_table = build_default_dispatch_table()
        _validate_dispatch_table(dispatch_table)
        self._dispatch = dispatch_table
        self._consumer = RedisMailboxConsumer(
            ctx.redis, ctx.root_session_id, ctx.pod_id, ctx.instance_id
        )
        self._stopping = asyncio.Event()
        self._stopped = asyncio.Event()
        # XREADGROUP tuning seams. Production keeps the constants (1000ms blocking
        # read, batch=32). Tests pass ``block_ms=0`` because the fakeredis async
        # impl does NOT wake a blocked XREADGROUP on a concurrent XADD — a
        # blocking call would hang forever in unit tests. ``idle_poll_sleep_s``
        # adds a tiny back-off when running non-blocking so the loop doesn't
        # spin-burn CPU; production with block_ms>0 should keep it at 0.
        self._block_ms = block_ms
        self._count = count
        self._idle_poll_sleep_s = idle_poll_sleep_s

    async def run(self) -> None:
        try:
            await self._consumer.ensure_group()
            # R2 P2.1 — signal readiness to SupervisorRegistry.spawn (if it's
            # waiting via a ``_ready_event`` injection). Optional hook.
            ready = getattr(self, "_ready_event", None)
            if ready is not None:
                ready.set()
            # PR-3b: await self._initial_xautoclaim() goes here
            while not self._stopping.is_set():
                try:
                    entries = await self._consumer.read(
                        count=self._count,
                        block_ms=self._block_ms,
                    )
                    # Codex r10 [P1] fix — per-entry fault isolation. The
                    # outer try/except below catches read()/sleep() errors
                    # and applies a 0.5s back-off. Without an inner try
                    # around _handle_envelope, an exception from one entry
                    # (most likely a transient XACK failure, which is NOT
                    # caught inside _handle_envelope) would bubble out of
                    # the for-loop, sleep 0.5s, then the next iteration
                    # reads ">" and skips the unprocessed remainder of
                    # this batch — those entries sit in PEL until PR-3b
                    # XAUTOCLAIM picks them up. Isolating each call so a
                    # single failing entry doesn't starve the rest of the
                    # batch is part of the skeleton's contract; PEL
                    # redelivery semantics for the failing entry itself
                    # are unchanged (no ACK → XAUTOCLAIM retries in PR-3b).
                    for redis_id, envelope in entries:
                        try:
                            await self._handle_envelope(redis_id, envelope)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "supervisor entry handling failed root=%s id=%s "
                                "type=%s — leave in PEL, continue batch",
                                self._ctx.root_session_id,
                                redis_id,
                                envelope.type.value,
                            )
                    # PR-3b: await self._tick_check_orphans() / periodic autoclaim
                    if not entries and self._idle_poll_sleep_s > 0:
                        await asyncio.sleep(self._idle_poll_sleep_s)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "supervisor loop iteration failed root=%s; sleep 0.5s",
                        self._ctx.root_session_id,
                    )
                    await asyncio.sleep(0.5)
        finally:
            self._stopped.set()

    async def stop(self, *, drain_timeout_s: float = 5.0) -> None:
        """Request stop and wait for the run-loop to actually exit.

        This is NOT a fire-and-forget flag — callers can rely on
        ``await sup.stop()`` to drain in-flight handler work, up to
        ``drain_timeout_s`` seconds. Past that we log a warning and return
        (the orphaned task is the caller's problem; this method must not
        block shutdown indefinitely).
        """
        self._stopping.set()
        try:
            await asyncio.wait_for(self._stopped.wait(), timeout=drain_timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "supervisor stop drain timed out root=%s after %ss",
                self._ctx.root_session_id,
                drain_timeout_s,
            )

    async def _handle_envelope(
        self, redis_id: bytes, envelope: MailboxEnvelope
    ) -> None:
        """Dispatch one envelope through its handler and ACK on success.

        - Cross-root envelope (parent_session_id mismatch) → log + ACK to
          drain (defense-in-depth — see comment below).
        - Unknown envelope type → log warning + ACK (drain; don't loop).
        - Handler raises → leave in PEL (no ACK). PR-3b XAUTOCLAIM retries
          until ``reclaim_count > MAILBOX_POISON_MAX_RECLAIM`` then drops to
          dead-letter sink.
        - ``side_effect`` raises → same PEL retain (no ACK).
        - ``outcome.ack == False`` → leave in PEL deliberately (defer).

        **Consumer-side audit dedup is PR-3b scope.** Per spec §5.8 the
        audit-repo ``get_processed → upsert_processing → mark_processed``
        lifecycle wires in PR-3b reliability work alongside XAUTOCLAIM, so
        a redelivered terminal envelope doesn't re-fire destroy(). PR-3a
        is the skeleton (routing + ACK paths only); the publisher-side
        Redis SET NX dedup (PR-2 / spec §5.8 Layer 1) is the only
        idempotency guarantee until PR-3b lands. Do NOT add audit calls
        here — that crosses the §5.8 layer boundary.
        """
        # Codex r8 [P1] fix — cross-root defense-in-depth. The XREADGROUP
        # stream key already scopes reads to one root, so a mismatch implies
        # either a publisher bug or a misrouted/forged envelope. We refuse
        # to dispatch and ACK to drain instead of looping; the warning
        # surfaces the publisher bug in production logs. Without this guard
        # a misrouted envelope could trigger PR-4 terminal destruction
        # against the wrong root, violating the M1 per-root single-writer
        # invariant.
        if envelope.parent_session_id != self._ctx.root_session_id:
            logger.warning(
                "cross-root envelope refused — supervisor root=%s but "
                "envelope.parent_session_id=%s (id=%s type=%s) — dropping+ack",
                self._ctx.root_session_id,
                envelope.parent_session_id,
                envelope.envelope_id,
                envelope.type.value,
            )
            await self._consumer.ack(redis_id)
            return

        handler = self._dispatch.get(envelope.type)
        if handler is None:
            logger.warning(
                "no handler for envelope type %s root=%s id=%s — drop+ack",
                envelope.type.value,
                self._ctx.root_session_id,
                envelope.envelope_id,
            )
            await self._consumer.ack(redis_id)
            return

        try:
            outcome = await handler.handle(envelope, self._ctx)
        except Exception:
            logger.exception(
                "handler raised for envelope id=%s type=%s — leave in PEL for retry",
                envelope.envelope_id,
                envelope.type.value,
            )
            # Do NOT ACK; PR-3b XAUTOCLAIM will retry; reclaim_count > MAX → poison drop.
            return

        # Codex r2 [P1] fix: side_effect success implies ACK (per spec §6.x),
        # regardless of ``outcome.ack``. PR-4 terminal handlers return
        # ``ack=False, side_effect=destroy(...)`` — when destroy() returns
        # normally, the envelope MUST be ACKed to avoid redelivery + double
        # destruction. The prior implementation only ACKed when ``ack=True``,
        # which would have re-fired terminal side effects on the next
        # XAUTOCLAIM sweep once PR-3b landed.
        if outcome.side_effect is not None:
            try:
                await outcome.side_effect()
            except Exception:
                logger.exception(
                    "side_effect raised for envelope id=%s — leave in PEL",
                    envelope.envelope_id,
                )
                return  # do NOT ack — PEL retain for XAUTOCLAIM retry
            await self._consumer.ack(redis_id)
            return

        # No side_effect — honor ``outcome.ack`` as the explicit ACK/defer
        # signal. ``ack=False`` is the legitimate "defer" path PR-3b reliability
        # layer relies on (e.g., dedup hit waiting for the in-flight handler).
        if outcome.ack:
            await self._consumer.ack(redis_id)
