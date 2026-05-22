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
    CHILD_TO_PARENT_TYPES,
    MAILBOX_PEL_IDLE_MS_FOR_CLAIM,
    MAILBOX_POISON_MAX_RECLAIM,
    MAILBOX_STREAM_KEY_TEMPLATE,
    MAILBOX_XAUTOCLAIM_PERIODIC_INTERVAL_SECONDS,
    MAILBOX_XREADGROUP_BLOCK_MS,
    MAILBOX_XREADGROUP_COUNT,
    CancelPolicy,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
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


@dataclass
class _CancelState:
    """In-memory CancelRequest tracking — populated by PR-4 cancel handler,
    consumed by PR-4 auto-escalate tick (spec §7.5). PR-3b ships the store
    so the supervisor instance carries the per-child cancel state across
    XREADGROUP iterations; PR-4 wires the periodic tick that promotes
    REQUEST_CANCEL → TERMINATE after CHILD_CANCEL_ACK_TIMEOUT_MS.
    """

    child_session_id: str
    policy: CancelPolicy
    requested_at_mono: float


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
    envelope stays in PEL for XAUTOCLAIM retry — PR-3b) and against
    transient Redis errors (logged + 0.5s back-off).

    ``CancelledError`` is the ONLY exception that escapes the inner loop —
    the parent task must be cancellable for clean asyncio shutdown.
    """

    # PR-3b class-level tunables — tests monkeypatch these to drive periodic
    # autoclaim into the test window. Production keeps the spec constants.
    _XAUTOCLAIM_INTERVAL_S: float = MAILBOX_XAUTOCLAIM_PERIODIC_INTERVAL_SECONDS
    _XAUTOCLAIM_MIN_IDLE_MS: int = MAILBOX_PEL_IDLE_MS_FOR_CLAIM

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
        # PR-3b reliability layer state ──────────────────────────────────────
        # last_seen[child_session_id] = monotonic seconds of latest *child-
        # origin* envelope (spec §9.2). SUPERVISOR_ECHO must NOT update —
        # otherwise a supervisor's own CANCEL_ACK after destroy() would
        # falsely report the child as alive.
        self._last_seen_mono: dict[str, float] = {}
        # _cancel_states[child_session_id] = _CancelState set by PR-4 cancel
        # handler. PR-3b ships the store; PR-4 wires the periodic auto-
        # escalate tick (spec §7.5: REQUEST_CANCEL → TERMINATE after
        # CHILD_CANCEL_ACK_TIMEOUT_MS).
        self._cancel_states: dict[str, _CancelState] = {}
        # XAUTOCLAIM throttle — set to last_run_mono so the *first* loop
        # iteration after _initial_xautoclaim() must wait one interval
        # before running the periodic sweep again.
        self._last_autoclaim_mono: float = 0.0
        # _known_children is populated by SupervisorRegistry.spawn at PR-3c
        # so _restore_last_seen_after_pod_restart() can scan XREVRANGE for
        # the children we care about. Defaults to empty (no restore work).
        self._known_children: list[str] = []

    async def run(self) -> None:
        try:
            await self._consumer.ensure_group()
            # R2 P2.1 — signal readiness to SupervisorRegistry.spawn (if it's
            # waiting via a ``_ready_event`` injection). Optional hook.
            ready = getattr(self, "_ready_event", None)
            if ready is not None:
                ready.set()
            # PR-3b spec §5.6 — startup XAUTOCLAIM drains any PEL entries
            # left over by a dead consumer on the same root (min_idle_ms=0
            # claims everything regardless of idle time, because the previous
            # consumer is by definition no longer reading).
            await self._initial_xautoclaim()
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
                    # PR-3b spec §5.6 — periodic XAUTOCLAIM (every
                    # _XAUTOCLAIM_INTERVAL_S seconds). Inline so the cadence
                    # is driven by the same loop's clock and we don't need a
                    # separate task.
                    await self._maybe_periodic_xautoclaim()
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
        a redelivered terminal envelope doesn't re-fire destroy(). PR-3b
        ships this in two places:

        1. ``get_processed`` check between the cross-root guard and the
           poison gate — short-circuits redelivery of envelopes that
           already ran their side_effect (codex r6 [HIGH] fix).
        2. ``mark_processed`` immediately before XACK on the
           side_effect-success path — the marker the get_processed check
           consumes on redelivery (codex r6 [HIGH] fix).

        XAUTOCLAIM bumps go through ``upsert_processing`` +
        ``increment_reclaim`` (PR-3b Phase B) so reclaim_count is
        durable across pod restarts.
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

        # PR-3b spec §5.8 layer 2 — consumer-side audit dedup (codex r6
        # [HIGH] fix). The crash window between ``side_effect`` success and
        # XACK is recoverable iff the supervisor marked the envelope as
        # processed BEFORE the XACK. On crash-then-redelivery via XAUTOCLAIM,
        # this check finds the existing ``processed_at`` row and ACKs without
        # re-firing the handler — preventing double destroy() in PR-4 once
        # terminal handlers wire real destructive side effects.
        #
        # ``get_processed`` failure is logged + treated as "not processed"
        # (proceed to handler). The cost of a false-negative is at-most-once
        # re-fire (current state). A false-positive (treating fresh as
        # processed) would silently lose work, so degrade safely toward the
        # already-current behavior.
        #
        # Placed BEFORE the poison gate: a poison envelope that was already
        # processed should drain via this short-circuit, not via the poison
        # path's telemetry+hook+ACK chain (no need to fire poison cascade
        # for an envelope whose terminal side_effect already ran).
        try:
            already_processed = await self._ctx.audit_repo.get_processed(
                envelope.parent_session_id, envelope.envelope_id
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "audit_repo.get_processed failed envelope=%s — proceeding "
                "to handler (degrade safely; PR-4 dedup may double-fire)",
                envelope.envelope_id,
            )
            already_processed = False
        if already_processed:
            logger.info(
                "envelope already processed (audit dedup hit) — ACK without "
                "re-firing handler id=%s type=%s",
                envelope.envelope_id,
                envelope.type.value,
            )
            await self._consumer.ack(redis_id)
            return

        # PR-3b spec §5.7 — poison drop. An envelope that has been reclaimed
        # more than MAILBOX_POISON_MAX_RECLAIM times is treated as stuck (the
        # handler chain is consistently failing for some non-transient reason)
        # — ACK + drop + telemetry so the stream stops re-fighting it.
        # Placed BEFORE last_seen refresh: a poison envelope's reclaim_count
        # implies it's been stuck for at least MAX sweeps, so treating it as
        # a fresh heartbeat would be misleading.
        if envelope.reclaim_count > MAILBOX_POISON_MAX_RECLAIM:
            # Codex r4 [P1] fix — telemetry.emit is best-effort here. If the
            # OTel/exporter backend raises (network blip, sink misconfig),
            # we still MUST reach the ACK below, otherwise the same
            # redelivery loop the hook isolation was meant to prevent
            # comes back via the telemetry path: envelope stays in PEL →
            # XAUTOCLAIM next cycle → poison gate re-fires → emit raises
            # again. Same risk as r1 [P1] for ``_on_poison_drop``; same
            # treatment (CancelledError escapes, broad Exception logged).
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.poison_message_dropped",
                    {
                        "envelope_id": envelope.envelope_id,
                        "type": envelope.type.value,
                        "child_session_id": envelope.child_session_id,
                        "reclaim_count": envelope.reclaim_count,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "poison drop telemetry raised envelope=%s — ACK anyway",
                    envelope.envelope_id,
                )
            # Spec §5.7 step 3 hook — if terminal type AND child still
            # RUNNING, trigger orphan cascade. PR-4 overrides this method.
            # Codex r1 [P1] fix — isolate hook exception. If a PR-4 override
            # raises (e.g., orphan-cascade publisher errors), we MUST still
            # ACK; otherwise the envelope stays in PEL, XAUTOCLAIM picks it
            # up next cycle, poison gate re-fires, hook re-raises — infinite
            # loop with ever-growing reclaim_count that never reaches ACK.
            try:
                await self._on_poison_drop(envelope)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "poison drop hook raised envelope=%s — ACK anyway to "
                    "break the redelivery loop",
                    envelope.envelope_id,
                )
            await self._consumer.ack(redis_id)
            return

        # PR-3b spec §9.2 — last_seen refresh BEFORE dispatch. Only child-
        # origin envelopes advance the heartbeat clock. ``_handle_envelope``
        # is the single funnel for both XREADGROUP fresh-delivery AND
        # XAUTOCLAIM redelivery, so this captures both paths.
        await self._refresh_last_seen_if_child_origin(envelope)

        # PR-3b spec §5.8 layer 2 — pre-stage the audit row BEFORE dispatch
        # so the side_effect-success path's ``mark_processed`` finds an
        # existing row (codex r8 [HIGH] fix). The DB impl raises ``ValueError``
        # on missing row (db_mailbox_envelope_audit_repository.py:141-145),
        # so without this upsert the fresh XREADGROUP delivery path would
        # silently lose its dedup marker when ``mark_processed`` raises and
        # the broad-except below logs + ACKs anyway — re-opening the
        # double-destroy hole codex r6 had closed for the XAUTOCLAIM paths
        # (XAUTOCLAIM already calls upsert before increment_reclaim).
        #
        # ``upsert_processing`` is a real PG ``ON CONFLICT DO UPDATE`` (see
        # db_mailbox_envelope_audit_repository.py lines 63-98), idempotent
        # whether the row exists or not. On failure we proceed to dispatch
        # without the row — best-effort degradation that preserves the
        # pre-r8 (broken) behavior, NOT a regression vs current state.
        try:
            await self._ctx.audit_repo.upsert_processing(
                envelope, processing_at=self._ctx.now()
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "audit_repo.upsert_processing failed envelope=%s — "
                "proceeding without dedup row (mark_processed may raise "
                "later; ACK still fires, accepts at-most-once dedup-loss)",
                envelope.envelope_id,
            )

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
            # PR-3b spec §5.8 layer 2 — write the dedup marker BEFORE XACK
            # (codex r6 [HIGH] fix). A crash window between this call and
            # the XACK below is recoverable: XAUTOCLAIM redelivers, the
            # ``get_processed`` check at envelope entry finds the row, ACKs
            # without re-firing side_effect.
            #
            # ``mark_processed`` failure: log + ACK anyway. Skipping the ACK
            # would force redelivery — and since the side_effect already
            # ran, the next pass (with ``get_processed=False`` because the
            # marker write failed) would re-fire destroy(). ACKing on
            # mark_processed failure accepts at-most-once dedup-loss in the
            # rare audit-DB-down case, but avoids guaranteed double-destroy.
            #
            # KNOWN PR-4 TRADEOFF (codex r7 [MEDIUM CONTRACT]): the dedup
            # contract for destructive terminal handlers is silently
            # downgraded in this code path — we ACK with no durable marker
            # iff ``mark_processed`` raises. The cleanest fix is making
            # ``side_effect`` + ``mark_processed`` atomic in the same DB
            # transaction (e.g., the PR-4 terminal handler writes both the
            # destroy outcome and the audit marker in a single SQLAlchemy
            # tx). PR-4 scope MUST address this — tracked in TODO2.md (C3
            # PR-4 acceptance gate). Until then, the at-most-once dedup-loss
            # is the lesser evil vs guaranteed double-destroy on redelivery.
            try:
                await self._ctx.audit_repo.mark_processed(
                    envelope.parent_session_id,
                    envelope.envelope_id,
                    processed_at=self._ctx.now(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "audit_repo.mark_processed failed envelope=%s — ACK "
                    "anyway to avoid double-fire on next redelivery; "
                    "PR-4 should make side_effect+mark_processed transactional",
                    envelope.envelope_id,
                )
            await self._consumer.ack(redis_id)
            return

        # No side_effect — honor ``outcome.ack`` as the explicit ACK/defer
        # signal. ``ack=False`` is the legitimate "defer" path PR-3b reliability
        # layer relies on (e.g., dedup hit waiting for the in-flight handler).
        if outcome.ack:
            await self._consumer.ack(redis_id)

    # ──────────────────────────────────────────────────────────────────────
    # PR-3b Phase A — heartbeat last_seen gate (spec §9.2)
    # ──────────────────────────────────────────────────────────────────────

    def _is_child_origin(self, envelope: MailboxEnvelope) -> bool:
        """Spec §9.2 — only envelopes that (i) belong to a child session,
        (ii) are in the child→parent direction (CHILD_TO_PARENT_TYPES), and
        (iii) were produced by the child agent itself (NOT the supervisor's
        echo / external publisher) tell us the child is alive.
        """
        return (
            bool(envelope.child_session_id)
            and envelope.type in CHILD_TO_PARENT_TYPES
            and envelope.producer_role == ProducerRole.CHILD_AGENT
        )

    async def _refresh_last_seen_if_child_origin(
        self, envelope: MailboxEnvelope
    ) -> None:
        if self._is_child_origin(envelope):
            self._last_seen_mono[envelope.child_session_id] = self._ctx.clock()

    def get_last_seen(self, child_session_id: str) -> Optional[float]:
        """Public read accessor for the orphan-watch / cascade tick logic
        (PR-4 wires the consumer). Returns ``None`` if we've never observed
        a child-origin envelope for this child on this supervisor instance.
        """
        return self._last_seen_mono.get(child_session_id)

    # ──────────────────────────────────────────────────────────────────────
    # PR-3b Phase B — XAUTOCLAIM startup + periodic (spec §5.6)
    # ──────────────────────────────────────────────────────────────────────

    async def _initial_xautoclaim(self) -> None:
        """Spec §5.6 startup XAUTOCLAIM — bring orphaned PEL from prior dead
        consumer into this supervisor's ownership.

        Startup sweep claims every entry in PEL regardless of idle time
        (``min_idle_ms=0``). Used when this supervisor instance takes over a
        root from a dead consumer (the dead consumer can't ACK, so XPENDING
        will still show its idle entries). ``count=1000`` is a soft batch
        cap; if there are more we'll catch the rest on the periodic tick.

        INVARIANT: ``min_idle_ms=0`` relies on the M1 single-writer-per-root
        invariant (one MailboxSupervisor per ``root_session_id`` globally,
        enforced by PR-3c's agent_task_runner integration + reconcile_orphans).
        If M1 is violated (e.g., overlap during rolling pod restart with a
        stale supervisor still active on a different pod), this WILL steal
        entries from the active sibling. The ``SupervisorRegistry`` is
        per-pod only; cross-pod M1 enforcement is the deploy-time
        responsibility (pod lifecycle hooks + reconcile_orphans).
        Spec-design decision per plan §5.6; codex r2 [P1] verified as not an
        impl bug — documenting reliance here so future operators don't relax
        ``min_idle_ms`` without first re-evaluating the M1 contract.
        """
        try:
            claimed = await self._consumer.autoclaim(
                min_idle_ms=0,  # see INVARIANT above — M1 ensures sibling is dead
                count=1000,
            )
            for redis_id, envelope in claimed:
                # Codex r2 [P1] fix — ensure audit row exists BEFORE
                # ``increment_reclaim``. The DB impl raises ``ValueError`` on
                # missing row (see ``db_mailbox_envelope_audit_repository.py``
                # lines 141-145). Without this upsert the broad-except below
                # would mask that ValueError, falling back to
                # ``envelope.reclaim_count + 1`` — but ``envelope.reclaim_count``
                # is fresh-parsed from the stream entry on every XAUTOCLAIM
                # sweep, so the fallback would forever return 1 and the poison
                # drop gate (``> MAILBOX_POISON_MAX_RECLAIM``) could never trip.
                # ``upsert_processing`` is a real PG ``ON CONFLICT DO UPDATE``
                # (idempotent — see same file lines 63-98), so calling it
                # before every reclaim is safe even when the row already
                # exists.
                #
                # Codex r1 [P0] fix (preserved) — persist reclaim_count via
                # audit_repo. Without DB persistence the in-memory
                # ``model_copy`` increment was lost on the next XAUTOCLAIM
                # sweep (the stream entry's raw envelope is re-parsed each
                # time), so poison drop never fired. Initial sweep entries
                # are by definition reclaims (the dead consumer couldn't
                # ACK), so each claimed entry counts as a retry for the
                # poison threshold.
                try:
                    await self._ctx.audit_repo.upsert_processing(
                        envelope, processing_at=self._ctx.now()
                    )
                    persisted_count = await self._ctx.audit_repo.increment_reclaim(
                        envelope.parent_session_id,
                        envelope.envelope_id,
                        "initial_xautoclaim_after_restart",
                    )
                except Exception:
                    logger.exception(
                        "audit_repo upsert+increment_reclaim failed envelope=%s "
                        "— best-effort fallback to in-memory bump",
                        envelope.envelope_id,
                    )
                    persisted_count = envelope.reclaim_count + 1
                envelope = envelope.model_copy(
                    update={"reclaim_count": persisted_count}
                )
                # Codex r3 [P1] fix — per-entry exception isolation. XAUTOCLAIM
                # has already transferred ownership of the entire claimed batch
                # to THIS supervisor; if ``_handle_envelope`` raises for one
                # entry (e.g., XACK transient failure) and the loop aborts,
                # the rest of the batch is now pending under this consumer
                # and will NOT be redelivered via XREADGROUP ``>``. The skipped
                # entries would only resurface on the next XAUTOCLAIM sweep
                # after ``MAILBOX_PEL_IDLE_MS_FOR_CLAIM`` (60s default), so
                # one transient failure could starve up to 999 startup-claimed
                # orphans for a minute. Mirror the main-loop pattern (codex
                # r10 [P1] fix in PR-3a): re-raise CancelledError, broadly
                # log+continue everything else.
                try:
                    await self._handle_envelope(redis_id, envelope)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "initial XAUTOCLAIM entry handling failed root=%s "
                        "id=%s type=%s — leave in PEL, continue batch",
                        self._ctx.root_session_id,
                        redis_id,
                        envelope.type.value,
                    )
            # Codex r2 [P2] + r5 [P2] fix — anchor the periodic-tick clock so
            # the first ``_maybe_periodic_xautoclaim`` call waits one full
            # interval instead of firing immediately. ``time.monotonic()``
            # returns process-uptime in seconds (typically >> 30 immediately
            # after boot of any non-trivial process), so leaving
            # ``self._last_autoclaim_mono`` at ``0.0`` causes the periodic
            # tick to fire on the very first loop iteration — doubling the
            # XAUTOCLAIM work right after ``_initial_xautoclaim`` (which
            # already claimed everything with ``min_idle=0``).
            #
            # r5 [P2]: anchor now lands BEFORE the closing telemetry emit.
            # If telemetry raises (network blip, sink misconfig), the outer
            # broad-except below catches it and the anchor would otherwise
            # be skipped — re-opening r2's premature-tick bug via the
            # observability path. Placement is still INSIDE the try so an
            # earlier XAUTOCLAIM failure leaves the anchor at 0.0 (degrades
            # safely — periodic attempts ASAP).
            self._last_autoclaim_mono = self._ctx.clock()
            # Codex r5 [P2] — telemetry.emit is best-effort here. Isolating
            # it separately means a sink crash doesn't get re-logged as
            # "initial XAUTOCLAIM failed" via the outer broad-except, which
            # would be misleading observability for an observability fault.
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.initial_autoclaim_completed",
                    {"root": self._ctx.root_session_id, "count": len(claimed)},
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "initial_autoclaim_completed telemetry raised root=%s "
                    "— continuing without the emit",
                    self._ctx.root_session_id,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "initial XAUTOCLAIM failed root=%s", self._ctx.root_session_id
            )

    async def _maybe_periodic_xautoclaim(self) -> None:
        """Periodic sweep — runs every ``_XAUTOCLAIM_INTERVAL_S`` seconds
        from the main loop. Uses ``MAILBOX_PEL_IDLE_MS_FOR_CLAIM`` so we
        don't steal entries the original consumer might still be processing.
        """
        now = self._ctx.clock()
        if now - self._last_autoclaim_mono < self._XAUTOCLAIM_INTERVAL_S:
            return
        self._last_autoclaim_mono = now
        try:
            claimed = await self._consumer.autoclaim(
                min_idle_ms=self._XAUTOCLAIM_MIN_IDLE_MS,
                count=100,
            )
            for redis_id, envelope in claimed:
                # Codex r2 [P1] fix — ensure audit row exists BEFORE
                # ``increment_reclaim``. The DB impl raises ``ValueError``
                # on missing row (``db_mailbox_envelope_audit_repository.py``
                # lines 141-145). Without this upsert the broad-except below
                # would mask that ValueError, falling back to
                # ``envelope.reclaim_count + 1`` which is forever 1 (the
                # stream entry's raw envelope is fresh-parsed each sweep),
                # so the poison drop gate (``> MAILBOX_POISON_MAX_RECLAIM``)
                # could never trip. ``upsert_processing`` is a real PG
                # ``ON CONFLICT DO UPDATE`` (idempotent — see same file
                # lines 63-98), safe to call before every reclaim.
                #
                # Codex r1 [P0] fix (preserved) — persist reclaim_count via
                # audit_repo. See ``_initial_xautoclaim`` for the full
                # rationale: the prior ``model_copy`` rebind was a transient
                # bump that never made it back to the stream entry, so
                # subsequent XAUTOCLAIM sweeps re-parsed the original
                # ``reclaim_count`` and the poison drop gate
                # (``> MAILBOX_POISON_MAX_RECLAIM``) was effectively dead
                # code in production.
                try:
                    await self._ctx.audit_repo.upsert_processing(
                        envelope, processing_at=self._ctx.now()
                    )
                    persisted_count = await self._ctx.audit_repo.increment_reclaim(
                        envelope.parent_session_id,
                        envelope.envelope_id,
                        "xautoclaim_redelivered",
                    )
                except Exception:
                    logger.exception(
                        "audit_repo upsert+increment_reclaim failed envelope=%s "
                        "— best-effort fallback to in-memory bump",
                        envelope.envelope_id,
                    )
                    persisted_count = envelope.reclaim_count + 1
                envelope = envelope.model_copy(
                    update={"reclaim_count": persisted_count}
                )
                # Codex r3 [P1] fix — per-entry exception isolation. Same
                # rationale as ``_initial_xautoclaim``: XAUTOCLAIM transferred
                # ownership of the batch to THIS supervisor; aborting the
                # loop on one entry's failure would leave the rest pending
                # under this consumer (not redelivered via XREADGROUP ``>``)
                # for another ``MAILBOX_PEL_IDLE_MS_FOR_CLAIM`` cycle.
                try:
                    await self._handle_envelope(redis_id, envelope)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "periodic XAUTOCLAIM entry handling failed root=%s "
                        "id=%s type=%s — leave in PEL, continue batch",
                        self._ctx.root_session_id,
                        redis_id,
                        envelope.type.value,
                    )
            await self._ctx.telemetry.emit(
                "mailbox.periodic_autoclaim_tick",
                {"root": self._ctx.root_session_id, "count": len(claimed)},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "periodic XAUTOCLAIM failed root=%s", self._ctx.root_session_id
            )

    # ──────────────────────────────────────────────────────────────────────
    # PR-3b Phase C — poison drop hook (spec §5.7)
    # ──────────────────────────────────────────────────────────────────────

    async def _on_poison_drop(self, envelope: MailboxEnvelope) -> None:
        """Hook for PR-4 to trigger orphan cascade when a *terminal-type*
        poison envelope is dropped while its child is still RUNNING.

        PR-3b ships the empty body — telemetry alone surfaces the drop in
        the spec §5.7 step 1/2 path. PR-4 overrides to call the
        cancel-cascade handler so the orphaned child is destroyed.
        """
        return None

    # ──────────────────────────────────────────────────────────────────────
    # PR-3b Phase D — pod-restart clock recovery (spec §9.3)
    # ──────────────────────────────────────────────────────────────────────

    async def _restore_last_seen_after_pod_restart(self) -> None:
        """Spec §9.3 — explicit Redis-time-anchored clock recovery.

        After a pod restart, monotonic clocks reset to 0. We can't trust the
        envelope's ``emitted_at`` (wall-clock from a possibly-skewed publisher),
        so we use the redis stream's entry id (millis since Unix epoch on
        the Redis server) as the reference point:

            age_ms = redis_now_ms - entry_ms        # how long ago the envelope was added
            last_seen_mono = clock_now - age_ms/1000  # convert into local monotonic frame

        PR-3c will wire this in ``SupervisorRegistry.spawn`` (after
        populating ``_known_children`` from the session repo at supervisor
        startup). PR-3b ships the function unused — the unit tests cover
        it in isolation; the registry integration is deferred so we don't
        spread session-repo dependencies into PR-3b scope.

        Idempotent — if XREVRANGE / TIME both fail we just skip; the next
        child-origin envelope will refresh ``_last_seen_mono`` naturally.
        """
        now_mono = self._ctx.clock()
        try:
            redis_time = await self._ctx.redis.time()  # (seconds, microseconds)
            redis_now_ms = redis_time[0] * 1000 + redis_time[1] // 1000
        except Exception:
            logger.warning(
                "redis TIME failed during clock recovery root=%s — skipping",
                self._ctx.root_session_id,
            )
            return

        stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
            root_session_id=self._ctx.root_session_id
        )
        for child_id in self._known_children:
            entry_ms = await self._latest_child_origin_entry_ms(
                stream_key, child_id
            )
            if entry_ms is None:
                continue
            age_ms = max(0, redis_now_ms - entry_ms)
            self._last_seen_mono[child_id] = now_mono - (age_ms / 1000.0)

    async def _latest_child_origin_entry_ms(
        self, stream_key: str, child_id: str
    ) -> Optional[int]:
        """XREVRANGE scan for the latest child-origin envelope for one
        child_id. Returns the millisecond component of the Redis stream
        entry id (which is ``<ms>-<seq>``).

        Caps at 100 entries scanned per child — assumes a per-child
        progress cadence faster than that across the recent stream history,
        which the spec §4.3 SUBAGENT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS=15
        guarantees in practice.
        """
        try:
            entries = await self._ctx.redis.xrevrange(stream_key, count=100)
        except Exception:
            return None
        for redis_id, fields in entries:
            raw = fields.get(b"envelope") or fields.get("envelope")
            if raw is None:
                continue
            try:
                env = MailboxEnvelope.model_validate_json(
                    raw.decode() if isinstance(raw, bytes) else raw
                )
            except Exception:
                continue
            if (
                env.child_session_id == child_id
                and env.type in CHILD_TO_PARENT_TYPES
                and env.producer_role == ProducerRole.CHILD_AGENT
            ):
                id_str = (
                    redis_id.decode()
                    if isinstance(redis_id, bytes)
                    else redis_id
                )
                ms_part = id_str.split("-")[0]
                try:
                    return int(ms_part)
                except ValueError:
                    return None
        return None
