"""C2 PR-3 §7.5 — CoordinatorTerminalEnvelopeWaiter.

Awaits the terminal envelope (RESULT_READY | CANCEL_ACK) for a given child
session from the root-scoped mailbox stream, using an **independent** consumer
group ``coordinator:waiter:{child_session_id}`` so the supervisor's group
(``actus:mailbox-supervisor:v1``) still receives its own copy + does
destroy/persist/mark_processed.

XREADGROUP semantics — Redis delivers each message once per group; multiple
groups receive independent copies. The waiter must not use ``RedisMailboxConsumer``
directly because that consumer is hard-wired to the supervisor's fixed group
name (would race the supervisor for terminal envelope delivery).

With a None/zero timeout the waiter consumes until a terminal envelope arrives.
Callers that opt into a positive timeout are bounded via ``asyncio.wait`` so an
inner ``TimeoutError`` remains distinguishable from the waiter's own deadline.
``cancel_event`` is informational (the child uses it to stop + emit CANCEL_ACK);
the waiter itself keeps consuming until terminal arrives or an explicit timeout
fires.

[Round 7 P2] After the waiter's job is done (terminal arrives, explicit timeout,
caller cancellation, or exception), the per-waiter consumer group is destroyed
in a ``finally`` block so dead groups don't accumulate under the long-lived root
mailbox stream.
The destroy is gated on a ``subscribed=True`` flag — if ``subscribe`` itself
raised, no group exists yet and ``destroy_group`` would emit a spurious
NOGROUP roundtrip. ``destroy_group`` failures are logged + swallowed so the
caller-facing exception (TimeoutError, etc.) propagates unmodified.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Awaitable, Callable, Mapping

from app.domain.external.mailbox_subscriber import MailboxSubscriber
from app.domain.models.mailbox_envelope import MailboxEnvelope, MailboxEnvelopeType


logger = logging.getLogger(__name__)


class CoordinatorTerminalEnvelopeRejected(ValueError):
    """A parsed terminal candidate does not match the expected run identity."""


_RAW_TERMINAL_TYPES = frozenset({
    MailboxEnvelopeType.RESULT_READY.value,
    MailboxEnvelopeType.CANCEL_ACK.value,
})


def _raw_terminal_identity_matches(
    value: object,
    *,
    child_session_id: str,
    root_session_id: str,
    coordinator_run_id: str,
) -> bool:
    """Non-throwing prefilter for RedisMailboxSubscriber predicates."""
    if not isinstance(value, Mapping):
        return False
    try:
        return bool(
            value.get("type") in _RAW_TERMINAL_TYPES
            and value.get("child_session_id") == child_session_id
            and value.get("parent_session_id") == root_session_id
            and value.get("correlation_id") == coordinator_run_id
        )
    except Exception:
        return False


def _validate_expected_terminal(
    value: MailboxEnvelope | Mapping[str, Any],
    *,
    child_session_id: str,
    root_session_id: str,
    coordinator_run_id: str,
) -> MailboxEnvelope:
    """Parse and validate the full identity shared by stream and DB recovery."""
    candidate = (
        value.model_dump(mode="python")
        if isinstance(value, MailboxEnvelope)
        else value
    )
    envelope = MailboxEnvelope.model_validate(candidate)
    if envelope.type not in {
        MailboxEnvelopeType.RESULT_READY,
        MailboxEnvelopeType.CANCEL_ACK,
    }:
        raise CoordinatorTerminalEnvelopeRejected(
            "terminal envelope has a non-terminal type"
        )
    if envelope.child_session_id != child_session_id:
        raise CoordinatorTerminalEnvelopeRejected(
            "terminal envelope child identity mismatch"
        )
    if envelope.parent_session_id != root_session_id:
        raise CoordinatorTerminalEnvelopeRejected(
            "terminal envelope root identity mismatch"
        )
    if envelope.correlation_id != coordinator_run_id:
        raise CoordinatorTerminalEnvelopeRejected(
            "terminal envelope coordinator run identity mismatch"
        )
    return envelope


def _normalize_timeout(timeout: float | None) -> float | None:
    """Return a finite positive deadline, or ``None`` for unlimited."""
    if timeout is None or timeout == 0:
        return None
    if timeout < 0 or not math.isfinite(timeout):
        raise ValueError("timeout must be None, zero, or a finite positive number")
    return float(timeout)


class CoordinatorTerminalEnvelopeWaiter:
    def __init__(
        self,
        *,
        subscriber: MailboxSubscriber,
        liveness_service: Any | None = None,
        orphan_reconciler: Callable[..., Awaitable[None]] | None = None,
        persisted_terminal_reader: Callable[..., Awaitable[Any | None]] | None = None,
    ) -> None:
        self._subscriber = subscriber
        self._liveness = liveness_service
        self._orphan_reconciler = orphan_reconciler
        self._persisted_terminal_reader = persisted_terminal_reader

    async def await_terminal(
        self,
        *,
        child_session_id: str,
        root_session_id: str,
        cancel_event: asyncio.Event,
        coordinator_run_id: str,
        timeout: float | None = None,
    ) -> MailboxEnvelope:
        """Block until a terminal envelope arrives for this child.

        Returns the validated ``MailboxEnvelope``. ``None`` and zero mean
        unlimited and await the subscriber directly. A finite positive timeout
        uses ``asyncio.wait`` and raises ``asyncio.TimeoutError`` when no
        terminal envelope is observed within that interval. Negative and
        non-finite values are rejected before subscribing.

        ``coordinator_run_id`` is required and the predicate matches
        ``env['correlation_id']`` so the waiter never accepts a stale
        RESULT_READY from a different
        coordinator run sharing the same child_session_id (e.g. PR-7
        rehydrate of a different attempt, or a re-issued child). PR-5
        applies patch_manifests from these envelopes directly to the
        parent sandbox, so a mismatched run_id would cross-contaminate
        apply plans.
        """
        normalized_timeout = _normalize_timeout(timeout)
        stream_key = f"actus:child:{root_session_id}:mailbox"
        consumer_group = f"coordinator:waiter:{child_session_id}"
        consumer_name = f"waiter-{child_session_id}"
        subscribed = False
        await self._subscriber.subscribe(
            stream_key=stream_key,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
        )
        subscribed = True

        async def _is_terminal_for_this_child(env: object) -> bool:
            return _raw_terminal_identity_matches(
                env,
                child_session_id=child_session_id,
                root_session_id=root_session_id,
                coordinator_run_id=coordinator_run_id,
            )

        async def _consume_one() -> MailboxEnvelope:
            async for env_dict in self._subscriber.consume(
                stream_key=stream_key,
                consumer_group=consumer_group,
                consumer_name=consumer_name,
                predicate=_is_terminal_for_this_child,
            ):
                return _validate_expected_terminal(
                    env_dict,
                    child_session_id=child_session_id,
                    root_session_id=root_session_id,
                    coordinator_run_id=coordinator_run_id,
                )
            raise asyncio.TimeoutError(
                f"subscriber exhausted before terminal for {child_session_id}"
            )

        async def _read_persisted_terminal() -> MailboxEnvelope | None:
            reader = self._persisted_terminal_reader
            if reader is None:
                return None
            value = await reader(
                child_session_id=child_session_id,
                root_session_id=root_session_id,
                coordinator_run_id=coordinator_run_id,
            )
            if value is None:
                return None
            if not isinstance(value, (MailboxEnvelope, Mapping)):
                raise TypeError("persisted terminal reader returned unsupported value")
            return _validate_expected_terminal(
                value,
                child_session_id=child_session_id,
                root_session_id=root_session_id,
                coordinator_run_id=coordinator_run_id,
            )

        async def _cancel_and_drain(task: asyncio.Task[Any]) -> None:
            if not task.done():
                task.cancel()
            try:
                await task
            except BaseException:
                pass

        async def _consume_vs_stale() -> MailboxEnvelope:
            terminal_task = asyncio.create_task(_consume_one())
            stale_task = asyncio.create_task(
                self._liveness.await_stale(child_session_id)
            )
            try:
                done, _ = await asyncio.wait(
                    {terminal_task, stale_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # A real stream terminal is already authoritative. If both
                # complete in the same loop turn, this branch wins without an
                # orphan side effect.
                if terminal_task in done:
                    return terminal_task.result()

                # Stale detection is an authority read, not a best-effort
                # hint. Propagate Redis/decode/cancellation failures before
                # consulting persisted state or emitting an orphan action.
                stale_task.result()

                # Consumer lag and the terminal DB transaction can race the
                # stale edge. Consult persistence before any orphan action.
                persisted = await _read_persisted_terminal()
                if persisted is not None:
                    return persisted

                # Persistence I/O may have yielded long enough for the stream
                # terminal (or its failure) to finish. It remains authoritative
                # and must win before any orphan side effect.
                if terminal_task.done():
                    return terminal_task.result()

                if self._orphan_reconciler is not None:
                    await self._orphan_reconciler(
                        child_session_id=child_session_id,
                        root_session_id=root_session_id,
                        coordinator_run_id=coordinator_run_id,
                    )
                # Staleness never fabricates a TIMED_OUT result. The existing
                # supervisor cascade eventually publishes authoritative
                # CANCEL_ACK, which this pre-created group consumes.
                return await terminal_task
            finally:
                await _cancel_and_drain(terminal_task)
                await _cancel_and_drain(stale_task)

        try:
            consume = (
                _consume_one
                if self._liveness is None
                else _consume_vs_stale
            )
            if normalized_timeout is None:
                return await consume()
            consume_task = asyncio.create_task(consume())
            try:
                done, _ = await asyncio.wait(
                    {consume_task},
                    timeout=normalized_timeout,
                )
                if consume_task in done:
                    return consume_task.result()
                raise asyncio.TimeoutError(
                    f"no terminal envelope for {child_session_id} "
                    f"after {normalized_timeout}s"
                )
            finally:
                await _cancel_and_drain(consume_task)
        finally:
            # [Round 7 P2] Destroy the per-waiter consumer group so it doesn't
            # accumulate as a dead XPENDING/group-metadata entry under the
            # long-lived root stream. Skip when ``subscribed=False`` (subscribe
            # itself raised — nothing to destroy). Best-effort: log + swallow
            # on any error — the group is per-waiter and idempotent destroy is
            # the adapter contract.
            if subscribed:
                try:
                    await self._subscriber.destroy_group(
                        stream_key=stream_key,
                        consumer_group=consumer_group,
                    )
                except Exception:
                    logger.warning(
                        "CoordinatorTerminalEnvelopeWaiter: destroy_group"
                        " failed child=%s group=%s — dead group may accumulate"
                        " in Redis",
                        child_session_id, consumer_group,
                        exc_info=True,
                    )
