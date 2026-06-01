"""C2 PR-6 §11.3-§11.4 + §11.8 — CoordinatorRunOrchestrator.

PR-6 fleshes out the PR-3 skeleton with:
  - ``should_trigger_sibling_cancel(envelope)`` predicate (spec §11.8 r15) —
    the 5-case invariant matrix that decides whether one work-unit's terminal
    envelope should fan out CANCEL_REQUEST to its remaining siblings.
  - Observer + cancel-watcher concurrent loops driven by ``asyncio.wait``
    with ``FIRST_COMPLETED`` semantics, gated by ``timeout_seconds``.
  - RESULT_READY / CANCEL_ACK observation via an injected
    ``MailboxSubscriber`` port; envelopes arrive as Redis wire dicts and
    are re-validated through ``MailboxEnvelope.model_validate`` so the
    payload-normalization invariant fires before the predicate inspects it.
  - ``emit_event`` async hook (PR-8 prep) that fires exactly when sibling
    cancel is triggered.
  - ``_published`` deduplication so the same wu_id cannot have CANCEL_REQUEST
    issued twice from observer + watcher races (spec §11.5).

Backward compatibility: when ``mailbox_subscriber=None`` (or omitted) the
orchestrator preserves the PR-3 parent-cancel-only loop. PR-3 ``run()``
semantics ("all-failed → raise RuntimeError", per-envelope failure swallowed,
empty pending no-op) are preserved on that path.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from app.application.services.coordinator_envelope_factory import (
    CoordinatorEnvelopeFactory,
)
from app.domain.external.mailbox_publisher import MailboxPublisher
from app.domain.external.mailbox_subscriber import MailboxSubscriber
from app.domain.models.mailbox_envelope import (
    MAILBOX_STREAM_KEY_TEMPLATE,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ResultReadyOutcome,
)

logger = logging.getLogger(__name__)


# ── Sibling-cancel predicate (spec §11.8 r15) ────────────────────────────────


_HARD_TERMINAL_OUTCOMES: frozenset[ResultReadyOutcome] = frozenset(
    {
        ResultReadyOutcome.FAILED,
        ResultReadyOutcome.TIMED_OUT,
        ResultReadyOutcome.CANCELLED,
        ResultReadyOutcome.NEEDS_AUTHORIZATION,
    }
)


def _coerce_outcome(raw: Any) -> Optional[ResultReadyOutcome]:
    """Normalize ``payload['outcome']`` to a ``ResultReadyOutcome`` enum.

    Defensive: ``MailboxEnvelope._validate_payload_matches_type`` round-trips
    the payload via ``model_dump(mode="python")`` which leaves enums as enum
    instances, but a caller may hand us a still-on-wire dict where the
    outcome arrived as a plain ``str`` (e.g. ``"failed"``). Both shapes must
    classify identically.
    """
    if isinstance(raw, ResultReadyOutcome):
        return raw
    if isinstance(raw, str):
        try:
            return ResultReadyOutcome(raw)
        except ValueError:
            return None
    return None


def _extract_outcome(payload: Any) -> Optional[ResultReadyOutcome]:
    """Read ``outcome`` from a payload that may be a dict OR a Pydantic model.

    The post-validate canonical shape is dict (see
    ``_validate_payload_matches_type``), but the predicate is also called
    from places that hand in a raw ``ResultReadyPayload`` instance — handle
    both defensively.
    """
    if payload is None:
        return None
    raw = None
    if isinstance(payload, dict):
        raw = payload.get("outcome")
    else:
        raw = getattr(payload, "outcome", None)
    return _coerce_outcome(raw)


def _extract_needs_auth_reason(payload: Any) -> Optional[str]:
    """Return ``needs_authorization_details.reason`` if present.

    Like ``_extract_outcome``, handles both dict (post-validate) and
    Pydantic model (pre-validate / direct call) shapes. Returns ``None``
    if the field is absent or has no ``reason``.
    """
    if payload is None:
        return None
    details: Any
    if isinstance(payload, dict):
        details = payload.get("needs_authorization_details")
    else:
        details = getattr(payload, "needs_authorization_details", None)
    if details is None:
        return None
    if isinstance(details, dict):
        reason = details.get("reason")
    else:
        reason = getattr(details, "reason", None)
    return reason if isinstance(reason, str) else None


def should_trigger_sibling_cancel(envelope: MailboxEnvelope) -> bool:
    """[spec §11.8 r15] Decide whether to fan out CANCEL_REQUEST to siblings.

    Returns ``True`` iff the envelope is a RESULT_READY with a hard-terminal
    outcome, with one carve-out: ``NEEDS_AUTHORIZATION`` with
    ``reason == "exploration_proposal"`` is the orderly hand-off path
    (planner-facing write plan), not a failure — siblings must continue.

    5-case matrix:
      1. RESULT_READY(SUCCESS)                        → False
      2. RESULT_READY(FAILED|TIMED_OUT|CANCELLED)     → True
      3. RESULT_READY(NEEDS_AUTHORIZATION,
           reason="exploration_proposal")             → False  (carve-out)
      4. RESULT_READY(NEEDS_AUTHORIZATION, other)     → True
      5. CANCEL_ACK / SPAWN_ACK / others              → False
    """
    if envelope.type != MailboxEnvelopeType.RESULT_READY:
        return False
    outcome = _extract_outcome(envelope.payload)
    if outcome is None or outcome not in _HARD_TERMINAL_OUTCOMES:
        return False
    if outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION:
        reason = _extract_needs_auth_reason(envelope.payload)
        if reason == "exploration_proposal":
            return False
    return True


# ── Orchestrator ─────────────────────────────────────────────────────────────


# [C2 PR-8 §13 Task 8.4] Widened from ``Callable[[dict], Awaitable[None]]``
# (PR-6 placeholder shape) to ``Callable[[Any], Awaitable[None]]`` so the
# orchestrator can emit typed ``CoordinatorSiblingCancelEvent``. The
# composition root binds this to ``event_queue.put`` which accepts any
# domain event; the only previous caller passed an ad-hoc dict.
EmitEvent = Callable[[Any], Awaitable[None]]


class CoordinatorRunOrchestrator:
    """Owns the parent-side observation + cancel orchestration for a single
    coordinator run.

    PR-3 contract (preserved):
      - ``parent_session_id`` + ``coordinator_run_id`` non-empty in ctor
        (publisher derives Redis stream key from envelope.parent_session_id).
      - ``run()`` with ``mailbox_subscriber=None`` waits ``cancel_event`` with
        timeout, fans out CANCEL_REQUEST × pending. All-failed publishes raise.
      - ``shutdown()`` is a no-op (background tasks are scoped to ``run()``).

    PR-6 extension (subscriber wired):
      - Spawns an ``observer_loop`` over
        ``actus:child:{root_session_id}:mailbox`` with consumer group
        ``coordinator:{coordinator_run_id}`` and consumer name
        ``orch-{coordinator_run_id}``.
      - Each RESULT_READY / CANCEL_ACK matching a pending wu discards that
        wu from ``pending``; if the envelope triggers
        ``should_trigger_sibling_cancel`` the remaining pending siblings are
        cancelled with ``reason=f"sibling_terminal_{outcome.value}"``.
      - A second ``cancel_watcher`` task awaits ``cancel_event``; on fire it
        cancels all remaining pending with ``reason="parent_cancel"``.
      - Both tasks run under ``asyncio.wait(..., timeout=timeout_seconds,
        return_when=FIRST_COMPLETED)`` and the loser is cancelled + drained
        via ``gather(return_exceptions=True)``.
      - ``_published`` set + ``_publish_lock`` ensure single-publish-per-wu
        across the two paths (spec §11.5).
    """

    def __init__(
        self,
        *,
        publisher: MailboxPublisher,
        parent_session_id: str,
        coordinator_run_id: str,
        envelope_factory: Optional[CoordinatorEnvelopeFactory] = None,
        mailbox_subscriber: Optional[MailboxSubscriber] = None,
        emit_event: Optional[EmitEvent] = None,
    ) -> None:
        """r3 P1-2: ``parent_session_id`` / ``coordinator_run_id`` are required
        non-empty. Live publisher derives the Redis stream key from
        ``envelope.parent_session_id`` (see
        ``RedisMailboxPublisher.publish``); an empty string would route
        CANCEL_REQUEST to ``actus:child::mailbox`` and never reach the child.

        PR-6 added kwargs:
          - ``mailbox_subscriber``: when ``None`` (default) ``run()`` falls
            back to the PR-3 parent-cancel-only path so existing callers
            need not change. When set, the full observer/watcher loop is
            wired.
          - ``emit_event``: optional async hook fired exactly when sibling
            cancel decision is made (NOT on success terminal, NOT on
            ``cancel_event`` fire).
        """
        if not parent_session_id:
            raise ValueError(
                "CoordinatorRunOrchestrator: parent_session_id must be non-empty"
            )
        if not coordinator_run_id:
            raise ValueError(
                "CoordinatorRunOrchestrator: coordinator_run_id must be non-empty"
            )
        self._publisher = publisher
        self._published: set[str] = set()
        self._publish_lock = asyncio.Lock()
        self._envelope_factory = envelope_factory or CoordinatorEnvelopeFactory()
        self._parent_session_id = parent_session_id
        self._coordinator_run_id = coordinator_run_id
        self._subscriber = mailbox_subscriber
        self._emit_event = emit_event

    async def run(
        self,
        *,
        coordinator_run_id: str,
        root_session_id: str,
        work_units_pending: list[str],
        child_session_ids: dict[str, str],
        cancel_event: asyncio.Event,
        timeout_seconds: float = 600.0,
        observer_group_precreated: bool = False,
    ) -> None:
        """Drive the coordinator run loop.

        - ``subscriber is None`` → PR-3 parent-cancel-only path (preserved
          verbatim, including "all-failed → raise RuntimeError" semantics).
        - ``subscriber is set`` → PR-6 observer + cancel_watcher concurrent
          loops gated by ``timeout_seconds``.

        ``observer_group_precreated`` (finish-core §5.4 G4-min): when True,
        ``dispatch_node`` already created the observer consumer group
        SYNCHRONOUSLY before any child task launched, so ``_run_with_observer``
        must NOT subscribe again. Only forwarded to the observer path.
        """
        if self._subscriber is None:
            await self._run_parent_cancel_only(
                coordinator_run_id=coordinator_run_id,
                work_units_pending=work_units_pending,
                child_session_ids=child_session_ids,
                cancel_event=cancel_event,
                timeout_seconds=timeout_seconds,
            )
            return

        await self._run_with_observer(
            coordinator_run_id=coordinator_run_id,
            root_session_id=root_session_id,
            work_units_pending=work_units_pending,
            child_session_ids=child_session_ids,
            cancel_event=cancel_event,
            timeout_seconds=timeout_seconds,
            observer_group_precreated=observer_group_precreated,
        )

    # ── PR-3 path (preserved verbatim from skeleton) ─────────────────────

    async def _run_parent_cancel_only(
        self,
        *,
        coordinator_run_id: str,
        work_units_pending: list[str],
        child_session_ids: dict[str, str],
        cancel_event: asyncio.Event,
        timeout_seconds: float,
    ) -> None:
        """PR-3 parent-cancel loop.

        Wait for ``cancel_event``; on set, publish CANCEL_REQUEST × all
        pending. Timeout → log + return (orchestrator finishes; child
        finalizers report outcome). Per-envelope publish failure is logged
        + swallowed so a single broken child does not stop CANCEL fan-out
        to the rest. If *every* publish fails, raise so the caller's
        done-callback sees the failure instead of silently losing the
        parent cancel.
        """
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.info(
                "CoordinatorRunOrchestrator: timeout (no cancel) for %s",
                coordinator_run_id,
            )
            return

        attempted = 0
        succeeded = 0
        last_exc: Optional[BaseException] = None
        for wu_id in work_units_pending:
            child_sid = child_session_ids.get(wu_id)
            if child_sid is None or wu_id in self._published:
                continue
            attempted += 1
            try:
                envelope = self._envelope_factory.make_cancel_request(
                    parent_session_id=self._parent_session_id,
                    child_session_id=child_sid,
                    correlation_id=coordinator_run_id,
                    reason="parent_cancel",
                )
                await self._publisher.publish(envelope)
                self._published.add(wu_id)
                succeeded += 1
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "CoordinatorRunOrchestrator: cancel publish failed wu=%s child=%s: %s",
                    wu_id, child_sid, exc,
                )
        if attempted > 0 and succeeded == 0:
            raise RuntimeError(
                f"CoordinatorRunOrchestrator: all {attempted} CANCEL_REQUEST "
                f"publishes failed for run {coordinator_run_id}; parent cancel lost. "
                f"Last error: {last_exc!r}"
            )

    # ── PR-6 path (observer + watcher) ───────────────────────────────────

    async def _run_with_observer(
        self,
        *,
        coordinator_run_id: str,
        root_session_id: str,
        work_units_pending: list[str],
        child_session_ids: dict[str, str],
        cancel_event: asyncio.Event,
        timeout_seconds: float,
        observer_group_precreated: bool = False,
    ) -> None:
        # Mutable working copy: observer + watcher both narrow ``pending``
        # as wus are accounted for.
        pending: set[str] = set(work_units_pending)
        # Reverse map child_session_id -> wu_id so the predicate can look
        # up wu membership quickly.
        child_to_wu: dict[str, str] = {
            child_sid: wu for wu, child_sid in child_session_ids.items()
        }

        stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
            root_session_id=root_session_id,
        )
        consumer_group = f"coordinator:{coordinator_run_id}"
        consumer_name = f"orch-{coordinator_run_id}"

        subscriber = self._subscriber
        assert subscriber is not None  # narrowed by caller
        subscribed = True
        if not observer_group_precreated:
            try:
                await subscriber.subscribe(
                    stream_key=stream_key,
                    consumer_group=consumer_group,
                    consumer_name=consumer_name,
                    start_id="$",
                )
            except Exception as exc:
                subscribed = False
                # Subscribe failure is non-fatal for parent cancel: skip the
                # observer (its `consume` would hit NOGROUP and exit silently
                # via the catch-all, which would trip asyncio.wait
                # FIRST_COMPLETED and cancel the watcher before parent_cancel
                # could fire). Only spawn the watcher so cancel_event is still
                # honoured.
                logger.warning(
                    "CoordinatorRunOrchestrator: subscribe failed run=%s: %s",
                    coordinator_run_id, exc,
                )
        # else: dispatch pre-created the group (§5.4 G4-min); ``subscribed``
        # stays True so the observer spawns AND the finally: destroy_group
        # still fires on completion (dispatch owns ONLY the
        # pre-orchestrator-start rollback).

        tasks: list[asyncio.Task[None]] = []
        if subscribed:
            observer_task = asyncio.create_task(
                self._observe_loop(
                    subscriber=subscriber,
                    stream_key=stream_key,
                    consumer_group=consumer_group,
                    consumer_name=consumer_name,
                    pending=pending,
                    child_to_wu=child_to_wu,
                    child_session_ids=child_session_ids,
                    coordinator_run_id=coordinator_run_id,
                    root_session_id=root_session_id,
                ),
                name=f"coord-orch-observer-{coordinator_run_id}",
            )
            tasks.append(observer_task)
        watcher_task = asyncio.create_task(
            self._cancel_watcher(
                cancel_event=cancel_event,
                pending=pending,
                child_session_ids=child_session_ids,
                coordinator_run_id=coordinator_run_id,
            ),
            name=f"coord-orch-watcher-{coordinator_run_id}",
        )
        tasks.append(watcher_task)

        try:
            done, _pending_tasks = await asyncio.wait(
                set(tasks),
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                logger.info(
                    "CoordinatorRunOrchestrator: timeout (no terminal, no cancel)"
                    " run=%s",
                    coordinator_run_id,
                )
        finally:
            # Cancel whichever task hasn't completed + drain exceptions so
            # CancelledError doesn't escape the orchestrator.
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # [codex R2 P1-6] Surface programming-error task exceptions
            # that would otherwise be silently swallowed by
            # ``return_exceptions=True``. ``CancelledError`` from the
            # ``task.cancel()`` above is the normal "loser" outcome and
            # is NOT logged. Anything else (AttributeError, TypeError,
            # KeyError, etc.) indicates a real bug in observer_loop /
            # cancel_watcher and must be visible in logs at ERROR level.
            for task, result in zip(tasks, results):
                if (
                    isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                ):
                    logger.error(
                        "CoordinatorRunOrchestrator: task %s ended with"
                        " exception run=%s",
                        task.get_name(),
                        coordinator_run_id,
                        exc_info=result,
                    )
            # [Round 6 P2] Destroy the per-run consumer group so it doesn't
            # accumulate as a dead XPENDING/group-metadata entry under the
            # long-lived root stream. Skip when ``subscribed=False`` (the
            # group was never created). Best-effort: log + swallow on any
            # error — the group is per-coordinator-run and idempotent
            # destroy is the adapter contract. Positioned AFTER the
            # post-gather exception inspection so log ordering is:
            # task errors logged FIRST, then group destroy attempt.
            if subscribed:
                try:
                    await subscriber.destroy_group(
                        stream_key=stream_key,
                        consumer_group=consumer_group,
                    )
                except Exception:
                    logger.warning(
                        "CoordinatorRunOrchestrator: destroy_group failed"
                        " run=%s group=%s — dead group may accumulate"
                        " in Redis",
                        coordinator_run_id, consumer_group,
                        exc_info=True,
                    )

    async def _observe_loop(
        self,
        *,
        subscriber: MailboxSubscriber,
        stream_key: str,
        consumer_group: str,
        consumer_name: str,
        pending: set[str],
        child_to_wu: dict[str, str],
        child_session_ids: dict[str, str],
        coordinator_run_id: str,
        root_session_id: str,
    ) -> None:
        """Consume mailbox envelopes for this run.

        ``predicate`` is called by the subscriber on each delivered envelope
        dict; we only accept RESULT_READY / CANCEL_ACK whose
        ``child_session_id`` belongs to a pending work unit. For each
        accepted envelope we deserialize via ``MailboxEnvelope.model_validate``
        (which runs the payload normalization model_validator) before
        deciding whether sibling cancel fires.

        Returns naturally when ``pending`` is drained.
        """

        async def predicate(env_dict: dict[str, Any]) -> bool:
            env_type = env_dict.get("type")
            # The wire dict carries the type as the str enum value (after
            # ``model_dump(mode="json")``) OR as the enum instance (under
            # ``mode="python"``); compare against both.
            if env_type not in (
                MailboxEnvelopeType.RESULT_READY,
                MailboxEnvelopeType.RESULT_READY.value,
                MailboxEnvelopeType.CANCEL_ACK,
                MailboxEnvelopeType.CANCEL_ACK.value,
            ):
                return False
            child_id = env_dict.get("child_session_id")
            if not isinstance(child_id, str):
                return False
            wu_id = child_to_wu.get(child_id)
            return wu_id is not None and wu_id in pending

        try:
            async for env_dict in subscriber.consume(
                stream_key=stream_key,
                consumer_group=consumer_group,
                consumer_name=consumer_name,
                predicate=predicate,
            ):
                try:
                    envelope = MailboxEnvelope.model_validate(env_dict)
                except Exception as exc:
                    logger.warning(
                        "CoordinatorRunOrchestrator: malformed envelope dropped"
                        " run=%s: %s",
                        coordinator_run_id, exc,
                    )
                    continue

                wu_id = child_to_wu.get(envelope.child_session_id)
                if wu_id is None or wu_id not in pending:
                    # Race: another path retired this wu while the envelope
                    # was in flight. Skip silently.
                    continue
                pending.discard(wu_id)

                if should_trigger_sibling_cancel(envelope):
                    outcome = _extract_outcome(envelope.payload)
                    await self._fan_out_sibling_cancel(
                        triggered_by_wu=wu_id,
                        outcome=outcome,
                        pending=pending,
                        child_session_ids=child_session_ids,
                        coordinator_run_id=coordinator_run_id,
                        root_session_id=root_session_id,
                    )

                if not pending:
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            # [codex R2 P1-6] Log + re-raise instead of silently exiting.
            # The previous catch-all swallowed programming errors
            # (AttributeError on a typo, KeyError on a missing field,
            # etc.) and quietly terminated the observer, which then
            # tripped ``asyncio.wait(FIRST_COMPLETED)`` and cancelled
            # the watcher — parent_cancel would be lost.
            #
            # By re-raising we surface the bug as a task exception
            # collected by ``return_exceptions=True`` in
            # ``_run_with_observer``'s ``finally``; the post-gather
            # inspection logs it at ERROR with full traceback. The
            # orchestrator's outer ``run()`` still returns normally
            # (so callers of ``await orchestrator.run(...)`` are not
            # surprised by a propagated exception from an in-process
            # background task) but the failure is no longer invisible.
            logger.exception(
                "CoordinatorRunOrchestrator: observer loop errored run=%s",
                coordinator_run_id,
            )
            raise

    async def _cancel_watcher(
        self,
        *,
        cancel_event: asyncio.Event,
        pending: set[str],
        child_session_ids: dict[str, str],
        coordinator_run_id: str,
    ) -> None:
        """Independent watcher: await ``cancel_event`` then fan-out CANCEL
        × remaining pending with ``reason="parent_cancel"``.
        """
        try:
            await cancel_event.wait()
        except asyncio.CancelledError:
            raise
        # Snapshot pending under no lock — set iteration is fine because
        # the observer only ever discards (never adds); even a torn read
        # would only miss a freshly-retired wu which the ``_published``
        # guard already filters out.
        remaining = list(pending)
        await self._publish_cancel_to_pending(
            wu_ids=remaining,
            child_session_ids=child_session_ids,
            coordinator_run_id=coordinator_run_id,
            reason="parent_cancel",
        )

    async def _fan_out_sibling_cancel(
        self,
        *,
        triggered_by_wu: str,
        outcome: Optional[ResultReadyOutcome],
        pending: set[str],
        child_session_ids: dict[str, str],
        coordinator_run_id: str,
        root_session_id: Optional[str] = None,
    ) -> None:
        """Cancel all remaining ``pending`` siblings + fire ``emit_event``."""
        remaining = sorted(pending)
        outcome_label = outcome.value if outcome is not None else "unknown"
        reason = f"sibling_terminal_{outcome_label}"
        cancelled = await self._publish_cancel_to_pending(
            wu_ids=remaining,
            child_session_ids=child_session_ids,
            coordinator_run_id=coordinator_run_id,
            reason=reason,
        )
        # C2 PR-8 §13 Task 8.4 — emit typed CoordinatorSiblingCancelEvent.
        #
        # Guard ``outcome is not None`` because the event schema requires a
        # non-None ResultReadyOutcome (the cascade represents a triggered
        # cancel, and "unknown outcome" is not a meaningful frontend
        # signal). When ``outcome is None`` we still publish the
        # CANCEL_REQUEST envelopes above but skip the SSE event — this is
        # a defensive edge that the observer loop normally avoids by
        # extracting a real outcome before reaching here.
        if (
            self._emit_event is not None
            and cancelled
            and outcome is not None
        ):
            try:
                from app.domain.models.event import (
                    CoordinatorSiblingCancelEvent,
                )
                # [PR-9b-B codex F3 — MEDIUM] Thread lineage onto the only live
                # coordinator event that was previously un-attributed. Now that
                # A6 wired ``emit_event`` to the orchestrator this event reaches
                # SSE; ``CoordinatorSiblingCancelEvent`` mixes in
                # ``CoordinatorLineageMixin`` so it accepts root/parent. The
                # cancel is group-level (like apply/reduce) so child_session_id
                # / work_unit_id stay None. ``parent_session_id`` is the ctor
                # invariant (``self._parent_session_id``, required non-empty);
                # ``root_session_id`` is threaded from ``run(...)`` through the
                # observer loop.
                await self._emit_event(CoordinatorSiblingCancelEvent(
                    triggered_by_work_unit_id=triggered_by_wu,
                    triggered_by_outcome=outcome,
                    cancelled_work_unit_ids=sorted(cancelled),
                    reason=reason,
                    coordinator_run_id=coordinator_run_id,
                    root_session_id=root_session_id,
                    parent_session_id=self._parent_session_id,
                ))
            except Exception as exc:
                logger.warning(
                    "CoordinatorRunOrchestrator: emit "
                    "CoordinatorSiblingCancelEvent raised run=%s: %s",
                    coordinator_run_id, exc,
                )

    async def _publish_cancel_to_pending(
        self,
        *,
        wu_ids: list[str],
        child_session_ids: dict[str, str],
        coordinator_run_id: str,
        reason: str,
    ) -> list[str]:
        """Publish CANCEL_REQUEST × ``wu_ids`` with single-publish dedup
        guarded by ``_publish_lock`` (observer + watcher may race for the
        same wu).

        Returns the list of wu_ids actually published (post-dedup, post
        per-envelope-failure filter).
        """
        published: list[str] = []
        for wu_id in wu_ids:
            child_sid = child_session_ids.get(wu_id)
            if child_sid is None:
                continue
            # Lock-guarded check + insert so observer/watcher can't both
            # pass the dedup check for the same wu.
            async with self._publish_lock:
                if wu_id in self._published:
                    continue
                self._published.add(wu_id)
            envelope = self._envelope_factory.make_cancel_request(
                parent_session_id=self._parent_session_id,
                child_session_id=child_sid,
                correlation_id=coordinator_run_id,
                reason=reason,
            )
            # [codex R3 P1-3] try/except over publish MUST catch
            # ``BaseException`` (not ``Exception``). On Py3.12
            # ``asyncio.CancelledError`` is a ``BaseException`` subclass
            # — if a parent task gets cancelled while ``await
            # publisher.publish(...)`` is suspended, an
            # ``except Exception`` rollback DOES NOT FIRE, leaving the
            # wu_id in ``self._published`` while no envelope was ever
            # delivered. The watcher / a retry path would then observe
            # the wu_id as "already published" and skip republishing →
            # child stays running. Catching ``BaseException`` rolls back
            # the dedup on cancellation (and any other exit path) before
            # re-raising so cancellation still propagates.
            try:
                await self._publisher.publish(envelope)
            except BaseException as exc:  # noqa: BLE001 — rollback dedup on ANY exit
                async with self._publish_lock:
                    self._published.discard(wu_id)
                if isinstance(exc, Exception):
                    # Recoverable: log + continue (mirrors PR-3
                    # swallow-and-continue policy across remaining wu_ids).
                    logger.warning(
                        "CoordinatorRunOrchestrator: cancel publish failed"
                        " wu=%s child=%s reason=%s run=%s: %s",
                        wu_id, child_sid, reason, coordinator_run_id, exc,
                    )
                    continue
                # Non-Exception BaseException (CancelledError / KeyboardInterrupt
                # / SystemExit): rollback already happened above, re-raise so
                # the cancellation actually aborts the orchestrator instead of
                # being silently turned into a "skipped wu".
                raise
            # Only append on a fully successful publish (no exception).
            published.append(wu_id)
        return published

    async def shutdown(self, timeout: float = 5.0) -> None:
        """PR-6 owns the background tasks inside ``run()`` itself (created +
        drained in the same call), so the externally visible ``shutdown()``
        remains a no-op for API compatibility.
        """
        return None
