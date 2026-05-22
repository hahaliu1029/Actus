"""C3 PR-3a/PR-3b — MailboxSupervisor main loop + reliability layer.

PR-3a tests use stubbed handlers — only verify routing + ACK. PR-3b extends
with last_seen heartbeat tracking, XAUTOCLAIM startup + periodic, poison
drop after MAILBOX_POISON_MAX_RECLAIM, and pod-restart clock recovery.
PR-4 tests exercise full handler behavior including destroy hooks.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.application.services.mailbox_supervisor import (
    HandlerOutcome,
    MailboxSupervisor,
    SupervisorContext,
)
from app.domain.models.mailbox_envelope import (
    MAILBOX_POISON_MAX_RECLAIM,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
    RedisMailboxPublisher,
)
from tests.domain.services.conftest import _InMemoryAuditRepo


class _StrictAuditRepo(_InMemoryAuditRepo):
    """Codex r2 [P1] — DB-faithful audit repo stub.

    The default ``_InMemoryAuditRepo.increment_reclaim`` uses
    ``setdefault({"reclaim_count": 0})`` which auto-creates a row if missing,
    masking the production bug where the real DB impl raises ``ValueError``
    on missing rows (see ``db_mailbox_envelope_audit_repository.py`` lines
    141-145).

    This strict variant mimics DB semantics: ``increment_reclaim`` requires
    a row that has already gone through ``upsert_processing`` (i.e.
    ``processing_at`` is set). Used in tests that assert the supervisor
    correctly upserts before incrementing — without that ordering, the
    poison-drop chain stays forever stuck at ``reclaim_count == 1`` and the
    ``> MAILBOX_POISON_MAX_RECLAIM`` gate never trips in production.
    """

    async def increment_reclaim(
        self, parent_session_id: str, envelope_id: str, last_error: str
    ) -> int:
        key = (parent_session_id, envelope_id)
        row = self.rows.get(key)
        if row is None or row.get("processing_at") is None:
            raise ValueError(
                f"increment_reclaim on missing audit row "
                f"({parent_session_id}, {envelope_id})"
            )
        return await super().increment_reclaim(
            parent_session_id, envelope_id, last_error
        )

    async def mark_processed(
        self,
        parent_session_id: str,
        envelope_id: str,
        *,
        processed_at,
    ) -> None:
        """Codex r9 [LOW] — DB-faithful ``mark_processed``.

        The parent ``_InMemoryAuditRepo.mark_processed`` uses
        ``setdefault((parent, env_id), {})`` which auto-creates a row when
        missing — that auto-create masks the codex r8 [HIGH] fresh-delivery
        bug (mark_processed would have raised, but parent stub silently
        creates the row instead). Mirror the DB impl: require ``upsert_processing``
        to have run first (``processing_at`` non-None) before allowing
        ``mark_processed`` to set ``processed_at``.
        """
        key = (parent_session_id, envelope_id)
        row = self.rows.get(key)
        if row is None or row.get("processing_at") is None:
            raise ValueError(
                f"mark_processed on missing audit row "
                f"({parent_session_id}, {envelope_id})"
            )
        await super().mark_processed(
            parent_session_id, envelope_id, processed_at=processed_at
        )


class _StubTelemetry:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, name: str, data: dict) -> None:
        self.emitted.append((name, data))


def _env(
    t: MailboxEnvelopeType = MailboxEnvelopeType.PROGRESS_UPDATE,
    eid: str = "01HSPYU0000000000000000001",
    payload: dict | None = None,
    *,
    parent_session_id: str = "root-1",
    child_session_id: str = "child-1",
    producer_role: ProducerRole = ProducerRole.CHILD_AGENT,
    reclaim_count: int = 0,
) -> MailboxEnvelope:
    if payload is None:
        if t == MailboxEnvelopeType.PROGRESS_UPDATE:
            payload = {"kind": "heartbeat", "visibility": "hidden"}
        elif t == MailboxEnvelopeType.RESULT_READY:
            payload = {"summary": "done", "outcome": "success"}
        elif t == MailboxEnvelopeType.SPAWN_REQUEST:
            payload = {"agent_kind": "research", "task_prompt": "x"}
        elif t == MailboxEnvelopeType.CANCEL_ACK:
            payload = {"final_state": "cancelled"}
        else:  # pragma: no cover — defensive
            payload = {}
    return MailboxEnvelope(
        envelope_id=eid,
        type=t,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        correlation_id="01HSPYU0000000000000000002",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=producer_role,
        payload=payload,
        reclaim_count=reclaim_count,
    )


@pytest.fixture
async def supervisor_ctx(fake_redis, audit_repo, stub_lifecycle, stub_agent_callback):
    return SupervisorContext(
        root_session_id="root-1",
        pod_id="pod-a",
        instance_id="i1",
        redis=fake_redis,
        audit_repo=audit_repo,
        publisher=RedisMailboxPublisher(fake_redis),
        sandbox_lifecycle=stub_lifecycle,
        agent_service_callback=stub_agent_callback,
        telemetry=_StubTelemetry(),
    )


@pytest.mark.anyio
async def test_supervisor_runs_and_ensures_group(supervisor_ctx, fake_redis):
    sup = MailboxSupervisor(
        supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
    )
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.1)
    await sup.stop()
    await task
    groups = await fake_redis.xinfo_groups("actus:child:root-1:mailbox")
    # fakeredis returns dicts keyed by str with bytes values when
    # decode_responses=False; coalesce defensively in case of mode change.
    names = [
        g["name"].decode() if isinstance(g.get("name"), bytes) else g.get("name")
        for g in groups
    ]
    assert "actus:mailbox-supervisor:v1" in names


@pytest.mark.anyio
async def test_supervisor_dispatches_progress_update_to_callback(
    supervisor_ctx, fake_redis, stub_agent_callback
):
    sup = MailboxSupervisor(
        supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
    )
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    pub = RedisMailboxPublisher(fake_redis)
    await pub.publish(_env())
    # Wait for one tick — block_ms=1000 is the supervisor's XREADGROUP block.
    await asyncio.sleep(0.5)
    await sup.stop()
    await task
    assert len(stub_agent_callback.received) == 1
    assert stub_agent_callback.received[0].type == MailboxEnvelopeType.PROGRESS_UPDATE


@pytest.mark.anyio
async def test_supervisor_acks_nonterminal_after_callback(supervisor_ctx, fake_redis):
    sup = MailboxSupervisor(
        supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
    )
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    pub = RedisMailboxPublisher(fake_redis)
    await pub.publish(_env())
    await asyncio.sleep(0.5)
    await sup.stop()
    await task
    pending = await fake_redis.xpending(
        "actus:child:root-1:mailbox", "actus:mailbox-supervisor:v1"
    )
    # xpending summary form returns str-keyed dict regardless of decode_responses.
    assert pending["pending"] == 0


@pytest.mark.anyio
async def test_supervisor_dispatches_terminal_envelope_to_stub_handler(
    supervisor_ctx, fake_redis
):
    """PR-3a uses stub terminal handler that records but doesn't actually destroy."""
    sup = MailboxSupervisor(
        supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
    )
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    pub = RedisMailboxPublisher(fake_redis)
    await pub.publish(
        _env(t=MailboxEnvelopeType.RESULT_READY, eid="01HSPYU0R00000000000000000")
    )
    await asyncio.sleep(0.5)
    await sup.stop()
    await task
    emitted = supervisor_ctx.telemetry.emitted
    assert any(
        name == "mailbox.terminal_envelope_dispatched_stub" for name, _ in emitted
    )


def _make_dummy_ctx():
    """Build a minimal SupervisorContext for sync validation tests.

    Codex r2 [P2] fix — `test_supervisor_rejects_*` are sync (the validation
    raises in `__init__` before any await), so they can't depend on the
    async `supervisor_ctx` fixture. Tests that exercise loop / dispatch
    behavior keep using the async fixture; tests that only verify ctor
    validation use this dummy.

    Codex r6 [P2] fix — also drop the ``fake_redis`` dependency. Validation
    runs before any Redis I/O, so passing a sentinel object satisfies the
    SupervisorContext dataclass without requiring an async fixture that a
    sync test can't await. The cast(Redis, ...) keeps the type checker
    quiet without ever touching a real client.
    """
    from typing import cast as _cast

    from redis.asyncio import Redis as _Redis

    async def _noop_callback(env):
        return None

    class _T:
        async def emit(self, n, d):
            return None

    class _L:
        async def destroy(self, sid, reason):
            return None

    return SupervisorContext(
        root_session_id="root-dummy",
        pod_id="pod-d",
        instance_id="id",
        redis=_cast(_Redis, object()),  # never touched on the validation path
        audit_repo=None,  # validation runs before audit is touched
        publisher=None,  # ditto
        sandbox_lifecycle=_L(),
        agent_service_callback=_noop_callback,
        telemetry=_T(),
    )


def test_supervisor_rejects_partial_dispatch_table():
    """Codex r1 [P1] regression — partial dispatch_table MUST fail fast at
    __init__, never run-through to the ACK-drop unknown-type branch.

    If a future caller forgets to copy an entry from build_default_dispatch_table
    (e.g., PR-4 swaps RESULT_READY but accidentally drops HANDOFF_REQUEST), the
    supervisor would historically ACK-drop legitimate envelopes of the missing
    type and the caller would never see a stack trace. Validation at __init__
    surfaces the bug at construction, not in production.
    """
    from app.application.services.mailbox_supervisor import (
        _StubNonTerminalHandler,
    )

    ctx = _make_dummy_ctx()
    # Build a deliberately incomplete table (only 1 of 10 types).
    partial = {MailboxEnvelopeType.RESULT_READY: _StubNonTerminalHandler()}
    with pytest.raises(ValueError) as exc:
        MailboxSupervisor(ctx, dispatch_table=partial)
    msg = str(exc.value)
    # Error must name at least one of the missing types so the developer
    # sees what to add.
    assert "missing" in msg.lower()
    assert "spawn_request" in msg.lower()


def test_supervisor_rejects_sync_def_handle():
    """Codex r4 [P2] regression — a handler with a sync ``def handle``
    would be ``callable`` (passing the r2 fix) but would crash at
    ``await handler.handle(...)`` with TypeError. That TypeError would
    be swallowed as a "handler raised" event and the envelope would
    loop in PEL until poison-drop. Validate ``async def`` at __init__.
    """
    from app.application.services.mailbox_supervisor import (
        HandlerOutcome,
        build_default_dispatch_table,
    )

    class _SyncHandle:
        def handle(self, env, ctx):  # NOT async — bug we want to catch
            return HandlerOutcome(ack=True)

    ctx = _make_dummy_ctx()
    tbl = build_default_dispatch_table()
    tbl[MailboxEnvelopeType.PROGRESS_UPDATE] = _SyncHandle()
    with pytest.raises(ValueError) as exc:
        MailboxSupervisor(ctx, dispatch_table=tbl)
    msg = str(exc.value).lower()
    assert "async def handle" in msg or "async" in msg
    assert "progress_update" in msg


def test_supervisor_rejects_wrong_arity_handle():
    """Codex r5 [P2] regression — a handler with the wrong number of
    positional parameters passes ``callable`` + ``iscoroutinefunction``
    but crashes at ``await handler.handle(envelope, ctx)`` with TypeError
    inside ``_handle_envelope``. Validate arity at construction.
    """
    from app.application.services.mailbox_supervisor import (
        build_default_dispatch_table,
    )

    class _OneArgHandle:
        async def handle(self, env):  # missing ctx — bug we want to catch
            return None

    class _ThreeArgHandle:
        async def handle(self, env, ctx, extra):  # extra mandatory arg
            return None

    ctx = _make_dummy_ctx()

    tbl_one = build_default_dispatch_table()
    tbl_one[MailboxEnvelopeType.SPAWN_ACK] = _OneArgHandle()
    with pytest.raises(ValueError) as exc:
        MailboxSupervisor(ctx, dispatch_table=tbl_one)
    msg = str(exc.value).lower()
    assert "signature" in msg
    assert "spawn_ack" in msg

    tbl_three = build_default_dispatch_table()
    tbl_three[MailboxEnvelopeType.HANDOFF_REQUEST] = _ThreeArgHandle()
    with pytest.raises(ValueError) as exc:
        MailboxSupervisor(ctx, dispatch_table=tbl_three)
    msg = str(exc.value).lower()
    assert "signature" in msg
    assert "handoff_request" in msg


def test_default_dispatch_table_passes_validation():
    """Codex r5 [P2] lock — the default dispatch table built by
    build_default_dispatch_table() must itself pass the same validator
    that user-supplied tables go through. Otherwise a future regression
    that drops a key, replaces a stub with a sync def, or changes the
    handle signature would silently ACK-drop envelopes when the
    supervisor is constructed without an explicit dispatch_table arg.
    """
    ctx = _make_dummy_ctx()
    # No exception = default table is well-formed and the validator is
    # actually wired into the no-arg construction path.
    sup = MailboxSupervisor(ctx)
    # Sanity: all 10 types reachable through the dispatch attr.
    assert set(sup._dispatch) == set(MailboxEnvelopeType)  # noqa: SLF001


def test_default_dispatch_path_actually_invokes_validator(monkeypatch):
    """Codex r6 [P2] lock — the prior test only proves the default table
    happens to be valid; it does NOT prove the no-arg ``MailboxSupervisor(ctx)``
    path runs the validator. If a future refactor skips
    ``_validate_dispatch_table()`` for the default-built table (e.g.,
    "we know the default is fine"), the prior test still passes silently.

    Monkeypatch the builder to return a deliberately invalid default and
    assert ValueError fires through the no-arg construction path. This
    actually exercises the validator-on-default code path.
    """
    from app.application.services import mailbox_supervisor as ms

    def _bad_default():
        # Missing 9 of 10 types — triggers the key-coverage branch.
        return {MailboxEnvelopeType.RESULT_READY: ms._StubTerminalHandler()}

    monkeypatch.setattr(ms, "build_default_dispatch_table", _bad_default)
    ctx = _make_dummy_ctx()
    with pytest.raises(ValueError) as exc:
        ms.MailboxSupervisor(ctx)  # no dispatch_table arg → default path
    assert "missing" in str(exc.value).lower()


def test_supervisor_rejects_none_dispatch_handler():
    """Codex r2 [P1] regression — even a fully-keyed dispatch_table must
    reject ``None`` (or non-handler) values. Otherwise ``.get(type)``
    returns ``None`` at runtime and the supervisor ACK-drops the envelope
    through the unknown-type branch.
    """
    from app.application.services.mailbox_supervisor import (
        build_default_dispatch_table,
    )

    ctx = _make_dummy_ctx()
    tbl = build_default_dispatch_table()
    # Inject a poison value — keys still fully cover MailboxEnvelopeType,
    # so the "missing key" check passes; the value check must catch it.
    tbl[MailboxEnvelopeType.PROGRESS_UPDATE] = None  # type: ignore[assignment]
    with pytest.raises(ValueError) as exc:
        MailboxSupervisor(ctx, dispatch_table=tbl)
    assert "invalid" in str(exc.value).lower()
    assert "progress_update" in str(exc.value).lower()


@pytest.mark.anyio
async def test_supervisor_acks_side_effect_success_even_when_ack_false(
    supervisor_ctx, fake_redis
):
    """Codex r2 [P1] regression — spec §6.x: ``ack=False`` + side_effect
    successful completion MUST ACK. PR-4 terminal handlers return this
    exact shape (destroy() runs in side_effect; on success the envelope
    must leave PEL or the next XAUTOCLAIM sweep re-fires destruction).

    The pre-fix code only ACKed when ``outcome.ack=True``, so a successful
    terminal handler would silently retain the envelope in PEL.
    """
    from app.application.services.mailbox_supervisor import (
        HandlerOutcome,
        build_default_dispatch_table,
    )

    side_effect_calls: list[str] = []

    class _AckFalseSideEffectHandler:
        async def handle(self, env, ctx) -> HandlerOutcome:
            async def _se() -> None:
                side_effect_calls.append(env.envelope_id)

            return HandlerOutcome(ack=False, side_effect=_se)

    tbl = build_default_dispatch_table()
    tbl[MailboxEnvelopeType.RESULT_READY] = _AckFalseSideEffectHandler()
    sup = MailboxSupervisor(
        supervisor_ctx, dispatch_table=tbl, block_ms=0, idle_poll_sleep_s=0.01
    )
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    pub = RedisMailboxPublisher(fake_redis)
    await pub.publish(
        _env(t=MailboxEnvelopeType.RESULT_READY, eid="01HSPYU0S00000000000000000")
    )
    await asyncio.sleep(0.5)
    await sup.stop()
    await task

    # Side effect ran exactly once.
    assert side_effect_calls == ["01HSPYU0S00000000000000000"]
    # And the envelope was ACKed — pending==0 (NOT lingering in PEL).
    pending = await fake_redis.xpending(
        "actus:child:root-1:mailbox", "actus:mailbox-supervisor:v1"
    )
    assert pending["pending"] == 0, (
        "side_effect success must ACK regardless of outcome.ack — "
        "otherwise PR-4 terminal handlers re-fire destruction on retry"
    )


@pytest.mark.anyio
async def test_supervisor_retains_pel_when_side_effect_raises(
    supervisor_ctx, fake_redis
):
    """Side_effect raising MUST leave envelope in PEL (no ACK). PR-3b
    XAUTOCLAIM relies on this for retry; poison drop only fires after
    reclaim_count > MAILBOX_POISON_MAX_RECLAIM.
    """
    from app.application.services.mailbox_supervisor import (
        HandlerOutcome,
        build_default_dispatch_table,
    )

    # Codex r3 [P2] lock: prove side_effect actually RAN before raising —
    # otherwise a regression that ``return``-s before calling side_effect
    # would also produce pending==1 and pass this test.
    side_effect_attempts: list[str] = []

    class _RaisingSideEffectHandler:
        async def handle(self, env, ctx) -> HandlerOutcome:
            async def _se() -> None:
                side_effect_attempts.append(env.envelope_id)
                raise RuntimeError("destroy() failed transiently")

            return HandlerOutcome(ack=False, side_effect=_se)

    tbl = build_default_dispatch_table()
    tbl[MailboxEnvelopeType.RESULT_READY] = _RaisingSideEffectHandler()
    sup = MailboxSupervisor(
        supervisor_ctx, dispatch_table=tbl, block_ms=0, idle_poll_sleep_s=0.01
    )
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    pub = RedisMailboxPublisher(fake_redis)
    await pub.publish(
        _env(t=MailboxEnvelopeType.RESULT_READY, eid="01HSPYU0T00000000000000000")
    )
    await asyncio.sleep(0.5)
    await sup.stop()
    await task

    # The supervisor MUST have invoked side_effect — that's what generates
    # the raise → PEL retain path.
    assert side_effect_attempts == ["01HSPYU0T00000000000000000"]
    pending = await fake_redis.xpending(
        "actus:child:root-1:mailbox", "actus:mailbox-supervisor:v1"
    )
    # Exactly one envelope still in PEL — XAUTOCLAIM (PR-3b) will retry it.
    assert pending["pending"] == 1


@pytest.mark.anyio
async def test_supervisor_refuses_cross_root_envelope(supervisor_ctx, fake_redis):
    """Codex r8 [P1] regression — defense-in-depth against publisher bugs /
    misrouted envelopes. The XREADGROUP stream key scopes reads to one root,
    but if a publisher accidentally XADDs an envelope whose
    parent_session_id doesn't match the stream's root, the supervisor MUST
    refuse to dispatch it (otherwise PR-4 terminal handlers would destroy
    the wrong root's sandbox).

    Verify:
    1. The agent_service_callback is NOT invoked for the cross-root envelope.
    2. The envelope is ACKed (drain — don't loop forever).
    """
    sup = MailboxSupervisor(
        supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
    )
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)

    # Hand-craft a forged envelope whose parent_session_id is NOT the
    # supervisor's bound root. Inject via raw XADD to bypass the publisher
    # (which would correctly route by parent_session_id) — this simulates
    # a publisher bug / mis-routed envelope landing on our stream.
    forged = MailboxEnvelope(
        envelope_id="01HSPYU0X00000000000000000",
        type=MailboxEnvelopeType.PROGRESS_UPDATE,
        parent_session_id="root-FORGED",  # MISMATCH vs supervisor root-1
        child_session_id="child-x",
        correlation_id="01HSPYU0Y00000000000000000",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload={"kind": "heartbeat", "visibility": "hidden"},
    )
    await fake_redis.xadd(
        "actus:child:root-1:mailbox",
        fields={"envelope": forged.model_dump_json()},
    )
    await asyncio.sleep(0.5)
    await sup.stop()
    await task

    # Callback NOT invoked — handler was bypassed.
    assert supervisor_ctx.agent_service_callback.received == []
    # ACKed — drained instead of looping.
    pending = await fake_redis.xpending(
        "actus:child:root-1:mailbox", "actus:mailbox-supervisor:v1"
    )
    assert pending["pending"] == 0


@pytest.mark.anyio
async def test_supervisor_isolates_per_entry_failures_in_batch(
    supervisor_ctx, fake_redis, monkeypatch
):
    """Codex r10 [P1] regression — a single failing entry in a batch must
    NOT starve the rest of the batch. Pre-fix, an exception bubbling out
    of _handle_envelope (e.g., transient XACK failure) would hit the outer
    loop except, sleep 0.5s, and the next ``read(">")`` would skip the
    unprocessed remainder of the batch — leaving them in PEL until PR-3b
    XAUTOCLAIM picks them up.

    Simulate: monkeypatch the consumer's ack() to raise on the FIRST call
    only, publish two envelopes, then verify both callbacks ran (entry #2
    dispatched even though entry #1's ACK blew up).
    """
    # Codex r11 [P2] fix — pre-publish BOTH envelopes before the supervisor
    # starts. ensure_group() is invoked at run() entry with id="0-0" +
    # mkstream=True, so both pre-existing entries are delivered in the
    # FIRST XREADGROUP read as one batch. If we published after start, the
    # supervisor might read entry #1 alone, fail its ACK, then on the next
    # iteration read entry #2 in a *separate* batch — and the test would
    # pass even if per-entry isolation were removed. Pre-publish locks the
    # "same batch remainder starvation" semantics this test claims to cover.
    pub = RedisMailboxPublisher(fake_redis)
    await pub.publish(_env(eid="01HSPYU0A00000000000000001"))
    await pub.publish(_env(eid="01HSPYU0A00000000000000002"))

    sup = MailboxSupervisor(
        supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
    )

    # Inject a transient ACK failure on the first ack call. The second
    # call (for entry #2 or for a retry-via-XAUTOCLAIM in PR-3b) succeeds.
    original_ack = sup._consumer.ack  # noqa: SLF001
    ack_calls: list[int] = [0]

    async def _flaky_ack(redis_id):
        ack_calls[0] += 1
        if ack_calls[0] == 1:
            raise RuntimeError("transient XACK failure")
        await original_ack(redis_id)

    monkeypatch.setattr(sup._consumer, "ack", _flaky_ack)  # noqa: SLF001

    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.5)
    await sup.stop()
    await task

    # Both envelopes' handlers ran — entry #2 was NOT starved by entry
    # #1's ACK failure, even though they were delivered in the same batch.
    received_ids = [e.envelope_id for e in supervisor_ctx.agent_service_callback.received]
    assert "01HSPYU0A00000000000000001" in received_ids
    assert "01HSPYU0A00000000000000002" in received_ids


@pytest.mark.anyio
async def test_supervisor_stop_drains_pending(supervisor_ctx):
    sup = MailboxSupervisor(
        supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
    )
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    await sup.stop(drain_timeout_s=2.0)
    await task
    # No exception; task exits cleanly. The drain is a real await on
    # ``self._stopped`` — confirms ``stop()`` is not a fire-and-forget flag.


# ──────────────────────────────────────────────────────────────────────────────
# PR-3b — Phase A: heartbeat last_seen + producer_role gate (spec §9.2)
# ──────────────────────────────────────────────────────────────────────────────


class TestLastSeenGate:
    """Spec §9.2 — last_seen refresh MUST gate on (child-origin envelope type)
    AND (producer_role == CHILD_AGENT). SUPERVISOR_ECHO must NOT advance the
    child's heartbeat clock (otherwise a supervisor sending its own CANCEL_ACK
    after a destroy() would falsely report the child as alive)."""

    @pytest.mark.anyio
    async def test_child_origin_progress_update_refreshes_last_seen(
        self, supervisor_ctx, fake_redis
    ):
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        # No need to start the loop — _handle_envelope is the unit under test
        # and last_seen tracking is an instance-state side effect.
        envelope = _env(
            t=MailboxEnvelopeType.PROGRESS_UPDATE,
            eid="01HSPYU0L00000000000000001",
            producer_role=ProducerRole.CHILD_AGENT,
            child_session_id="child-1",
        )
        # ensure_group so cross-root guard doesn't bypass
        await sup._consumer.ensure_group()  # noqa: SLF001
        # Drive directly — _handle_envelope is the single touchpoint.
        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001
        assert sup.get_last_seen("child-1") is not None

    @pytest.mark.anyio
    async def test_supervisor_echo_cancel_ack_does_not_refresh_last_seen(
        self, supervisor_ctx
    ):
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        envelope = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            eid="01HSPYU0L00000000000000002",
            producer_role=ProducerRole.SUPERVISOR_ECHO,
            child_session_id="child-1",
        )
        await sup._consumer.ensure_group()  # noqa: SLF001
        await sup._handle_envelope(b"0-2", envelope)  # noqa: SLF001
        # SUPERVISOR_ECHO must NOT advance the child clock — spec §9.2.
        assert sup.get_last_seen("child-1") is None

    @pytest.mark.anyio
    async def test_real_child_cancel_ack_refreshes_last_seen(
        self, supervisor_ctx
    ):
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        envelope = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            eid="01HSPYU0L00000000000000003",
            producer_role=ProducerRole.CHILD_AGENT,
            child_session_id="child-1",
        )
        await sup._consumer.ensure_group()  # noqa: SLF001
        await sup._handle_envelope(b"0-3", envelope)  # noqa: SLF001
        assert sup.get_last_seen("child-1") is not None

    @pytest.mark.anyio
    async def test_parent_to_child_envelope_does_not_refresh_last_seen(
        self, supervisor_ctx
    ):
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        # SPAWN_REQUEST flows parent → child; producer_role=PARENT_AGENT.
        envelope = _env(
            t=MailboxEnvelopeType.SPAWN_REQUEST,
            eid="01HSPYU0L00000000000000004",
            producer_role=ProducerRole.PARENT_AGENT,
            child_session_id="child-1",
        )
        await sup._consumer.ensure_group()  # noqa: SLF001
        await sup._handle_envelope(b"0-4", envelope)  # noqa: SLF001
        # Parent → child envelopes don't tell us anything about child liveness.
        assert sup.get_last_seen("child-1") is None


# ──────────────────────────────────────────────────────────────────────────────
# PR-3b — Phase B: XAUTOCLAIM startup + periodic (spec §5.6 + §5.8 step 3)
# ──────────────────────────────────────────────────────────────────────────────


class TestXAutoclaim:
    """Spec §5.6 — initial XAUTOCLAIM at supervisor start drains PEL entries
    left by a dead consumer; periodic XAUTOCLAIM runs every
    MAILBOX_XAUTOCLAIM_PERIODIC_INTERVAL_SECONDS to claim idle entries."""

    @pytest.mark.anyio
    async def test_initial_xautoclaim_picks_up_orphan_pel(
        self, supervisor_ctx, fake_redis
    ):
        # Phase 1: consumer A reads the entry but never ACKs (simulating
        # a crashed supervisor on a previous pod).
        from app.infrastructure.external.mailbox.redis_mailbox_consumer import (
            RedisMailboxConsumer,
        )

        consumer_a = RedisMailboxConsumer(
            fake_redis, "root-1", "pod-old", "i-old"
        )
        await consumer_a.ensure_group()
        pub = RedisMailboxPublisher(fake_redis)
        await pub.publish(_env(eid="01HSPYU0M00000000000000001"))
        await consumer_a.read(count=10, block_ms=0)

        # Phase 2: fresh supervisor on a new instance_id starts. The initial
        # XAUTOCLAIM (min_idle_ms=0) MUST pull the orphan entry out of A's
        # PEL slot and dispatch it through our handler chain.
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        task = asyncio.create_task(sup.run())
        await asyncio.sleep(0.3)
        await sup.stop()
        await task

        received_ids = [
            e.envelope_id for e in supervisor_ctx.agent_service_callback.received
        ]
        assert "01HSPYU0M00000000000000001" in received_ids

    @pytest.mark.anyio
    async def test_periodic_xautoclaim_runs_on_interval(
        self, supervisor_ctx, monkeypatch
    ):
        # Pin the periodic interval very short so the loop fires within
        # the test window.
        monkeypatch.setattr(
            MailboxSupervisor, "_XAUTOCLAIM_INTERVAL_S", 0.05, raising=True
        )
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        task = asyncio.create_task(sup.run())
        await asyncio.sleep(0.5)
        await sup.stop()
        await task

        emitted = supervisor_ctx.telemetry.emitted
        periodic_ticks = [
            name for name, _ in emitted if name == "mailbox.periodic_autoclaim_tick"
        ]
        assert len(periodic_ticks) >= 1, (
            f"periodic XAUTOCLAIM never emitted telemetry; emitted={emitted!r}"
        )

    @pytest.mark.anyio
    async def test_periodic_tick_waits_full_interval_after_initial_xautoclaim(
        self, supervisor_ctx
    ):
        """Codex r2 [P2] regression — _initial_xautoclaim must anchor
        _last_autoclaim_mono so the periodic tick doesn't fire immediately
        on startup.

        ``time.monotonic()`` returns process-uptime in seconds (typically
        >> 30 immediately after boot of any non-trivial process), so if
        ``_last_autoclaim_mono`` stays at the default ``0.0``, the
        ``now - last < _XAUTOCLAIM_INTERVAL_S=30.0`` gate evaluates to
        False on the first loop iteration and the periodic sweep fires
        immediately — doubling the XAUTOCLAIM work right after
        ``_initial_xautoclaim`` already drained the PEL with ``min_idle=0``.

        Run the supervisor for 200ms with the production interval
        (30s) and assert ``mailbox.periodic_autoclaim_tick`` was NOT
        emitted. ``mailbox.initial_autoclaim_completed`` must still appear
        — that's the smoke test that the startup sweep itself ran.
        """
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.05
        )
        # Lock the production interval — without the r2 [P2] anchor fix,
        # the periodic tick would fire on iteration 1 since
        # time.monotonic() - 0.0 >> 30s.
        sup._XAUTOCLAIM_INTERVAL_S = 30.0  # noqa: SLF001
        task = asyncio.create_task(sup.run())
        await asyncio.sleep(0.2)
        await sup.stop()
        await task

        names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        # Smoke: startup sweep actually ran.
        assert "mailbox.initial_autoclaim_completed" in names, (
            f"initial_autoclaim_completed missing; emitted={names!r}"
        )
        # The fix: periodic tick did NOT fire prematurely.
        assert "mailbox.periodic_autoclaim_tick" not in names, (
            "periodic tick fired prematurely after _initial_xautoclaim — "
            "r2 [P2] anchor fix missing; "
            f"emitted={names!r}"
        )

    @pytest.mark.anyio
    async def test_initial_xautoclaim_anchors_clock_even_when_telemetry_raises(
        self, monkeypatch, supervisor_ctx
    ):
        """Codex r5 [P2] regression — closing ``initial_autoclaim_completed``
        telemetry emit must NOT block ``_last_autoclaim_mono`` anchor.

        Pre-fix order was: for-loop → telemetry emit → anchor. The anchor was
        inside the outer broad-except scope, so if telemetry raised, the
        outer except swallowed it and the anchor was skipped — bringing back
        r2's "periodic tick fires immediately on startup" bug via the
        observability path (telemetry sink crash re-opens correctness regression).

        Fix order: for-loop → anchor → isolated telemetry emit. Verify the
        anchor lands at clock() even when telemetry raises.
        """
        sup = MailboxSupervisor(supervisor_ctx)
        # Force the closing telemetry emit to raise.
        original_emit = supervisor_ctx.telemetry.emit

        async def _boom_emit(name, data):
            if name == "mailbox.initial_autoclaim_completed":
                raise RuntimeError("synthetic telemetry sink crash")
            await original_emit(name, data)

        monkeypatch.setattr(supervisor_ctx.telemetry, "emit", _boom_emit)  # noqa: SLF001

        # Run the startup sweep. The supervisor still needs the group to
        # exist before XAUTOCLAIM can return anything; just call directly —
        # empty PEL is fine, the closing telemetry still fires.
        await sup._consumer.ensure_group()  # noqa: SLF001
        before = sup._last_autoclaim_mono  # noqa: SLF001
        assert before == 0.0  # baseline
        await sup._initial_xautoclaim()  # noqa: SLF001
        # Anchor must be set despite telemetry raising — proves anchor
        # ordering moved BEFORE the telemetry emit.
        assert sup._last_autoclaim_mono > 0.0, (  # noqa: SLF001
            "r5 [P2] anchor missing — telemetry emit raised and the outer "
            "broad-except skipped the anchor; periodic tick will fire "
            "prematurely on next loop iteration"
        )

    @pytest.mark.anyio
    async def test_initial_xautoclaim_isolates_per_entry_failures(
        self, monkeypatch, supervisor_ctx
    ):
        """Codex r3 [P1] regression — XAUTOCLAIM transfers PEL ownership to
        this supervisor in one batch. If ``_handle_envelope`` raises for one
        entry and the for-loop aborts, the rest of the batch is now pending
        under THIS consumer (not redelivered via XREADGROUP ``>``) and would
        only resurface after ``MAILBOX_PEL_IDLE_MS_FOR_CLAIM`` — starving up
        to 999 startup-claimed orphans on one transient failure.

        Mirror PR-3a main-loop pattern: each ``_handle_envelope`` is wrapped
        in CancelledError-passes / broad-Exception-logs.

        Stub ``_consumer.autoclaim`` to return TWO claimed entries; monkeypatch
        ``_handle_envelope`` to raise on the first call. Assert the second
        call still ran.
        """
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.05
        )
        env_a = _env(MailboxEnvelopeType.PROGRESS_UPDATE, eid="01HSPYU0i00000000000ITER01")
        env_b = _env(MailboxEnvelopeType.PROGRESS_UPDATE, eid="01HSPYU0i00000000000ITER02")

        async def _fake_autoclaim(*, min_idle_ms, count, start_id="0-0"):
            return [(b"0-1", env_a), (b"0-2", env_b)]

        monkeypatch.setattr(sup._consumer, "autoclaim", _fake_autoclaim)

        seen: list[str] = []
        original_handle = sup._handle_envelope

        async def _flaky_handle(redis_id, envelope):
            seen.append(envelope.envelope_id)
            if envelope.envelope_id == env_a.envelope_id:
                raise RuntimeError("synthetic per-entry crash")
            await original_handle(redis_id, envelope)

        monkeypatch.setattr(sup, "_handle_envelope", _flaky_handle)

        await sup._initial_xautoclaim()
        # Without the r3 [P1] fix, the for-loop aborts on env_a's raise and
        # env_b is never touched — `seen` would be one element.
        assert seen == [env_a.envelope_id, env_b.envelope_id], (
            "r3 [P1] per-entry isolation missing — second claimed entry was "
            "skipped after first raised; "
            f"seen={seen!r}"
        )

    @pytest.mark.anyio
    async def test_periodic_xautoclaim_isolates_per_entry_failures(
        self, monkeypatch, supervisor_ctx
    ):
        """Codex r3 [P1] regression — same as initial, but for the periodic
        sweep path. ``_maybe_periodic_xautoclaim`` must wrap each
        ``_handle_envelope`` in per-entry try/except.
        """
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.05
        )
        # Force the periodic tick to fire immediately
        sup._XAUTOCLAIM_INTERVAL_S = 0.0  # noqa: SLF001
        sup._last_autoclaim_mono = 0.0  # noqa: SLF001

        env_a = _env(MailboxEnvelopeType.PROGRESS_UPDATE, eid="01HSPYU0p00000000000ITER01")
        env_b = _env(MailboxEnvelopeType.PROGRESS_UPDATE, eid="01HSPYU0p00000000000ITER02")

        async def _fake_autoclaim(*, min_idle_ms, count, start_id="0-0"):
            return [(b"0-1", env_a), (b"0-2", env_b)]

        monkeypatch.setattr(sup._consumer, "autoclaim", _fake_autoclaim)

        seen: list[str] = []
        original_handle = sup._handle_envelope

        async def _flaky_handle(redis_id, envelope):
            seen.append(envelope.envelope_id)
            if envelope.envelope_id == env_a.envelope_id:
                raise RuntimeError("synthetic per-entry crash")
            await original_handle(redis_id, envelope)

        monkeypatch.setattr(sup, "_handle_envelope", _flaky_handle)

        await sup._maybe_periodic_xautoclaim()
        assert seen == [env_a.envelope_id, env_b.envelope_id], (
            "r3 [P1] per-entry isolation missing in periodic sweep; "
            f"seen={seen!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# PR-3b — Phase C: poison-message drop (spec §5.7)
# ──────────────────────────────────────────────────────────────────────────────


class TestPoisonDrop:
    @pytest.mark.anyio
    async def test_envelope_reclaimed_max_times_is_dropped_with_telemetry(
        self, supervisor_ctx, fake_redis
    ):
        """Spec §5.7 — after reclaim_count > MAILBOX_POISON_MAX_RECLAIM the
        envelope MUST be dropped + ACKed + telemetry emitted, NOT re-dispatched."""
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        await sup._consumer.ensure_group()  # noqa: SLF001
        # Publish an envelope first so it has a real PEL slot.
        pub = RedisMailboxPublisher(fake_redis)
        await pub.publish(_env(eid="01HSPYU0P00000000000000001"))
        entries = await sup._consumer.read(count=10, block_ms=0)  # noqa: SLF001
        redis_id, envelope = entries[0]
        # Stamp reclaim_count above the poison threshold.
        envelope = envelope.model_copy(
            update={"reclaim_count": MAILBOX_POISON_MAX_RECLAIM + 1}
        )
        await sup._handle_envelope(redis_id, envelope)  # noqa: SLF001

        # Telemetry emitted with the right name.
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert "mailbox.poison_message_dropped" in emitted_names
        # Handler was NOT invoked — the agent_service_callback received nothing.
        assert supervisor_ctx.agent_service_callback.received == []
        # Envelope was ACKed — pending count is zero after the drop.
        pending = await fake_redis.xpending(
            "actus:child:root-1:mailbox", "actus:mailbox-supervisor:v1"
        )
        assert pending["pending"] == 0

    @pytest.mark.anyio
    async def test_poison_drop_isolates_hook_exception_and_still_acks(
        self, supervisor_ctx, monkeypatch
    ):
        """Codex r1 [P1] regression — if a PR-4 ``_on_poison_drop`` override
        raises (e.g., cascade publisher errors), the envelope MUST still be
        ACKed. Without this isolation, the envelope stays in PEL, the next
        XAUTOCLAIM sweep re-reads it, the poison gate re-fires, the hook
        re-raises — infinite loop with no ACK ever happening.
        """
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        await sup._consumer.ensure_group()  # noqa: SLF001

        async def _boom(envelope):
            raise RuntimeError("PR-4 override blew up")

        # Override hook to always raise — simulate a PR-4 cascade failure.
        sup._on_poison_drop = _boom  # noqa: SLF001

        envelope = _env(
            t=MailboxEnvelopeType.PROGRESS_UPDATE,
            eid="01HSPYU0P00000000HOOKBOOM1",
            reclaim_count=MAILBOX_POISON_MAX_RECLAIM + 1,
        )

        # Spy on consumer.ack so we can assert it ran exactly once.
        ack_calls: list[bytes] = []
        original_ack = sup._consumer.ack  # noqa: SLF001

        async def _spy_ack(rid):
            ack_calls.append(rid)
            await original_ack(rid)

        monkeypatch.setattr(sup._consumer, "ack", _spy_ack)  # noqa: SLF001

        # Inject directly — _handle_envelope is the single funnel for
        # both fresh delivery and XAUTOCLAIM redelivery.
        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        # ACK ran exactly once despite the hook raising.
        assert len(ack_calls) == 1
        # Telemetry was emitted BEFORE the hook raised — observability is
        # preserved even when the hook fails.
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert "mailbox.poison_message_dropped" in emitted_names

    @pytest.mark.anyio
    async def test_poison_drop_isolates_telemetry_exception_and_still_acks(
        self, monkeypatch, supervisor_ctx
    ):
        """Codex r4 [P1] regression — telemetry.emit is best-effort.

        If the OTel/exporter backend raises during ``mailbox.poison_message_dropped``
        emit (network blip, sink misconfig), execution must NOT skip the
        ``_on_poison_drop`` hook OR ``_consumer.ack``. Otherwise the envelope
        stays in PEL, XAUTOCLAIM re-grabs it, poison gate re-fires, telemetry
        emit raises again — same infinite-loop class as r1's hook isolation,
        but via the observability path.
        """
        sup = MailboxSupervisor(supervisor_ctx)

        # Force telemetry.emit to raise — instance-level override beats the
        # stub's normal list-append.
        emit_calls: list[tuple[str, dict]] = []

        async def _boom_emit(name, data):
            emit_calls.append((name, data))
            raise RuntimeError("synthetic telemetry sink crash")

        monkeypatch.setattr(supervisor_ctx.telemetry, "emit", _boom_emit)  # noqa: SLF001

        # Spy hook to verify it still ran AFTER telemetry raised.
        hook_calls: list[str] = []

        async def _spy_hook(envelope):
            hook_calls.append(envelope.envelope_id)

        sup._on_poison_drop = _spy_hook  # noqa: SLF001

        # Spy ack to verify it still ran AFTER both telemetry and hook.
        ack_calls: list[bytes] = []
        original_ack = sup._consumer.ack  # noqa: SLF001

        async def _spy_ack(rid):
            ack_calls.append(rid)
            await original_ack(rid)

        monkeypatch.setattr(sup._consumer, "ack", _spy_ack)  # noqa: SLF001

        envelope = _env(
            t=MailboxEnvelopeType.PROGRESS_UPDATE,
            eid="01HSPYU0T00000000TLMBOOM01",
            reclaim_count=MAILBOX_POISON_MAX_RECLAIM + 1,
        )
        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        # Telemetry was attempted (proves the spy fired)
        assert len(emit_calls) == 1
        assert emit_calls[0][0] == "mailbox.poison_message_dropped"
        # Hook still ran despite telemetry raising
        assert hook_calls == [envelope.envelope_id]
        # ACK still ran — the loop-breaking invariant
        assert len(ack_calls) == 1

    @pytest.mark.anyio
    async def test_reclaim_count_persists_across_periodic_xautoclaim_sweeps(
        self, fake_redis, stub_lifecycle, stub_agent_callback, monkeypatch
    ):
        """Codex r1 [P0] + r2 [P1] regression — multi-sweep accumulation
        against a DB-faithful audit repo.

        Before r1 the per-sweep ``model_copy`` increment was thrown away
        (only the in-memory envelope carried it), so each new XAUTOCLAIM
        re-parsed ``reclaim_count=0`` and poison drop was effectively dead
        code.

        After r1 the supervisor called ``increment_reclaim`` directly. But
        the conftest's lenient ``_InMemoryAuditRepo.increment_reclaim``
        auto-creates the row via ``setdefault``, masking the production
        bug — the real DB impl raises ``ValueError`` on missing row
        (``db_mailbox_envelope_audit_repository.py`` lines 141-145), the
        broad-except fallback then returned ``envelope.reclaim_count + 1``
        which is forever 1 (the stream entry's envelope is re-parsed each
        sweep), so poison drop STILL could never trip.

        r2 fixes this by calling ``upsert_processing`` BEFORE
        ``increment_reclaim`` in every reclaim path. This test now uses the
        ``_StrictAuditRepo`` to lock the fix down: assert ``reclaim_count``
        AND ``processing_at`` are set on the audit row after each sweep,
        and the count grows monotonically 1 → 2 → 3.
        """
        # Use the DB-faithful strict stub — increment_reclaim raises on
        # missing row, so we prove upsert_processing actually runs first.
        strict_repo = _StrictAuditRepo()
        ctx = SupervisorContext(
            root_session_id="root-1",
            pod_id="pod-a",
            instance_id="i1",
            redis=fake_redis,
            audit_repo=strict_repo,
            publisher=RedisMailboxPublisher(fake_redis),
            sandbox_lifecycle=stub_lifecycle,
            agent_service_callback=stub_agent_callback,
            telemetry=_StubTelemetry(),
        )

        monkeypatch.setattr(
            MailboxSupervisor, "_XAUTOCLAIM_INTERVAL_S", 0.0, raising=True
        )
        sup = MailboxSupervisor(ctx, block_ms=0, idle_poll_sleep_s=0.01)

        # Stub the consumer's autoclaim() to always re-yield the same
        # entry. This simulates the production case where the handler
        # consistently fails and the envelope keeps coming back through
        # XAUTOCLAIM. We bypass _handle_envelope by overriding it so the
        # only side effect we observe is the audit_repo write made inside
        # _maybe_periodic_xautoclaim.
        envelope = _env(
            t=MailboxEnvelopeType.PROGRESS_UPDATE,
            eid="01HSPYU0RECURSEINPEL000001",
            reclaim_count=0,
        )

        async def _stub_autoclaim(*, min_idle_ms, count):
            return [(b"0-7", envelope)]

        monkeypatch.setattr(sup._consumer, "autoclaim", _stub_autoclaim)  # noqa: SLF001

        # Suppress real dispatch — we only care that the count persists.
        async def _noop_handle(rid, env):
            return None

        monkeypatch.setattr(sup, "_handle_envelope", _noop_handle)  # noqa: SLF001

        key = (envelope.parent_session_id, envelope.envelope_id)
        # Run the periodic sweep three times; reclaim_count must climb.
        # Each sweep must:
        #   1. upsert_processing (sets processing_at — without this the
        #      strict stub's increment_reclaim raises ValueError),
        #   2. increment_reclaim (bumps reclaim_count by 1).
        for _expected in (1, 2, 3):
            await sup._maybe_periodic_xautoclaim()  # noqa: SLF001
            assert strict_repo.rows[key]["reclaim_count"] == _expected, (
                f"sweep {_expected}: reclaim_count must grow monotonically; "
                f"got {strict_repo.rows[key]['reclaim_count']}"
            )
            assert strict_repo.rows[key]["last_error"] == "xautoclaim_redelivered"
            # processing_at is set by upsert_processing — without the r2
            # fix this row would not exist at all (strict stub raises on
            # missing row, broad-except masks the fault, count stays at 1).
            assert strict_repo.rows[key].get("processing_at") is not None, (
                "upsert_processing must run before increment_reclaim — "
                "otherwise DB impl raises ValueError on missing row"
            )

    @pytest.mark.anyio
    async def test_initial_xautoclaim_persists_reclaim_count_via_audit_repo(
        self, fake_redis, stub_lifecycle, stub_agent_callback
    ):
        """Codex r2 [P1] regression — startup XAUTOCLAIM must upsert audit
        row BEFORE incrementing reclaim_count.

        Pre-stage a PEL entry by having a dead consumer read but never ACK.
        Start a fresh supervisor with the DB-faithful strict stub. The
        initial sweep claims the orphan entry; if the supervisor calls
        ``increment_reclaim`` without first calling ``upsert_processing``,
        the strict stub raises ValueError, the broad-except falls back to
        in-memory bump, and the audit row never gets created.

        Asserts both ``reclaim_count`` >= 1 and ``processing_at`` is set —
        proves both the upsert AND the increment ran against the real
        audit_repo on the startup path.
        """
        from app.infrastructure.external.mailbox.redis_mailbox_consumer import (
            RedisMailboxConsumer,
        )

        # Phase 1: dead consumer reads + leaves entry in its PEL.
        consumer_dead = RedisMailboxConsumer(
            fake_redis, "root-1", "pod-old", "i-old"
        )
        await consumer_dead.ensure_group()
        pub = RedisMailboxPublisher(fake_redis)
        envelope_id = "01HSPYU0INITXACSTRICT0001"
        await pub.publish(_env(eid=envelope_id))
        await consumer_dead.read(count=10, block_ms=0)

        # Phase 2: fresh supervisor wired with strict stub.
        strict_repo = _StrictAuditRepo()
        ctx = SupervisorContext(
            root_session_id="root-1",
            pod_id="pod-new",
            instance_id="i-new",
            redis=fake_redis,
            audit_repo=strict_repo,
            publisher=RedisMailboxPublisher(fake_redis),
            sandbox_lifecycle=stub_lifecycle,
            agent_service_callback=stub_agent_callback,
            telemetry=_StubTelemetry(),
        )
        sup = MailboxSupervisor(ctx, block_ms=0, idle_poll_sleep_s=0.01)
        task = asyncio.create_task(sup.run())
        await asyncio.sleep(0.3)
        await sup.stop()
        await task

        key = ("root-1", envelope_id)
        assert key in strict_repo.rows, (
            "audit_repo row must exist after _initial_xautoclaim — "
            "upsert_processing wasn't called before increment_reclaim"
        )
        row = strict_repo.rows[key]
        assert row["reclaim_count"] >= 1, (
            f"reclaim_count must persist via audit_repo; got {row['reclaim_count']}"
        )
        assert row.get("processing_at") is not None, (
            "processing_at must be set by upsert_processing — "
            "absence proves the r2 [P1] fix is missing"
        )


# ──────────────────────────────────────────────────────────────────────────────
# PR-3b — Phase D: pod-restart clock recovery (spec §9.3)
# ──────────────────────────────────────────────────────────────────────────────


class TestClockRecovery:
    @pytest.mark.anyio
    async def test_last_seen_recovered_from_redis_entry_id(
        self, supervisor_ctx, fake_redis
    ):
        """Spec §9.3 — supervisor restart on the same pod recovers the
        child's last_seen heartbeat from the stream's XREVRANGE history."""
        # Pre-populate the stream with a real child-origin envelope.
        pub = RedisMailboxPublisher(fake_redis)
        await pub.publish(
            _env(
                t=MailboxEnvelopeType.PROGRESS_UPDATE,
                eid="01HSPYU0Q00000000000000001",
                producer_role=ProducerRole.CHILD_AGENT,
                child_session_id="child-1",
            )
        )

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._known_children = ["child-1"]  # noqa: SLF001 — registry would set this
        await sup._restore_last_seen_after_pod_restart()  # noqa: SLF001

        ls = sup.get_last_seen("child-1")
        assert ls is not None
        # Should be within ~2s of clock() now (fakeredis TIME returns ~now).
        now = supervisor_ctx.clock()
        assert now - 2.0 <= ls <= now + 0.5

    @pytest.mark.anyio
    async def test_last_seen_not_initialized_when_no_history(
        self, supervisor_ctx, fake_redis
    ):
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._known_children = ["child-without-history"]  # noqa: SLF001
        await sup._restore_last_seen_after_pod_restart()  # noqa: SLF001
        assert sup.get_last_seen("child-without-history") is None


# ──────────────────────────────────────────────────────────────────────────────
# PR-3b §5.8 layer 2 — consumer-side audit dedup (codex r6 [HIGH] fix)
# ──────────────────────────────────────────────────────────────────────────────


class TestAuditDedup:
    """Codex r6 [HIGH CONTRACT] regression — the crash window between
    side_effect success and XACK is recoverable iff the supervisor marked
    the envelope as processed BEFORE the XACK. On XAUTOCLAIM redelivery,
    a ``get_processed`` check at envelope entry must short-circuit and ACK
    without re-firing the handler — preventing double destroy() in PR-4 once
    terminal handlers wire real destructive side effects.

    Two halves:
    1. ``get_processed=True`` at envelope entry → ACK + skip handler
    2. side_effect success → ``mark_processed`` → ACK (in that order)
    """

    @pytest.mark.anyio
    async def test_handle_envelope_short_circuits_on_processed_dedup(
        self, monkeypatch, supervisor_ctx
    ):
        """Half 1 — pre-mark envelope as processed; assert handler never
        runs but XACK still happens.

        Simulates the crash-then-redelivery case: prior pod ran
        side_effect + mark_processed, crashed before XACK; XAUTOCLAIM
        redelivers; this pod sees ``get_processed=True`` and ACKs the
        dead envelope without re-firing the destructive handler.
        """
        sup = MailboxSupervisor(supervisor_ctx)

        envelope = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0D00000000DEDUPHIT01",
        )

        # Pre-mark as processed via the in-memory stub (matches the DB
        # impl's ON-CONFLICT-DO-UPDATE: row exists with processed_at set).
        await supervisor_ctx.audit_repo.upsert_processing(
            envelope, processing_at=supervisor_ctx.now()
        )
        await supervisor_ctx.audit_repo.mark_processed(
            envelope.parent_session_id,
            envelope.envelope_id,
            processed_at=supervisor_ctx.now(),
        )

        # Spy: handler must NOT run.
        handler_calls: list[str] = []
        original_handle = sup._dispatch[MailboxEnvelopeType.RESULT_READY].handle  # noqa: SLF001

        async def _spy_handle(env, ctx):
            handler_calls.append(env.envelope_id)
            return await original_handle(env, ctx)

        sup._dispatch[MailboxEnvelopeType.RESULT_READY].handle = _spy_handle  # noqa: SLF001 — type: ignore[method-assign]

        # Spy: ACK must run exactly once.
        ack_calls: list[bytes] = []
        original_ack = sup._consumer.ack  # noqa: SLF001

        async def _spy_ack(rid):
            ack_calls.append(rid)
            await original_ack(rid)

        monkeypatch.setattr(sup._consumer, "ack", _spy_ack)  # noqa: SLF001

        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        assert handler_calls == [], (
            "dedup hit — handler must NOT run; r6 [HIGH] short-circuit missing"
        )
        assert len(ack_calls) == 1, (
            "dedup hit — ACK must drain the redelivered envelope; "
            f"ack_calls={ack_calls!r}"
        )

    @pytest.mark.anyio
    async def test_side_effect_success_writes_mark_processed_before_xack(
        self, monkeypatch, supervisor_ctx
    ):
        """Half 2 — side_effect success path runs in order:
        side_effect → mark_processed → ACK.

        This is the WRITE half of the dedup pair. Without
        ``mark_processed`` between side_effect and XACK, the crash window
        is unrecoverable: redelivery hits ``get_processed=False`` and
        re-fires destroy().
        """
        sup = MailboxSupervisor(supervisor_ctx)

        envelope = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0D00000000MARK00001",
        )

        # Build a handler that returns ``ack=False, side_effect=<spy>``
        # so we hit the side_effect-success branch (not the no-side_effect
        # branch which just respects outcome.ack).
        call_order: list[str] = []

        async def _side_effect():
            call_order.append("side_effect")

        _side_effect_ref = _side_effect

        class _TestHandler:
            async def handle(self, env, ctx):
                return HandlerOutcome(
                    ack=False,
                    side_effect=_side_effect_ref,
                    audit_payload={},
                )

        sup._dispatch[MailboxEnvelopeType.RESULT_READY] = _TestHandler()  # noqa: SLF001

        # Spy: mark_processed must run between side_effect and ACK.
        original_mark = supervisor_ctx.audit_repo.mark_processed

        async def _spy_mark_processed(parent, env_id, *, processed_at):
            call_order.append("mark_processed")
            await original_mark(parent, env_id, processed_at=processed_at)

        monkeypatch.setattr(  # noqa: SLF001
            supervisor_ctx.audit_repo, "mark_processed", _spy_mark_processed
        )

        # Spy: ACK call order.
        original_ack = sup._consumer.ack  # noqa: SLF001

        async def _spy_ack(rid):
            call_order.append("ack")
            await original_ack(rid)

        monkeypatch.setattr(sup._consumer, "ack", _spy_ack)  # noqa: SLF001

        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        assert call_order == ["side_effect", "mark_processed", "ack"], (
            "r6 [HIGH] CONTRACT — side_effect success path must run in "
            f"order side_effect → mark_processed → ACK; got: {call_order!r}"
        )

        # mark_processed actually persisted (audit row has processed_at).
        row = await supervisor_ctx.audit_repo.fetch_raw(
            envelope.parent_session_id, envelope.envelope_id
        )
        assert row.get("processed_at") is not None, (
            "audit row missing processed_at after side_effect success — "
            "dedup marker not written"
        )

    @pytest.mark.anyio
    async def test_fresh_delivery_upserts_audit_row_before_dispatch(
        self, fake_redis, stub_lifecycle, stub_agent_callback
    ):
        """Codex r8 [HIGH] regression — fresh XREADGROUP delivery path
        MUST call ``upsert_processing`` before dispatch.

        Pre-r8: only the XAUTOCLAIM paths called ``upsert_processing``.
        Fresh delivery → handler → side_effect → mark_processed (DB raises
        ValueError on missing row) → broad-except logs + ACKs → no dedup
        marker. Next XAUTOCLAIM redelivery re-fires destroy().

        This test locks the fix down with the DB-faithful ``_StrictAuditRepo``:
        ``mark_processed`` raises if no row exists or ``processing_at`` is
        None. The supervisor MUST upsert_processing first; otherwise the
        whole §5.8 layer 2 dedup contract is silently broken.
        """
        strict_repo = _StrictAuditRepo()
        ctx = SupervisorContext(
            root_session_id="root-1",
            pod_id="pod-a",
            instance_id="i1",
            redis=fake_redis,
            audit_repo=strict_repo,
            publisher=RedisMailboxPublisher(fake_redis),
            sandbox_lifecycle=stub_lifecycle,
            agent_service_callback=stub_agent_callback,
            telemetry=_StubTelemetry(),
        )
        sup = MailboxSupervisor(ctx)

        envelope = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0R8000000FRESHDELIV1",
        )

        side_effect_ran = False

        async def _side_effect():
            nonlocal side_effect_ran
            side_effect_ran = True

        _side_effect_ref = _side_effect

        class _TestHandler:
            async def handle(self, env, ctx):
                return HandlerOutcome(
                    ack=False,
                    side_effect=_side_effect_ref,
                    audit_payload={},
                )

        sup._dispatch[MailboxEnvelopeType.RESULT_READY] = _TestHandler()  # noqa: SLF001

        # Drive a FRESH delivery — no prior XAUTOCLAIM upsert.
        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        # side_effect must have run.
        assert side_effect_ran, "handler dispatch failed — side_effect didn't fire"

        # The DB-faithful strict stub raises on mark_processed if no row.
        # If upsert_processing was NOT called before dispatch, the strict
        # stub's mark_processed would have raised, and the broad-except in
        # supervisor would have swallowed it, leaving processed_at unset.
        # Assert the row exists AND has processed_at set — proves both
        # upsert_processing AND mark_processed ran successfully.
        key = (envelope.parent_session_id, envelope.envelope_id)
        assert key in strict_repo.rows, (
            "r8 [HIGH] regression — upsert_processing must run before "
            "dispatch on fresh delivery; audit row missing entirely"
        )
        assert strict_repo.rows[key].get("processing_at") is not None, (
            "r8 [HIGH] regression — upsert_processing must set processing_at "
            "before dispatch; row exists but processing_at is None"
        )
        assert strict_repo.rows[key].get("processed_at") is not None, (
            "r8 [HIGH] regression — mark_processed should succeed because "
            "upsert_processing pre-staged the row; if processed_at is None, "
            "the strict stub's mark_processed raised and the broad-except "
            "swallowed it (the very bug r8 closed)"
        )

    @pytest.mark.anyio
    async def test_handle_envelope_proceeds_when_get_processed_raises(
        self, monkeypatch, supervisor_ctx
    ):
        """Codex r6 follow-up — ``get_processed`` failure must degrade safely
        toward handler-runs (the pre-r6 behavior), not silently skip the
        envelope. A false-positive (treating fresh as processed) would lose
        work; a false-negative (proceeding to handler) re-runs at-most-once.
        """
        sup = MailboxSupervisor(supervisor_ctx)

        async def _boom_get_processed(parent, env_id):
            raise RuntimeError("synthetic audit-DB crash")

        monkeypatch.setattr(  # noqa: SLF001
            supervisor_ctx.audit_repo, "get_processed", _boom_get_processed
        )

        envelope = _env(
            t=MailboxEnvelopeType.PROGRESS_UPDATE,
            eid="01HSPYU0D00000000GETPROC01",
        )

        # The non-terminal stub handler forwards to agent_service_callback;
        # assert it received the envelope (proves we didn't short-circuit).
        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        received_ids = [
            e.envelope_id for e in supervisor_ctx.agent_service_callback.received
        ]
        assert envelope.envelope_id in received_ids, (
            "get_processed failure should NOT short-circuit — handler must "
            "still run; received_ids={received_ids!r}"
        )

    @pytest.mark.anyio
    async def test_side_effect_acks_even_when_mark_processed_raises(
        self, monkeypatch, supervisor_ctx
    ):
        """Codex r6 follow-up — ``mark_processed`` failure must still ACK.

        Refusing to ACK on mark_processed failure would force redelivery,
        and since side_effect already ran, the next pass (with
        ``get_processed=False`` because the marker write failed) would
        re-fire destroy(). ACK-on-mark-failure accepts at-most-once
        dedup-loss in the rare audit-DB-down case in exchange for
        avoiding guaranteed double-destroy. PR-4 should make
        side_effect + mark_processed transactional.
        """
        sup = MailboxSupervisor(supervisor_ctx)

        envelope = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0D00000000MARKFAIL1",
        )

        async def _side_effect():
            pass  # success

        _side_effect_ref = _side_effect

        class _TestHandler:
            async def handle(self, env, ctx):
                return HandlerOutcome(
                    ack=False,
                    side_effect=_side_effect_ref,
                    audit_payload={},
                )

        sup._dispatch[MailboxEnvelopeType.RESULT_READY] = _TestHandler()  # noqa: SLF001

        async def _boom_mark_processed(parent, env_id, *, processed_at):
            raise RuntimeError("synthetic audit-DB crash")

        monkeypatch.setattr(  # noqa: SLF001
            supervisor_ctx.audit_repo, "mark_processed", _boom_mark_processed
        )

        ack_calls: list[bytes] = []
        original_ack = sup._consumer.ack  # noqa: SLF001

        async def _spy_ack(rid):
            ack_calls.append(rid)
            await original_ack(rid)

        monkeypatch.setattr(sup._consumer, "ack", _spy_ack)  # noqa: SLF001

        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        assert len(ack_calls) == 1, (
            "mark_processed failure must NOT block ACK — refusing to ACK "
            "would guarantee double-destroy on redelivery"
        )
