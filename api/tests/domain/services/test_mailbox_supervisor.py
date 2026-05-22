"""C3 PR-3a — MailboxSupervisor main loop + dispatch table (spec §6.1/§6.3/§6.4).

PR-3a tests use stubbed handlers — only verify routing + ACK. PR-4 tests
exercise full handler behavior including destroy hooks.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.application.services.mailbox_supervisor import (
    MailboxSupervisor,
    SupervisorContext,
)
from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
    RedisMailboxPublisher,
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
) -> MailboxEnvelope:
    if payload is None:
        if t == MailboxEnvelopeType.PROGRESS_UPDATE:
            payload = {"kind": "heartbeat", "visibility": "hidden"}
        elif t == MailboxEnvelopeType.RESULT_READY:
            payload = {"summary": "done", "outcome": "success"}
        else:  # pragma: no cover — defensive
            payload = {}
    return MailboxEnvelope(
        envelope_id=eid,
        type=t,
        parent_session_id="root-1",
        child_session_id="child-1",
        correlation_id="01HSPYU0000000000000000002",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=payload,
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
