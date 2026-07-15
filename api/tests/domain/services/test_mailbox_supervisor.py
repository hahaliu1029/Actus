"""C3 PR-3a/PR-3b — MailboxSupervisor main loop + reliability layer.

PR-3a tests use stubbed handlers — only verify routing + ACK. PR-3b extends
with last_seen heartbeat tracking, XAUTOCLAIM startup + periodic, poison
drop after MAILBOX_POISON_MAX_RECLAIM, and pod-restart clock recovery.
PR-4 tests exercise full handler behavior including destroy hooks.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.coordinator_liveness_lease_service import (
    CoordinatorChildLease,
)
from app.application.services.mailbox_supervisor import (
    HandlerOutcome,
    MailboxSupervisor,
    ResultReadyHandler,
    SupervisorContext,
)
from app.domain.models.mailbox_envelope import (
    MAILBOX_POISON_MAX_RECLAIM,
    CancelPolicy,
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


@pytest.mark.anyio
async def test_startup_restore_registers_running_children_without_respawn(
    supervisor_ctx,
) -> None:
    row = SimpleNamespace(
        session_id="child-restore",
        coordinator_run_id="run-restore",
        work_unit_id="wu-restore",
    )
    supervisor_ctx.session_repo = SimpleNamespace(
        find_running_mailbox_children_for_parent=AsyncMock(return_value=[row]),
    )
    lease = CoordinatorChildLease(
        root_session_id="root-1",
        parent_session_id="root-1",
        child_session_id="child-restore",
        coordinator_run_id="run-restore",
        work_unit_id="wu-restore",
        last_seen_epoch=1.0,
        phase="starting",
    )
    liveness = SimpleNamespace(
        get_lease=AsyncMock(return_value=lease),
        is_stale=MagicMock(return_value=False),
        record_startup_lease=AsyncMock(),
        record_heartbeat=AsyncMock(),
        mark_terminal=AsyncMock(),
    )
    supervisor_ctx.liveness_service = liveness
    sup = MailboxSupervisor(supervisor_ctx)

    await sup._restore_coordinator_liveness_after_startup()  # noqa: SLF001

    assert sup._known_children == ["child-restore"]  # noqa: SLF001
    liveness.record_startup_lease.assert_not_awaited()
    assert not hasattr(supervisor_ctx, "child_runner_starter")


@pytest.mark.anyio
async def test_startup_restore_missing_lease_gets_full_startup_grace(
    supervisor_ctx,
) -> None:
    row = SimpleNamespace(
        session_id="child-missing",
        coordinator_run_id="run-missing",
        work_unit_id="wu-missing",
    )
    supervisor_ctx.session_repo = SimpleNamespace(
        find_running_mailbox_children_for_parent=AsyncMock(return_value=[row]),
    )
    liveness = SimpleNamespace(
        get_lease=AsyncMock(return_value=None),
        is_stale=MagicMock(),
        record_startup_lease=AsyncMock(return_value=SimpleNamespace()),
        record_heartbeat=AsyncMock(),
        mark_terminal=AsyncMock(),
    )
    supervisor_ctx.liveness_service = liveness
    sup = MailboxSupervisor(supervisor_ctx)

    await sup._restore_coordinator_liveness_after_startup()  # noqa: SLF001

    liveness.record_startup_lease.assert_awaited_once_with(
        root_session_id="root-1",
        parent_session_id="root-1",
        child_session_id="child-missing",
        coordinator_run_id="run-missing",
        work_unit_id="wu-missing",
        last_seen_age_seconds=None,
    )
    assert sup.get_last_seen("child-missing") is not None


@pytest.mark.anyio
@pytest.mark.parametrize("application_clock", [-1_000_000.0, 1_000_000.0])
async def test_startup_restore_uses_redis_time_stream_age_across_clock_offsets(
    supervisor_ctx,
    application_clock: float,
) -> None:
    row = SimpleNamespace(
        session_id="child-history",
        coordinator_run_id="run-history",
        work_unit_id="wu-history",
    )
    supervisor_ctx.session_repo = SimpleNamespace(
        find_running_mailbox_children_for_parent=AsyncMock(return_value=[row]),
    )
    liveness = SimpleNamespace(
        get_lease=AsyncMock(return_value=None),
        record_startup_lease=AsyncMock(return_value=SimpleNamespace()),
    )
    supervisor_ctx.liveness_service = liveness
    supervisor_ctx.clock = lambda: application_clock
    supervisor_ctx.redis.time = AsyncMock(
        return_value=(1_720_000_095, 623_000),
    )
    sup = MailboxSupervisor(supervisor_ctx)
    sup._latest_child_origin_entry_ms = AsyncMock(  # type: ignore[method-assign]
        return_value=1_720_000_000_123,
    )

    await sup._restore_coordinator_liveness_after_startup()  # noqa: SLF001

    liveness.record_startup_lease.assert_awaited_once_with(
        root_session_id="root-1",
        parent_session_id="root-1",
        child_session_id="child-history",
        coordinator_run_id="run-history",
        work_unit_id="wu-history",
        last_seen_age_seconds=95.5,
    )


@pytest.mark.anyio
async def test_startup_restore_propagates_redis_time_failure_for_stream_age(
    supervisor_ctx,
) -> None:
    row = SimpleNamespace(
        session_id="child-history",
        coordinator_run_id="run-history",
        work_unit_id="wu-history",
    )
    supervisor_ctx.session_repo = SimpleNamespace(
        find_running_mailbox_children_for_parent=AsyncMock(return_value=[row]),
    )
    supervisor_ctx.liveness_service = SimpleNamespace(
        get_lease=AsyncMock(return_value=None),
        record_startup_lease=AsyncMock(return_value=SimpleNamespace()),
    )
    supervisor_ctx.redis.time = AsyncMock(
        side_effect=ConnectionError("redis TIME unavailable"),
    )
    sup = MailboxSupervisor(supervisor_ctx)
    sup._latest_child_origin_entry_ms = AsyncMock(  # type: ignore[method-assign]
        return_value=1_720_000_000_123,
    )

    with pytest.raises(ConnectionError, match="redis TIME unavailable"):
        await sup._restore_coordinator_liveness_after_startup()  # noqa: SLF001


@pytest.mark.anyio
async def test_startup_restore_clamps_future_stream_id_to_zero_age(
    supervisor_ctx,
) -> None:
    row = SimpleNamespace(
        session_id="child-future",
        coordinator_run_id="run-future",
        work_unit_id="wu-future",
    )
    supervisor_ctx.session_repo = SimpleNamespace(
        find_running_mailbox_children_for_parent=AsyncMock(return_value=[row]),
    )
    liveness = SimpleNamespace(
        get_lease=AsyncMock(return_value=None),
        record_startup_lease=AsyncMock(return_value=SimpleNamespace()),
    )
    supervisor_ctx.liveness_service = liveness
    supervisor_ctx.redis.time = AsyncMock(return_value=(1_000, 0))
    sup = MailboxSupervisor(supervisor_ctx)
    sup._latest_child_origin_entry_ms = AsyncMock(  # type: ignore[method-assign]
        return_value=1_000_001,
    )

    await sup._restore_coordinator_liveness_after_startup()  # noqa: SLF001

    assert liveness.record_startup_lease.await_args.kwargs[
        "last_seen_age_seconds"
    ] == 0.0


@pytest.mark.anyio
async def test_stream_fallback_skips_newer_untrusted_entries_for_older_heartbeat(
    supervisor_ctx,
    fake_redis,
) -> None:
    stream_key = "actus:child:root-1:mailbox"
    valid = _env().model_copy(update={
        "correlation_id": "hb:child-1",
        "emitted_at": datetime(2099, 1, 1, tzinfo=timezone.utc),
    })
    valid_id = await fake_redis.xadd(
        stream_key,
        {"envelope": valid.model_dump_json()},
        id="1720000000000-0",
    )
    newer_untrusted = [
        _env(t=MailboxEnvelopeType.RESULT_READY).model_copy(update={
            "correlation_id": "run-1",
        }),
        _env(parent_session_id="other-root").model_copy(update={
            "correlation_id": "hb:child-1",
        }),
        _env(producer_role=ProducerRole.PARENT_AGENT).model_copy(update={
            "correlation_id": "hb:child-1",
        }),
        _env().model_copy(update={"correlation_id": "run-1"}),
    ]
    for index, envelope in enumerate(newer_untrusted, start=1):
        await fake_redis.xadd(
            stream_key,
            {"envelope": envelope.model_dump_json()},
            id=f"{1720000000000 + index}-0",
        )
    sup = MailboxSupervisor(supervisor_ctx)

    restored_ms = await sup._latest_child_origin_entry_ms(  # noqa: SLF001
        stream_key,
        "child-1",
    )

    raw_id = valid_id.decode() if isinstance(valid_id, bytes) else valid_id
    assert restored_ms == int(raw_id.split("-", 1)[0])


@pytest.mark.anyio
async def test_stale_restored_lease_reuses_existing_orphan_cascade(
    supervisor_ctx,
) -> None:
    lease = CoordinatorChildLease(
        root_session_id="root-1",
        parent_session_id="root-1",
        child_session_id="child-stale",
        coordinator_run_id="run-stale",
        work_unit_id="wu-stale",
        last_seen_epoch=1.0,
        phase="running",
    )
    liveness = SimpleNamespace(
        get_lease=AsyncMock(return_value=lease),
        is_stale=MagicMock(return_value=True),
        record_startup_lease=AsyncMock(),
        record_heartbeat=AsyncMock(),
        mark_terminal=AsyncMock(),
    )
    supervisor_ctx.liveness_service = liveness
    sup = MailboxSupervisor(supervisor_ctx)
    sup._known_children = ["child-stale"]  # noqa: SLF001
    sup._last_orphan_check_mono = -100.0  # noqa: SLF001
    sup._emit_cascade_terminate = AsyncMock()  # type: ignore[method-assign]

    await sup._maybe_tick_check_orphans()  # noqa: SLF001

    sup._emit_cascade_terminate.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_rejected_coordinator_heartbeat_does_not_extend_local_liveness(
    supervisor_ctx,
) -> None:
    liveness = SimpleNamespace(
        get_lease=AsyncMock(),
        record_heartbeat=AsyncMock(return_value=False),
    )
    supervisor_ctx.liveness_service = liveness
    sup = MailboxSupervisor(supervisor_ctx)
    envelope = _env(payload={"kind": "heartbeat", "visibility": "hidden"})

    await sup._refresh_last_seen_if_child_origin(envelope)  # noqa: SLF001

    assert sup.get_last_seen("child-1") is None
    assert sup._known_children == []  # noqa: SLF001
    liveness.get_lease.assert_not_awaited()


@pytest.mark.anyio
async def test_ordinary_research_heartbeat_refreshes_local_liveness(
    supervisor_ctx,
) -> None:
    liveness = SimpleNamespace(
        record_heartbeat=AsyncMock(return_value=None),
    )
    supervisor_ctx.liveness_service = liveness
    supervisor_ctx.clock = lambda: 456.0
    sup = MailboxSupervisor(supervisor_ctx)
    envelope = _env(payload={"kind": "heartbeat", "visibility": "hidden"})

    await sup._refresh_last_seen_if_child_origin(envelope)  # noqa: SLF001

    assert sup.get_last_seen("child-1") == 456.0
    assert sup._known_children == []  # noqa: SLF001


@pytest.mark.anyio
async def test_first_coordinator_heartbeat_tracks_child_only_after_durable_acceptance(
    supervisor_ctx,
) -> None:
    liveness = SimpleNamespace(record_heartbeat=AsyncMock(return_value=True))
    supervisor_ctx.liveness_service = liveness
    supervisor_ctx.clock = lambda: 123.0
    sup = MailboxSupervisor(supervisor_ctx)
    envelope = _env(payload={"kind": "heartbeat", "visibility": "hidden"})

    await sup._refresh_last_seen_if_child_origin(envelope)  # noqa: SLF001

    liveness.record_heartbeat.assert_awaited_once_with(envelope)
    assert sup.get_last_seen("child-1") == 123.0
    assert sup._known_children == ["child-1"]  # noqa: SLF001


@pytest.mark.anyio
async def test_first_nonheartbeat_coordinator_event_switches_to_durable_authority(
    supervisor_ctx,
) -> None:
    lease = CoordinatorChildLease(
        root_session_id="root-1",
        parent_session_id="root-1",
        child_session_id="child-1",
        coordinator_run_id="run-1",
        work_unit_id="wu-1",
        last_seen_epoch=1.0,
        phase="starting",
        authority_age_seconds=0.0,
    )
    liveness = SimpleNamespace(
        get_lease=AsyncMock(return_value=lease),
        is_stale=MagicMock(return_value=False),
        record_heartbeat=AsyncMock(),
    )
    supervisor_ctx.liveness_service = liveness
    supervisor_ctx.clock = lambda: 123.0
    sup = MailboxSupervisor(supervisor_ctx)
    envelope = _env(
        payload={"kind": "tool_started", "visibility": "hidden"},
    )

    await sup._refresh_last_seen_if_child_origin(envelope)  # noqa: SLF001

    assert sup._known_children == ["child-1"]  # noqa: SLF001
    assert sup.get_last_seen("child-1") is None
    liveness.record_heartbeat.assert_not_awaited()

    sup._last_orphan_check_mono = 0.0  # noqa: SLF001
    sup._emit_cascade_terminate = AsyncMock()  # type: ignore[method-assign]
    await sup._maybe_tick_check_orphans()  # noqa: SLF001

    liveness.is_stale.assert_called_once_with(lease)
    sup._emit_cascade_terminate.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_first_nonheartbeat_legacy_event_keeps_local_liveness_fallback(
    supervisor_ctx,
) -> None:
    liveness = SimpleNamespace(
        get_lease=AsyncMock(return_value=None),
        record_heartbeat=AsyncMock(),
    )
    supervisor_ctx.liveness_service = liveness
    supervisor_ctx.clock = lambda: 123.0
    sup = MailboxSupervisor(supervisor_ctx)
    envelope = _env(
        payload={"kind": "tool_started", "visibility": "hidden"},
    )

    await sup._refresh_last_seen_if_child_origin(envelope)  # noqa: SLF001

    liveness.get_lease.assert_awaited_once_with("child-1")
    assert sup._known_children == []  # noqa: SLF001
    assert sup.get_last_seen("child-1") == 123.0


@pytest.mark.anyio
async def test_terminal_tombstone_failure_keeps_tracking_and_pel_side_effect(
    supervisor_ctx,
    stub_lifecycle,
) -> None:
    supervisor_ctx.terminalize_child = AsyncMock(return_value=True)
    supervisor_ctx.liveness_service = SimpleNamespace(
        mark_terminal=AsyncMock(side_effect=RuntimeError("redis down")),
    )
    sup = MailboxSupervisor(supervisor_ctx)
    sup._last_seen_mono["child-1"] = 1.0  # noqa: SLF001
    outcome = await ResultReadyHandler().handle(
        _env(t=MailboxEnvelopeType.RESULT_READY), supervisor_ctx,
    )

    with pytest.raises(RuntimeError, match="redis down"):
        await outcome.side_effect()

    assert sup.get_last_seen("child-1") == 1.0
    assert stub_lifecycle.destroy_calls == []


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
async def test_supervisor_dispatches_terminal_envelope_to_real_handler(
    supervisor_ctx, fake_redis, stub_lifecycle, stub_agent_callback
):
    """PR-4 swaps the stub terminal handler for the real ``ResultReadyHandler``.
    A fresh RESULT_READY envelope must drive ``destroy()`` with
    ``SUBAGENT_TERMINAL_RESULT`` and dispatch the in-process callback.
    """
    from app.domain.models.session import DestroyReason

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
    assert ("child-1", DestroyReason.SUBAGENT_TERMINAL_RESULT) in (
        stub_lifecycle.destroy_calls
    )
    assert any(
        e.envelope_id == "01HSPYU0R00000000000000000"
        for e in stub_agent_callback.received
    )
    # Envelope was ACKed — no lingering PEL entry.
    pending = await fake_redis.xpending(
        "actus:child:root-1:mailbox", "actus:mailbox-supervisor:v1"
    )
    assert pending["pending"] == 0


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
        return {MailboxEnvelopeType.RESULT_READY: ms._StubNonTerminalHandler()}

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


@pytest.mark.anyio
async def test_supervisor_run_loop_iteration_broad_except_swallows_runtime_error(
    supervisor_ctx, monkeypatch
):
    """codex r5 [R5-1, HIGH TEST] — complementary lock for T5
    (integration ``test_mailbox_crash_recovery.py``).

    T5 validates the **registry's** restart input contract by injecting a
    pre-failed task into the registry slot. The other half of spec §13.1
    is the **supervisor's** outer broad-except guard at
    ``MailboxSupervisor.run`` (line ~1347): any non-CancelledError raised
    during a single iteration (``read``, ``_maybe_periodic_xautoclaim``,
    ``_maybe_tick_cancel_check``, ``_maybe_tick_check_orphans``) MUST be
    logged + 0.5s sleep + continue — i.e., the supervisor task itself
    keeps running. That guard is what makes synthesizable raises (the
    most common test injection vector) STRUCTURALLY unable to crash a
    real ``MailboxSupervisor``, which is why T5 had to drop down to
    direct slot injection.

    This test pins that guard: monkeypatch ``_maybe_tick_check_orphans``
    to raise once and then succeed; the task MUST survive the raise and
    process a follow-up envelope through the dispatch path. If a future
    refactor removes the outer broad-except (or narrows it), the
    supervisor task starts dying on transient faults → registry restart
    storms → load amplification on every transient. T5 + this test are
    the joint lock on spec §13.1.
    """
    sup = MailboxSupervisor(
        supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
    )

    # Inject a transient raise into the periodic orphan tick. The first
    # iteration raises; subsequent iterations succeed. The outer broad-
    # except in run() should catch + log + sleep + continue.
    tick_calls = {"n": 0}
    original_tick = sup._maybe_tick_check_orphans  # noqa: SLF001

    async def _flaky_orphan_tick():
        tick_calls["n"] += 1
        if tick_calls["n"] == 1:
            raise RuntimeError(
                "simulated transient fault in orphan tick (test injection)"
            )
        await original_tick()

    monkeypatch.setattr(sup, "_maybe_tick_check_orphans", _flaky_orphan_tick)  # noqa: SLF001

    task = asyncio.create_task(sup.run())
    try:
        # Give the loop time to hit the flaky tick at least twice (so the
        # second call succeeds and proves the task survived the raise).
        for _ in range(40):
            await asyncio.sleep(0.05)
            if tick_calls["n"] >= 2:
                break

        # Supervisor task must STILL be running (broad-except swallowed
        # the RuntimeError, slept 0.5s, then continued).
        assert not task.done(), (
            "outer broad-except in MailboxSupervisor.run must keep the "
            "task alive across non-CancelledError raises; instead the "
            "task terminated — this means the broad-except guard at "
            "lines ~1347 is missing or narrowed, which would cause "
            "supervisor task to die on every transient fault and force "
            "the registry restart loop to handle workloads it shouldn't"
        )
        assert tick_calls["n"] >= 2, (
            "loop must continue past the first raise so the second "
            "iteration's orphan tick runs; otherwise the broad-except "
            "broke the loop without re-entering"
        )

        # Now drive a regular envelope through to confirm the task is
        # not just "alive but stuck" — handler dispatch still works.
        pub = RedisMailboxPublisher(supervisor_ctx.redis)
        await pub.publish(_env(eid="01HSPYU0R51000000BROADEXC01"))
        for _ in range(40):
            await asyncio.sleep(0.05)
            received_ids = [
                e.envelope_id
                for e in supervisor_ctx.agent_service_callback.received
            ]
            if "01HSPYU0R51000000BROADEXC01" in received_ids:
                break
        received_ids = [
            e.envelope_id
            for e in supervisor_ctx.agent_service_callback.received
        ]
        assert "01HSPYU0R51000000BROADEXC01" in received_ids, (
            "post-raise iteration must dispatch a fresh envelope — "
            "confirms the broad-except continued the run-loop, not just "
            "kept the asyncio.Task alive in a stalled state"
        )
    finally:
        await sup.stop(drain_timeout_s=2.0)
        await task


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
        """The heartbeat refresh fires BEFORE dispatch, but a successful
        CancelAck terminal destroy then runs ``ctx.clear_child_tracking``
        (codex F9 HIGH) to remove ``_last_seen_mono`` so the orphan
        detector doesn't fire a spurious cascade against an already-dead
        child. Drive the refresh path directly (no terminal handler) to
        verify the heartbeat-gate wiring stays intact for the refresh
        layer that ``_handle_envelope`` runs at line 1016.
        """
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        envelope = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            eid="01HSPYU0L00000000000000003",
            producer_role=ProducerRole.CHILD_AGENT,
            child_session_id="child-1",
        )
        # Exercise just the heartbeat-refresh helper (the same helper
        # ``_handle_envelope`` calls before dispatch). Routing through the
        # full handler would also run CancelAckHandler's terminal destroy
        # → ``clear_child_tracking`` → wipe the refresh we just made.
        await sup._refresh_last_seen_if_child_origin(envelope)  # noqa: SLF001
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
            ).model_copy(update={"correlation_id": "hb:child-1"})
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


class TestSupervisorPodRestartClockRecovery:
    """codex r9b [R9b-1, HIGH ARCH] — caller wiring deferred to PR-5.

    PR-4 ships ``_restore_last_seen_after_pod_restart`` plus its in-isolation
    coverage above (``TestClockRecovery``) but the actual caller wiring
    (``SupervisorRegistry.spawn`` populates ``_known_children`` from the
    session repo at supervisor startup, then invokes the routine after
    ``ensure_group()``) is deferred to PR-5 to avoid widening PR-4's
    plumbing budget. Runtime safety while unwired depends on the empty-
    list default making the routine a no-op. This test locks that
    invariant — if a future change accidentally populates
    ``_known_children`` from an unintended source, or changes the
    routine to perform work even for an empty input set, this test
    breaks first.
    """

    @pytest.mark.anyio
    async def test_unwired_known_children_default_is_safe_noop(
        self, supervisor_ctx, fake_redis
    ):
        """An unwired supervisor (``_known_children`` is the default empty
        list, no registry → session_repo plumbing yet) MUST run
        ``_restore_last_seen_after_pod_restart`` as a complete no-op:

        * no exception (even if ``redis.time`` / ``xrevrange`` would
          fail — we never reach them on the empty-list path);
        * ``_last_seen_mono`` is not mutated;
        * the routine returns cleanly so the caller (eventually PR-5's
          registry wiring) can treat it as idempotent.
        """
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        # Default state — exactly as ``__init__`` leaves it.
        assert sup._known_children == []  # noqa: SLF001 — locked default
        # Snapshot ``_last_seen_mono`` so we can assert no mutation.
        before_snapshot = dict(sup._last_seen_mono)  # noqa: SLF001

        # No exception. Returns None.
        result = await sup._restore_last_seen_after_pod_restart()  # noqa: SLF001
        assert result is None

        # No mutation of the heartbeat ledger.
        assert sup._last_seen_mono == before_snapshot  # noqa: SLF001


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
    async def test_side_effect_success_with_mark_processed_failure_retains_pel(
        self, monkeypatch, supervisor_ctx
    ):
        """Codex r4 [R4-3, HIGH CONTRACT] — ``mark_processed`` failure on the
        side_effect-success path MUST NOT ACK.

        Prior to R4-3 the supervisor ACKed anyway, sacrificing the §5.8
        durable-dedup contract to "avoid double-destroy on redelivery".
        That tradeoff was inverted: the terminal handlers' destroy() is
        idempotent (``SandboxAlreadyDestroyed`` / ``SandboxBindingMissing``
        are explicitly classified as terminal-success), so leaving the
        entry in the PEL is safe — XAUTOCLAIM redelivers, destroy raises
        AlreadyDestroyed, terminal-success ACKs the entry, and
        mark_processed is retried until the DB recovers (or
        ``MAILBOX_POISON_MAX_RECLAIM`` trips and poison-drop fires).

        This test verifies the inverted contract: no ACK, no broken
        audit row, entry stays in PEL.
        """
        sup = MailboxSupervisor(supervisor_ctx)

        envelope = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0R430MARKFAILPEL0001",
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

        # Must NOT raise — the failure is swallowed and the entry is
        # left in the PEL.
        await sup._handle_envelope(b"0-r43-1", envelope)  # noqa: SLF001

        assert ack_calls == [], (
            "mark_processed failure on the side_effect-success path MUST "
            "leave the envelope in the PEL — R4-3 inverts the prior policy "
            "that ACKed without a durable dedup marker"
        )

    @pytest.mark.anyio
    async def test_side_effect_success_with_mark_processed_success_acks(
        self, monkeypatch, supervisor_ctx
    ):
        """Codex r4 [R4-3, HIGH CONTRACT] — regression guard for the happy
        path. When ``mark_processed`` succeeds after a successful
        side_effect, ACK fires + the audit row's ``processed_at`` is
        populated (the dedup contract holds).
        """
        sup = MailboxSupervisor(supervisor_ctx)

        envelope = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0R430MARKOKACKOK00001",
        )

        async def _side_effect():
            pass  # success — no DB writes, just the outer mark_processed

        _side_effect_ref = _side_effect

        class _TestHandler:
            async def handle(self, env, ctx):
                return HandlerOutcome(
                    ack=False,
                    side_effect=_side_effect_ref,
                    audit_payload={},
                )

        sup._dispatch[MailboxEnvelopeType.RESULT_READY] = _TestHandler()  # noqa: SLF001
        # The audit repo needs an existing row for mark_processed to
        # update; mirror handle_envelope's pre-stage upsert.
        await supervisor_ctx.audit_repo.upsert_processing(
            envelope, processing_at=supervisor_ctx.now()
        )

        ack_calls: list[bytes] = []
        original_ack = sup._consumer.ack  # noqa: SLF001

        async def _spy_ack(rid):
            ack_calls.append(rid)
            await original_ack(rid)

        monkeypatch.setattr(sup._consumer, "ack", _spy_ack)  # noqa: SLF001

        await sup._handle_envelope(b"0-r43-2", envelope)  # noqa: SLF001

        assert ack_calls == [b"0-r43-2"], (
            "side_effect success + mark_processed success must ACK"
        )
        assert await supervisor_ctx.audit_repo.get_processed(
            envelope.parent_session_id, envelope.envelope_id
        ), "mark_processed success must leave a durable dedup marker"


# ──────────────────────────────────────────────────────────────────────────────
# PR-4 Phase A — ResultReadyHandler (spec §7.3)
# ──────────────────────────────────────────────────────────────────────────────


class TestResultReadyHandlerDestroyClassification:
    """Spec §7.3 — ResultReadyHandler classifies destroy outcomes:
      - success         → mark_processed + ACK + dispatch ChildDoneEvent
      - AlreadyDestroyed → mark_processed + ACK (idempotent success)
      - BindingMissing   → mark_processed + ACK (terminal-success)
      - retryable        → no mark_processed, no ACK (PEL retry)
    """

    @pytest.mark.anyio
    async def test_success_path_marks_processed_acks_and_dispatches(
        self, supervisor_ctx, stub_lifecycle, stub_agent_callback
    ):
        from app.application.services.mailbox_supervisor import (
            ResultReadyHandler,
        )
        from app.domain.models.session import DestroyReason

        handler = ResultReadyHandler()
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0R10000000000000001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        # side_effect runs first; supervisor ACKs after on success.
        assert outcome.ack is False
        assert outcome.side_effect is not None
        await outcome.side_effect()
        assert (env.child_session_id, DestroyReason.SUBAGENT_TERMINAL_RESULT) in (
            stub_lifecycle.destroy_calls
        )
        assert any(
            e.envelope_id == env.envelope_id for e in stub_agent_callback.received
        )
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_already_destroyed_is_treated_as_terminal_success(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            ResultReadyHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed

        async def _raise_already(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_already
        )
        handler = ResultReadyHandler()
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0R20000000000000001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        # Terminal-success: mark_processed must be set even on idempotent no-op.
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert "mailbox.destroy_idempotent_noop" in emitted_names

    @pytest.mark.anyio
    async def test_binding_missing_is_treated_as_terminal_success(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            ResultReadyHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxBindingMissing

        async def _raise_missing(session_id, reason):
            raise SandboxBindingMissing(session_id)

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_missing
        )
        handler = ResultReadyHandler()
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0R30000000000000001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert "mailbox.destroy_binding_missing" in emitted_names

    @pytest.mark.anyio
    async def test_retryable_failure_does_not_mark_processed_and_raises(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            ResultReadyHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError

        async def _raise_transient(session_id, reason):
            raise SandboxLifecycleError("transient")

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_transient
        )
        handler = ResultReadyHandler()
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0R40000000000000001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        with pytest.raises(SandboxLifecycleError):
            await outcome.side_effect()
        # Retryable: must NOT mark_processed so XAUTOCLAIM redelivers.
        assert not await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert "mailbox.destroy_retryable_failed" in emitted_names

    @pytest.mark.anyio
    async def test_dedup_hit_returns_ack_without_side_effect(self, supervisor_ctx):
        """Side-effect-first idempotency — re-delivery after mark_processed → ACK,
        no destroy."""
        from app.application.services.mailbox_supervisor import (
            ResultReadyHandler,
        )

        handler = ResultReadyHandler()
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0R50000000000000001",
        )
        await supervisor_ctx.audit_repo.upsert_processing(
            env, processing_at=supervisor_ctx.now()
        )
        await supervisor_ctx.audit_repo.mark_processed(
            env.parent_session_id,
            env.envelope_id,
            processed_at=supervisor_ctx.now(),
        )
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is True
        assert outcome.side_effect is None


# ──────────────────────────────────────────────────────────────────────────────
# PR-4 Phase B — CancelAckHandler (spec §7.4)
# ──────────────────────────────────────────────────────────────────────────────


class TestCancelAckHandlerDestroyClassification:
    """Spec §7.4 — same shape as ResultReady but destroy reason CANCEL_ACK_OBSERVED."""

    @pytest.mark.anyio
    async def test_success_path_marks_processed_and_dispatches(
        self, supervisor_ctx, stub_lifecycle, stub_agent_callback
    ):
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
        )
        from app.domain.models.session import DestroyReason

        handler = CancelAckHandler()
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0C10000000000000001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is False
        assert outcome.side_effect is not None
        await outcome.side_effect()
        assert (env.child_session_id, DestroyReason.CANCEL_ACK_OBSERVED) in (
            stub_lifecycle.destroy_calls
        )
        assert any(
            e.envelope_id == env.envelope_id for e in stub_agent_callback.received
        )
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_already_destroyed_is_terminal_success(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed

        async def _raise(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(supervisor_ctx.sandbox_lifecycle, "destroy", _raise)
        handler = CancelAckHandler()
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0C20000000000000001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_retryable_failure_raises_without_marking(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError

        async def _raise(session_id, reason):
            raise SandboxLifecycleError("transient")

        monkeypatch.setattr(supervisor_ctx.sandbox_lifecycle, "destroy", _raise)
        handler = CancelAckHandler()
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0C30000000000000001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        with pytest.raises(SandboxLifecycleError):
            await outcome.side_effect()
        assert not await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    # ──────────────────────────────────────────────────────────────────────
    # codex r2 [R2-4, HIGH CONTRACT] — supervisor-echo short-circuit.
    # ──────────────────────────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_supervisor_echo_cancel_ack_short_circuits_destroy(
        self, supervisor_ctx, stub_lifecycle, stub_agent_callback
    ):
        """``CancelRequestHandler._terminate_outcome`` publishes a synthetic
        ``producer_role=SUPERVISOR_ECHO`` CANCEL_ACK so external observers
        (frontend SSE bridge, parent audit trail) see the terminal
        transition (spec §9.2). The same supervisor reads that envelope
        back via XREADGROUP and dispatches into CancelAckHandler. Without
        the short-circuit, every FORCE_TERMINATE produces a second
        destroy attempt that raises SandboxAlreadyDestroyed (idempotent
        terminal-success) → wasted lifecycle work + duplicate telemetry.
        """
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
        )

        handler = CancelAckHandler()
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.SUPERVISOR_ECHO,
            eid="01HSPYU0C40000ECHO0000000001",
            payload={"final_state": "force_terminated"},
        )
        outcome = await handler.handle(env, supervisor_ctx)
        # ack=True + terminalize-only side_effect. Production runs its
        # independently row-guarded CAS; this legacy context has no terminalizer
        # port, so awaiting it is a no-op. Duplicate destroy/callback stay skipped.
        assert outcome.ack is True
        assert outcome.side_effect is not None
        await outcome.side_effect()
        assert outcome.audit_payload == {"supervisor_echo": True}
        # The destroy must NOT fire — terminal handler already destroyed
        # the child on the originating TERMINATE side_effect.
        assert stub_lifecycle.destroy_calls == []
        # The agent_service_callback must NOT fire either — this is an
        # echo for external observers; the original TERMINATE already
        # cooperatively stopped the agent in step 1.
        assert stub_agent_callback.received == []
        # upsert_processing happened so the audit trail records the echo.
        row = await supervisor_ctx.audit_repo.fetch_raw(
            env.parent_session_id, env.envelope_id
        )
        assert row.get("processing_at") is not None

    @pytest.mark.anyio
    async def test_child_agent_cancel_ack_still_destroys_after_r2_4(
        self, supervisor_ctx, stub_lifecycle, stub_agent_callback
    ):
        """codex r2 [R2-4] regression cover — the supervisor-echo guard
        must NOT swallow real CHILD_AGENT cancellation acks. Those still
        fire destroy(CANCEL_ACK_OBSERVED) per spec §7.4.
        """
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
        )
        from app.domain.models.session import DestroyReason

        handler = CancelAckHandler()
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0C50000CHILD0000000001",
            payload={"final_state": "cancelled"},
        )
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is False
        assert outcome.side_effect is not None
        await outcome.side_effect()
        assert (env.child_session_id, DestroyReason.CANCEL_ACK_OBSERVED) in (
            stub_lifecycle.destroy_calls
        )
        # Agent callback also fires on the real child-cancellation path.
        assert any(
            e.envelope_id == env.envelope_id
            for e in stub_agent_callback.received
        )


class TestCancelAckHandlerEchoTrust:
    """codex r9b [R9b-2, HIGH CONTRACT] — make the SUPERVISOR_ECHO cleanup
    trust contract explicit (see ``CancelAckHandler`` for the full rationale).

    The supervisor-echo short-circuit (R2-4) is load-bearing for the
    FORCE_TERMINATE flow and still trusts the wire ``producer_role`` for the
    narrow decision to suppress duplicate destroy/callback. Task 4 terminal DB
    authority is not covered by that trust: when the port is wired, the
    application helper independently verifies the authoritative session row.

    This test locks only the duplicate-cleanup behaviour so a future origin
    verification change has to flip an explicit assertion.
    """

    @pytest.mark.anyio
    async def test_supervisor_echo_short_circuit_is_trust_based(
        self, supervisor_ctx, stub_lifecycle, stub_agent_callback
    ):
        """The short-circuit fires purely on ``producer_role``. The wire
        envelope_id pattern (``ack:{sha256...}`` after R3-7) is NOT a
        discriminator — producers MUST be trusted to reserve the
        producer_role per the spec §9.2 publisher contract.

        A forged envelope presenting ``producer_role=SUPERVISOR_ECHO``
        with an arbitrary envelope_id (one that does NOT match the
        supervisor's synthetic ``ack:{hash}`` pattern) IS still honoured.
        This documents the cleanup trust assumption; it does not bypass the
        separate row guard for Task 4 terminal DB ownership.
        """
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
        )

        handler = CancelAckHandler()
        # envelope_id deliberately does NOT match the supervisor's
        # synthetic ``ack:{hash}`` pattern — a producer that knows the
        # contract could mint any 26-char ULID-shaped id here.
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.SUPERVISOR_ECHO,
            eid="01HSPYU0R9B2ECHOFORGEDID01",
            payload={"final_state": "force_terminated"},
        )
        outcome = await handler.handle(env, supervisor_ctx)
        # Cleanup trust contract: SUPERVISOR_ECHO short-circuits duplicate
        # destroy/callback regardless of envelope_id shape. This legacy context
        # has no terminalizer port, so the side_effect is a no-op; production
        # performs a separately row-guarded terminal CAS.
        assert outcome.ack is True
        assert outcome.side_effect is not None
        await outcome.side_effect()
        assert outcome.audit_payload == {"supervisor_echo": True}
        assert stub_lifecycle.destroy_calls == []
        assert stub_agent_callback.received == []


# ──────────────────────────────────────────────────────────────────────────────
# PR-4 Phase C — CancelRequestHandler (spec §7.6 + §8)
# ──────────────────────────────────────────────────────────────────────────────


def _cancel_env(
    *,
    policy,
    eid: str = "01HSPYU0X1CANCELREQ000000001",
    parent_session_id: str = "root-1",
    child_session_id: str = "child-1",
    producer_role: ProducerRole = ProducerRole.PARENT_AGENT,
    reason: str = "test_reason",
) -> MailboxEnvelope:
    return MailboxEnvelope(
        envelope_id=eid,
        type=MailboxEnvelopeType.CANCEL_REQUEST,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        correlation_id=eid + "C",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=producer_role,
        payload={"reason": reason, "policy": policy.value},
    )


class TestCancelRequestHandlerTerminate:
    """Spec §7.6 + §8.2 — TERMINATE policy drives stop → destroy → echo CANCEL_ACK."""

    @pytest.mark.anyio
    async def test_terminate_policy_destroys_with_force_terminate_reason(
        self, supervisor_ctx, stub_lifecycle, stub_agent_callback, fake_redis
    ):
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy
        from app.domain.models.session import DestroyReason

        # Need supervisor instance to wire register_cancel_state hook.
        MailboxSupervisor(supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01)

        handler = CancelRequestHandler()
        env = _cancel_env(policy=CancelPolicy.TERMINATE)
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is False
        assert outcome.side_effect is not None
        await outcome.side_effect()

        # stop_session (via agent_service_callback) ran first; then destroy.
        assert any(
            e.envelope_id == env.envelope_id for e in stub_agent_callback.received
        ), "agent_service_callback must be invoked first to stop the agent task"
        assert (env.child_session_id, DestroyReason.FORCE_TERMINATE) in (
            stub_lifecycle.destroy_calls
        )

        # mark_processed completed.
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

        # Synthetic CANCEL_ACK echo published (producer_role=SUPERVISOR_ECHO).
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        ack_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_ACK"
        ]
        assert len(ack_entries) == 1, (
            "TERMINATE branch must publish exactly one synthetic CANCEL_ACK"
        )
        # The producer_role is recorded in the envelope payload — decode it.
        import json as _json

        ack_envelope = _json.loads(ack_entries[0][1][b"envelope"])
        assert ack_envelope["producer_role"] == "supervisor_echo", (
            "synthetic ACK must use SUPERVISOR_ECHO (not CHILD_AGENT) or "
            "last_seen heartbeat invariant breaks"
        )

    @pytest.mark.anyio
    async def test_terminate_already_destroyed_idempotent(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed
        from app.domain.models.mailbox_envelope import CancelPolicy

        MailboxSupervisor(supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01)

        async def _raise(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(supervisor_ctx.sandbox_lifecycle, "destroy", _raise)

        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0X1CANCELREQ000000002",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        # Idempotent: still marked processed.
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_terminate_retryable_failure_raises(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError
        from app.domain.models.mailbox_envelope import CancelPolicy

        MailboxSupervisor(supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01)

        async def _raise(session_id, reason):
            raise SandboxLifecycleError("transient")

        monkeypatch.setattr(supervisor_ctx.sandbox_lifecycle, "destroy", _raise)
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0X1CANCELREQ000000003",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        with pytest.raises(SandboxLifecycleError):
            await outcome.side_effect()
        assert not await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )


class TestCancelRequestHandlerRequestCancel:
    """Spec §8.3 — REQUEST_CANCEL forwards + records cancel_state, no destroy."""

    @pytest.mark.anyio
    async def test_request_cancel_forwards_and_records_state(
        self, supervisor_ctx, stub_lifecycle, stub_agent_callback
    ):
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        handler = CancelRequestHandler()
        env = _cancel_env(policy=CancelPolicy.REQUEST_CANCEL)
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()

        # No destroy yet — child still has CHILD_CANCEL_ACK_TIMEOUT_MS.
        assert stub_lifecycle.destroy_calls == []
        # Forwarded to in-process callback so child agent can cooperate.
        assert any(
            e.envelope_id == env.envelope_id for e in stub_agent_callback.received
        )
        # Cancel state recorded with REQUEST_CANCEL.
        state = sup._cancel_states.get(env.child_session_id)  # noqa: SLF001
        assert state is not None
        assert state.policy == CancelPolicy.REQUEST_CANCEL
        # mark_processed completed.
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_request_cancel_registers_state_before_callback_failure(
        self, supervisor_ctx, monkeypatch
    ):
        """codex r5 [R5-3, HIGH CONTRACT] — register_cancel_state MUST run
        before agent_service_callback so the §8.4 auto-escalate tick is
        armed even when the callback hangs or raises.

        Pre-fix ordering: callback → register_cancel_state. A callback that
        raised (e.g. agent_service down, SSE bridge bug) left
        ``_cancel_states[child]`` empty → §8.4 tick saw no state for this
        child → REQUEST_CANCEL never escalated to TERMINATE → child stuck
        until the 90s orphan tick. Fix: register state FIRST, wrap the
        callback in try/except so a callback fault doesn't abort
        mark_processed.

        This regression test injects a callback that raises and asserts
        ``_cancel_states[child]`` IS registered + ``mark_processed`` IS
        completed (proving the side_effect ran to completion despite the
        callback raise).
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        # Replace agent_service_callback with one that always raises.
        async def _failing_callback(envelope):
            raise RuntimeError(
                "simulated callback failure (agent_service unreachable)"
            )

        monkeypatch.setattr(
            supervisor_ctx, "agent_service_callback", _failing_callback
        )

        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.REQUEST_CANCEL,
            eid="01HSPYU0R53000000CALLBACKFAIL",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        # MUST NOT raise — callback failure is wrapped so mark_processed
        # still runs (otherwise the envelope would loop in PEL forever).
        await outcome.side_effect()

        # R5-3 invariant — cancel state IS registered even though the
        # callback raised. Pre-fix this would be ``state is None`` and
        # the auto-escalate tick would never see the child.
        state = sup._cancel_states.get(env.child_session_id)  # noqa: SLF001
        assert state is not None, (
            "register_cancel_state MUST run before the callback so the "
            "§8.4 auto-escalate tick is armed even on callback fault; "
            "pre-fix ordering (callback → register_cancel_state) would "
            "leave _cancel_states empty after a callback raise"
        )
        assert state.policy == CancelPolicy.REQUEST_CANCEL, (
            "cancel state must record REQUEST_CANCEL policy so the §8.4 "
            "tick knows to escalate to TERMINATE after the ACK timeout"
        )
        # mark_processed STILL ran — the side_effect completed end-to-end
        # despite the callback raise.
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        ), (
            "mark_processed must complete despite callback failure so "
            "the envelope dedups on redelivery; otherwise XAUTOCLAIM "
            "would refire the callback in a loop"
        )


class TestCancelRequestHandlerDedup:
    @pytest.mark.anyio
    async def test_dedup_hit_returns_ack_without_side_effect(
        self, supervisor_ctx
    ):
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        MailboxSupervisor(supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01)
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0X1CANCELREQ000000DUP",
        )
        await supervisor_ctx.audit_repo.upsert_processing(
            env, processing_at=supervisor_ctx.now()
        )
        await supervisor_ctx.audit_repo.mark_processed(
            env.parent_session_id,
            env.envelope_id,
            processed_at=supervisor_ctx.now(),
        )
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is True
        assert outcome.side_effect is None


# ──────────────────────────────────────────────────────────────────────────────
# PR-4 Phase D — ApprovalRequestHandler stub (spec §10.2)
# ──────────────────────────────────────────────────────────────────────────────


def _approval_env(
    *,
    eid: str = "01HSPYU0A10000APPROVAL000001",
    correlation_id: str = "01HSPYU0A10000APPROVAL000CID",
    tool_call_id: str = "tc-1",
    parent_session_id: str = "root-1",
    child_session_id: str = "child-1",
) -> MailboxEnvelope:
    from app.domain.models.mailbox_envelope import (
        APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
        ApprovalRequestPayload,
    )

    payload = ApprovalRequestPayload(
        tool_name="shell_execute",
        tool_args_snapshot={"cmd": "ls"},
        risk_tier="medium",
        rationale="test",
        correlation_id=correlation_id,
        tool_call_id=tool_call_id,
        requested_at=datetime.now(tz=timezone.utc),
        timeout_seconds=APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
    )
    return MailboxEnvelope(
        envelope_id=eid,
        type=MailboxEnvelopeType.APPROVAL_REQUEST,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        correlation_id=correlation_id,
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=payload.model_dump(mode="json"),
    )


class TestApprovalRequestHandlerStub:
    """Spec §10.2 — immediate deny (NOT 300s wait); paired APPROVAL_RESPONSE
    envelope with decided_by=AUTO_POLICY."""

    @pytest.mark.anyio
    async def test_immediate_deny_publishes_paired_response(
        self, supervisor_ctx, fake_redis
    ):
        from app.application.services.mailbox_supervisor import (
            ApprovalRequestHandler,
        )

        handler = ApprovalRequestHandler()
        env = _approval_env(correlation_id="cid-deny-1", tool_call_id="tc-deny-1")
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is True
        assert outcome.side_effect is None

        # Verify response envelope was published.
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        responses = [
            e for e in entries if e[1].get(b"type") == b"APPROVAL_RESPONSE"
        ]
        assert len(responses) == 1, (
            "stub handler must emit exactly one paired APPROVAL_RESPONSE"
        )
        import json as _json

        resp = _json.loads(responses[0][1][b"envelope"])
        assert resp["payload"]["correlation_id"] == "cid-deny-1"
        assert resp["payload"]["approved"] is False
        assert resp["payload"]["decided_by"] == "auto_policy"


class TestApprovalRequestCorrelationIdMismatch:
    """Codex r4 [R4-2, HIGH CONTRACT] — spec §13.1 T7 invariant: when an
    APPROVAL_REQUEST envelope carries a ``payload.correlation_id`` that
    diverges from ``envelope.correlation_id``, the supervisor MUST refuse
    to publish a response and instead ACK+drop with telemetry. Publishing
    a deny response keyed to either id would either (a) leave the child's
    in-flight tool call awaiting a never-arriving response or (b) drift a
    response with the wrong key into the stream where the SSE bridge
    might mis-route it to a different tool call.
    """

    @pytest.mark.anyio
    async def test_mismatched_correlation_id_publishes_deny_keyed_to_envelope_cid(
        self, supervisor_ctx, fake_redis
    ):
        """codex r6 [R6-4, HIGH CONTRACT] — payload.correlation_id !=
        envelope.correlation_id → ACK + publish an immediate deny keyed
        to ``envelope.correlation_id`` (supervisor's trusted source).
        The pre-R6-4 behavior (ACK without publish) forced the child to
        wait the full 300s timeout; R6-4 unblocks it immediately.
        """
        from app.application.services.mailbox_supervisor import (
            ApprovalRequestHandler,
        )

        handler = ApprovalRequestHandler()
        env = _approval_env(
            eid="01HSPYU0R420APPROVAL0MISMATCH",
            correlation_id="envelope-cid-good",
        )
        env = env.model_copy(
            update={
                "payload": {
                    **env.payload,
                    "correlation_id": "payload-cid-bad",
                },
            }
        )
        assert env.correlation_id != env.payload["correlation_id"]

        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is True
        assert outcome.side_effect is None
        assert outcome.audit_payload == {"correlation_id_mismatch": True}

        # codex r6 [R6-4] — APPROVAL_RESPONSE IS published, keyed to
        # envelope.correlation_id (NOT the bogus payload one).
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        responses = [
            e for e in entries if e[1].get(b"type") == b"APPROVAL_RESPONSE"
        ]
        assert len(responses) == 1, (
            "R6-4 — mismatch path must publish a deny keyed to "
            "envelope.correlation_id so the child unblocks immediately "
            "instead of timing out at 300s"
        )
        import json as _json

        resp_env = _json.loads(responses[0][1][b"envelope"])
        assert resp_env["correlation_id"] == "envelope-cid-good", (
            f"deny envelope must be keyed to envelope.correlation_id "
            f"(trusted source), got {resp_env['correlation_id']!r}"
        )
        assert resp_env["payload"]["correlation_id"] == "envelope-cid-good"
        assert resp_env["payload"]["approved"] is False
        assert resp_env["payload"]["decided_by"] == "auto_policy"
        assert resp_env["producer_role"] == "supervisor"
        assert "correlation_id_mismatch" in resp_env["payload"]["reason"]
        # Synthetic envelope_id stays ≤ 64 chars (R3-7 invariant).
        assert resp_env["envelope_id"].startswith("mismatch_deny:")
        assert len(resp_env["envelope_id"]) <= 64

        # Telemetry fired with both ids + producer_role for ops triage.
        emitted = supervisor_ctx.telemetry.emitted
        mismatch_events = [
            data
            for name, data in emitted
            if name == "mailbox.approval_correlation_id_mismatch"
        ]
        assert len(mismatch_events) == 1
        ev = mismatch_events[0]
        assert ev["envelope_correlation_id"] == "envelope-cid-good"
        assert ev["payload_correlation_id"] == "payload-cid-bad"
        assert ev["producer_role"] == ProducerRole.CHILD_AGENT.value
        assert ev["envelope_id"] == env.envelope_id

    @pytest.mark.anyio
    async def test_matched_correlation_ids_publish_response_normally(
        self, supervisor_ctx, fake_redis
    ):
        """The happy path is unchanged: payload.correlation_id ==
        envelope.correlation_id → paired response published as before.
        Regression guard so the R4-2 mismatch check doesn't accidentally
        block the normal flow.
        """
        from app.application.services.mailbox_supervisor import (
            ApprovalRequestHandler,
        )

        handler = ApprovalRequestHandler()
        env = _approval_env(
            eid="01HSPYU0R420APPROVAL0MATCHED",
            correlation_id="cid-matched-r42",
        )
        # Sanity — factory wires them in lock-step.
        assert env.correlation_id == env.payload["correlation_id"]

        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is True
        assert outcome.side_effect is None
        assert outcome.audit_payload == {"pe_stub_denied": True}

        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        responses = [
            e for e in entries if e[1].get(b"type") == b"APPROVAL_RESPONSE"
        ]
        assert len(responses) == 1
        import json as _json

        resp = _json.loads(responses[0][1][b"envelope"])
        assert resp["payload"]["correlation_id"] == "cid-matched-r42"

    def test_envelope_rejects_approval_request_missing_payload_correlation_id(
        self,
    ):
        """codex r9b [R9b-3, MEDIUM TEST] — ``ApprovalRequestPayload``'s
        ``correlation_id`` is a required field (see
        ``api/app/domain/models/mailbox_envelope.py`` :class:`ApprovalRequestPayload`),
        and the ``MailboxEnvelope._validate_payload_matches_type`` validator
        re-runs the typed payload check at envelope-construction time. Any
        APPROVAL_REQUEST envelope reaching the supervisor over the wire
        therefore MUST carry ``payload.correlation_id``; producers that omit
        it cannot construct a ``MailboxEnvelope`` at all.

        Pre-R9b this position was tested via ``model_copy(update={...})``
        which bypasses the model validator and exercised a "fallback to
        ``envelope.correlation_id``" branch in ``ApprovalRequestHandler``
        that is unreachable from real wire input. R9b deletes that dead
        handler branch and flips this test into a CONTRACT GUARD: assert
        the envelope-level validator REJECTS payloads without
        ``correlation_id``. If a future spec change relaxes the required
        constraint, this test breaks first and forces a reviewer to
        revisit the handler branch.

        Companion model-level guard:
        ``tests/domain/services/test_mailbox_envelope.py::test_approval_request_correlation_id_required``
        already asserts ``ApprovalRequestPayload`` itself rejects the
        missing field; this guard adds the envelope-level layer so both
        construction paths (direct payload model + ``MailboxEnvelope``
        validator) stay locked together.
        """
        from datetime import datetime, timezone

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MailboxEnvelope(
                envelope_id="01HSPYU0R9B3APPROVALMISSINGC",
                type=MailboxEnvelopeType.APPROVAL_REQUEST,
                parent_session_id="root-1",
                child_session_id="child-1",
                correlation_id="envelope-cid",
                emitted_at=datetime.now(tz=timezone.utc),
                producer_role=ProducerRole.CHILD_AGENT,
                payload={
                    "tool_name": "shell_execute",
                    "tool_args_snapshot": {"cmd": "ls"},
                    "risk_tier": "medium",
                    "rationale": "test",
                    # correlation_id deliberately omitted — must fail.
                    "tool_call_id": "tc-1",
                    "requested_at": datetime.now(tz=timezone.utc).isoformat(),
                    "timeout_seconds": 60,
                },
            )

    @pytest.mark.anyio
    async def test_mismatch_telemetry_failure_still_publishes_deny(
        self, supervisor_ctx, monkeypatch, fake_redis
    ):
        """codex r6 [R6-4] — telemetry sink failure on the mismatch path
        MUST NOT block the deny publish. Telemetry is best-effort; the
        deny keyed to envelope.correlation_id is the load-bearing
        invariant so the child unblocks immediately.
        """
        from app.application.services.mailbox_supervisor import (
            ApprovalRequestHandler,
        )

        async def _raise(*args, **kwargs):
            raise RuntimeError("OTel down")

        monkeypatch.setattr(supervisor_ctx.telemetry, "emit", _raise)

        handler = ApprovalRequestHandler()
        env = _approval_env(
            eid="01HSPYU0R420APPROVAL0TELRAIS",
            correlation_id="envelope-cid",
        )
        env = env.model_copy(
            update={
                "payload": {
                    **env.payload,
                    "correlation_id": "payload-cid-bad",
                },
            }
        )

        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is True
        assert outcome.audit_payload == {"correlation_id_mismatch": True}

        # codex r6 [R6-4] — deny IS published even when telemetry raises.
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        responses = [
            e for e in entries if e[1].get(b"type") == b"APPROVAL_RESPONSE"
        ]
        assert len(responses) == 1, (
            "R6-4 — telemetry fault must not block the deny publish; "
            "the child still unblocks immediately"
        )
        import json as _json

        resp_env = _json.loads(responses[0][1][b"envelope"])
        assert resp_env["correlation_id"] == "envelope-cid"
        assert resp_env["payload"]["approved"] is False


# ──────────────────────────────────────────────────────────────────────────────
# PR-4 Phase E — HandoffRequestHandler stub (spec §6.6)
# ──────────────────────────────────────────────────────────────────────────────


def _handoff_env(
    eid: str = "01HSPYU0H10000HANDOFF0000001",
) -> MailboxEnvelope:
    from app.domain.models.mailbox_envelope import HandoffRequestPayload

    payload = HandoffRequestPayload(
        handoff_target="human",
        reason="example",
        context_summary="some context",
    )
    return MailboxEnvelope(
        envelope_id=eid,
        type=MailboxEnvelopeType.HANDOFF_REQUEST,
        parent_session_id="root-1",
        child_session_id="child-1",
        correlation_id=eid + "C",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=payload.model_dump(mode="python"),
    )


class TestHandoffRequestHandlerStub:
    """Spec §6.6 — C3 ship only freezes envelope schema; handler emits telemetry."""

    @pytest.mark.anyio
    async def test_handoff_request_emits_telemetry_and_acks(
        self, supervisor_ctx, stub_lifecycle, stub_agent_callback
    ):
        from app.application.services.mailbox_supervisor import (
            HandoffRequestHandler,
        )

        handler = HandoffRequestHandler()
        env = _handoff_env()
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is True
        assert outcome.side_effect is None
        # Telemetry recorded as unsupported.
        emitted = supervisor_ctx.telemetry.emitted
        assert any(
            n == "mailbox.handoff_request_unsupported" for n, _ in emitted
        )
        # No destruction.
        assert stub_lifecycle.destroy_calls == []
        # No agent callback dispatched — telemetry-only stub.
        assert stub_agent_callback.received == []


# ──────────────────────────────────────────────────────────────────────────────
# PR-4 Phase F — Orphan tick (spec §7.5 + §3.2 M8)
# ──────────────────────────────────────────────────────────────────────────────


class TestOrphanTick:
    """Spec §7.5 — children whose ``last_seen_mono`` is stale longer than
    SUBAGENT_PROGRESS_STALE_AFTER_SECONDS get a synthetic CANCEL_REQUEST(TERMINATE)
    cascaded onto their mailbox stream."""

    @pytest.mark.anyio
    async def test_stale_child_triggers_cascade_terminate(
        self, supervisor_ctx, fake_redis
    ):
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.mailbox_envelope import (
            SUBAGENT_PROGRESS_STALE_AFTER_SECONDS,
        )

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._ORPHAN_CHECK_INTERVAL_S = 0.0  # noqa: SLF001
        sup._last_orphan_check_mono = 0.0  # noqa: SLF001
        # Make child appear stale.
        now = supervisor_ctx.clock()
        stale_mono = now - (SUBAGENT_PROGRESS_STALE_AFTER_SECONDS + 5)
        sup._last_seen_mono["child-stale"] = stale_mono  # noqa: SLF001

        await sup._maybe_tick_check_orphans()  # noqa: SLF001

        emitted = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert "mailbox.orphan_detected" in emitted
        # Synthetic CANCEL_REQUEST published.
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert len(cancel_entries) == 1
        import json as _json

        env = _json.loads(cancel_entries[0][1][b"envelope"])
        assert env["payload"]["policy"] == "TERMINATE"
        assert env["payload"]["reason"] == "orphan_timeout"
        assert env["producer_role"] == "supervisor"
        # Child cleared from tracking so we don't double-cascade.
        assert "child-stale" not in sup._last_seen_mono  # noqa: SLF001

    @pytest.mark.anyio
    async def test_fresh_child_not_cascaded(self, supervisor_ctx, fake_redis):
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._ORPHAN_CHECK_INTERVAL_S = 0.0  # noqa: SLF001
        sup._last_orphan_check_mono = 0.0  # noqa: SLF001
        sup._last_seen_mono["child-fresh"] = supervisor_ctx.clock()  # noqa: SLF001

        await sup._maybe_tick_check_orphans()  # noqa: SLF001

        emitted = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert "mailbox.orphan_detected" not in emitted
        # No cascade publish.
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert cancel_entries == []
        # Child still tracked.
        assert "child-fresh" in sup._last_seen_mono  # noqa: SLF001


# ──────────────────────────────────────────────────────────────────────────────
# PR-4 Phase G — _on_poison_drop overrides emit cascade for terminal types
# ──────────────────────────────────────────────────────────────────────────────


class TestPoisonDropCascade:
    """Spec §5.7 step 3 — when a terminal-type envelope reaches poison drop AND
    a child_session_id is present, supervisor fires synthetic CANCEL_REQUEST
    (TERMINATE) so the orphaned child still gets destroyed."""

    @pytest.mark.anyio
    async def test_poison_drop_on_terminal_envelope_fires_cascade(
        self, supervisor_ctx, fake_redis
    ):
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0POISONCASCADE00001",
            child_session_id="child-cascade",
        )
        await sup._on_poison_drop(env)  # noqa: SLF001
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert len(cancel_entries) == 1
        import json as _json

        cascaded = _json.loads(cancel_entries[0][1][b"envelope"])
        assert cascaded["payload"]["policy"] == "TERMINATE"
        assert cascaded["payload"]["reason"] == "poison_drop_terminal_envelope"

    @pytest.mark.anyio
    async def test_poison_drop_on_nonterminal_envelope_does_not_cascade(
        self, supervisor_ctx, fake_redis
    ):
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        env = _env(
            t=MailboxEnvelopeType.PROGRESS_UPDATE,
            eid="01HSPYU0POISONNOCAS0000001",
            child_session_id="child-x",
        )
        await sup._on_poison_drop(env)  # noqa: SLF001
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert cancel_entries == []

    @pytest.mark.anyio
    async def test_poison_drop_empty_child_session_id_does_not_cascade(
        self, supervisor_ctx, fake_redis
    ):
        """Spec §5.7 step 3 guards on non-empty ``child_session_id`` — synthetic
        cross-root forgeries (already filtered earlier) and any future envelope
        path that allows an empty id should NOT trigger a cascade. The wire
        contract forbids empty ``child_session_id`` for legitimate envelopes,
        but the supervisor's poison-drop hook codes defensively because the
        cost of a wrong-cascade is destroying an unrelated child.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        # Build a real envelope, then bypass the frozen guard via
        # ``object.__setattr__`` to force the empty-id condition the spec
        # gate is written to defend against.
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0POISONNOCHILD00001",
            child_session_id="placeholder",
        )
        object.__setattr__(env, "child_session_id", "")
        await sup._on_poison_drop(env)  # noqa: SLF001
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert cancel_entries == []


# ──────────────────────────────────────────────────────────────────────────────
# codex r2 [R2-5, HIGH CONTRACT] — cascade envelope_id must fit String(64).
# ──────────────────────────────────────────────────────────────────────────────


class TestCascadeEnvelopeIdBudget:
    """Cascade envelope_id must always fit the audit table's
    ``String(64)`` column. Prior format
    ``cascade:{reason}:{child_id}:{ts_ms}`` could exceed 64 chars with
    UUID child_ids + verbose reasons.

    New format: ``cascade:{sha256(reason:child_id:ts_ms)[:32]}`` →
    always 39 chars. Idempotent across (reason, child_id, time) and
    collision-resistant across distinct times.
    """

    @pytest.mark.anyio
    async def test_cascade_envelope_id_fits_audit_column_with_uuid_child(
        self, supervisor_ctx, fake_redis
    ):
        """Any (reason, UUID-child_id) pair must produce ≤ 64 chars."""
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        # Replicate the codex r2 [R2-5] worst-case child id (UUID, 36 char)
        # plus the longest cascade reason currently used:
        # "poison_drop_terminal_envelope" (29 char).
        for reason in (
            "cancel_ack_timeout",
            "orphan_timeout",
            "poison_drop_terminal_envelope",
        ):
            await sup._emit_cascade_terminate(  # noqa: SLF001
                "550e8400-e29b-41d4-a716-446655440000",
                reason=reason,
            )

        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        import json as _json

        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert len(cancel_entries) == 3
        for _, fields in cancel_entries:
            env = _json.loads(fields[b"envelope"])
            envelope_id = env["envelope_id"]
            assert envelope_id.startswith("cascade:")
            assert len(envelope_id) <= 64, (
                f"cascade envelope_id {envelope_id!r} (len={len(envelope_id)}) "
                f"exceeds the audit table's String(64) column budget"
            )

    @pytest.mark.anyio
    async def test_cascade_envelope_id_idempotent_per_clock_tick(
        self, supervisor_ctx, fake_redis, monkeypatch
    ):
        """Two cascades with the same (reason, child_id, clock_ms) tuple
        produce identical envelope_ids (deterministic hash). The audit
        repo's PK ensures the second publish stays out via dedup; the
        idempotency on the publisher side is a nice-to-have but the
        load-bearing guarantee is in the audit upsert.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        # Freeze the clock so two cascades land in the same ms bucket.
        frozen = 12345.6789
        monkeypatch.setattr(supervisor_ctx, "clock", lambda: frozen)
        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-1", reason="orphan_timeout"
        )
        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-1", reason="orphan_timeout"
        )
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        import json as _json

        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        # Two publishes; both must carry the SAME envelope_id (the audit
        # repo would catch the second on upsert via PK conflict).
        ids = {
            _json.loads(fields[b"envelope"])["envelope_id"]
            for _, fields in cancel_entries
        }
        assert len(ids) == 1

    @pytest.mark.anyio
    async def test_cascade_envelope_id_differs_across_clock_ticks(
        self, supervisor_ctx, fake_redis, monkeypatch
    ):
        """Same (reason, child_id) at two different millisecond ticks
        must produce different envelope_ids — otherwise a stuck cascade
        retry would collide with the first publish's audit row + the
        operator would see a single record instead of separate attempts.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        clock_state = {"t": 1000.0}
        monkeypatch.setattr(
            supervisor_ctx, "clock", lambda: clock_state["t"]
        )
        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-1", reason="orphan_timeout"
        )
        clock_state["t"] = 2000.0
        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-1", reason="orphan_timeout"
        )
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        import json as _json

        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        ids = {
            _json.loads(fields[b"envelope"])["envelope_id"]
            for _, fields in cancel_entries
        }
        assert len(ids) == 2, (
            f"two cascades with different clock ticks must produce "
            f"distinct envelope_ids; got {ids!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# codex r2 [R2-6] superseded by codex r6 [R6-2, HIGH CONTRACT] —
# orphan/poison cascades thread ``DestroyReason.ORPHAN_TIMEOUT`` via the
# supervisor-private ``_cascade_destroy_overrides`` side-table (NOT the
# wire payload, which is frozen at C3 ship as ``{reason, policy}``).
# Parent cascades have no side-table entry → handler defaults to
# FORCE_TERMINATE.
# ──────────────────────────────────────────────────────────────────────────────


class TestCascadeDestroyReason:
    """codex r6 [R6-2] — side-table semantics for orphan/poison cascades.

    The publisher (``_emit_cascade_terminate``) stamps an entry in
    ``MailboxSupervisor._cascade_destroy_overrides`` BEFORE
    ``publisher.publish``. The handler (``CancelRequestHandler.
    _terminate_outcome``) reads+pops the entry on dispatch. The wire
    payload is unchanged at ``{reason, policy}``; external producers
    cannot reach the side-table, so a hostile override attack is
    structurally impossible (no R3-6 producer_role guard needed).
    """

    @pytest.mark.anyio
    async def test_orphan_cascade_stamps_side_table_with_orphan_timeout(
        self, supervisor_ctx, fake_redis
    ):
        """codex r6 [R6-2] — orphan tick stamps the supervisor-private
        side-table BEFORE publish; payload stays frozen at
        ``{reason, policy}``.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.mailbox_envelope import (
            SUBAGENT_PROGRESS_STALE_AFTER_SECONDS,
        )
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._ORPHAN_CHECK_INTERVAL_S = 0.0  # noqa: SLF001
        sup._last_orphan_check_mono = 0.0  # noqa: SLF001
        now = supervisor_ctx.clock()
        sup._last_seen_mono["child-orphan"] = (  # noqa: SLF001
            now - (SUBAGENT_PROGRESS_STALE_AFTER_SECONDS + 5)
        )
        await sup._maybe_tick_check_orphans()  # noqa: SLF001

        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        import json as _json

        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert len(cancel_entries) == 1
        env = _json.loads(cancel_entries[0][1][b"envelope"])
        # codex r6 [R6-2] — wire payload is frozen at {reason, policy}.
        # destroy_reason must NOT appear on the wire.
        assert set(env["payload"].keys()) == {"reason", "policy"}, (
            f"payload schema regressed; got {env['payload']!r}"
        )
        # The override lives in the supervisor-private side-table,
        # keyed by the synthetic envelope_id.
        synthetic_eid = env["envelope_id"]
        assert (
            sup._cascade_destroy_overrides.get(synthetic_eid)  # noqa: SLF001
            == DestroyReason.ORPHAN_TIMEOUT
        ), (
            f"_cascade_destroy_overrides missing entry for {synthetic_eid!r}; "
            f"got {sup._cascade_destroy_overrides!r}"  # noqa: SLF001
        )

    @pytest.mark.anyio
    async def test_poison_drop_cascade_stamps_side_table_with_orphan_timeout(
        self, supervisor_ctx, fake_redis
    ):
        """codex r6 [R6-2] — poison-drop fallback stamps the side-table
        with ORPHAN_TIMEOUT. Payload remains ``{reason, policy}``.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0R26POISONORPHAN0001",
            child_session_id="child-poison",
        )
        await sup._on_poison_drop(env)  # noqa: SLF001
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        import json as _json

        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert len(cancel_entries) == 1
        cascaded = _json.loads(cancel_entries[0][1][b"envelope"])
        # Wire payload is unchanged at {reason, policy}.
        assert set(cascaded["payload"].keys()) == {"reason", "policy"}
        # Override is in the side-table, keyed by synthetic envelope_id.
        assert (
            sup._cascade_destroy_overrides.get(cascaded["envelope_id"])  # noqa: SLF001
            == DestroyReason.ORPHAN_TIMEOUT
        )

    @pytest.mark.anyio
    async def test_orphan_cascade_handler_destroys_with_orphan_timeout_reason(
        self, supervisor_ctx, stub_lifecycle
    ):
        """codex r6 [R6-2] — end-to-end: orphan tick stamps the
        side-table, ``CancelRequestHandler._terminate_outcome`` reads+pops
        it on dispatch and destroys with ``ORPHAN_TIMEOUT``.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
        )
        from app.domain.models.session import DestroyReason

        # Bind ctx hooks (register_cancel_state / clear_child_tracking +
        # cascade_destroy_overrides) so the TERMINATE side_effect can
        # read the override.
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0R26ORPHANHND00000001",
            producer_role=ProducerRole.SUPERVISOR,
            reason="orphan_timeout",
        )
        # Stamp the side-table BEFORE handler dispatch (mimics what
        # _emit_cascade_terminate does before publish).
        sup._cascade_destroy_overrides[env.envelope_id] = (  # noqa: SLF001
            DestroyReason.ORPHAN_TIMEOUT
        )
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.side_effect is not None
        await outcome.side_effect()
        # Handler must use ORPHAN_TIMEOUT, NOT FORCE_TERMINATE.
        assert (env.child_session_id, DestroyReason.ORPHAN_TIMEOUT) in (
            stub_lifecycle.destroy_calls
        )
        assert (env.child_session_id, DestroyReason.FORCE_TERMINATE) not in (
            stub_lifecycle.destroy_calls
        )
        # Side-table entry was popped (not re-applied on redelivery).
        assert env.envelope_id not in sup._cascade_destroy_overrides, (  # noqa: SLF001
            "side-table entry must be popped on dispatch so XAUTOCLAIM "
            "redelivery doesn't double-apply"
        )

    @pytest.mark.anyio
    async def test_parent_terminate_without_side_table_entry_falls_back_force_terminate(
        self, supervisor_ctx, stub_lifecycle
    ):
        """codex r6 [R6-2] — parent-originated CANCEL_REQUEST(TERMINATE)
        has no side-table entry → handler defaults to FORCE_TERMINATE.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
        )
        from app.domain.models.session import DestroyReason

        MailboxSupervisor(supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01)
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0R26PARENTNOR000001",
            producer_role=ProducerRole.PARENT_AGENT,
            reason="user_cancel",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.side_effect is not None
        await outcome.side_effect()
        assert (env.child_session_id, DestroyReason.FORCE_TERMINATE) in (
            stub_lifecycle.destroy_calls
        )

    @pytest.mark.anyio
    async def test_payload_destroy_reason_field_no_longer_honored(
        self, supervisor_ctx, stub_lifecycle
    ):
        """codex r6 [R6-2] — even a SUPERVISOR-originated envelope with a
        payload-level ``destroy_reason`` is IGNORED. The wire schema is
        frozen at ``{reason, policy}`` and the handler only consults the
        supervisor-private side-table. This test exercises the
        backwards-compat path: an old in-process cascade caller still
        passing the field on the payload (instead of the side-table)
        must now degrade to FORCE_TERMINATE rather than the
        ORPHAN_TIMEOUT the obsolete code path expected.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
        )
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0R26UNKNOWNREASON001",
            producer_role=ProducerRole.SUPERVISOR,
            reason="some_synthetic_reason",
        )
        object.__setattr__(
            env,
            "payload",
            {
                "reason": "some_synthetic_reason",
                "policy": CancelPolicy.TERMINATE.value,
                # Payload-level destroy_reason is ignored under R6-2.
                "destroy_reason": DestroyReason.ORPHAN_TIMEOUT.value,
            },
        )
        # No side-table entry — handler must default to FORCE_TERMINATE.
        assert env.envelope_id not in sup._cascade_destroy_overrides  # noqa: SLF001
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert (env.child_session_id, DestroyReason.FORCE_TERMINATE) in (
            stub_lifecycle.destroy_calls
        )
        assert (env.child_session_id, DestroyReason.ORPHAN_TIMEOUT) not in (
            stub_lifecycle.destroy_calls
        )


class TestCascadeOverrideRetryAndRestart:
    """codex r7 [R7-4 / R7-5] — side-table semantics under failure + restart.

    R7-4: ``CancelRequestHandler._terminate_outcome`` must pop the side-table
    entry only AFTER all destroy/publish steps succeed. If destroy raises
    (retryable ``SandboxLifecycleError``), the side_effect re-raises → outer
    ``_handle_envelope`` leaves the envelope in PEL → XAUTOCLAIM redelivers
    → handler runs again → same ``DestroyReason`` is re-applied.

    R7-5: The dict is in-process. Across supervisor restart it is empty;
    the durable envelope replays via XAUTOCLAIM but the override is lost
    and the handler falls back to ``FORCE_TERMINATE``. Accepted drift —
    documented in ``SupervisorContext.cascade_destroy_overrides`` docstring
    and asserted here as a regression guard so a future refactor doesn't
    silently change the fallback semantics.
    """

    @pytest.mark.anyio
    async def test_destroy_failure_retains_override_for_pel_retry(
        self, supervisor_ctx, stub_lifecycle
    ):
        """codex r7 [R7-4] — destroy raises mid-flow → side_effect re-raises
        → outer loop retains entry in PEL → next dispatch re-reads the same
        override (the pop only happens on the all-steps-succeeded path).
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0R74RETRYORPHAN00001",
            producer_role=ProducerRole.SUPERVISOR,
            reason="orphan_timeout",
        )
        # Stamp the side-table BEFORE handler dispatch (mimics
        # _emit_cascade_terminate's pre-publish stamp).
        sup._cascade_destroy_overrides[env.envelope_id] = (  # noqa: SLF001
            DestroyReason.ORPHAN_TIMEOUT
        )

        # First attempt — destroy raises retryable SandboxLifecycleError.
        attempts = {"count": 0}

        async def _destroy_raising(session_id, reason):
            attempts["count"] += 1
            stub_lifecycle.destroy_calls.append((session_id, reason))
            raise SandboxLifecycleError("transient docker daemon error")

        supervisor_ctx.sandbox_lifecycle.destroy = _destroy_raising

        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.side_effect is not None
        with pytest.raises(SandboxLifecycleError):
            await outcome.side_effect()

        # codex r7 [R7-4] — the override MUST still be present so a
        # redelivery applies the same DestroyReason. The pop is gated
        # on all steps succeeding.
        assert (
            sup._cascade_destroy_overrides.get(env.envelope_id)  # noqa: SLF001
            == DestroyReason.ORPHAN_TIMEOUT
        ), (
            "R7-4 — side-table override must be retained when destroy "
            "raises so XAUTOCLAIM redelivery re-applies the same reason; "
            f"got {sup._cascade_destroy_overrides!r}"  # noqa: SLF001
        )

        # Second attempt — destroy succeeds. Override should be consumed
        # (popped only on the success path).
        async def _destroy_ok(session_id, reason):
            stub_lifecycle.destroy_calls.append((session_id, reason))

        supervisor_ctx.sandbox_lifecycle.destroy = _destroy_ok
        outcome2 = await handler.handle(env, supervisor_ctx)
        await outcome2.side_effect()

        # Now the override should be gone (consumed on success).
        assert env.envelope_id not in sup._cascade_destroy_overrides, (  # noqa: SLF001
            "R7-4 — side-table override must be popped on the success "
            "path so a later non-cascade envelope sharing the id doesn't "
            "accidentally inherit"
        )
        # Both attempts used ORPHAN_TIMEOUT (not the FORCE_TERMINATE fallback).
        orphan_calls = [
            c
            for c in stub_lifecycle.destroy_calls
            if c[1] == DestroyReason.ORPHAN_TIMEOUT
        ]
        assert len(orphan_calls) == 2, (
            f"both attempts must call destroy with ORPHAN_TIMEOUT; "
            f"got destroy_calls={stub_lifecycle.destroy_calls!r}"
        )

    @pytest.mark.anyio
    async def test_cascade_override_lost_after_supervisor_restart_falls_back_to_force_terminate(
        self, supervisor_ctx, stub_lifecycle
    ):
        """codex r7 [R7-5, ACCEPTED DRIFT] — supervisor crash between
        ``_emit_cascade_terminate``'s publish and the handler dispatching
        the synthetic envelope loses the in-process side-table. The
        restarting supervisor reads the durable envelope from Redis but
        sees an empty ``_cascade_destroy_overrides`` → handler falls back
        to ``FORCE_TERMINATE``. The destroy still fires (load-bearing
        invariant); only the audit reason annotation differs from the
        pre-crash intent. Regression guard.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
        )
        from app.domain.models.session import DestroyReason

        # Simulate the restart: a fresh supervisor with an empty
        # side-table dispatches the synthetic envelope that a prior
        # supervisor instance had published.
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        assert sup._cascade_destroy_overrides == {}  # noqa: SLF001
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="cascade:abc123restartlost000000000001",
            producer_role=ProducerRole.SUPERVISOR,
            reason="orphan_timeout",
        )
        # No side-table entry — restart path. Handler MUST default to
        # FORCE_TERMINATE rather than the original ORPHAN_TIMEOUT.
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert (env.child_session_id, DestroyReason.FORCE_TERMINATE) in (
            stub_lifecycle.destroy_calls
        ), (
            "R7-5 — supervisor restart must fall back to FORCE_TERMINATE "
            "since the in-process side-table did not survive the crash; "
            f"got destroy_calls={stub_lifecycle.destroy_calls!r}"
        )
        # Documented degradation: ORPHAN_TIMEOUT is NOT applied.
        assert (env.child_session_id, DestroyReason.ORPHAN_TIMEOUT) not in (
            stub_lifecycle.destroy_calls
        ), (
            "R7-5 — accepted drift: post-restart cascade cannot recover "
            "the original DestroyReason; if this assertion ever flips, "
            "either Option B (envelope_id-encoded) or Option C (producer_role "
            "variant) was added back without updating this regression test"
        )


class TestApprovalMismatchPublishFailureNoAck:
    """codex r7 [R7-6, HIGH CONTRACT] — when the paired deny publish on
    the mismatch path raises, the envelope MUST NOT be ACKed. Defer to
    XAUTOCLAIM retry; the publisher's SET NX dedup makes repeated publish
    attempts idempotent. Earlier R6-4 behavior ACKed regardless, which
    silently lost the deny and forced the child to wait the full 300s.
    """

    @pytest.mark.anyio
    async def test_mismatch_publish_failure_returns_ack_false_and_retains_pel(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            ApprovalRequestHandler,
        )

        original_publish = supervisor_ctx.publisher.publish

        async def _publish_raising(envelope):
            # Only the mismatch_deny envelope should raise; let any other
            # publish through (defensive — current code path only emits
            # one publish, but keeps the stub honest).
            if envelope.envelope_id.startswith("mismatch_deny:"):
                raise RuntimeError("publisher down (test)")
            return await original_publish(envelope)

        monkeypatch.setattr(
            supervisor_ctx.publisher, "publish", _publish_raising
        )

        handler = ApprovalRequestHandler()
        env = _approval_env(
            eid="01HSPYU0R76APPROVAL0PUBRAIS",
            correlation_id="envelope-cid-r76",
        )
        env = env.model_copy(
            update={
                "payload": {
                    **env.payload,
                    "correlation_id": "payload-cid-bad",
                },
            }
        )

        outcome = await handler.handle(env, supervisor_ctx)
        # R7-6 — MUST NOT ACK; envelope stays in PEL for XAUTOCLAIM.
        assert outcome.ack is False, (
            "R7-6 — mismatch deny publish failure must NOT ACK; the "
            "envelope stays in PEL so XAUTOCLAIM redelivers and the "
            "paired deny is retried (publisher SET NX dedup makes "
            "repeated publishes safe)"
        )
        assert outcome.side_effect is None
        assert outcome.audit_payload == {
            "correlation_id_mismatch": True,
            "mismatch_deny_publish_failed": True,
        }


class TestCascadeRetryablePublishFailurePreservesTracking:
    """codex r7 [R7-7, HIGH CONTRACT] — when ``_emit_cascade_terminate``'s
    publish raises AND the direct-kill fallback's destroy also raises
    retryable ``SandboxLifecycleError``, the cascade signals failure via
    ``_CascadeFailedError`` so callers preserve per-child tracking
    (``_last_seen_mono`` / ``_cancel_states``). The next supervisor tick
    re-attempts the cascade. Earlier rounds silently swallowed the
    failure and cleared tracking → cascade lost.
    """

    @pytest.mark.anyio
    async def test_orphan_tick_preserves_last_seen_when_cascade_and_directkill_both_fail(
        self, supervisor_ctx, monkeypatch
    ):
        """R7-7 Option A+C — orphan tick MUST keep ``_last_seen_mono``
        when both XADD and direct-destroy fail so the next tick retries.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError
        from app.domain.models.mailbox_envelope import (
            SUBAGENT_PROGRESS_STALE_AFTER_SECONDS,
        )

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._ORPHAN_CHECK_INTERVAL_S = 0.0  # noqa: SLF001
        sup._last_orphan_check_mono = 0.0  # noqa: SLF001
        now = supervisor_ctx.clock()
        sup._last_seen_mono["child-stuck"] = (  # noqa: SLF001
            now - (SUBAGENT_PROGRESS_STALE_AFTER_SECONDS + 5)
        )

        # Force publish to fail (XADD raises).
        async def _publish_raising(envelope):
            raise RuntimeError("redis XADD down (test)")

        monkeypatch.setattr(
            supervisor_ctx.publisher, "publish", _publish_raising
        )

        # Force fallback destroy to also raise retryable error.
        async def _destroy_raising(session_id, reason):
            raise SandboxLifecycleError("docker daemon down (test)")

        supervisor_ctx.sandbox_lifecycle.destroy = _destroy_raising

        await sup._maybe_tick_check_orphans()  # noqa: SLF001

        # R7-7 — last_seen MUST be retained for next-tick retry.
        assert "child-stuck" in sup._last_seen_mono, (  # noqa: SLF001
            "R7-7 — orphan tick must preserve _last_seen_mono when "
            "cascade publish + direct-kill both fail so the next "
            "_ORPHAN_CHECK_INTERVAL_S tick re-attempts the cascade; "
            f"got _last_seen_mono={sup._last_seen_mono!r}"  # noqa: SLF001
        )

    @pytest.mark.anyio
    async def test_cancel_auto_escalate_preserves_cancel_state_on_cascade_failure(
        self, supervisor_ctx, monkeypatch
    ):
        """R7-7 Option A+C — auto-escalate tick MUST keep ``_cancel_states``
        when both XADD and direct-destroy fail so the next tick retries.
        """
        from app.application.services.mailbox_supervisor import (
            MailboxSupervisor,
            _CancelState,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError
        from app.domain.models.mailbox_envelope import (
            CHILD_CANCEL_ACK_TIMEOUT_MS,
        )

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._CANCEL_CHECK_INTERVAL_S = 0.0  # noqa: SLF001
        elapsed_seconds = (CHILD_CANCEL_ACK_TIMEOUT_MS / 1000.0) + 0.5
        sup._cancel_states["child-stuck"] = _CancelState(  # noqa: SLF001
            child_session_id="child-stuck",
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock() - elapsed_seconds,
        )

        async def _publish_raising(envelope):
            raise RuntimeError("redis XADD down (test)")

        monkeypatch.setattr(
            supervisor_ctx.publisher, "publish", _publish_raising
        )

        async def _destroy_raising(session_id, reason):
            raise SandboxLifecycleError("docker daemon down (test)")

        supervisor_ctx.sandbox_lifecycle.destroy = _destroy_raising

        await sup._maybe_tick_cancel_check()  # noqa: SLF001

        # R7-7 — cancel_state MUST be retained for next-tick retry.
        assert "child-stuck" in sup._cancel_states, (  # noqa: SLF001
            "R7-7 — auto-escalate tick must preserve _cancel_states when "
            "cascade publish + direct-kill both fail so the next "
            "tick re-attempts the cascade; "
            f"got _cancel_states={sup._cancel_states!r}"  # noqa: SLF001
        )

    @pytest.mark.anyio
    async def test_orphan_tick_clears_last_seen_when_cascade_recovers_via_directkill(
        self, supervisor_ctx, monkeypatch
    ):
        """R7-7 Option A+C — when publish fails but direct-kill succeeds
        (no retryable error), the cascade is considered done → caller
        clears tracking. Regression guard so the new ``_CascadeFailedError``
        signal doesn't accidentally preserve tracking on successful
        fallback paths.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.mailbox_envelope import (
            SUBAGENT_PROGRESS_STALE_AFTER_SECONDS,
        )

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._ORPHAN_CHECK_INTERVAL_S = 0.0  # noqa: SLF001
        sup._last_orphan_check_mono = 0.0  # noqa: SLF001
        now = supervisor_ctx.clock()
        sup._last_seen_mono["child-recoverable"] = (  # noqa: SLF001
            now - (SUBAGENT_PROGRESS_STALE_AFTER_SECONDS + 5)
        )

        async def _publish_raising(envelope):
            raise RuntimeError("redis XADD down (test)")

        monkeypatch.setattr(
            supervisor_ctx.publisher, "publish", _publish_raising
        )

        # destroy succeeds — fallback kills the orphan directly.
        await sup._maybe_tick_check_orphans()  # noqa: SLF001

        assert "child-recoverable" not in sup._last_seen_mono, (  # noqa: SLF001
            "R7-7 — when direct-kill succeeds, the cascade is done; "
            "tracking MUST be cleared. Otherwise the next tick would "
            "re-attempt a destroyed child. "
            f"got _last_seen_mono={sup._last_seen_mono!r}"  # noqa: SLF001
        )


# ──────────────────────────────────────────────────────────────────────────────
# codex r2 [R2-7, MEDIUM CONTRACT] — HandoffRequestHandler telemetry isolation.
# ──────────────────────────────────────────────────────────────────────────────


class TestHandoffRequestHandlerTelemetryIsolation:
    """Spec §6.6 — handoff handler is telemetry-only. If the sink raises,
    the exception MUST NOT propagate; ACK still fires so XAUTOCLAIM
    doesn't loop the envelope (same fail-open pattern as codex r1 F5).
    """

    @pytest.mark.anyio
    async def test_handoff_request_acks_even_when_telemetry_raises(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            HandoffRequestHandler,
        )

        async def _raise(name, data):  # noqa: ANN001
            raise RuntimeError("telemetry sink down (test)")

        monkeypatch.setattr(supervisor_ctx.telemetry, "emit", _raise)
        handler = HandoffRequestHandler()
        env = _handoff_env(eid="01HSPYU0R27HANDOFFRAISE0001")
        # MUST NOT raise; ACK must still fire.
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is True
        assert outcome.side_effect is None
        assert outcome.audit_payload == {
            "unsupported": True,
            "reason": "c3_ship_does_not_implement_handoff",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Codex F1 / F5 / F6 — telemetry isolation in terminal handlers + best-effort
# inner mark_processed
# ──────────────────────────────────────────────────────────────────────────────


class _RaisingTelemetry:
    """Telemetry stub that raises ``RuntimeError`` from every ``emit``.

    Used to verify the side_effect isolation introduced by codex F1/F5 — a
    sink fault MUST NOT escape the side_effect, otherwise XAUTOCLAIM
    redelivers a terminal envelope whose destroy already ran (or whose
    destroy classification was AlreadyDestroyed), looping forever on the
    observability path.
    """

    def __init__(self) -> None:
        self.attempts: list[tuple[str, dict]] = []

    async def emit(self, name: str, data: dict) -> None:
        self.attempts.append((name, data))
        raise RuntimeError(f"telemetry sink down (test): {name}")


class TestTerminalHandlerTelemetryIsolation:
    """Codex F1 / F5 (HIGH) — telemetry faults inside terminal handler
    side_effects must NOT propagate. The supervisor's PEL-retain semantics
    would otherwise loop a destroyed child forever via the observability path.
    """

    @pytest.mark.anyio
    async def test_result_ready_already_destroyed_branch_swallows_telemetry_raise(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            ResultReadyHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed

        async def _raise_already(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_already
        )
        supervisor_ctx.telemetry = _RaisingTelemetry()
        handler = ResultReadyHandler()
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F50000RESULT0DESTROY",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        # MUST NOT raise — telemetry isolation is the load-bearing fix.
        await outcome.side_effect()
        # Verify the emit was attempted and swallowed.
        attempted = [n for n, _ in supervisor_ctx.telemetry.attempts]
        assert "mailbox.destroy_idempotent_noop" in attempted
        # mark_processed still runs because terminal-success continues.
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_result_ready_binding_missing_branch_swallows_telemetry_raise(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            ResultReadyHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxBindingMissing

        async def _raise_missing(session_id, reason):
            raise SandboxBindingMissing(session_id)

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_missing
        )
        supervisor_ctx.telemetry = _RaisingTelemetry()
        handler = ResultReadyHandler()
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F50000RESULT0BINDING",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_result_ready_retryable_branch_preserves_sandbox_raise(
        self, supervisor_ctx, monkeypatch
    ):
        """Codex F5 retryable sub-branch — telemetry MUST be isolated but the
        downstream ``raise`` of the SandboxLifecycleError MUST still escape so
        the side_effect raises and the envelope stays in PEL for retry."""
        from app.application.services.mailbox_supervisor import (
            ResultReadyHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError

        async def _raise_transient(session_id, reason):
            raise SandboxLifecycleError("transient")

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_transient
        )
        supervisor_ctx.telemetry = _RaisingTelemetry()
        handler = ResultReadyHandler()
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F50000RESULT0TRANS01",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        with pytest.raises(SandboxLifecycleError):
            await outcome.side_effect()
        # No mark_processed on retryable; PEL retain semantics preserved.
        assert not await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_cancel_ack_already_destroyed_branch_swallows_telemetry_raise(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed

        async def _raise(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise
        )
        supervisor_ctx.telemetry = _RaisingTelemetry()
        handler = CancelAckHandler()
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F50000CANCEL0ACK0001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_terminate_callback_fail_telemetry_isolation(
        self, supervisor_ctx, monkeypatch
    ):
        """Codex F1 (HIGH) — when the agent_service_callback raises during
        TERMINATE, the supervisor emits ``mailbox.force_terminate_stop_failed``.
        If THAT emit also raises, the cascade must still continue to destroy
        (spec §7.6 invariant: stop failure must not prevent destroy).
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        async def _raise(envelope):
            raise RuntimeError("agent_service callback crashed")

        # Replace the agent_service_callback on the ctx with one that raises.
        supervisor_ctx.agent_service_callback = _raise
        supervisor_ctx.telemetry = _RaisingTelemetry()
        handler = CancelRequestHandler()
        env = _cancel_env(policy=CancelPolicy.TERMINATE)
        outcome = await handler.handle(env, supervisor_ctx)
        # Must not raise — destroy must complete despite both the callback
        # and the stop_failed telemetry emit raising.
        await outcome.side_effect()
        # destroy was reached + recorded by the stub lifecycle.
        assert any(
            sid == env.child_session_id
            for sid, _reason in supervisor_ctx.sandbox_lifecycle.destroy_calls
        )

    @pytest.mark.anyio
    async def test_terminate_already_destroyed_branch_swallows_telemetry_raise(
        self, supervisor_ctx, monkeypatch
    ):
        """Codex F5 (HIGH) — TERMINATE's own AlreadyDestroyed branch emits
        telemetry; that emit must be isolated.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed
        from app.domain.models.mailbox_envelope import CancelPolicy

        async def _raise_already(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_already
        )
        supervisor_ctx.telemetry = _RaisingTelemetry()
        handler = CancelRequestHandler()
        env = _cancel_env(policy=CancelPolicy.TERMINATE)
        outcome = await handler.handle(env, supervisor_ctx)
        # Must not raise on telemetry fault.
        await outcome.side_effect()


class TestTerminalHandlerInnerMarkProcessedIsolation:
    """Codex F6 (HIGH) — the belt-and-suspenders inner ``mark_processed``
    inside terminal handler side_effects must NOT propagate DB faults.
    The outer ``_handle_envelope`` runs an idempotent mark_processed
    post-side_effect; raising from the inner one would leave the envelope
    in PEL → XAUTOCLAIM redelivers → re-fires destroy callback chain.
    """

    @pytest.mark.anyio
    async def test_result_ready_inner_mark_processed_failure_swallowed(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            ResultReadyHandler,
        )

        async def _raise(*args, **kwargs):
            raise RuntimeError("audit DB blip")

        # Monkeypatch the repo's mark_processed only after upsert_processing
        # already wrote the staging row (handler does that before side_effect).
        handler = ResultReadyHandler()
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F60000RESULT0MARK001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        # Replace ONLY the mark_processed after the initial upsert.
        monkeypatch.setattr(
            supervisor_ctx.audit_repo, "mark_processed", _raise
        )
        # MUST NOT raise — the inner mark_processed failure is best-effort.
        await outcome.side_effect()
        # destroy completed.
        assert any(
            sid == env.child_session_id
            for sid, _reason in supervisor_ctx.sandbox_lifecycle.destroy_calls
        )

    @pytest.mark.anyio
    async def test_cancel_ack_inner_mark_processed_failure_swallowed(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
        )

        async def _raise(*args, **kwargs):
            raise RuntimeError("audit DB blip")

        handler = CancelAckHandler()
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F60000CANCELACK0001",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        monkeypatch.setattr(
            supervisor_ctx.audit_repo, "mark_processed", _raise
        )
        await outcome.side_effect()
        assert any(
            sid == env.child_session_id
            for sid, _reason in supervisor_ctx.sandbox_lifecycle.destroy_calls
        )

    @pytest.mark.anyio
    async def test_terminate_inner_mark_processed_failure_swallowed(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        async def _raise(*args, **kwargs):
            raise RuntimeError("audit DB blip")

        handler = CancelRequestHandler()
        env = _cancel_env(policy=CancelPolicy.TERMINATE)
        outcome = await handler.handle(env, supervisor_ctx)
        monkeypatch.setattr(
            supervisor_ctx.audit_repo, "mark_processed", _raise
        )
        await outcome.side_effect()
        # destroy still completed.
        assert any(
            sid == env.child_session_id
            for sid, _reason in supervisor_ctx.sandbox_lifecycle.destroy_calls
        )


# ──────────────────────────────────────────────────────────────────────────────
# Codex F7 + F9 — terminal handlers clear per-child tracking state
# ──────────────────────────────────────────────────────────────────────────────


class TestTerminalHandlerClearsChildTracking:
    """Codex F7 + F9 (HIGH) — terminal destroy success MUST drop both
    ``_cancel_states[child]`` and ``_last_seen_mono[child]`` so the auto-
    escalate tick and orphan detector don't fire spurious cascades for an
    already-dead child.

    Coverage matrix:
    * ResultReady success / AlreadyDestroyed / retryable raise
    * CancelAck   success / retryable raise
    * TERMINATE   success / AlreadyDestroyed
    """

    @pytest.mark.anyio
    async def test_result_ready_success_clears_tracking(
        self, supervisor_ctx
    ):
        from app.application.services.mailbox_supervisor import (
            MailboxSupervisor,
            ResultReadyHandler,
            _CancelState,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        # Pre-populate tracking state for the child that will be destroyed.
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F70000CLEAR00RESULT1",
        )
        sup._last_seen_mono[env.child_session_id] = supervisor_ctx.clock()  # noqa: SLF001
        sup._cancel_states[env.child_session_id] = _CancelState(  # noqa: SLF001
            child_session_id=env.child_session_id,
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock(),
        )

        handler = ResultReadyHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()

        # Both tracking maps cleared.
        assert env.child_session_id not in sup._cancel_states  # noqa: SLF001
        assert env.child_session_id not in sup._last_seen_mono  # noqa: SLF001

    @pytest.mark.anyio
    async def test_result_ready_already_destroyed_clears_tracking(
        self, supervisor_ctx, monkeypatch
    ):
        from app.application.services.mailbox_supervisor import (
            MailboxSupervisor,
            ResultReadyHandler,
            _CancelState,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed
        from app.domain.models.mailbox_envelope import CancelPolicy

        async def _raise_already(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_already
        )
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F70000CLEAR0ALREADY",
        )
        sup._last_seen_mono[env.child_session_id] = supervisor_ctx.clock()  # noqa: SLF001
        sup._cancel_states[env.child_session_id] = _CancelState(  # noqa: SLF001
            child_session_id=env.child_session_id,
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock(),
        )

        handler = ResultReadyHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        # Terminal-success path also clears tracking.
        assert env.child_session_id not in sup._cancel_states  # noqa: SLF001
        assert env.child_session_id not in sup._last_seen_mono  # noqa: SLF001

    @pytest.mark.anyio
    async def test_result_ready_retryable_raise_preserves_tracking(
        self, supervisor_ctx, monkeypatch
    ):
        """Retryable failure: ``raise`` short-circuits BEFORE cleanup so
        the state is preserved for the next XAUTOCLAIM retry."""
        from app.application.services.mailbox_supervisor import (
            MailboxSupervisor,
            ResultReadyHandler,
            _CancelState,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError
        from app.domain.models.mailbox_envelope import CancelPolicy

        async def _raise_transient(session_id, reason):
            raise SandboxLifecycleError("transient")

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_transient
        )
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F70000CLEAR0RETRY01",
        )
        sup._last_seen_mono[env.child_session_id] = supervisor_ctx.clock()  # noqa: SLF001
        sup._cancel_states[env.child_session_id] = _CancelState(  # noqa: SLF001
            child_session_id=env.child_session_id,
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock(),
        )

        handler = ResultReadyHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        with pytest.raises(SandboxLifecycleError):
            await outcome.side_effect()
        # Tracking preserved — retry must run on next XAUTOCLAIM redelivery.
        assert env.child_session_id in sup._cancel_states  # noqa: SLF001
        assert env.child_session_id in sup._last_seen_mono  # noqa: SLF001

    @pytest.mark.anyio
    async def test_cancel_ack_success_clears_tracking(
        self, supervisor_ctx
    ):
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
            MailboxSupervisor,
            _CancelState,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0F70000CLEAR0CANCEL1",
        )
        sup._last_seen_mono[env.child_session_id] = supervisor_ctx.clock()  # noqa: SLF001
        sup._cancel_states[env.child_session_id] = _CancelState(  # noqa: SLF001
            child_session_id=env.child_session_id,
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock(),
        )

        handler = CancelAckHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert env.child_session_id not in sup._cancel_states  # noqa: SLF001
        assert env.child_session_id not in sup._last_seen_mono  # noqa: SLF001

    @pytest.mark.anyio
    async def test_terminate_success_clears_tracking(self, supervisor_ctx):
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
            _CancelState,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        env = _cancel_env(policy=CancelPolicy.TERMINATE)
        sup._last_seen_mono[env.child_session_id] = supervisor_ctx.clock()  # noqa: SLF001
        sup._cancel_states[env.child_session_id] = _CancelState(  # noqa: SLF001
            child_session_id=env.child_session_id,
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock(),
        )

        handler = CancelRequestHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert env.child_session_id not in sup._cancel_states  # noqa: SLF001
        assert env.child_session_id not in sup._last_seen_mono  # noqa: SLF001

    @pytest.mark.anyio
    async def test_terminate_already_destroyed_clears_tracking(
        self, supervisor_ctx, monkeypatch
    ):
        """Same as success — terminal-success after AlreadyDestroyed must
        still drop tracking so the cascade tick doesn't fire later.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
            _CancelState,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed
        from app.domain.models.mailbox_envelope import CancelPolicy

        async def _raise_already(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _raise_already
        )
        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        env = _cancel_env(policy=CancelPolicy.TERMINATE)
        sup._last_seen_mono[env.child_session_id] = supervisor_ctx.clock()  # noqa: SLF001
        sup._cancel_states[env.child_session_id] = _CancelState(  # noqa: SLF001
            child_session_id=env.child_session_id,
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock(),
        )

        handler = CancelRequestHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert env.child_session_id not in sup._cancel_states  # noqa: SLF001
        assert env.child_session_id not in sup._last_seen_mono  # noqa: SLF001


class TestTerminalHandlerClearsTrackingDespiteCallbackFailure:
    """Codex r4 [R4-4, MEDIUM CONTRACT] — destroy success + callback
    raising MUST still clear per-child tracking state. The previous
    ordering placed ``clear_child_tracking`` AFTER the callback, so a
    callback failure (agent_service down, SSE bridge raises, ...) caused
    the tracking dict to leak: the destroyed child stayed in
    ``_last_seen_mono`` and ``_cancel_states`` and the auto-escalate /
    orphan ticks would fire spurious cascades against it.

    R4-4 fix moves the cleanup BEFORE the callback (or, for
    CancelRequestHandler.TERMINATE, before the synthetic CANCEL_ACK
    publish — the only un-wrapped raise site after destroy).
    """

    @pytest.mark.anyio
    async def test_result_ready_callback_failure_still_clears_tracking(
        self, supervisor_ctx
    ):
        from app.application.services.mailbox_supervisor import (
            MailboxSupervisor,
            ResultReadyHandler,
            _CancelState,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _raise_callback(envelope) -> None:
            raise RuntimeError("agent_service down")

        # Replace the supervisor's agent_service_callback so the
        # downstream notification raises.
        supervisor_ctx.agent_service_callback = _raise_callback  # type: ignore[assignment]

        env = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0R440RESULT0CALLBACK0",
        )
        sup._last_seen_mono[env.child_session_id] = supervisor_ctx.clock()  # noqa: SLF001
        sup._cancel_states[env.child_session_id] = _CancelState(  # noqa: SLF001
            child_session_id=env.child_session_id,
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock(),
        )

        handler = ResultReadyHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        # The callback raises but the side_effect itself MUST NOT raise —
        # the R4-4 fix wraps the callback in try/except so cleanup +
        # mark_processed continue to run.
        await outcome.side_effect()

        # Tracking cleared despite callback raising.
        assert env.child_session_id not in sup._cancel_states  # noqa: SLF001
        assert env.child_session_id not in sup._last_seen_mono  # noqa: SLF001

    @pytest.mark.anyio
    async def test_cancel_ack_callback_failure_still_clears_tracking(
        self, supervisor_ctx
    ):
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
            MailboxSupervisor,
            _CancelState,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _raise_callback(envelope) -> None:
            raise RuntimeError("agent_service down")

        supervisor_ctx.agent_service_callback = _raise_callback  # type: ignore[assignment]

        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0R440CANCEL0CALLBACK0",
        )
        sup._last_seen_mono[env.child_session_id] = supervisor_ctx.clock()  # noqa: SLF001
        sup._cancel_states[env.child_session_id] = _CancelState(  # noqa: SLF001
            child_session_id=env.child_session_id,
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock(),
        )

        handler = CancelAckHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()

        assert env.child_session_id not in sup._cancel_states  # noqa: SLF001
        assert env.child_session_id not in sup._last_seen_mono  # noqa: SLF001

    @pytest.mark.anyio
    async def test_terminate_synthetic_publish_failure_still_clears_tracking(
        self, supervisor_ctx, monkeypatch
    ):
        """CancelRequest TERMINATE: destroy succeeds, the synthetic
        CANCEL_ACK echo publish (un-wrapped) raises — R4-4 moves
        ``clear_child_tracking`` BEFORE the publish so cleanup runs
        regardless.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
            _CancelState,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _raise_publish(envelope) -> None:
            raise RuntimeError("redis publish blip")

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _raise_publish)

        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0R440TERMINATE0PUBLISH",
        )
        sup._last_seen_mono[env.child_session_id] = supervisor_ctx.clock()  # noqa: SLF001
        sup._cancel_states[env.child_session_id] = _CancelState(  # noqa: SLF001
            child_session_id=env.child_session_id,
            policy=CancelPolicy.REQUEST_CANCEL,
            requested_at_mono=supervisor_ctx.clock(),
        )

        handler = CancelRequestHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        # publisher.publish is the only un-wrapped step after destroy in
        # the TERMINATE branch — when it raises, the side_effect
        # propagates the exception (no ACK, PEL retry). But cleanup
        # MUST have run before the raise (R4-4 invariant).
        with pytest.raises(RuntimeError, match="redis publish blip"):
            await outcome.side_effect()
        assert env.child_session_id not in sup._cancel_states  # noqa: SLF001
        assert env.child_session_id not in sup._last_seen_mono  # noqa: SLF001


# ──────────────────────────────────────────────────────────────────────────────
# Codex F8 — ApprovalRequestHandler ack=True path writes mark_processed
# ──────────────────────────────────────────────────────────────────────────────


class TestAckTruePathMarksProcessed:
    """Codex F8 (HIGH) — handlers that return ``ack=True`` with no
    side_effect (ApprovalRequestHandler stub, _StubNonTerminalHandler,
    dedup-hit returns) MUST leave a ``processed_at`` row. Without it the
    audit-dedup get_processed short-circuit never trips for these types.
    """

    @pytest.mark.anyio
    async def test_approval_request_marks_processed_via_handle_envelope(
        self, supervisor_ctx, fake_redis
    ):
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        await sup._consumer.ensure_group()  # noqa: SLF001
        env = _approval_env(
            eid="01HSPYU0F80000APPROVAL0PROC1",
            correlation_id="cid-f8-1",
            tool_call_id="tc-f8-1",
        )
        await sup._handle_envelope(b"f8-0-1", env)  # noqa: SLF001
        # processed_at written via the new ack=True path.
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_progress_update_marks_processed_via_handle_envelope(
        self, supervisor_ctx
    ):
        """The _StubNonTerminalHandler also returns ack=True / no side_effect.
        Same dedup requirement — without it, every redelivered heartbeat
        would re-fire the agent_service_callback."""
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        await sup._consumer.ensure_group()  # noqa: SLF001
        env = _env(
            t=MailboxEnvelopeType.PROGRESS_UPDATE,
            eid="01HSPYU0F80000PROGRESS0PROC",
            producer_role=ProducerRole.CHILD_AGENT,
        )
        await sup._handle_envelope(b"f8-0-2", env)  # noqa: SLF001
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    @pytest.mark.anyio
    async def test_ack_true_path_swallows_mark_processed_db_failure(
        self, supervisor_ctx, monkeypatch
    ):
        """codex r5 [R5-4, HIGH CONTRACT] — ack=True path MUST NOT ACK when
        mark_processed raises (mirror of R4-3 fix on the side_effect path).

        Prior behavior ACKed anyway with a logged warning, which broke the
        spec §5.8 "marker is the truth" invariant: the envelope would be
        gone from the PEL but the audit row had no ``processed_at`` →
        on a publisher-cross-pod replay the handler re-fires without the
        dedup short-circuit. Fix: leave entry in PEL for XAUTOCLAIM retry;
        handlers are idempotent (publisher SET NX dedups APPROVAL_RESPONSE,
        stub callback re-fire is cheap, dedup-hit returns are no-ops).

        Regression assertion: mark_processed raises → handler returns
        WITHOUT calling consumer.ack. The original "swallow + ACK" was the
        very bug R5-4 closed.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        await sup._consumer.ensure_group()  # noqa: SLF001
        env = _env(
            t=MailboxEnvelopeType.PROGRESS_UPDATE,
            eid="01HSPYU0F80000PROGRESS0FAIL",
            producer_role=ProducerRole.CHILD_AGENT,
        )

        original = supervisor_ctx.audit_repo.mark_processed
        call_log = {"n": 0}

        async def _raise_then_pass(*args, **kwargs):
            call_log["n"] += 1
            # First call (from the ack=True branch) raises; subsequent
            # calls (e.g. a future side_effect path) pass through.
            if call_log["n"] == 1:
                raise RuntimeError("audit DB blip")
            return await original(*args, **kwargs)

        monkeypatch.setattr(
            supervisor_ctx.audit_repo, "mark_processed", _raise_then_pass
        )

        # R5-4 — spy on consumer.ack to prove the supervisor does NOT
        # ACK when mark_processed raised. Pre-fix the ACK would fire here,
        # breaking the dedup contract.
        ack_calls: list[bytes] = []
        original_ack = sup._consumer.ack  # noqa: SLF001

        async def _ack_spy(redis_id):
            ack_calls.append(redis_id)
            await original_ack(redis_id)

        monkeypatch.setattr(sup._consumer, "ack", _ack_spy)  # noqa: SLF001

        # MUST NOT raise (broad-except in _handle_envelope catches the
        # mark_processed failure and returns early — entry stays in PEL).
        await sup._handle_envelope(b"f8-0-3", env)  # noqa: SLF001

        # mark_processed was attempted once (the raising call).
        assert call_log["n"] == 1
        # R5-4 invariant — NO ACK when mark_processed failed on ack=True
        # path. Pre-fix this would be ``len(ack_calls) == 1``.
        assert ack_calls == [], (
            "ack=True path must NOT call consumer.ack when "
            "mark_processed raised (spec §5.8 layer 2 dedup marker is "
            "the truth — without it, ACKing strips XAUTOCLAIM's ability "
            "to retry). The R5-4 fix returns early on the audit "
            f"failure; got ack_calls={ack_calls!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# codex r3 [R3-5 / R3-7, HIGH CONTRACT] — synthetic envelope_id +
# correlation_id MUST fit the audit table's String(64) bounds.
# ──────────────────────────────────────────────────────────────────────────────


class TestSyntheticEnvelopeIdLengthInvariants:
    """Audit table ``envelope_id`` and ``correlation_id`` are both
    ``String(64)`` (``api/app/infrastructure/models/mailbox_envelope_audit.py
    :47,51``). Every supervisor-synthesised envelope ID — cascade, ACK
    echo, deny response — MUST stay inside that bound across realistic
    inputs.
    """

    _AUDIT_COLUMN_MAX = 64

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "reason,child_id",
        [
            # Worst-case operational inputs:
            #  - long reason from poison-drop fallback
            #  - full UUID child_id (36 chars) AND a 64-char child_id
            ("poison_drop_terminal_envelope", "ed3a8f00-1111-2222-3333-444455556666"),
            ("cancel_ack_timeout", "ed3a8f00-1111-2222-3333-444455556666"),
            ("orphan_timeout", "c" * 64),
            ("poison_drop_terminal_envelope", "c" * 64),
        ],
    )
    async def test_cascade_envelope_and_correlation_ids_fit_audit_column(
        self, supervisor_ctx, fake_redis, reason, child_id
    ):
        """R3-5 + R3-7 regression — cascade envelope_id and correlation_id
        both stay ≤ 64 across realistic (reason, child_id) combinations.
        Pre-fix, ``correlation_id = f"cascade:{reason}:{child_id}"`` could
        reach 73+ chars with ``poison_drop_terminal_envelope`` + UUID
        child.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason
        import json as _json

        sup = MailboxSupervisor(supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01)
        await sup._emit_cascade_terminate(  # noqa: SLF001
            child_id, reason=reason, destroy_reason=DestroyReason.ORPHAN_TIMEOUT
        )
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert cancel_entries, "_emit_cascade_terminate must publish one envelope"
        env_blob = cancel_entries[-1][1][b"envelope"]
        env = _json.loads(env_blob)

        assert len(env["envelope_id"]) <= self._AUDIT_COLUMN_MAX, (
            f"cascade envelope_id={env['envelope_id']!r} length "
            f"{len(env['envelope_id'])} exceeds {self._AUDIT_COLUMN_MAX} "
            f"(reason={reason!r}, child_id_len={len(child_id)})"
        )
        assert len(env["correlation_id"]) <= self._AUDIT_COLUMN_MAX, (
            f"cascade correlation_id={env['correlation_id']!r} length "
            f"{len(env['correlation_id'])} exceeds {self._AUDIT_COLUMN_MAX} "
            f"(reason={reason!r}, child_id_len={len(child_id)})"
        )
        # Stability sanity check — supervisor cascade envelope_id starts
        # with ``cascade:`` so ops grep stays meaningful even after the
        # length-bound hash refactor.
        assert env["envelope_id"].startswith("cascade:")
        assert env["correlation_id"].startswith("cascade:")

    @pytest.mark.anyio
    async def test_synthetic_ack_envelope_id_fits_when_incoming_id_is_max_length(
        self, supervisor_ctx, fake_redis, stub_lifecycle
    ):
        """R3-7 regression — when the incoming TERMINATE envelope's
        envelope_id is at the 64-char limit, the synthetic CANCEL_ACK echo
        ``ack:{sha256...}`` MUST still fit the audit column.

        Pre-fix, ``f"{envelope.envelope_id}:ack"`` would overflow to 68.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy
        import json as _json

        # Need supervisor instance for register_cancel_state hook + handler.
        MailboxSupervisor(supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01)
        handler = CancelRequestHandler()

        # Construct an incoming envelope at the 64-char limit.
        max_envelope_id = "a" * 64
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid=max_envelope_id,
            producer_role=ProducerRole.PARENT_AGENT,
            reason="user_cancel",
        )
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()

        # Read the synthetic CANCEL_ACK echo from the stream.
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        ack_envs = [
            _json.loads(fields[b"envelope"])
            for _, fields in entries
            if fields.get(b"type") == b"CANCEL_ACK"
        ]
        assert ack_envs, "TERMINATE branch must publish synthetic CANCEL_ACK"
        ack = ack_envs[-1]
        assert len(ack["envelope_id"]) <= 64, (
            f"synthetic CANCEL_ACK envelope_id={ack['envelope_id']!r} "
            f"length {len(ack['envelope_id'])} exceeds 64 "
            f"(incoming envelope_id length was {len(max_envelope_id)})"
        )
        assert ack["envelope_id"].startswith("ack:"), (
            f"synthetic envelope_id should preserve 'ack:' tag prefix; "
            f"got {ack['envelope_id']!r}"
        )

    @pytest.mark.anyio
    async def test_synthetic_deny_envelope_id_fits_when_incoming_id_is_max_length(
        self, supervisor_ctx, fake_redis
    ):
        """R3-7 regression for ApprovalRequestHandler's deny path — same
        invariant. Incoming envelope_id at 64 chars must not overflow the
        synthetic APPROVAL_RESPONSE echo.
        """
        from app.application.services.mailbox_supervisor import (
            ApprovalRequestHandler,
            MailboxSupervisor,
        )
        import json as _json

        MailboxSupervisor(supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01)
        handler = ApprovalRequestHandler()

        max_envelope_id = "b" * 64
        approval_env = MailboxEnvelope(
            envelope_id=max_envelope_id,
            type=MailboxEnvelopeType.APPROVAL_REQUEST,
            parent_session_id="root-1",
            child_session_id="child-1",
            correlation_id="corr-approval-001",
            emitted_at=datetime.now(tz=timezone.utc),
            producer_role=ProducerRole.CHILD_AGENT,
            payload={
                "tool_name": "shell.exec",
                "tool_args_snapshot": {},
                "risk_tier": "high",
                "rationale": "high-risk tool",
                "correlation_id": "corr-approval-001",
                "tool_call_id": "tc-001",
                "requested_at": datetime.now(tz=timezone.utc).isoformat(),
                "timeout_seconds": 300,
            },
        )
        await handler.handle(approval_env, supervisor_ctx)

        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        deny_envs = [
            _json.loads(fields[b"envelope"])
            for _, fields in entries
            if fields.get(b"type") == b"APPROVAL_RESPONSE"
        ]
        assert deny_envs, "ApprovalRequestHandler must publish deny response"
        deny = deny_envs[-1]
        assert len(deny["envelope_id"]) <= 64, (
            f"synthetic deny envelope_id={deny['envelope_id']!r} length "
            f"{len(deny['envelope_id'])} exceeds 64"
        )
        assert deny["envelope_id"].startswith("deny:"), (
            f"synthetic deny envelope_id should preserve 'deny:' tag "
            f"prefix; got {deny['envelope_id']!r}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# codex r3 [R3-6] (superseded by codex r6 [R6-2]) — the R3-6 producer_role
# guard defended against external producers writing
# ``CancelRequestPayload.destroy_reason``. R6-2 moved the override off
# the wire entirely (now an in-process supervisor-private side-table),
# which makes the attack structurally impossible — external producers
# cannot reach the dict. The guard is therefore moot; tests below
# assert the new contract: payload-level destroy_reason is ignored
# regardless of producer_role, and only the side-table override fires.
# ──────────────────────────────────────────────────────────────────────────────


class TestCancelRequestDestroyReasonProducerRoleGuard:
    """codex r6 [R6-2] — payload-level ``destroy_reason`` is ignored
    by the handler regardless of producer_role. The supervisor-private
    side-table is the only honored override path.
    """

    @pytest.mark.anyio
    async def test_side_table_override_for_supervisor_cascade_destroys_with_orphan_timeout(
        self, supervisor_ctx, stub_lifecycle
    ):
        """Positive: side-table override (stamped by orphan tick / poison
        drop / ``_emit_cascade_terminate``) → destroy with ORPHAN_TIMEOUT.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0R3SUPERVISOROK0001",
            producer_role=ProducerRole.SUPERVISOR,
            reason="orphan_timeout",
        )
        sup._cascade_destroy_overrides[env.envelope_id] = (  # noqa: SLF001
            DestroyReason.ORPHAN_TIMEOUT
        )
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert (env.child_session_id, DestroyReason.ORPHAN_TIMEOUT) in (
            stub_lifecycle.destroy_calls
        )
        assert (env.child_session_id, DestroyReason.FORCE_TERMINATE) not in (
            stub_lifecycle.destroy_calls
        )

    @pytest.mark.anyio
    async def test_parent_agent_payload_destroy_reason_ignored_even_with_extra_field(
        self, supervisor_ctx, stub_lifecycle
    ):
        """codex r6 [R6-2] — a hostile PARENT_AGENT injecting
        ``destroy_reason`` directly into the payload (bypassing the
        pydantic ``extra="forbid"`` schema via direct dict mutation)
        cannot influence the handler: side-table is the only path,
        external producers cannot reach it.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0R3PARENTNOOVER0001",
            producer_role=ProducerRole.PARENT_AGENT,
            reason="user_cancel",
        )
        # Simulate a hostile producer bypassing pydantic validation.
        object.__setattr__(
            env,
            "payload",
            {
                "reason": "user_cancel",
                "policy": CancelPolicy.TERMINATE.value,
                "destroy_reason": DestroyReason.ORPHAN_TIMEOUT.value,
            },
        )
        # No side-table entry — handler must default to FORCE_TERMINATE.
        assert env.envelope_id not in sup._cascade_destroy_overrides  # noqa: SLF001
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert (env.child_session_id, DestroyReason.FORCE_TERMINATE) in (
            stub_lifecycle.destroy_calls
        )
        assert (env.child_session_id, DestroyReason.ORPHAN_TIMEOUT) not in (
            stub_lifecycle.destroy_calls
        )

    @pytest.mark.anyio
    async def test_supervisor_producer_unknown_destroy_reason_falls_back(
        self, supervisor_ctx, stub_lifecycle
    ):
        """codex r6 [R6-2] — payload-level destroy_reason is ignored
        entirely. Unknown values that used to be a ValueError path are
        moot; the handler never looks at the payload field.
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.models.mailbox_envelope import CancelPolicy
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        handler = CancelRequestHandler()
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0R3SUPERVISORBAD001",
            producer_role=ProducerRole.SUPERVISOR,
            reason="orphan_timeout",
        )
        object.__setattr__(
            env,
            "payload",
            {
                "reason": "orphan_timeout",
                "policy": CancelPolicy.TERMINATE.value,
                # Payload-level destroy_reason is ignored under R6-2;
                # whether the value is sentinel-valid or junk has no effect.
                "destroy_reason": "not_a_real_enum_value",
            },
        )
        # No side-table entry — handler must default to FORCE_TERMINATE.
        assert env.envelope_id not in sup._cascade_destroy_overrides  # noqa: SLF001
        outcome = await handler.handle(env, supervisor_ctx)
        await outcome.side_effect()
        assert (env.child_session_id, DestroyReason.FORCE_TERMINATE) in (
            stub_lifecycle.destroy_calls
        )


# ──────────────────────────────────────────────────────────────────────────────
# codex r6 [R6-1, HIGH ARCH] — _register_cancel_state preserves the earliest
# requested_at_mono so XAUTOCLAIM redelivery can't reset the auto-escalate
# timer.
# ──────────────────────────────────────────────────────────────────────────────


class TestRegisterCancelStateTimestampPreservation:
    """codex r6 [R6-1] — when the same child receives a second
    REQUEST_CANCEL (XAUTOCLAIM redelivery, duplicate envelope), the
    register-cancel hook MUST NOT overwrite the existing
    ``requested_at_mono``. Pre-fix the timestamp reset → §8.4 tick saw
    "fresh" state → auto-escalate timer never fired.
    """

    @pytest.mark.anyio
    async def test_second_request_cancel_preserves_first_timestamp(
        self, supervisor_ctx
    ):
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        child = "child-r61"

        first_mono = 100.0
        await sup._register_cancel_state(  # noqa: SLF001
            child, CancelPolicy.REQUEST_CANCEL, first_mono
        )
        state_after_first = sup._cancel_states.get(child)  # noqa: SLF001
        assert state_after_first is not None
        assert state_after_first.requested_at_mono == first_mono

        # Simulate XAUTOCLAIM redelivery: same child, same policy, much
        # later mono clock. The entry MUST NOT be overwritten.
        second_mono = 100.0 + 60.0
        await sup._register_cancel_state(  # noqa: SLF001
            child, CancelPolicy.REQUEST_CANCEL, second_mono
        )
        state_after_second = sup._cancel_states.get(child)  # noqa: SLF001
        assert state_after_second is not None
        assert state_after_second.requested_at_mono == first_mono, (
            "second REQUEST_CANCEL must preserve the first timestamp so "
            "auto-escalate tick still fires; got "
            f"{state_after_second.requested_at_mono!r}"
        )

    @pytest.mark.anyio
    async def test_terminate_clear_then_fresh_request_cancel_registers_new(
        self, supervisor_ctx
    ):
        """codex r6 [R6-1] — the legitimate state transition path is
        TERMINATE → ``clear_child_tracking`` (pops the entry) → next
        REQUEST_CANCEL for the same child registers a fresh slot. This
        is the only path that re-keys cancel state; the no-op guard
        does NOT block it because the slot is empty by the time the
        second register call lands.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        child = "child-r61-cleared"

        # First REQUEST_CANCEL registers.
        await sup._register_cancel_state(  # noqa: SLF001
            child, CancelPolicy.REQUEST_CANCEL, 50.0
        )
        assert sup._cancel_states[child].requested_at_mono == 50.0  # noqa: SLF001

        # Simulate the TERMINATE → clear path.
        sup._clear_child_tracking(child)  # noqa: SLF001
        assert child not in sup._cancel_states  # noqa: SLF001

        # Fresh REQUEST_CANCEL after clear must register with NEW
        # timestamp (the slot is empty).
        await sup._register_cancel_state(  # noqa: SLF001
            child, CancelPolicy.REQUEST_CANCEL, 999.0
        )
        assert (
            sup._cancel_states[child].requested_at_mono == 999.0  # noqa: SLF001
        ), "fresh register after clear must use the new timestamp"


# ──────────────────────────────────────────────────────────────────────────────
# codex r6 [R6-3, HIGH CONTRACT] — _emit_cascade_terminate falls back to
# direct destroy when publish raises. Spec §7.5/§7.6: "XADD failure still
# kills."
# ──────────────────────────────────────────────────────────────────────────────


class TestCascadeTerminatePublishFailureFallback:
    """codex r6 [R6-3] — when ``publisher.publish`` raises inside
    ``_emit_cascade_terminate``, the supervisor falls back to a direct
    ``sandbox_lifecycle.destroy`` so the orphaned child still gets
    cleaned up. Spec §7.5/§7.6 hard rule: XADD is best-effort, the kill
    is load-bearing.
    """

    @pytest.mark.anyio
    async def test_publish_success_does_not_invoke_direct_destroy(
        self, supervisor_ctx, stub_lifecycle, fake_redis
    ):
        """Regression cover — when publish succeeds, the synthetic envelope
        flows through CancelRequestHandler normally and the direct-kill
        fallback MUST NOT fire.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-r63-ok",
            reason="orphan_timeout",
            destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
        )
        assert stub_lifecycle.destroy_calls == [], (
            "publish success path must NOT invoke direct destroy "
            "fallback; the handler picks it up via the envelope"
        )
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        cancel_entries = [
            e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
        ]
        assert len(cancel_entries) == 1
        import json as _json

        env = _json.loads(cancel_entries[0][1][b"envelope"])
        assert (
            sup._cascade_destroy_overrides.get(env["envelope_id"])  # noqa: SLF001
            == DestroyReason.ORPHAN_TIMEOUT
        )

    @pytest.mark.anyio
    async def test_publish_failure_falls_back_to_direct_destroy_with_override(
        self, supervisor_ctx, stub_lifecycle, monkeypatch
    ):
        """codex r6 [R6-3] — XADD failure triggers a direct destroy with
        the cascade's destroy_reason override.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _raise(env):
            raise RuntimeError("simulated Redis XADD failure")

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _raise)

        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-r63-publishfail",
            reason="orphan_timeout",
            destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
        )

        assert ("child-r63-publishfail", DestroyReason.ORPHAN_TIMEOUT) in (
            stub_lifecycle.destroy_calls
        ), (
            f"publish failure must fall back to direct destroy with "
            f"ORPHAN_TIMEOUT; got destroy_calls="
            f"{stub_lifecycle.destroy_calls!r}"
        )

        # Side-table was cleaned up (handler will never see this envelope).
        assert all(
            "publishfail" not in k
            for k in sup._cascade_destroy_overrides.keys()  # noqa: SLF001
        ), "side-table must be cleaned up after publish-failure fallback"

        fallback_events = [
            data
            for name, data in supervisor_ctx.telemetry.emitted
            if name == "mailbox.cascade_xadd_failed_direct_kill"
        ]
        assert len(fallback_events) == 1
        assert fallback_events[0]["child_session_id"] == "child-r63-publishfail"
        assert fallback_events[0]["destroy_reason"] == "orphan_timeout"
        assert fallback_events[0]["reason"] == "orphan_timeout"

    @pytest.mark.anyio
    async def test_publish_failure_no_override_defaults_to_force_terminate(
        self, supervisor_ctx, stub_lifecycle, monkeypatch
    ):
        """codex r6 [R6-3] — XADD failure for a cascade without an
        explicit override (e.g. cancel_ack_timeout) falls back to
        FORCE_TERMINATE (the same default the TERMINATE handler would
        have applied).
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _raise(env):
            raise RuntimeError("simulated XADD fail")

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _raise)

        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-r63-no-override",
            reason="cancel_ack_timeout",
        )
        assert (
            "child-r63-no-override",
            DestroyReason.FORCE_TERMINATE,
        ) in stub_lifecycle.destroy_calls

    @pytest.mark.anyio
    async def test_publish_failure_then_direct_destroy_already_destroyed(
        self, supervisor_ctx, monkeypatch
    ):
        """codex r6 [R6-3] — direct destroy raises ``SandboxAlreadyDestroyed``
        → telemetry, no exception propagated up to the orphan-tick caller.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _publish_raise(env):
            raise RuntimeError("XADD fail")

        async def _destroy_raise(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _publish_raise)
        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _destroy_raise
        )

        # MUST NOT raise.
        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-r63-already",
            reason="orphan_timeout",
            destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
        )

        already_events = [
            data
            for name, data in supervisor_ctx.telemetry.emitted
            if name == (
                "mailbox.cascade_xadd_failed_direct_kill_already_destroyed"
            )
        ]
        assert len(already_events) == 1
        assert already_events[0]["child_session_id"] == "child-r63-already"

    @pytest.mark.anyio
    async def test_publish_failure_then_direct_destroy_lifecycle_error_logged(
        self, supervisor_ctx, monkeypatch, caplog
    ):
        """codex r6 [R6-3] — direct destroy raises
        ``SandboxLifecycleError`` → telemetry recorded, exception logged.

        codex r7 [R7-7, HIGH CONTRACT] — earlier behavior swallowed the
        failure so the caller's ``continue`` loop kept moving. That meant
        a cascade lost to XADD + lifecycle failure cleared tracking and
        was never retried. The new contract raises
        ``_CascadeFailedError`` so callers preserve per-child tracking
        for next-tick retry. The telemetry emit + exception logging are
        still load-bearing; this test asserts both AND the new raise.
        """
        import logging

        from app.application.services.mailbox_supervisor import (
            MailboxSupervisor,
            _CascadeFailedError,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _publish_raise(env):
            raise RuntimeError("XADD fail")

        async def _destroy_raise(session_id, reason):
            raise SandboxLifecycleError("docker daemon dead")

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _publish_raise)
        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _destroy_raise
        )
        caplog.set_level(logging.ERROR)

        # codex r7 [R7-7] — raises so caller preserves tracking. The
        # telemetry emit still fires *before* the raise.
        with pytest.raises(_CascadeFailedError):
            await sup._emit_cascade_terminate(  # noqa: SLF001
                "child-r63-lifecycle",
                reason="orphan_timeout",
                destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
            )

        retryable_events = [
            data
            for name, data in supervisor_ctx.telemetry.emitted
            if name == "mailbox.cascade_xadd_failed_direct_kill_retryable_failed"
        ]
        assert len(retryable_events) == 1
        assert retryable_events[0]["child_session_id"] == "child-r63-lifecycle"


# ──────────────────────────────────────────────────────────────────────────────
# codex r8 [R8-2/R8-3, HIGH TEST] — close the 4-way destroy classification
# coverage gap for terminal handlers. The four branches every terminal
# handler must cover:
#   1. destroy() returns normally → mark_processed + ACK
#   2. destroy() raises SandboxAlreadyDestroyed → terminal-success
#   3. destroy() raises SandboxBindingMissing → terminal-success
#   4. destroy() raises SandboxLifecycleError (retryable) → re-raise (PEL)
#
# R8-2 fills the BindingMissing gap in CancelAckHandler; R8-3 fills the
# corresponding gap in CancelRequestHandler's TERMINATE branch.
# ──────────────────────────────────────────────────────────────────────────────


class TestCancelAckHandlerR8_2_BindingMissing:
    """codex r8 [R8-2, HIGH TEST] — fourth destroy-classification branch.

    CancelAckHandler already covers success / AlreadyDestroyed / retryable
    (test_mailbox_supervisor.py::TestCancelAckHandlerDestroyClassification);
    this test fills the SandboxBindingMissing branch so all four
    classification arms are locked. Branch lives at
    ``mailbox_supervisor.py:~550``.
    """

    @pytest.mark.anyio
    async def test_binding_missing_is_treated_as_terminal_success(
        self, supervisor_ctx, monkeypatch
    ):
        """destroy() raising SandboxBindingMissing → terminal-success:
        - outcome.ack=False, side_effect set (same shape as success path);
        - side_effect runs to completion without re-raising;
        - mark_processed lands;
        - telemetry ``mailbox.cancel_ack_binding_missing`` emitted exactly
          once.

        The branch is "the child's binding is already gone" → from a
        cleanup POV that's success-equivalent. Retrying would burn
        cycles with no chance of success.
        """
        from app.application.services.mailbox_supervisor import (
            CancelAckHandler,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxBindingMissing

        async def _raise(session_id, reason):
            raise SandboxBindingMissing(session_id)

        monkeypatch.setattr(supervisor_ctx.sandbox_lifecycle, "destroy", _raise)
        handler = CancelAckHandler()
        env = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            producer_role=ProducerRole.CHILD_AGENT,
            eid="01HSPYU0R82BINDINGMISS00001",
        )

        outcome = await handler.handle(env, supervisor_ctx)
        # Same shape as the success path: ack=False so side_effect drives
        # the ACK after destroy+callback complete.
        assert outcome.ack is False
        assert outcome.side_effect is not None

        # side_effect MUST NOT re-raise — BindingMissing is terminal-success.
        await outcome.side_effect()

        # mark_processed landed (terminal-success).
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

        # Telemetry recorded the BindingMissing classification.
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert emitted_names.count("mailbox.cancel_ack_binding_missing") == 1, (
            "BindingMissing branch must emit exactly one "
            "'mailbox.cancel_ack_binding_missing' telemetry event; "
            f"got emitted={supervisor_ctx.telemetry.emitted!r}"
        )


class TestCancelRequestHandlerR8_3_TerminateBindingMissing:
    """codex r8 [R8-3, HIGH TEST] — fourth destroy-classification branch
    for CancelRequestHandler's TERMINATE path. Mirrors R8-2.

    Branch lives at ``mailbox_supervisor.py:~805``.
    """

    @pytest.mark.anyio
    async def test_terminate_binding_missing_is_treated_as_terminal_success(
        self, supervisor_ctx, monkeypatch, fake_redis
    ):
        """destroy() raising SandboxBindingMissing inside TERMINATE side_effect:
        - side_effect completes without re-raising;
        - synthetic CANCEL_ACK echo IS published (terminal-success continues
          downstream so external observers / SSE bridges see the transition);
        - telemetry ``mailbox.force_terminate_binding_missing`` emitted
          exactly once;
        - ``_clear_child_tracking`` ran (the supervisor's per-child tracking
          slot for the child is dropped — proves the cleanup-before-publish
          R4-4 invariant fires on the BindingMissing branch too).
        """
        from app.application.services.mailbox_supervisor import (
            CancelRequestHandler,
            MailboxSupervisor,
        )
        from app.domain.errors.sandbox_lifecycle import SandboxBindingMissing
        from app.domain.models.mailbox_envelope import CancelPolicy

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        # Seed per-child tracking so we can verify clear_child_tracking ran.
        env = _cancel_env(
            policy=CancelPolicy.TERMINATE,
            eid="01HSPYU0R83BINDINGMISS00001",
            child_session_id="child-r83-binding-miss",
        )
        sup._last_seen_mono[env.child_session_id] = (  # noqa: SLF001
            supervisor_ctx.clock()
        )

        async def _raise(session_id, reason):
            raise SandboxBindingMissing(session_id)

        monkeypatch.setattr(supervisor_ctx.sandbox_lifecycle, "destroy", _raise)

        handler = CancelRequestHandler()
        outcome = await handler.handle(env, supervisor_ctx)
        assert outcome.ack is False
        assert outcome.side_effect is not None

        # side_effect MUST NOT re-raise — BindingMissing is terminal-success.
        await outcome.side_effect()

        # mark_processed landed.
        assert await supervisor_ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

        # Telemetry recorded the BindingMissing classification.
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert (
            emitted_names.count("mailbox.force_terminate_binding_missing") == 1
        ), (
            "TERMINATE BindingMissing branch must emit exactly one "
            "'mailbox.force_terminate_binding_missing' telemetry event; "
            f"got emitted={supervisor_ctx.telemetry.emitted!r}"
        )

        # The supervisor's per-child tracking was cleared (R4-4 invariant
        # mirrored on the BindingMissing branch).
        assert env.child_session_id not in sup._last_seen_mono, (  # noqa: SLF001
            "TERMINATE BindingMissing branch must call clear_child_tracking "
            "so the orphan tick doesn't re-cascade against the already-"
            "gone child; "
            f"got _last_seen_mono={sup._last_seen_mono!r}"
        )

        # Synthetic CANCEL_ACK echo IS published — terminal-success
        # continues downstream so external observers see the transition.
        entries = await fake_redis.xrange(
            f"actus:child:{supervisor_ctx.root_session_id}:mailbox"
        )
        import json as _json

        ack_envs = []
        for _eid, fields in entries:
            if fields.get(b"type") != b"CANCEL_ACK":
                continue
            ack_envs.append(_json.loads(fields[b"envelope"]))
        assert len(ack_envs) == 1, (
            "TERMINATE BindingMissing branch must still publish the "
            "synthetic CANCEL_ACK echo so SSE bridge / parent audit "
            "observes the terminal transition; "
            f"got entries={entries!r}"
        )
        assert ack_envs[0]["producer_role"] == "supervisor_echo"


# ──────────────────────────────────────────────────────────────────────────────
# codex r8 [R8-4, MEDIUM PERF] — poison-drop must pop the cascade override
# side-table entry to prevent a per-poison leak in long-lived supervisors.
# ──────────────────────────────────────────────────────────────────────────────


class TestPoisonDropCleansCascadeOverride:
    """codex r8 [R8-4, MEDIUM PERF] — when a synthetic cascade
    CANCEL_REQUEST poisons (its reclaim_count exceeds
    MAILBOX_POISON_MAX_RECLAIM and the supervisor drops it), the matching
    entry in ``_cascade_destroy_overrides`` must be popped. Otherwise
    long-lived supervisors accumulate one stale entry per poisoned
    cascade — slow memory leak.
    """

    @pytest.mark.anyio
    async def test_poison_dropped_cascade_envelope_pops_override(
        self, supervisor_ctx, monkeypatch
    ):
        """Pre-seed the override side-table (mirroring what
        ``_emit_cascade_terminate`` does before publish). Inject a poison
        envelope with the matching ``envelope_id``. Assert the entry is
        removed after the poison-drop hook runs.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        # Pre-seed the override as if _emit_cascade_terminate had run.
        synthetic_eid = "cascade:r84abcdef0123456789abcdef0123"
        sup._cascade_destroy_overrides[synthetic_eid] = (  # noqa: SLF001
            DestroyReason.ORPHAN_TIMEOUT
        )

        # Force the cascade republish (inside _on_poison_drop's terminal
        # branch) to no-op so the test isolates the override-pop behavior
        # from the cascade re-emission side-effect. The poison envelope
        # is itself a CANCEL_REQUEST (not a terminal type), so the
        # terminal-cascade branch is skipped anyway — but stub the call
        # defensively in case the type is changed.
        async def _noop_cascade(*args, **kwargs):  # pragma: no cover
            return None

        monkeypatch.setattr(sup, "_emit_cascade_terminate", _noop_cascade)

        poisoned = MailboxEnvelope(
            envelope_id=synthetic_eid,
            type=MailboxEnvelopeType.CANCEL_REQUEST,
            parent_session_id=supervisor_ctx.root_session_id,
            child_session_id="child-r84",
            correlation_id="cascade:r84correlation0000000000000",
            emitted_at=datetime.now(tz=timezone.utc),
            producer_role=ProducerRole.SUPERVISOR,
            payload={"reason": "orphan_timeout", "policy": "TERMINATE"},
            reclaim_count=MAILBOX_POISON_MAX_RECLAIM + 1,
        )

        # Sanity — entry is present before the poison drop.
        assert synthetic_eid in sup._cascade_destroy_overrides  # noqa: SLF001

        # Run the poison-drop hook directly.
        await sup._on_poison_drop(poisoned)  # noqa: SLF001

        # R8-4 — the override MUST be popped to prevent the leak.
        assert synthetic_eid not in sup._cascade_destroy_overrides, (  # noqa: SLF001
            "R8-4 — _on_poison_drop must pop the matching "
            "_cascade_destroy_overrides entry; otherwise long-lived "
            "supervisors leak one entry per poisoned cascade. "
            f"got _cascade_destroy_overrides={sup._cascade_destroy_overrides!r}"  # noqa: SLF001
        )

    @pytest.mark.anyio
    async def test_poison_drop_non_cascade_envelope_does_not_disturb_other_overrides(
        self, supervisor_ctx
    ):
        """Regression — popping is keyed on the dropped envelope's
        ``envelope_id``; an unrelated override for a different cascade
        envelope_id must remain.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        unrelated_eid = "cascade:unrelated0123456789abcdef01"
        sup._cascade_destroy_overrides[unrelated_eid] = (  # noqa: SLF001
            DestroyReason.ORPHAN_TIMEOUT
        )

        # Non-cascade envelope (regular PROGRESS_UPDATE that poisoned).
        poisoned = _env(
            t=MailboxEnvelopeType.PROGRESS_UPDATE,
            eid="01HSPYU0R84UNRELATED000000001",
            reclaim_count=MAILBOX_POISON_MAX_RECLAIM + 1,
        )

        await sup._on_poison_drop(poisoned)  # noqa: SLF001

        # Unrelated override remains.
        assert unrelated_eid in sup._cascade_destroy_overrides  # noqa: SLF001


# ──────────────────────────────────────────────────────────────────────────────
# codex r8 [R8-5, HIGH CONTRACT] — _CascadeFailedError from the poison-drop
# hook must surface as a dedicated telemetry event before being swallowed.
# ──────────────────────────────────────────────────────────────────────────────


class TestPoisonDropCascadeFailedCriticalTelemetry:
    """codex r8 [R8-5, HIGH CONTRACT] — when the poison-drop hook fires
    a cascade and BOTH the XADD publish AND the direct-kill fallback
    fail (the fallback raises ``SandboxLifecycleError`` → wrapped in
    ``_CascadeFailedError`` per R7-7), the supervisor MUST emit a
    distinguishable telemetry event so ops alerts can fire on the
    silent cleanup loss. The earlier broad ``except Exception`` swallowed
    the signal entirely → only a generic logger.exception line.

    Spec §5.7 step 3 mandates that the poison drop hook fire cascade so
    every running child reaches destroy(). If the cascade fails, that's
    a critical operational signal — not a routine warning.
    """

    @pytest.mark.anyio
    async def test_poison_cascade_failed_critical_telemetry_fires_and_ack_still_proceeds(
        self, supervisor_ctx, fake_redis, monkeypatch
    ):
        """End-to-end via ``_handle_envelope``:
        - Inject a terminal-type poison envelope (CANCEL_ACK).
        - Force ``publisher.publish`` to raise so the cascade XADD fails.
        - Force ``sandbox_lifecycle.destroy`` to raise
          ``SandboxLifecycleError`` so the direct-kill fallback also fails.
        - Assert ``mailbox.poison_cascade_failed_critical`` telemetry was
          emitted with the right envelope_id + child_session_id + error.
        - Assert ``_consumer.ack`` still ran (the loop-breaking invariant).
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        await sup._consumer.ensure_group()  # noqa: SLF001

        async def _publish_raise(env):
            raise RuntimeError("redis XADD down (R8-5 test)")

        async def _destroy_raise(session_id, reason):
            raise SandboxLifecycleError("docker daemon dead (R8-5 test)")

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _publish_raise)
        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _destroy_raise
        )

        # Spy on consumer.ack so we can assert it ran exactly once.
        ack_calls: list[bytes] = []
        original_ack = sup._consumer.ack  # noqa: SLF001

        async def _spy_ack(rid):
            ack_calls.append(rid)
            await original_ack(rid)

        monkeypatch.setattr(sup._consumer, "ack", _spy_ack)  # noqa: SLF001

        # Use a terminal type (CANCEL_ACK) so _on_poison_drop's terminal-
        # cascade branch fires; that branch is what calls
        # _emit_cascade_terminate which is where _CascadeFailedError
        # originates.
        envelope = _env(
            t=MailboxEnvelopeType.CANCEL_ACK,
            eid="01HSPYU0R85POISON_CASCADE0001",
            child_session_id="child-r85-cascade-fail",
            reclaim_count=MAILBOX_POISON_MAX_RECLAIM + 1,
        )

        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        # Two telemetry events expected: the routine
        # poison_message_dropped + the new R8-5 critical event.
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert "mailbox.poison_message_dropped" in emitted_names
        critical_events = [
            data
            for name, data in supervisor_ctx.telemetry.emitted
            if name == "mailbox.poison_cascade_failed_critical"
        ]
        assert len(critical_events) == 1, (
            "R8-5 — exactly one mailbox.poison_cascade_failed_critical "
            "must be emitted when the poison-drop cascade raises "
            "_CascadeFailedError; otherwise ops alerts cannot detect "
            "the silent cleanup loss. "
            f"got emitted={supervisor_ctx.telemetry.emitted!r}"
        )
        event = critical_events[0]
        assert event["envelope_id"] == envelope.envelope_id
        assert event["child_session_id"] == envelope.child_session_id
        assert event["type"] == MailboxEnvelopeType.CANCEL_ACK.value
        assert "docker daemon dead" in event["error"]

        # ACK still ran — break-the-loop invariant preserved.
        assert len(ack_calls) == 1, (
            "R8-5 — ACK MUST still fire after the critical telemetry "
            "event so the poison envelope leaves PEL; otherwise the "
            "redelivery loop never breaks. "
            f"got ack_calls={ack_calls!r}"
        )

    @pytest.mark.anyio
    async def test_poison_cascade_failed_critical_isolates_telemetry_emit_failure(
        self, supervisor_ctx, monkeypatch
    ):
        """Regression — even if the
        ``mailbox.poison_cascade_failed_critical`` telemetry emit itself
        raises (sink down), the supervisor must still ACK. Mirrors the
        existing telemetry-isolation pattern around poison_message_dropped.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        await sup._consumer.ensure_group()  # noqa: SLF001

        async def _publish_raise(env):
            raise RuntimeError("redis XADD down")

        async def _destroy_raise(session_id, reason):
            raise SandboxLifecycleError("docker dead")

        # Force the critical-event telemetry emit to raise; routine
        # poison_message_dropped emit should still pass (we want to
        # exercise only the critical-event isolation branch).
        original_emit = supervisor_ctx.telemetry.emit

        async def _selective_boom(name, data):
            if name == "mailbox.poison_cascade_failed_critical":
                raise RuntimeError("OTel sink down (R8-5 isolation test)")
            await original_emit(name, data)

        monkeypatch.setattr(
            supervisor_ctx.telemetry, "emit", _selective_boom
        )
        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _publish_raise)
        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _destroy_raise
        )

        ack_calls: list[bytes] = []
        original_ack = sup._consumer.ack  # noqa: SLF001

        async def _spy_ack(rid):
            ack_calls.append(rid)
            await original_ack(rid)

        monkeypatch.setattr(sup._consumer, "ack", _spy_ack)  # noqa: SLF001

        envelope = _env(
            t=MailboxEnvelopeType.RESULT_READY,
            eid="01HSPYU0R85POISON_CRIT_BOOM01",
            child_session_id="child-r85-emit-boom",
            reclaim_count=MAILBOX_POISON_MAX_RECLAIM + 1,
        )

        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        # ACK still ran — telemetry isolation kept the loop-breaking
        # invariant intact.
        assert len(ack_calls) == 1


# ──────────────────────────────────────────────────────────────────────────────
# codex r8 [R8-6, MEDIUM CONTRACT] — direct-kill fallback must call
# _clear_child_tracking on terminal-success paths (destroy success /
# AlreadyDestroyed / BindingMissing). Without this the orphan tick re-
# cascades against an already-dead child.
# ──────────────────────────────────────────────────────────────────────────────


class TestCascadeDirectKillClearsChildTracking:
    """codex r8 [R8-6, MEDIUM CONTRACT] — direct-kill fallback hits
    three terminal-success branches (destroy returns / AlreadyDestroyed /
    BindingMissing) and one retryable branch (SandboxLifecycleError).
    The first three MUST drop per-child tracking; the fourth MUST
    preserve tracking (covered by existing R7-7 tests).
    """

    @pytest.mark.anyio
    async def test_direct_kill_success_clears_child_tracking(
        self, supervisor_ctx, monkeypatch
    ):
        """publish raises → fallback destroy returns normally → tracking
        cleared. Without R8-6 the next orphan tick would re-cascade
        against the same already-dead child.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        # Pre-seed tracking so we can verify it's dropped.
        sup._last_seen_mono["child-r86-success"] = (  # noqa: SLF001
            supervisor_ctx.clock()
        )

        async def _publish_raise(env):
            raise RuntimeError("XADD fail")

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _publish_raise)
        # destroy returns normally — terminal-success.

        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-r86-success",
            reason="orphan_timeout",
            destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
        )

        # destroy was invoked via direct-kill fallback.
        assert (
            "child-r86-success",
            DestroyReason.ORPHAN_TIMEOUT,
        ) in supervisor_ctx.sandbox_lifecycle.destroy_calls

        # R8-6 — tracking dropped.
        assert "child-r86-success" not in sup._last_seen_mono, (  # noqa: SLF001
            "R8-6 — direct-kill success must call _clear_child_tracking "
            "so the orphan tick doesn't re-cascade against the already-"
            "destroyed child; "
            f"got _last_seen_mono={sup._last_seen_mono!r}"  # noqa: SLF001
        )

    @pytest.mark.anyio
    async def test_direct_kill_already_destroyed_clears_child_tracking(
        self, supervisor_ctx, monkeypatch
    ):
        """publish raises → fallback destroy raises AlreadyDestroyed →
        terminal-success → tracking cleared.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.errors.sandbox_lifecycle import SandboxAlreadyDestroyed
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._last_seen_mono["child-r86-already"] = (  # noqa: SLF001
            supervisor_ctx.clock()
        )

        async def _publish_raise(env):
            raise RuntimeError("XADD fail")

        async def _destroy_already(session_id, reason):
            raise SandboxAlreadyDestroyed(session_id)

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _publish_raise)
        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _destroy_already
        )

        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-r86-already",
            reason="orphan_timeout",
            destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
        )

        # R8-6 — tracking dropped on AlreadyDestroyed branch too.
        assert "child-r86-already" not in sup._last_seen_mono  # noqa: SLF001

        # Confirm the right telemetry fired.
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert (
            "mailbox.cascade_xadd_failed_direct_kill_already_destroyed"
            in emitted_names
        )

    @pytest.mark.anyio
    async def test_direct_kill_binding_missing_clears_child_tracking(
        self, supervisor_ctx, monkeypatch
    ):
        """publish raises → fallback destroy raises BindingMissing →
        terminal-success → tracking cleared.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.errors.sandbox_lifecycle import SandboxBindingMissing
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        sup._last_seen_mono["child-r86-binding"] = (  # noqa: SLF001
            supervisor_ctx.clock()
        )

        async def _publish_raise(env):
            raise RuntimeError("XADD fail")

        async def _destroy_binding(session_id, reason):
            raise SandboxBindingMissing(session_id)

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _publish_raise)
        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _destroy_binding
        )

        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-r86-binding",
            reason="orphan_timeout",
            destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
        )

        # R8-6 — tracking dropped on BindingMissing branch too.
        assert "child-r86-binding" not in sup._last_seen_mono  # noqa: SLF001

        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert (
            "mailbox.cascade_xadd_failed_direct_kill_binding_missing"
            in emitted_names
        )


# ──────────────────────────────────────────────────────────────────────────────
# codex r10 [R10-2, HIGH CONTRACT] — direct-kill fallback must fire the
# agent_service_callback BEFORE destroy so the spec §7.6 stop-before-destroy
# order is preserved even in the XADD-failure emergency path. Earlier rounds
# (R6-3) skipped the callback on the rationale that we lacked a useful
# envelope; the cascade synthetic envelope is now threaded through so the
# callback can fire. Best-effort: a callback fault must NOT block destroy.
# ──────────────────────────────────────────────────────────────────────────────


class TestCascadeDirectKillFiresCallbackBeforeDestroy:
    """codex r10 [R10-2, HIGH CONTRACT] — spec §7.6 stop-before-destroy.

    The direct-kill fallback runs when the cascade XADD failed and the
    handler will never see the envelope. The callback (which signals
    stop_session in production) MUST still fire before destroy.
    """

    @pytest.mark.anyio
    async def test_direct_kill_calls_callback_then_destroy_in_order(
        self, supervisor_ctx, stub_lifecycle, stub_agent_callback, monkeypatch
    ):
        """The synthetic envelope reaches ``agent_service_callback`` BEFORE
        ``sandbox_lifecycle.destroy`` lands. We capture the ordering by
        timestamping each spy in a shared list.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _raise(env):
            raise RuntimeError("simulated XADD failure")

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _raise)

        ordering: list[str] = []

        # Wrap the existing _CollectingCallback so we capture order while
        # preserving the receive log.
        original_callback = supervisor_ctx.agent_service_callback

        async def _ordered_callback(env):
            ordering.append("callback")
            await original_callback(env)

        supervisor_ctx.agent_service_callback = _ordered_callback

        original_destroy = stub_lifecycle.destroy

        async def _ordered_destroy(session_id, reason):
            ordering.append("destroy")
            await original_destroy(session_id, reason)

        monkeypatch.setattr(stub_lifecycle, "destroy", _ordered_destroy)

        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-r10-callback-order",
            reason="orphan_timeout",
            destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
        )

        # R10-2 — callback fires BEFORE destroy per spec §7.6.
        assert ordering == ["callback", "destroy"], (
            f"R10-2 — direct-kill must fire agent_service_callback BEFORE "
            f"sandbox_lifecycle.destroy (spec §7.6 stop-before-destroy). "
            f"got ordering={ordering!r}"
        )

        # The callback received the cascade synthetic CANCEL_REQUEST envelope.
        assert len(stub_agent_callback.received) == 1
        assert (
            stub_agent_callback.received[0].type
            == MailboxEnvelopeType.CANCEL_REQUEST
        )
        assert (
            stub_agent_callback.received[0].child_session_id
            == "child-r10-callback-order"
        )

        # Destroy still landed.
        assert (
            "child-r10-callback-order",
            DestroyReason.ORPHAN_TIMEOUT,
        ) in stub_lifecycle.destroy_calls

    @pytest.mark.anyio
    async def test_direct_kill_callback_exception_does_not_block_destroy(
        self, supervisor_ctx, stub_lifecycle, monkeypatch
    ):
        """Best-effort callback: a RuntimeError from the callback MUST NOT
        block destroy (the load-bearing kill step). Logger.exception runs
        but the path continues to destroy.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _publish_raise(env):
            raise RuntimeError("XADD down")

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _publish_raise)

        async def _callback_raise(env):
            raise RuntimeError("agent_service_callback fault")

        supervisor_ctx.agent_service_callback = _callback_raise

        # MUST NOT raise — callback fault is logged + swallowed.
        await sup._emit_cascade_terminate(  # noqa: SLF001
            "child-r10-callback-boom",
            reason="orphan_timeout",
            destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
        )

        # Destroy ran despite the callback raising.
        assert (
            "child-r10-callback-boom",
            DestroyReason.ORPHAN_TIMEOUT,
        ) in stub_lifecycle.destroy_calls, (
            "R10-2 — callback exception MUST NOT block destroy (callback "
            "is best-effort; destroy is the load-bearing kill step)"
        )

    @pytest.mark.anyio
    async def test_direct_kill_callback_cancelled_error_propagates(
        self, supervisor_ctx, stub_lifecycle, monkeypatch
    ):
        """``CancelledError`` from the callback MUST propagate so caller-
        driven cancellation aborts cleanly — distinct from broad-exception
        swallowing.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        async def _publish_raise(env):
            raise RuntimeError("XADD down")

        monkeypatch.setattr(supervisor_ctx.publisher, "publish", _publish_raise)

        async def _callback_cancelled(env):
            raise asyncio.CancelledError()

        supervisor_ctx.agent_service_callback = _callback_cancelled

        with pytest.raises(asyncio.CancelledError):
            await sup._emit_cascade_terminate(  # noqa: SLF001
                "child-r10-callback-cancelled",
                reason="orphan_timeout",
                destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
            )

        # Destroy did NOT run — cancellation cut the path early.
        assert all(
            sid != "child-r10-callback-cancelled"
            for sid, _ in stub_lifecycle.destroy_calls
        )


# ──────────────────────────────────────────────────────────────────────────────
# codex r10 [R10-1, HIGH ARCH] — cascade CANCEL_REQUEST poison-drop branch.
#
# When the orphan tick or cancel auto-escalate tick publishes a synthetic
# cascade CANCEL_REQUEST(TERMINATE) AND clears per-child tracking on the
# happy path, the child has no remaining retry signal if the cascade
# envelope then exhausts XAUTOCLAIM retries and hits the poison gate.
# Previously _on_poison_drop only handled RESULT_READY / CANCEL_ACK; the
# CANCEL_REQUEST poison drop was silently ACKed.
#
# Fix: extend _on_poison_drop with a CANCEL_REQUEST(TERMINATE) branch that
# fires the direct-kill fallback (callback → destroy) so the orphan still
# gets destroyed. The destroy_reason is reused from the side-table override
# captured before pop; absent that, FORCE_TERMINATE is the conservative
# default (same as the handler default).
# ──────────────────────────────────────────────────────────────────────────────


class TestPoisonDropCascadeCancelRequestDirectKill:
    """codex r10 [R10-1, HIGH ARCH] — cascade CANCEL_REQUEST poison drop
    must fall back to direct-kill so the orphaned child still gets destroyed."""

    def _cascade_envelope(
        self,
        child_id: str,
        envelope_id: str = "cascade:r10test1",
        *,
        reason: str = "orphan_timeout",
        reclaim_count: int = 0,
    ) -> MailboxEnvelope:
        from app.domain.models.mailbox_envelope import CancelRequestPayload

        return MailboxEnvelope(
            envelope_id=envelope_id,
            type=MailboxEnvelopeType.CANCEL_REQUEST,
            parent_session_id="root-1",
            child_session_id=child_id,
            correlation_id="cascade:r10corr",
            emitted_at=datetime.now(tz=timezone.utc),
            producer_role=ProducerRole.SUPERVISOR,
            payload=CancelRequestPayload(
                reason=reason,
                policy=CancelPolicy.TERMINATE,
            ).model_dump(mode="json"),
            reclaim_count=reclaim_count,
        )

    @pytest.mark.anyio
    async def test_poison_drop_cascade_cancel_request_fires_direct_kill_with_override(
        self, supervisor_ctx, stub_lifecycle
    ):
        """Override is captured-then-popped from ``_cascade_destroy_overrides``
        BEFORE the direct-kill path runs, so the destroy lands with
        ORPHAN_TIMEOUT (the original orphan-tick tag) rather than the
        FORCE_TERMINATE default.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        envelope = self._cascade_envelope(
            "child-r10-poison-override", envelope_id="cascade:r10poisonOR1"
        )
        # Side-table populated by the original orphan tick before publish.
        sup._cascade_destroy_overrides[envelope.envelope_id] = (  # noqa: SLF001
            DestroyReason.ORPHAN_TIMEOUT
        )

        await sup._on_poison_drop(envelope)  # noqa: SLF001

        # Direct-kill fired with the captured override.
        assert (
            "child-r10-poison-override",
            DestroyReason.ORPHAN_TIMEOUT,
        ) in stub_lifecycle.destroy_calls, (
            "R10-1 — cascade CANCEL_REQUEST poison drop must reuse the "
            "side-table destroy_reason override; got destroy_calls="
            f"{stub_lifecycle.destroy_calls!r}"
        )

        # Side-table cleaned up (the side-table pop runs at the top of
        # ``_on_poison_drop`` before the cascade branch).
        assert (
            envelope.envelope_id
            not in sup._cascade_destroy_overrides  # noqa: SLF001
        )

        # Per-child tracking cleared on direct-kill success (R8-6 + R10-1).
        assert "child-r10-poison-override" not in sup._last_seen_mono  # noqa: SLF001
        assert "child-r10-poison-override" not in sup._cancel_states  # noqa: SLF001

        # Telemetry signals the direct-kill success path.
        emitted_names = [n for n, _ in supervisor_ctx.telemetry.emitted]
        assert "mailbox.cascade_xadd_failed_direct_kill" in emitted_names

    @pytest.mark.anyio
    async def test_poison_drop_cascade_cancel_request_no_override_defaults_to_force_terminate(
        self, supervisor_ctx, stub_lifecycle
    ):
        """A cascade CANCEL_REQUEST that never had a destroy_reason override
        (e.g. cancel_ack_timeout path) falls back to FORCE_TERMINATE —
        the same conservative default the TERMINATE handler would apply.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.session import DestroyReason

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )

        envelope = self._cascade_envelope(
            "child-r10-poison-default",
            envelope_id="cascade:r10poisonDEF",
            reason="cancel_ack_timeout",
        )
        # No side-table entry — exercises the FORCE_TERMINATE default branch.

        await sup._on_poison_drop(envelope)  # noqa: SLF001

        assert (
            "child-r10-poison-default",
            DestroyReason.FORCE_TERMINATE,
        ) in stub_lifecycle.destroy_calls

    @pytest.mark.anyio
    async def test_poison_drop_cascade_cancel_request_direct_kill_retryable_emits_critical(
        self, supervisor_ctx, monkeypatch
    ):
        """When the direct-kill fallback raises ``SandboxLifecycleError``
        (retryable), the cascade-CANCEL_REQUEST poison branch must
        propagate ``_CascadeFailedError`` so the outer ``_handle_envelope``
        wrapper emits ``mailbox.poison_cascade_failed_critical`` (R8-5).
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        await sup._consumer.ensure_group()  # noqa: SLF001

        async def _destroy_raise(session_id, reason):
            raise SandboxLifecycleError("docker daemon dead (R10-1 test)")

        monkeypatch.setattr(
            supervisor_ctx.sandbox_lifecycle, "destroy", _destroy_raise
        )

        # Use reclaim_count > MAX so the envelope hits the poison gate via
        # the full _handle_envelope wrapper (which is where the critical
        # telemetry emits live).
        envelope = self._cascade_envelope(
            "child-r10-poison-retryable",
            envelope_id="cascade:r10poisonRET",
            reclaim_count=MAILBOX_POISON_MAX_RECLAIM + 1,
        )

        await sup._handle_envelope(b"0-1", envelope)  # noqa: SLF001

        critical_events = [
            data
            for name, data in supervisor_ctx.telemetry.emitted
            if name == "mailbox.poison_cascade_failed_critical"
        ]
        assert len(critical_events) == 1, (
            "R10-1 — cascade CANCEL_REQUEST poison drop direct-kill "
            "retryable failure must emit the R8-5 critical telemetry. "
            f"got emitted={supervisor_ctx.telemetry.emitted!r}"
        )
        assert (
            critical_events[0]["envelope_id"] == envelope.envelope_id
        )
        assert (
            critical_events[0]["child_session_id"]
            == "child-r10-poison-retryable"
        )
        assert critical_events[0]["type"] == MailboxEnvelopeType.CANCEL_REQUEST.value

    @pytest.mark.anyio
    async def test_poison_drop_cascade_non_terminate_policy_does_not_fire_direct_kill(
        self, supervisor_ctx, stub_lifecycle
    ):
        """Defensive: a synthetic cascade with policy=REQUEST_CANCEL (not
        TERMINATE) is structurally not produced by the supervisor today,
        but the branch guards on TERMINATE only — verify the gate.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor
        from app.domain.models.mailbox_envelope import CancelRequestPayload

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        envelope = MailboxEnvelope(
            envelope_id="cascade:r10notterm",
            type=MailboxEnvelopeType.CANCEL_REQUEST,
            parent_session_id="root-1",
            child_session_id="child-r10-non-terminate",
            correlation_id="cascade:r10notterm",
            emitted_at=datetime.now(tz=timezone.utc),
            producer_role=ProducerRole.SUPERVISOR,
            payload=CancelRequestPayload(
                reason="orphan_timeout",
                policy=CancelPolicy.REQUEST_CANCEL,
            ).model_dump(mode="json"),
        )

        await sup._on_poison_drop(envelope)  # noqa: SLF001

        # No direct-kill fired for the child.
        assert all(
            sid != "child-r10-non-terminate"
            for sid, _ in stub_lifecycle.destroy_calls
        )

    @pytest.mark.anyio
    async def test_poison_drop_cascade_cancel_request_empty_child_does_not_fire(
        self, supervisor_ctx, stub_lifecycle
    ):
        """The cascade-CANCEL_REQUEST branch guards on a non-empty
        ``child_session_id`` — defensive gate matches the existing
        terminal-cascade branch's empty-id guard so cross-root forgeries
        cannot trip the direct-kill path.
        """
        from app.application.services.mailbox_supervisor import MailboxSupervisor

        sup = MailboxSupervisor(
            supervisor_ctx, block_ms=0, idle_poll_sleep_s=0.01
        )
        envelope = self._cascade_envelope(
            "placeholder", envelope_id="cascade:r10empty1"
        )
        # Force empty child id past the model validator.
        object.__setattr__(envelope, "child_session_id", "")

        await sup._on_poison_drop(envelope)  # noqa: SLF001

        assert stub_lifecycle.destroy_calls == [], (
            "R10-1 — empty child_session_id must NOT trip the direct-kill "
            "path; defensive gate matches the terminal-cascade branch."
        )
