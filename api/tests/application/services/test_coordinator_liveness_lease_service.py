from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.application.services.coordinator_liveness_lease_service import (
    CoordinatorChildLease,
    CoordinatorLivenessLeaseRejected,
    CoordinatorLivenessLeaseService,
)
from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.domain.models.session import Session, SessionStatus


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Clock:
    def __init__(self, *, wall: float = 1_000.0, monotonic: float = 50.0) -> None:
        self.wall = wall
        self.monotonic = monotonic

    def time(self) -> float:
        return self.wall

    def mono(self) -> float:
        return self.monotonic


class _SessionRepository:
    def __init__(self, row: Session | None) -> None:
        self.row = row
        self.lookups: list[str] = []
        self.lookup_started: asyncio.Event | None = None
        self.release_lookup: asyncio.Event | None = None

    async def get_by_id(self, session_id: str) -> Session | None:
        self.lookups.append(session_id)
        if self.lookup_started is not None:
            self.lookup_started.set()
        if self.release_lookup is not None:
            await self.release_lookup.wait()
        if self.row is None or self.row.id != session_id:
            return None
        return self.row


class _Pipeline:
    def __init__(self, redis: "_Redis") -> None:
        self.redis = redis
        self.commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def __aenter__(self) -> "_Pipeline":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def hset(self, *args: Any, **kwargs: Any) -> "_Pipeline":
        self.commands.append(("hset", args, kwargs))
        return self

    def expire(self, *args: Any, **kwargs: Any) -> "_Pipeline":
        self.commands.append(("expire", args, kwargs))
        return self

    async def execute(self) -> list[Any]:
        if self.redis.execute_error is not None:
            raise self.redis.execute_error
        if self.redis.fail_execute:
            raise RuntimeError("redis unavailable")
        results: list[Any] = []
        for name, args, kwargs in self.commands:
            results.append(await getattr(self.redis, name)(*args, **kwargs))
        self.redis.executed_pipelines.append(list(self.commands))
        return results


class _Redis:
    """Small Redis fake exercising the public hset/expire/pipeline contract."""

    def __init__(self, *, bytes_reads: bool = False) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.expiries: dict[str, int] = {}
        self.pttls: dict[str, int | str | None] = {}
        self.streams: dict[str, list[dict[str, Any]]] = {}
        self.executed_pipelines: list[
            list[tuple[str, tuple[Any, ...], dict[str, Any]]]
        ] = []
        self.pipeline_transaction_values: list[bool] = []
        self.fail_execute = False
        self.execute_error: BaseException | None = None
        self.bytes_reads = bytes_reads
        self._eval_lock = asyncio.Lock()

    def pipeline(self, *, transaction: bool) -> _Pipeline:
        self.pipeline_transaction_values.append(transaction)
        return _Pipeline(self)

    async def hset(self, key: str, *, mapping: dict[str, Any]) -> int:
        self.hashes[key] = {str(k): str(v) for k, v in mapping.items()}
        return len(mapping)

    async def expire(self, key: str, ttl: int) -> bool:
        self.expiries[key] = ttl
        self.pttls[key] = ttl * 1_000
        return True

    async def hgetall(self, key: str) -> dict[Any, Any]:
        values = self.hashes.get(key, {})
        if not self.bytes_reads:
            return dict(values)
        return {
            field.encode(): value.encode()
            for field, value in values.items()
        }

    async def delete(self, key: str) -> int:
        existed = key in self.hashes or key in self.expiries or key in self.pttls
        self.hashes.pop(key, None)
        self.expiries.pop(key, None)
        self.pttls.pop(key, None)
        return int(existed)

    async def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        if self.execute_error is not None:
            raise self.execute_error
        if self.fail_execute:
            raise RuntimeError("redis unavailable")
        assert numkeys == 1
        key = str(args[0])
        argv = [str(value) for value in args[1:]]
        async with self._eval_lock:
            self.executed_pipelines.append([("eval", (script, numkeys, *args), {})])
            if "coordinator-startup-cas-v1" in script:
                (
                    root_id,
                    parent_id,
                    child_id,
                    run_id,
                    work_unit_id,
                    last_seen,
                    phase,
                    ttl,
                ) = argv
                current = self.hashes.get(key)
                if current is not None:
                    if current.get("state") == "terminal":
                        return 0
                    expected = {
                        "root_session_id": root_id,
                        "parent_session_id": parent_id,
                        "child_session_id": child_id,
                        "coordinator_run_id": run_id,
                        "work_unit_id": work_unit_id,
                    }
                    expected_fields = {
                        *expected,
                        "last_seen_epoch",
                        "phase",
                    }
                    if (
                        set(current) != expected_fields
                        or any(
                            current.get(field) != value
                            for field, value in expected.items()
                        )
                    ):
                        return 0
                self.hashes[key] = {
                    "root_session_id": root_id,
                    "parent_session_id": parent_id,
                    "child_session_id": child_id,
                    "coordinator_run_id": run_id,
                    "work_unit_id": work_unit_id,
                    "last_seen_epoch": last_seen,
                    "phase": phase,
                }
                ttl_ms = int(ttl)
                self.expiries[key] = (ttl_ms + 999) // 1_000
                self.pttls[key] = ttl_ms
                return 1
            if "coordinator-refresh-cas-v1" in script:
                (
                    root_id,
                    parent_id,
                    child_id,
                    run_id,
                    work_unit_id,
                    last_seen,
                    phase,
                    ttl,
                ) = argv
                current = self.hashes.get(key)
                expected = {
                    "root_session_id": root_id,
                    "parent_session_id": parent_id,
                    "child_session_id": child_id,
                    "coordinator_run_id": run_id,
                    "work_unit_id": work_unit_id,
                }
                required_fields = {
                    *expected,
                    "last_seen_epoch",
                    "phase",
                }
                if (
                    current is None
                    or current.get("state") == "terminal"
                    or set(current) != required_fields
                    or any(
                        current.get(field) != value
                        for field, value in expected.items()
                    )
                ):
                    return 0
                current["last_seen_epoch"] = last_seen
                current["phase"] = phase
                ttl_ms = int(ttl)
                self.expiries[key] = (ttl_ms + 999) // 1_000
                self.pttls[key] = ttl_ms
                return 1
            if "coordinator-terminal-tombstone-v1" in script:
                child_id, ttl = argv
                self.hashes[key] = {
                    "state": "terminal",
                    "child_session_id": child_id,
                }
                ttl_ms = int(ttl)
                self.expiries[key] = (ttl_ms + 999) // 1_000
                self.pttls[key] = ttl_ms
                return 1
            if "coordinator-live-lease-read-pttl-v1" in script:
                current = self.hashes.get(key)
                required_fields = (
                    "root_session_id", "parent_session_id",
                    "child_session_id", "coordinator_run_id",
                    "work_unit_id", "last_seen_epoch", "phase",
                )
                if current is None or set(current) != set(required_fields):
                    return []
                pttl = self.pttls.get(key)
                maximum_ttl = int(argv[0])
                if pttl is None:
                    return []
                if isinstance(pttl, int) and (
                    pttl <= 0 or pttl > maximum_ttl
                ):
                    return []
                values: list[Any] = [current[field] for field in required_fields]
                if self.bytes_reads:
                    values = [value.encode() for value in values]
                return [*values, pttl]
            if "coordinator-live-lease-compare-delete-v1" in script:
                current = self.hashes.get(key)
                if current is None:
                    return -1
                required_fields = {
                    "root_session_id", "parent_session_id",
                    "child_session_id", "coordinator_run_id",
                    "work_unit_id", "last_seen_epoch", "phase",
                }
                if (
                    current.get("state") == "terminal"
                    or set(current) != required_fields
                ):
                    return 0
                expected = dict(zip(
                    (
                        "root_session_id", "parent_session_id",
                        "child_session_id", "coordinator_run_id",
                        "work_unit_id",
                    ),
                    argv,
                    strict=True,
                ))
                if any(
                    current.get(field) != value
                    for field, value in expected.items()
                ):
                    return 0
                await self.delete(key)
                return 1
        raise AssertionError("unexpected Redis script")

    def trim_stream(self, key: str) -> None:
        self.streams[key] = []


def _row(**updates: Any) -> Session:
    base = Session(
        id="child-1",
        parent_session_id="parent-1",
        worker_type="subagent",
        depth=1,
        root_session_id="root-1",
        subagent_control_plane="mailbox",
        tool_filter_preset="coordinator_step",
        coordinator_run_id="run-1",
        work_unit_id="wu-1",
        status=SessionStatus.RUNNING,
    )
    return base.model_copy(update=updates)


def _heartbeat(**updates: Any) -> MailboxEnvelope:
    values: dict[str, Any] = {
        "envelope_id": "hb-envelope-1",
        "type": MailboxEnvelopeType.PROGRESS_UPDATE,
        "parent_session_id": "parent-1",
        "child_session_id": "child-1",
        "correlation_id": "hb:child-1",
        "emitted_at": datetime.now(timezone.utc),
        "producer_role": ProducerRole.CHILD_AGENT,
        "payload": {
            "kind": "heartbeat",
            "visibility": "hidden",
            "phase": "in_tool",
            "tool_call_id": "tc-1",
        },
    }
    values.update(updates)
    return MailboxEnvelope(**values)


def _service(
    *,
    redis: _Redis | None = None,
    row: Session | None = None,
    clock: _Clock | None = None,
    touch_parent: AsyncMock | None = None,
    renew_child_sandbox: AsyncMock | None = None,
    renew_ordinary_sandboxes: AsyncMock | None = None,
    renew_quota: AsyncMock | None = None,
    sleep: AsyncMock | None = None,
    **config: Any,
) -> tuple[CoordinatorLivenessLeaseService, _Redis, _SessionRepository, _Clock]:
    redis = redis or _Redis()
    repo = _SessionRepository(_row() if row is None else row)
    clock = clock or _Clock()
    service = CoordinatorLivenessLeaseService(
        redis=redis,
        session_repository=repo,
        clock=clock.time,
        monotonic_clock=clock.mono,
        sleep=sleep or AsyncMock(),
        touch_parent=touch_parent,
        renew_child_sandbox=renew_child_sandbox,
        renew_ordinary_sandboxes=renew_ordinary_sandboxes,
        renew_quota=renew_quota,
        **config,
    )
    return service, redis, repo, clock


async def _record_startup(
    service: CoordinatorLivenessLeaseService,
) -> CoordinatorChildLease:
    return await service.record_startup_lease(
        root_session_id="root-1",
        parent_session_id="parent-1",
        child_session_id="child-1",
        coordinator_run_id="run-1",
        work_unit_id="wu-1",
    )


async def test_startup_writes_complete_lineage_and_minimum_ttl_atomically() -> None:
    service, redis, repo, _ = _service()

    lease = await _record_startup(service)

    assert lease == CoordinatorChildLease(
        root_session_id="root-1",
        parent_session_id="parent-1",
        child_session_id="child-1",
        coordinator_run_id="run-1",
        work_unit_id="wu-1",
        last_seen_epoch=1_000.0,
        phase="starting",
    )
    key = "coordinator:liveness:child:child-1"
    assert redis.hashes[key] == {
        "root_session_id": "root-1",
        "parent_session_id": "parent-1",
        "child_session_id": "child-1",
        "coordinator_run_id": "run-1",
        "work_unit_id": "wu-1",
        "last_seen_epoch": "1000.0",
        "phase": "starting",
    }
    assert redis.expiries[key] >= 180
    assert redis.pipeline_transaction_values == []
    assert [command[0] for command in redis.executed_pipelines[0]] == [
        "eval",
    ]
    assert repo.lookups == ["child-1"]


async def test_startup_can_restore_trusted_stream_age_without_fresh_grace() -> None:
    service, redis, _, _ = _service()

    lease = await service.record_startup_lease(
        root_session_id="root-1",
        parent_session_id="parent-1",
        child_session_id="child-1",
        coordinator_run_id="run-1",
        work_unit_id="wu-1",
        last_seen_age_seconds=90.0,
    )

    assert lease.last_seen_epoch == 910.0
    assert lease.authority_age_seconds == 90.0
    assert redis.pttls["coordinator:liveness:child:child-1"] == 90_000
    assert service.is_stale(lease) is True


@pytest.mark.parametrize(
    ("age_seconds", "remaining_ttl_ms", "authority_age", "is_stale"),
    [
        (None, 180_000, 0.0, False),
        (80.0, 100_000, 80.0, False),
        (95.0, 85_000, 95.0, True),
        (250.0, 1, 179.999, True),
    ],
)
async def test_startup_age_uses_bounded_remaining_redis_ttl(
    age_seconds: float | None,
    remaining_ttl_ms: int,
    authority_age: float,
    is_stale: bool,
) -> None:
    service, redis, _, _ = _service()

    lease = await service.record_startup_lease(
        root_session_id="root-1",
        parent_session_id="parent-1",
        child_session_id="child-1",
        coordinator_run_id="run-1",
        work_unit_id="wu-1",
        last_seen_age_seconds=age_seconds,
    )

    assert lease.authority_age_seconds == authority_age
    assert redis.pttls["coordinator:liveness:child:child-1"] == remaining_ttl_ms
    assert service.is_stale(lease) is is_stale


async def test_startup_clamps_future_stream_timestamp_to_effective_now() -> None:
    service, _, _, _ = _service()

    lease = await service.record_startup_lease(
        root_session_id="root-1",
        parent_session_id="parent-1",
        child_session_id="child-1",
        coordinator_run_id="run-1",
        work_unit_id="wu-1",
        last_seen_epoch=9_999.0,
    )

    assert lease.last_seen_epoch == 1_000.0


async def test_startup_core_grace_does_not_renew_peripheral_resources() -> None:
    callbacks = [AsyncMock(), AsyncMock(), AsyncMock()]
    service, redis, _, _ = _service(
        touch_parent=callbacks[0],
        renew_child_sandbox=callbacks[1],
        renew_quota=callbacks[2],
    )

    await _record_startup(service)

    assert "coordinator:liveness:child:child-1" in redis.hashes
    assert all(callback.await_count == 0 for callback in callbacks)


@pytest.mark.parametrize(
    ("row_update", "argument_update"),
    [
        ({"root_session_id": "other-root"}, {}),
        ({"parent_session_id": "other-parent"}, {}),
        ({"coordinator_run_id": "other-run"}, {}),
        ({"work_unit_id": "other-wu"}, {}),
        ({"worker_type": "root"}, {}),
        ({"subagent_control_plane": "legacy"}, {}),
        ({"tool_filter_preset": "subagent_research"}, {}),
        ({}, {"root_session_id": "claimed-root"}),
    ],
)
async def test_startup_rejects_non_authoritative_row_or_expected_lineage(
    row_update: dict[str, Any], argument_update: dict[str, str]
) -> None:
    service, redis, _, _ = _service(row=_row(**row_update))
    kwargs = {
        "root_session_id": "root-1",
        "parent_session_id": "parent-1",
        "child_session_id": "child-1",
        "coordinator_run_id": "run-1",
        "work_unit_id": "wu-1",
    }
    kwargs.update(argument_update)

    with pytest.raises(CoordinatorLivenessLeaseRejected):
        await service.record_startup_lease(**kwargs)

    assert redis.hashes == {}


async def test_startup_rejects_missing_db_row() -> None:
    service, redis, repo, _ = _service()
    repo.row = None

    with pytest.raises(CoordinatorLivenessLeaseRejected):
        await _record_startup(service)

    assert redis.hashes == {}


@pytest.mark.parametrize(
    "status",
    [
        status
        for status in SessionStatus
        if status not in {SessionStatus.PENDING, SessionStatus.RUNNING}
    ],
)
async def test_startup_rejects_non_startable_status(status: SessionStatus) -> None:
    callback = AsyncMock()
    service, redis, _, _ = _service(
        row=_row(status=status), touch_parent=callback,
    )

    with pytest.raises(CoordinatorLivenessLeaseRejected):
        await _record_startup(service)

    assert redis.hashes == {}
    callback.assert_not_awaited()


async def test_startup_accepts_real_pending_session_but_heartbeat_does_not() -> None:
    service, redis, repo, _ = _service(row=_row(status=SessionStatus.PENDING))

    lease = await _record_startup(service)

    assert lease.phase == "starting"
    assert "coordinator:liveness:child:child-1" in redis.hashes
    assert await service.record_heartbeat(_heartbeat()) is False
    assert repo.lookups == ["child-1", "child-1"]


@pytest.mark.parametrize(
    "malformed",
    [
        {"unknown": "value"},
        {
            "root_session_id": "root-1",
            "parent_session_id": "parent-1",
            "child_session_id": "child-1",
            "coordinator_run_id": "run-1",
            "work_unit_id": "wu-1",
            "last_seen_epoch": "900.0",
        },
        {
            "root_session_id": "root-1",
            "parent_session_id": "parent-1",
            "child_session_id": "child-1",
            "coordinator_run_id": "run-1",
            "work_unit_id": "wu-1",
            "last_seen_epoch": "900.0",
            "phase": "starting",
            "extra": "value",
        },
        {
            "root_session_id": "root-1",
            "parent_session_id": "parent-1",
            "child_session_id": "child-1",
            "coordinator_run_id": "run-1",
            "work_unit_id": "wu-1",
            "last_seen_epoch": "900.0",
            "unknown": "value",
        },
    ],
)
async def test_startup_rejects_existing_malformed_hash_fail_closed(
    malformed: dict[str, str],
) -> None:
    service, redis, _, _ = _service()
    key = "coordinator:liveness:child:child-1"
    redis.hashes[key] = dict(malformed)
    redis.expiries[key] = 37

    with pytest.raises(CoordinatorLivenessLeaseRejected):
        await _record_startup(service)

    assert redis.hashes[key] == malformed
    assert redis.expiries[key] == 37


async def test_startup_refreshes_exact_same_live_lineage() -> None:
    service, redis, _, clock = _service()
    await _record_startup(service)
    clock.monotonic = 65.0

    lease = await _record_startup(service)

    assert lease.last_seen_epoch == 1_015.0
    assert redis.hashes["coordinator:liveness:child:child-1"]["phase"] == "starting"


async def test_startup_rejects_model_construct_string_running_status() -> None:
    data = _row().model_dump(mode="python")
    data["status"] = "running"
    bypassed = Session.model_construct(**data)
    service, redis, _, _ = _service(row=bypassed)

    with pytest.raises(CoordinatorLivenessLeaseRejected):
        await _record_startup(service)

    assert redis.hashes == {}


async def test_startup_rejects_empty_lineage_even_if_db_row_matches() -> None:
    service, redis, _, _ = _service(row=_row(coordinator_run_id=""))

    with pytest.raises(CoordinatorLivenessLeaseRejected):
        await service.record_startup_lease(
            root_session_id="root-1",
            parent_session_id="parent-1",
            child_session_id="child-1",
            coordinator_run_id="",
            work_unit_id="wu-1",
        )

    assert redis.hashes == {}


async def test_heartbeat_refreshes_lease_from_db_authority_not_correlation() -> None:
    touch = AsyncMock()
    sandbox = AsyncMock()
    quota = AsyncMock()
    service, redis, _, clock = _service(
        touch_parent=touch,
        renew_child_sandbox=sandbox,
        renew_quota=quota,
    )
    await _record_startup(service)
    clock.wall = 1_015.0
    clock.monotonic = 65.0

    accepted = await service.record_heartbeat(_heartbeat())

    assert accepted is True
    lease = await service.get_lease("child-1")
    assert lease is not None
    assert lease.last_seen_epoch == 1_015.0
    assert lease.phase == "in_tool"
    assert lease.coordinator_run_id == "run-1"
    assert redis.expiries["coordinator:liveness:child:child-1"] == 180
    for callback in (touch, sandbox, quota):
        callback.assert_awaited()
        assert callback.await_args.args == (lease,)


async def test_get_lease_accepts_bytes_hash_fields_and_is_stream_independent() -> None:
    redis = _Redis(bytes_reads=True)
    service, _, _, _ = _service(redis=redis)
    expected = await _record_startup(service)
    redis.streams["actus:child:root-1:mailbox"] = [{"envelope": "heartbeat"}]
    redis.trim_stream("actus:child:root-1:mailbox")

    assert await service.get_lease("child-1") == expected


@pytest.mark.parametrize(
    "envelope",
    [
        _heartbeat(producer_role=ProducerRole.PARENT_AGENT),
        _heartbeat(
            payload={
                "kind": "tool_started",
                "visibility": "hidden",
                "phase": "in_tool",
            }
        ),
        MailboxEnvelope(
            envelope_id="not-progress",
            type=MailboxEnvelopeType.RESULT_READY,
            parent_session_id="parent-1",
            child_session_id="child-1",
            correlation_id="run-1",
            emitted_at=datetime.now(timezone.utc),
            producer_role=ProducerRole.CHILD_AGENT,
            payload={"summary": "done", "outcome": "success"},
        ),
        _heartbeat(parent_session_id="other-parent"),
        _heartbeat(child_session_id="other-child"),
    ],
)
async def test_heartbeat_rejects_wrong_wire_identity(envelope: MailboxEnvelope) -> None:
    service, redis, _, clock = _service()
    await _record_startup(service)
    key = "coordinator:liveness:child:child-1"
    original_hash = dict(redis.hashes[key])
    clock.wall = 1_030.0

    assert await service.record_heartbeat(envelope) is False

    assert redis.hashes[key] == original_hash
    assert not any(
        "coordinator-refresh-cas-v1" in command[1][0]
        for batch in redis.executed_pipelines[1:]
        for command in batch
    )


@pytest.mark.parametrize(
    "correlation_id",
    ["wrong-prefix:child-1", "hb:other-child", "run-1", ""],
)
async def test_heartbeat_rejects_noncanonical_correlation_without_side_effects(
    correlation_id: str,
) -> None:
    callback = AsyncMock()
    service, redis, _, clock = _service(touch_parent=callback)
    await _record_startup(service)
    callback.reset_mock()
    key = "coordinator:liveness:child:child-1"
    original_hash = dict(redis.hashes[key])
    clock.wall = 1_030.0

    assert await service.record_heartbeat(
        _heartbeat(correlation_id=correlation_id)
    ) is False

    assert redis.hashes[key] == original_hash
    assert not any(
        "coordinator-refresh-cas-v1" in command[1][0]
        for batch in redis.executed_pipelines[1:]
        for command in batch
    )
    callback.assert_not_awaited()


async def test_heartbeat_rejects_malformed_or_missing_schema_fields() -> None:
    service, redis, _, _ = _service()
    await _record_startup(service)
    raw = _heartbeat().model_dump(mode="python")
    raw.pop("producer_role")

    assert await service.record_heartbeat(raw) is False
    assert await service.record_heartbeat({"child_session_id": "child-1"}) is False
    assert len(redis.executed_pipelines) == 1


async def test_heartbeat_revalidates_model_construct_instance_fail_closed() -> None:
    service, redis, _, _ = _service()
    await _record_startup(service)
    bypassed = MailboxEnvelope.model_construct(
        envelope_id="bypassed",
        type=MailboxEnvelopeType.PROGRESS_UPDATE,
        parent_session_id="parent-1",
        child_session_id="child-1",
        correlation_id="hb:child-1",
        emitted_at=datetime.now(timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload={"kind": "heartbeat", "unexpected": "field"},
    )

    assert await service.record_heartbeat(bypassed) is False
    assert len(redis.executed_pipelines) == 1


@pytest.mark.parametrize(
    "row_update",
    [
        {"root_session_id": "other-root"},
        {"parent_session_id": "other-parent"},
        {"coordinator_run_id": "other-run"},
        {"work_unit_id": "other-wu"},
        {"worker_type": "root"},
        {"subagent_control_plane": "legacy"},
        {"tool_filter_preset": "subagent_research"},
    ],
)
async def test_heartbeat_rejects_db_drift_without_refreshing_terminal_row(
    row_update: dict[str, Any],
) -> None:
    callback = AsyncMock()
    service, redis, repo, clock = _service(touch_parent=callback)
    await _record_startup(service)
    callback.reset_mock()
    key = "coordinator:liveness:child:child-1"
    original_hash = dict(redis.hashes[key])
    original_expiry = redis.expiries[key]
    repo.row = _row(**row_update)
    clock.wall = 1_040.0

    assert await service.record_heartbeat(_heartbeat()) is False

    assert redis.hashes[key] == original_hash
    assert redis.expiries[key] == original_expiry
    callback.assert_not_awaited()


@pytest.mark.parametrize(
    "status",
    [status for status in SessionStatus if status is not SessionStatus.RUNNING],
)
async def test_heartbeat_rejects_every_non_running_db_status(
    status: SessionStatus,
) -> None:
    callback = AsyncMock()
    service, redis, repo, clock = _service(touch_parent=callback)
    await _record_startup(service)
    callback.reset_mock()
    key = "coordinator:liveness:child:child-1"
    original_hash = dict(redis.hashes[key])
    repo.row = _row(status=status)
    clock.wall = 1_040.0

    assert await service.record_heartbeat(_heartbeat()) is False

    assert redis.hashes[key] == original_hash
    assert not any(
        "coordinator-refresh-cas-v1" in command[1][0]
        for batch in redis.executed_pipelines[1:]
        for command in batch
    )
    callback.assert_not_awaited()


async def test_heartbeat_rejects_model_construct_string_running_status() -> None:
    callback = AsyncMock()
    service, redis, repo, _ = _service(touch_parent=callback)
    await _record_startup(service)
    callback.reset_mock()
    data = _row().model_dump(mode="python")
    data["status"] = "running"
    repo.row = Session.model_construct(**data)

    assert await service.record_heartbeat(_heartbeat()) is False
    assert not any(
        "coordinator-refresh-cas-v1" in command[1][0]
        for batch in redis.executed_pipelines[1:]
        for command in batch
    )
    callback.assert_not_awaited()


async def test_heartbeat_requires_existing_startup_lease() -> None:
    service, redis, _, _ = _service()

    assert await service.record_heartbeat(_heartbeat()) is False
    assert redis.hashes == {}


async def test_ordinary_mailbox_research_heartbeat_is_not_coordinator_applicable() -> None:
    row = _row(
        tool_filter_preset="subagent_research",
        coordinator_run_id=None,
        work_unit_id=None,
    )
    service, redis, repo, _ = _service(row=row)

    assert await service.record_heartbeat(_heartbeat()) is None
    assert repo.lookups == ["child-1"]
    assert redis.hashes == {}


async def test_ordinary_research_heartbeat_renews_its_sandbox_owners() -> None:
    row = _row(
        tool_filter_preset="subagent_research",
        coordinator_run_id=None,
        work_unit_id=None,
    )
    renew = AsyncMock()
    service, _, _, _ = _service(
        row=row,
        renew_ordinary_sandboxes=renew,
    )

    assert await service.record_heartbeat(_heartbeat()) is None
    renew.assert_awaited_once_with(row)


async def test_ordinary_sandbox_renew_failure_keeps_heartbeat_alive() -> None:
    row = _row(
        tool_filter_preset="subagent_research",
        coordinator_run_id=None,
        work_unit_id=None,
    )
    renew = AsyncMock(side_effect=RuntimeError("sandbox API unavailable"))
    service, _, _, _ = _service(
        row=row,
        renew_ordinary_sandboxes=renew,
    )

    assert await service.record_heartbeat(_heartbeat()) is None


async def test_ordinary_mailbox_heartbeat_with_wrong_parent_is_rejected() -> None:
    row = _row(
        tool_filter_preset="subagent_research",
        coordinator_run_id=None,
        work_unit_id=None,
    )
    service, _, _, _ = _service(row=row)

    assert await service.record_heartbeat(
        _heartbeat(parent_session_id="forged-parent"),
    ) is False


async def test_heartbeat_rejects_missing_authoritative_db_row() -> None:
    service, redis, repo, _ = _service()
    await _record_startup(service)
    repo.row = None

    assert await service.record_heartbeat(_heartbeat()) is False
    assert not any(
        "coordinator-refresh-cas-v1" in command[1][0]
        for batch in redis.executed_pipelines[1:]
        for command in batch
    )


async def test_peripheral_renew_failures_do_not_rollback_core_heartbeat() -> None:
    callbacks = [
        AsyncMock(side_effect=RuntimeError("parent failed")),
        AsyncMock(side_effect=RuntimeError("sandbox failed")),
        AsyncMock(side_effect=RuntimeError("quota failed")),
    ]
    service, _, _, clock = _service(
        touch_parent=callbacks[0],
        renew_child_sandbox=callbacks[1],
        renew_quota=callbacks[2],
    )
    await _record_startup(service)
    clock.wall = 1_020.0
    clock.monotonic = 70.0

    assert await service.record_heartbeat(_heartbeat()) is True
    lease = await service.get_lease("child-1")
    assert lease is not None
    assert lease.last_seen_epoch == 1_020.0
    assert all(callback.await_count == 1 for callback in callbacks)


async def test_core_redis_failure_propagates_and_skips_callbacks() -> None:
    callback = AsyncMock()
    redis = _Redis()
    service, _, _, _ = _service(redis=redis, touch_parent=callback)
    redis.fail_execute = True

    with pytest.raises(RuntimeError, match="redis unavailable"):
        await _record_startup(service)

    callback.assert_not_awaited()


async def test_heartbeat_core_redis_failure_does_not_claim_refresh() -> None:
    callback = AsyncMock()
    redis = _Redis()
    service, _, _, clock = _service(redis=redis, touch_parent=callback)
    await _record_startup(service)
    callback.reset_mock()
    original = dict(redis.hashes["coordinator:liveness:child:child-1"])
    redis.fail_execute = True
    clock.wall = 1_020.0

    with pytest.raises(RuntimeError, match="redis unavailable"):
        await service.record_heartbeat(_heartbeat())

    assert redis.hashes["coordinator:liveness:child:child-1"] == original
    callback.assert_not_awaited()


async def test_heartbeat_pipeline_cancellation_preserves_startup_core_state() -> None:
    callback = AsyncMock()
    redis = _Redis()
    service, _, _, clock = _service(redis=redis, touch_parent=callback)
    await _record_startup(service)
    key = "coordinator:liveness:child:child-1"
    startup_hash = dict(redis.hashes[key])
    startup_ttl = redis.expiries[key]
    startup_observation = service._observed_monotonic["child-1"]
    startup_pipeline_count = len(redis.executed_pipelines)
    clock.monotonic = 51.0
    redis.execute_error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await service.record_heartbeat(_heartbeat())

    callback.assert_not_awaited()
    assert redis.hashes[key] == startup_hash
    assert redis.expiries[key] == startup_ttl
    assert service._observed_monotonic["child-1"] == startup_observation
    assert len(redis.executed_pipelines) == startup_pipeline_count


async def test_terminal_tombstone_wins_after_heartbeat_db_check() -> None:
    callback = AsyncMock()
    service, _, repo, clock = _service(touch_parent=callback)
    await _record_startup(service)
    callback.reset_mock()
    repo.lookup_started = asyncio.Event()
    repo.release_lookup = asyncio.Event()
    clock.monotonic = 51.0

    heartbeat_task = asyncio.create_task(service.record_heartbeat(_heartbeat()))
    await repo.lookup_started.wait()
    await service.mark_terminal("child-1")
    repo.release_lookup.set()

    assert await heartbeat_task is False
    assert await service.get_lease("child-1") is None
    callback.assert_not_awaited()


async def test_malformed_hash_wins_after_heartbeat_db_check_fail_closed() -> None:
    callback = AsyncMock()
    service, redis, repo, _ = _service(touch_parent=callback)
    await _record_startup(service)
    callback.reset_mock()
    repo.lookup_started = asyncio.Event()
    repo.release_lookup = asyncio.Event()

    heartbeat_task = asyncio.create_task(service.record_heartbeat(_heartbeat()))
    await repo.lookup_started.wait()
    key = "coordinator:liveness:child:child-1"
    malformed = {
        "root_session_id": "root-1",
        "parent_session_id": "parent-1",
        "child_session_id": "child-1",
        "coordinator_run_id": "run-1",
        "work_unit_id": "wu-1",
        "last_seen_epoch": "1000.0",
        "unknown": "field",
    }
    redis.hashes[key] = dict(malformed)
    repo.release_lookup.set()

    assert await heartbeat_task is False
    assert redis.hashes[key] == malformed
    callback.assert_not_awaited()


async def test_terminal_tombstone_wins_after_startup_db_check() -> None:
    service, _, repo, _ = _service()
    repo.lookup_started = asyncio.Event()
    repo.release_lookup = asyncio.Event()

    startup_task = asyncio.create_task(_record_startup(service))
    await repo.lookup_started.wait()
    await service.mark_terminal("child-1")
    repo.release_lookup.set()

    with pytest.raises(CoordinatorLivenessLeaseRejected):
        await startup_task
    assert await service.get_lease("child-1") is None


async def test_refresh_first_then_terminal_still_finishes_missing() -> None:
    service, _, _, clock = _service()
    await _record_startup(service)
    clock.monotonic = 51.0

    assert await service.record_heartbeat(_heartbeat()) is True
    await service.mark_terminal("child-1")

    assert await service.get_lease("child-1") is None


async def test_mark_terminal_is_idempotent_and_blocks_new_startup() -> None:
    service, redis, _, _ = _service()

    await service.mark_terminal("child-1")
    await service.mark_terminal("child-1")

    assert redis.hashes["coordinator:liveness:child:child-1"] == {
        "state": "terminal",
        "child_session_id": "child-1",
    }
    assert redis.expiries["coordinator:liveness:child:child-1"] >= 180
    with pytest.raises(CoordinatorLivenessLeaseRejected):
        await _record_startup(service)


async def test_compare_delete_removes_only_exact_live_lineage() -> None:
    service, redis, _, _ = _service()
    expected = await _record_startup(service)

    assert await service.clear_if_matches(expected) is True
    assert redis.hashes == {}
    assert "child-1" not in service._observed_monotonic


async def test_compare_delete_cannot_remove_terminal_tombstone() -> None:
    service, redis, _, _ = _service()
    expected = await _record_startup(service)
    await service.mark_terminal("child-1")

    assert await service.clear_if_matches(expected) is False
    assert redis.hashes["coordinator:liveness:child:child-1"] == {
        "state": "terminal",
        "child_session_id": "child-1",
    }


async def test_compare_delete_preserves_other_attempt_and_local_observation() -> None:
    service, redis, _, _ = _service()
    current = await _record_startup(service)
    stale_attempt = replace(current, coordinator_run_id="older-run")

    assert await service.clear_if_matches(stale_attempt) is False
    assert redis.hashes["coordinator:liveness:child:child-1"][
        "coordinator_run_id"
    ] == "run-1"
    assert "child-1" in service._observed_monotonic


async def test_compare_delete_rejects_incomplete_expected_lineage() -> None:
    service, redis, _, _ = _service()
    current = await _record_startup(service)

    with pytest.raises(ValueError, match="expected lineage"):
        await service.clear_if_matches(replace(current, work_unit_id=""))

    assert "coordinator:liveness:child:child-1" in redis.hashes


@pytest.mark.parametrize(
    "malformed",
    [
        {"child_session_id": "child-1"},
        {
            "root_session_id": "root-1",
            "parent_session_id": "parent-1",
            "child_session_id": "child-1",
            "coordinator_run_id": "run-1",
            "work_unit_id": "wu-1",
            "last_seen_epoch": "1000.0",
        },
        {
            "root_session_id": "root-1",
            "parent_session_id": "parent-1",
            "child_session_id": "child-1",
            "coordinator_run_id": "run-1",
            "work_unit_id": "wu-1",
            "last_seen_epoch": "1000.0",
            "phase": "starting",
            "extra": "field",
        },
        {
            "root_session_id": "root-1",
            "parent_session_id": "parent-1",
            "child_session_id": "child-1",
            "coordinator_run_id": "run-1",
            "work_unit_id": "wu-1",
            "last_seen_epoch": "1000.0",
            "unknown": "field",
        },
    ],
)
async def test_compare_delete_preserves_malformed_hash(
    malformed: dict[str, str],
) -> None:
    service, redis, _, _ = _service()
    expected = await _record_startup(service)
    key = "coordinator:liveness:child:child-1"
    redis.hashes[key] = malformed

    assert await service.clear_if_matches(expected) is False
    assert redis.hashes[key] == malformed


async def test_compare_delete_missing_target_forgets_local_observation() -> None:
    service, redis, _, _ = _service()
    expected = await _record_startup(service)
    await redis.delete("coordinator:liveness:child:child-1")
    assert "child-1" in service._observed_monotonic

    assert await service.clear_if_matches(expected) is False
    assert "child-1" not in service._observed_monotonic


async def test_callback_cancellation_propagates_and_stops_later_callbacks() -> None:
    cancelled = AsyncMock(side_effect=asyncio.CancelledError())
    should_not_run = AsyncMock()
    service, redis, _, clock = _service(
        touch_parent=cancelled,
        renew_child_sandbox=should_not_run,
    )
    await _record_startup(service)
    clock.monotonic = 51.0

    with pytest.raises(asyncio.CancelledError):
        await service.record_heartbeat(_heartbeat())

    assert (
        redis.hashes["coordinator:liveness:child:child-1"]["last_seen_epoch"]
        == "1001.0"
    )
    cancelled.assert_awaited_once()
    should_not_run.assert_not_awaited()


async def test_stale_boundary_missing_semantics_and_clear_are_explicit() -> None:
    service, redis, _, clock = _service()
    lease = replace(
        await _record_startup(service),
        authority_age_seconds=None,
    )

    clock.wall = 1_089.999
    clock.monotonic = 139.999
    assert service.is_stale(lease) is False
    clock.wall = 1_090.0
    clock.monotonic = 140.0
    assert service.is_stale(lease) is True
    assert service.is_stale(None) is True

    await service.clear("child-1")
    await service.clear("child-1")
    assert await service.get_lease("child-1") is None
    assert "coordinator:liveness:child:child-1" not in redis.expiries


@pytest.mark.parametrize(
    ("pttl_ms", "expected_age", "expected_stale"),
    [
        (100_000, 80.0, False),
        (85_000, 95.0, True),
    ],
)
async def test_redis_pttl_is_cross_pod_age_authority_despite_app_clock_offset(
    pttl_ms: int,
    expected_age: float,
    expected_stale: bool,
) -> None:
    redis = _Redis()
    origin, _, _, _ = _service(
        redis=redis,
        clock=_Clock(wall=1_000.0, monotonic=50.0),
    )
    await _record_startup(origin)
    redis.pttls["coordinator:liveness:child:child-1"] = pttl_ms

    for wall in (-1_000_000.0, 1_000_000.0):
        observer, _, _, _ = _service(
            redis=redis,
            clock=_Clock(wall=wall, monotonic=7.0),
        )
        lease = await observer.get_lease("child-1")

        assert lease is not None
        assert lease.authority_age_seconds == expected_age
        assert observer.is_stale(lease) is expected_stale


@pytest.mark.parametrize("invalid_pttl", [None, -1, 0, 180_001, "bad"])
async def test_invalid_redis_pttl_fails_closed(invalid_pttl: Any) -> None:
    service, redis, _, _ = _service()
    await _record_startup(service)
    redis.pttls["coordinator:liveness:child:child-1"] = invalid_pttl

    assert await service.get_lease("child-1") is None
    assert "child-1" not in service._observed_monotonic


async def test_await_stale_polls_until_exact_boundary() -> None:
    clock = _Clock()
    redis = _Redis()
    key = "coordinator:liveness:child:child-1"

    async def advance(_: float) -> None:
        clock.wall += 30.0
        clock.monotonic += 30.0
        redis.pttls[key] = int(redis.pttls[key]) - 30_000

    sleep = AsyncMock(side_effect=advance)
    service, _, _, _ = _service(redis=redis, clock=clock, sleep=sleep)
    expected = await _record_startup(service)

    stale = await service.await_stale("child-1")

    assert stale == expected
    assert sleep.await_count == 3


async def test_await_stale_rereads_shared_pttl_each_poll() -> None:
    redis = _Redis()
    key = "coordinator:liveness:child:child-1"
    calls = 0

    async def advance(_: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            redis.pttls[key] = 90_000
            return
        raise AssertionError("await_stale did not re-read Redis PTTL")

    service, _, _, _ = _service(
        redis=redis,
        sleep=AsyncMock(side_effect=advance),
    )
    await _record_startup(service)

    stale = await service.await_stale("child-1")

    assert stale is not None
    assert stale.authority_age_seconds == 90.0
    assert calls == 1


async def test_clock_rollback_uses_monotonic_age_instead_of_extending_forever() -> None:
    clock = _Clock(wall=1_000.0, monotonic=10.0)
    service, _, _, _ = _service(clock=clock)
    lease = replace(
        await _record_startup(service),
        authority_age_seconds=None,
    )
    clock.wall = 900.0
    clock.monotonic = 99.999
    assert service.is_stale(lease) is False
    clock.monotonic = 100.0
    assert service.is_stale(lease) is True


async def test_wall_clock_forward_jump_does_not_create_false_staleness() -> None:
    clock = _Clock(wall=1_000.0, monotonic=10.0)
    service, _, _, _ = _service(clock=clock)
    lease = replace(
        await _record_startup(service),
        authority_age_seconds=None,
    )
    clock.wall = 4_600.0
    clock.monotonic = 11.0
    assert service.is_stale(lease) is False
    clock.monotonic = 99.999
    assert service.is_stale(lease) is False
    clock.monotonic = 100.0
    assert service.is_stale(lease) is True


async def test_restart_uses_durable_pttl_instead_of_future_app_epoch() -> None:
    redis = _Redis()
    original_clock = _Clock(wall=2_000.0, monotonic=10.0)
    original, _, _, _ = _service(redis=redis, clock=original_clock)
    expected = await _record_startup(original)
    key = "coordinator:liveness:child:child-1"
    redis.pttls[key] = 90_001

    restarted_clock = _Clock(wall=1_000.0, monotonic=50.0)
    restarted, _, _, _ = _service(redis=redis, clock=restarted_clock)
    restored = await restarted.get_lease("child-1")
    assert restored == expected
    assert restarted.is_stale(restored) is False
    redis.pttls[key] = 90_000
    assert restarted.is_stale(await restarted.get_lease("child-1")) is True


@pytest.mark.parametrize("bad_value", [None, 123, object(), b"\xff"])
async def test_corrupt_redis_field_fails_closed_and_forgets_local_observation(
    bad_value: Any,
) -> None:
    service, redis, _, _ = _service()
    await _record_startup(service)
    lease_hash = redis.hashes["coordinator:liveness:child:child-1"]
    lease_hash["phase"] = bad_value  # type: ignore[assignment]

    assert await service.get_lease("child-1") is None
    assert service.is_stale(None) is True
    assert "child-1" not in service._observed_monotonic


@pytest.mark.parametrize("shape", ["missing", "partial", "extra"])
async def test_missing_or_wrong_hash_shape_forgets_local_observation(
    shape: str,
) -> None:
    service, redis, _, _ = _service()
    await _record_startup(service)
    key = "coordinator:liveness:child:child-1"
    if shape == "missing":
        redis.hashes.pop(key)
    elif shape == "partial":
        redis.hashes[key].pop("work_unit_id")
    else:
        redis.hashes[key]["unexpected"] = "field"

    assert await service.get_lease("child-1") is None
    assert "child-1" not in service._observed_monotonic


async def test_redis_read_network_error_still_propagates() -> None:
    service, redis, _, _ = _service()
    redis.eval = AsyncMock(  # type: ignore[method-assign]
        side_effect=ConnectionError("redis down")
    )

    with pytest.raises(ConnectionError, match="redis down"):
        await service.get_lease("child-1")


async def test_redis_decode_response_error_fails_closed_and_forgets_observation() -> None:
    service, redis, _, _ = _service()
    await _record_startup(service)
    assert "child-1" in service._observed_monotonic
    redis.eval = AsyncMock(  # type: ignore[method-assign]
        side_effect=UnicodeDecodeError(
            "utf-8", b"\xff", 0, 1, "invalid start byte"
        )
    )

    assert await service.get_lease("child-1") is None
    assert "child-1" not in service._observed_monotonic


@pytest.mark.parametrize(
    "config",
    [
        {"stale_after_seconds": 0},
        {"stale_after_seconds": -1},
        {"stale_after_seconds": float("nan")},
        {"stale_after_seconds": float("inf")},
        {"poll_interval_seconds": 0},
        {"poll_interval_seconds": float("nan")},
        {"lease_ttl_seconds": 179},
        {"lease_ttl_seconds": 180.5},
        {"lease_ttl_seconds": True},
    ],
)
async def test_constructor_rejects_nonfinite_or_unsafe_timing(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _service(**config)
