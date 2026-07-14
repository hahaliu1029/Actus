"""Durable liveness leases for coordinator-owned child sessions.

The mailbox stream is transport history, not liveness authority: trimming or
consumer-group churn must not erase the latest trusted child heartbeat.  This
service therefore keeps one small Redis hash per child and refreshes its TTL
only after the envelope and the authoritative session row agree.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from pydantic import ValidationError

from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ProgressKind,
    SUBAGENT_PROGRESS_STALE_AFTER_SECONDS,
)
from app.domain.models.session import Session, SessionStatus
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET
from app.domain.repositories.session_repository import SessionRepository


logger = logging.getLogger(__name__)

_STARTUP_SESSION_STATUSES = frozenset({
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
})
_HEARTBEAT_SESSION_STATUSES = frozenset({SessionStatus.RUNNING})

COORDINATOR_CHILD_LEASE_KEY_TEMPLATE = "coordinator:liveness:child:{child_id}"
COORDINATOR_CHILD_LEASE_TTL_SECONDS = 180
COORDINATOR_LIVENESS_POLL_INTERVAL_SECONDS = 5.0

_STARTUP_LEASE_CAS_SCRIPT = """
-- coordinator-startup-cas-v1
local key = KEYS[1]
if redis.call('HGET', key, 'state') == 'terminal' then
  return 0
end
if redis.call('EXISTS', key) == 1 then
  if redis.call('HLEN', key) ~= 7 then
    return 0
  end
  if redis.call('HGET', key, 'last_seen_epoch') == false
    or redis.call('HGET', key, 'phase') == false then
    return 0
  end
  local fields = {
    'root_session_id', 'parent_session_id', 'child_session_id',
    'coordinator_run_id', 'work_unit_id'
  }
  for index, field in ipairs(fields) do
    if redis.call('HGET', key, field) ~= ARGV[index] then
      return 0
    end
  end
end
redis.call(
  'HSET', key,
  'root_session_id', ARGV[1],
  'parent_session_id', ARGV[2],
  'child_session_id', ARGV[3],
  'coordinator_run_id', ARGV[4],
  'work_unit_id', ARGV[5],
  'last_seen_epoch', ARGV[6],
  'phase', ARGV[7]
)
redis.call('PEXPIRE', key, tonumber(ARGV[8]))
return 1
"""

_REFRESH_LEASE_CAS_SCRIPT = """
-- coordinator-refresh-cas-v1
local key = KEYS[1]
if redis.call('EXISTS', key) ~= 1 or redis.call('HLEN', key) ~= 7 then
  return 0
end
if redis.call('HGET', key, 'state') == 'terminal' then
  return 0
end
if redis.call('HGET', key, 'last_seen_epoch') == false
  or redis.call('HGET', key, 'phase') == false then
  return 0
end
local fields = {
  'root_session_id', 'parent_session_id', 'child_session_id',
  'coordinator_run_id', 'work_unit_id'
}
for index, field in ipairs(fields) do
  if redis.call('HGET', key, field) ~= ARGV[index] then
    return 0
  end
end
redis.call('HSET', key, 'last_seen_epoch', ARGV[6], 'phase', ARGV[7])
redis.call('PEXPIRE', key, tonumber(ARGV[8]))
return 1
"""

_TERMINAL_TOMBSTONE_SCRIPT = """
-- coordinator-terminal-tombstone-v1
local key = KEYS[1]
redis.call('DEL', key)
redis.call(
  'HSET', key,
  'state', 'terminal',
  'child_session_id', ARGV[1]
)
redis.call('PEXPIRE', key, tonumber(ARGV[2]))
return 1
"""

_READ_LIVE_LEASE_WITH_PTTL_SCRIPT = """
-- coordinator-live-lease-read-pttl-v1
local key = KEYS[1]
if redis.call('EXISTS', key) ~= 1 or redis.call('HLEN', key) ~= 7 then
  return {}
end
if redis.call('HGET', key, 'state') == 'terminal' then
  return {}
end
local fields = {
  'root_session_id', 'parent_session_id', 'child_session_id',
  'coordinator_run_id', 'work_unit_id', 'last_seen_epoch', 'phase'
}
local values = {}
for index, field in ipairs(fields) do
  local value = redis.call('HGET', key, field)
  if value == false then
    return {}
  end
  values[index] = value
end
local pttl = redis.call('PTTL', key)
local maximum_ttl = tonumber(ARGV[1])
if pttl == false or pttl <= 0 or pttl > maximum_ttl then
  return {}
end
values[8] = pttl
return values
"""

_LIVE_LEASE_COMPARE_DELETE_SCRIPT = """
-- coordinator-live-lease-compare-delete-v1
local key = KEYS[1]
if redis.call('EXISTS', key) ~= 1 then
  return -1
end
if redis.call('HGET', key, 'state') == 'terminal' then
  return 0
end
if redis.call('HLEN', key) ~= 7 then
  return 0
end
if redis.call('HGET', key, 'last_seen_epoch') == false
  or redis.call('HGET', key, 'phase') == false then
  return 0
end
local fields = {
  'root_session_id', 'parent_session_id', 'child_session_id',
  'coordinator_run_id', 'work_unit_id'
}
for index, field in ipairs(fields) do
  if redis.call('HGET', key, field) ~= ARGV[index] then
    return 0
  end
end
redis.call('DEL', key)
return 1
"""

@dataclass(frozen=True)
class CoordinatorChildLease:
    root_session_id: str
    parent_session_id: str
    child_session_id: str
    coordinator_run_id: str
    work_unit_id: str
    last_seen_epoch: float
    phase: str
    # Derived atomically from Redis PTTL. It is intentionally not persisted
    # in the seven-field hash and excluded from identity/equality comparisons.
    # ``None`` is reserved for explicitly constructed local fallback objects.
    authority_age_seconds: float | None = field(default=None, compare=False)


class CoordinatorLivenessLeaseRejected(ValueError):
    """The supplied startup lineage is not backed by the session row."""


LeaseRenewCallback = Callable[[CoordinatorChildLease], Awaitable[None]]


class CoordinatorLivenessLeaseService:
    """Validate and persist coordinator child startup/heartbeat liveness."""

    def __init__(
        self,
        *,
        redis: Any,
        session_repository: SessionRepository,
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        stale_after_seconds: float = SUBAGENT_PROGRESS_STALE_AFTER_SECONDS,
        poll_interval_seconds: float = COORDINATOR_LIVENESS_POLL_INTERVAL_SECONDS,
        lease_ttl_seconds: int = COORDINATOR_CHILD_LEASE_TTL_SECONDS,
        touch_parent: LeaseRenewCallback | None = None,
        renew_child_sandbox: LeaseRenewCallback | None = None,
        renew_quota: LeaseRenewCallback | None = None,
    ) -> None:
        self._validate_positive_finite(
            stale_after_seconds, "stale_after_seconds"
        )
        self._validate_positive_finite(
            poll_interval_seconds, "poll_interval_seconds"
        )
        if (
            isinstance(lease_ttl_seconds, bool)
            or not isinstance(lease_ttl_seconds, int)
            or lease_ttl_seconds < 2 * stale_after_seconds
        ):
            raise ValueError(
                "lease_ttl_seconds must be an integer at least twice "
                "stale_after_seconds"
            )

        self._redis = redis
        self._sessions = session_repository
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
        self._stale_after = float(stale_after_seconds)
        self._poll_interval = float(poll_interval_seconds)
        self._lease_ttl_ms = lease_ttl_seconds * 1_000
        self._callbacks = tuple(
            callback
            for callback in (touch_parent, renew_child_sandbox, renew_quota)
            if callback is not None
        )

        # A monotonic projection prevents a backwards wall-clock adjustment
        # from making a lease immortal.  Per-child observations reset the
        # fallback age whenever this process successfully writes fresh liveness.
        self._anchor_wall = float(clock())
        self._anchor_monotonic = float(monotonic_clock())
        if not math.isfinite(self._anchor_wall):
            raise ValueError("clock must return a finite value")
        if not math.isfinite(self._anchor_monotonic):
            raise ValueError("monotonic_clock must return a finite value")
        self._observed_monotonic: dict[str, float] = {}

    @staticmethod
    def _validate_positive_finite(value: float, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")

    @staticmethod
    def _key(child_session_id: str) -> str:
        return COORDINATOR_CHILD_LEASE_KEY_TEMPLATE.format(
            child_id=child_session_id
        )

    def _effective_now(self) -> float:
        monotonic_now = float(self._monotonic_clock())
        if not math.isfinite(monotonic_now):
            raise ValueError("monotonic_clock must remain finite")
        # Wall time is sampled exactly once at construction.  Afterwards age
        # advances only with monotonic time, so NTP/manual wall-clock jumps in
        # either direction cannot make a healthy child instantly stale or keep
        # it alive indefinitely.
        return self._anchor_wall + max(
            0.0, monotonic_now - self._anchor_monotonic
        )

    async def record_startup_lease(
        self,
        *,
        root_session_id: str,
        parent_session_id: str,
        child_session_id: str,
        coordinator_run_id: str,
        work_unit_id: str,
        last_seen_epoch: float | None = None,
        last_seen_age_seconds: float | None = None,
    ) -> CoordinatorChildLease:
        """Record ``starting`` only for a DB-authorized coordinator child."""
        now = self._effective_now()
        if last_seen_epoch is not None and last_seen_age_seconds is not None:
            raise ValueError(
                "last_seen_epoch and last_seen_age_seconds are mutually exclusive"
            )
        if last_seen_age_seconds is None:
            authority_age = 0.0
        else:
            if (
                isinstance(last_seen_age_seconds, bool)
                or not isinstance(last_seen_age_seconds, (int, float))
                or not math.isfinite(last_seen_age_seconds)
                or last_seen_age_seconds < 0
            ):
                raise ValueError(
                    "last_seen_age_seconds must be finite and non-negative"
                )
            authority_age = float(last_seen_age_seconds)

        if last_seen_epoch is not None:
            if (
                isinstance(last_seen_epoch, bool)
                or not isinstance(last_seen_epoch, (int, float))
                or not math.isfinite(last_seen_epoch)
            ):
                raise ValueError("last_seen_epoch must be finite when provided")
            # Redis stream IDs are server timestamps. Clamp a future ID so a
            # malformed/restored stream entry cannot make startup grace
            # immortal after a host clock rollback.
            effective_last_seen = min(float(last_seen_epoch), now)
        else:
            # Audit/debug only. Staleness never compares this application
            # epoch with timestamps produced by another pod or by Redis.
            effective_last_seen = now - authority_age

        elapsed_ms = math.ceil(authority_age * 1_000)
        remaining_ttl_ms = max(1, self._lease_ttl_ms - elapsed_ms)
        persisted_authority_age = (
            self._lease_ttl_ms - remaining_ttl_ms
        ) / 1_000.0
        expected = CoordinatorChildLease(
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            coordinator_run_id=coordinator_run_id,
            work_unit_id=work_unit_id,
            last_seen_epoch=effective_last_seen,
            phase="starting",
            authority_age_seconds=persisted_authority_age,
        )
        row = await self._sessions.get_by_id(child_session_id)
        if not self._row_matches_lease(
            row,
            expected,
            allowed_statuses=_STARTUP_SESSION_STATUSES,
        ):
            raise CoordinatorLivenessLeaseRejected(
                f"startup liveness rejected for child={child_session_id}: "
                "session row is missing, terminal, or lineage/control-plane "
                "does not match"
            )
        accepted = await self._write_startup_core(
            expected,
            ttl_ms=remaining_ttl_ms,
        )
        if not accepted:
            raise CoordinatorLivenessLeaseRejected(
                f"startup liveness rejected for child={child_session_id}: "
                "an active lease with different lineage or a terminal "
                "tombstone won the Redis CAS"
            )
        # Startup is only a bounded grace marker.  Parent/sandbox/quota owners
        # are renewed after the first DB-authorized heartbeat proves the child
        # runner is actually alive.
        return expected

    async def record_heartbeat(
        self, envelope: MailboxEnvelope | Mapping[str, Any]
    ) -> bool:
        """Refresh a lease for one schema-valid, DB-authorized heartbeat.

        ``correlation_id`` must use the canonical ``hb:<child>`` wire identity;
        it is not treated as run identity. Run/work-unit authority comes from
        the existing startup lease plus the current session row.

        The DB read and Redis write are not one transaction, so the final write
        is a Redis compare-and-refresh script. A terminal tombstone or lineage
        change that wins after the DB read causes this method to return False;
        callbacks and local observations run only after a successful CAS.
        """
        try:
            candidate = (
                envelope.model_dump(mode="python")
                if isinstance(envelope, MailboxEnvelope)
                else envelope
            )
            parsed = MailboxEnvelope.model_validate(candidate)
        except (ValidationError, ValueError, TypeError):
            return False

        if (
            parsed.type != MailboxEnvelopeType.PROGRESS_UPDATE
            or parsed.producer_role != ProducerRole.CHILD_AGENT
            or parsed.payload.get("kind") != ProgressKind.HEARTBEAT
            or parsed.correlation_id != f"hb:{parsed.child_session_id}"
        ):
            return False

        current = await self.get_lease(parsed.child_session_id)
        if current is None:
            return False
        if (
            parsed.child_session_id != current.child_session_id
            or parsed.parent_session_id != current.parent_session_id
        ):
            return False

        row = await self._sessions.get_by_id(parsed.child_session_id)
        if not self._row_matches_lease(
            row,
            current,
            allowed_statuses=_HEARTBEAT_SESSION_STATUSES,
        ):
            return False

        phase_value = parsed.payload.get("phase") or "idle"
        phase = (
            phase_value.value
            if hasattr(phase_value, "value")
            else str(phase_value)
        )
        refreshed = CoordinatorChildLease(
            root_session_id=current.root_session_id,
            parent_session_id=current.parent_session_id,
            child_session_id=current.child_session_id,
            coordinator_run_id=current.coordinator_run_id,
            work_unit_id=current.work_unit_id,
            last_seen_epoch=max(current.last_seen_epoch, self._effective_now()),
            phase=phase,
            authority_age_seconds=0.0,
        )
        if not await self._refresh_core(refreshed):
            return False
        await self._run_callbacks(refreshed)
        return True

    @staticmethod
    def _row_matches_lease(
        row: Session | None,
        lease: CoordinatorChildLease,
        *,
        allowed_statuses: frozenset[SessionStatus],
    ) -> bool:
        lineage = (
            lease.root_session_id,
            lease.parent_session_id,
            lease.child_session_id,
            lease.coordinator_run_id,
            lease.work_unit_id,
        )
        return bool(
            all(isinstance(value, str) and value.strip() for value in lineage)
            and row is not None
            and row.id == lease.child_session_id
            and isinstance(row.status, SessionStatus)
            and row.status in allowed_statuses
            and row.worker_type == "subagent"
            and row.subagent_control_plane == "mailbox"
            and row.tool_filter_preset == COORDINATOR_STEP_PRESET
            and row.root_session_id == lease.root_session_id
            and row.parent_session_id == lease.parent_session_id
            and row.coordinator_run_id == lease.coordinator_run_id
            and row.work_unit_id == lease.work_unit_id
        )

    @staticmethod
    def _lease_script_args(
        lease: CoordinatorChildLease, ttl_ms: int
    ) -> tuple[str, ...]:
        return (
            lease.root_session_id,
            lease.parent_session_id,
            lease.child_session_id,
            lease.coordinator_run_id,
            lease.work_unit_id,
            str(lease.last_seen_epoch),
            lease.phase,
            str(ttl_ms),
        )

    async def _write_startup_core(
        self,
        lease: CoordinatorChildLease,
        *,
        ttl_ms: int,
    ) -> bool:
        accepted = await self._redis.eval(
            _STARTUP_LEASE_CAS_SCRIPT,
            1,
            self._key(lease.child_session_id),
            *self._lease_script_args(lease, ttl_ms),
        )
        if int(accepted) != 1:
            return False
        self._remember_observation(lease.child_session_id)
        return True

    async def _refresh_core(self, lease: CoordinatorChildLease) -> bool:
        accepted = await self._redis.eval(
            _REFRESH_LEASE_CAS_SCRIPT,
            1,
            self._key(lease.child_session_id),
            *self._lease_script_args(lease, self._lease_ttl_ms),
        )
        if int(accepted) != 1:
            self._observed_monotonic.pop(lease.child_session_id, None)
            return False
        self._remember_observation(lease.child_session_id)
        return True

    def _remember_observation(self, child_session_id: str) -> None:
        self._observed_monotonic[child_session_id] = float(
            self._monotonic_clock()
        )

    async def _run_callbacks(self, lease: CoordinatorChildLease) -> None:
        for callback in self._callbacks:
            try:
                await callback(lease)
            except Exception:
                logger.warning(
                    "coordinator liveness peripheral renew failed child=%s "
                    "callback=%r; core lease remains authoritative",
                    lease.child_session_id,
                    callback,
                    exc_info=True,
                )

    async def get_lease(
        self, child_session_id: str
    ) -> CoordinatorChildLease | None:
        try:
            raw = await self._redis.eval(
                _READ_LIVE_LEASE_WITH_PTTL_SCRIPT,
                1,
                self._key(child_session_id),
                str(self._lease_ttl_ms),
            )
        except UnicodeDecodeError:
            return self._forget_invalid_lease(child_session_id)
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 8
        ):
            return self._forget_invalid_lease(child_session_id)
        try:
            decoded_values = [self._decode(value) for value in raw[:7]]
            raw_pttl = raw[7]
            if isinstance(raw_pttl, bool):
                raise TypeError("Redis PTTL must be an integer")
            pttl_ms = (
                raw_pttl
                if isinstance(raw_pttl, int)
                else int(self._decode(raw_pttl))
            )
        except (TypeError, UnicodeDecodeError, ValueError):
            return self._forget_invalid_lease(child_session_id)
        fields = (
            "root_session_id",
            "parent_session_id",
            "child_session_id",
            "coordinator_run_id",
            "work_unit_id",
            "last_seen_epoch",
            "phase",
        )
        decoded = dict(zip(fields, decoded_values, strict=True))
        if pttl_ms <= 0 or pttl_ms > self._lease_ttl_ms:
            return self._forget_invalid_lease(child_session_id)
        try:
            last_seen = float(decoded["last_seen_epoch"])
        except (TypeError, ValueError):
            return self._forget_invalid_lease(child_session_id)
        if (
            not math.isfinite(last_seen)
            or not all(
                decoded[field] for field in fields if field != "last_seen_epoch"
            )
            or decoded["child_session_id"] != child_session_id
        ):
            return self._forget_invalid_lease(child_session_id)
        return CoordinatorChildLease(
            root_session_id=decoded["root_session_id"],
            parent_session_id=decoded["parent_session_id"],
            child_session_id=decoded["child_session_id"],
            coordinator_run_id=decoded["coordinator_run_id"],
            work_unit_id=decoded["work_unit_id"],
            last_seen_epoch=last_seen,
            phase=decoded["phase"],
            authority_age_seconds=(self._lease_ttl_ms - pttl_ms) / 1_000.0,
        )

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8")
        raise TypeError("Redis lease fields must be str or UTF-8 bytes")

    def _forget_invalid_lease(self, child_session_id: str) -> None:
        self._observed_monotonic.pop(child_session_id, None)
        return None

    def is_stale(self, lease: CoordinatorChildLease | None) -> bool:
        """Return true for missing leases and at the exact stale boundary."""
        if lease is None:
            return True
        authority_age = lease.authority_age_seconds
        if authority_age is not None:
            if (
                isinstance(authority_age, bool)
                or not isinstance(authority_age, (int, float))
                or not math.isfinite(authority_age)
                or authority_age < 0
            ):
                return True
            return authority_age >= self._stale_after
        now = self._effective_now()
        if now >= lease.last_seen_epoch:
            age = now - lease.last_seen_epoch
        else:
            # A persisted epoch may be ahead after an NTP/host clock rollback.
            # Fall back to elapsed monotonic time since this process last wrote
            # the child (or since service startup after a process restart).
            observed_at = self._observed_monotonic.get(
                lease.child_session_id, self._anchor_monotonic
            )
            age = max(0.0, float(self._monotonic_clock()) - observed_at)
        return age >= self._stale_after

    async def await_stale(
        self, child_session_id: str
    ) -> CoordinatorChildLease | None:
        while True:
            lease = await self.get_lease(child_session_id)
            if self.is_stale(lease):
                return lease
            await self._sleep(self._poll_interval)

    async def clear(self, child_session_id: str) -> None:
        await self._redis.delete(self._key(child_session_id))
        self._observed_monotonic.pop(child_session_id, None)

    async def clear_if_matches(self, expected: CoordinatorChildLease) -> bool:
        """Delete only the exact live attempt registered by this caller.

        ``clear`` remains the explicit user-level unconditional cleanup API.
        Dispatch rollback must use this compare-delete surface so a concurrent
        terminal tombstone or newer attempt cannot be erased.
        """
        lineage = (
            expected.root_session_id,
            expected.parent_session_id,
            expected.child_session_id,
            expected.coordinator_run_id,
            expected.work_unit_id,
        )
        if not all(isinstance(value, str) and value.strip() for value in lineage):
            raise ValueError("expected lineage must contain five non-empty strings")
        result = int(await self._redis.eval(
            _LIVE_LEASE_COMPARE_DELETE_SCRIPT,
            1,
            self._key(expected.child_session_id),
            *lineage,
        ))
        if result in {1, -1}:
            self._observed_monotonic.pop(expected.child_session_id, None)
        return result == 1

    async def mark_terminal(self, child_session_id: str) -> None:
        """Atomically replace any live lease with a bounded tombstone.

        The marker prevents an in-flight heartbeat or startup registration
        that already passed its DB check from resurrecting liveness. The
        persisted terminal row remains the authority after this short marker
        expires; new startup/heartbeat attempts still fail their DB gate.
        """
        await self._redis.eval(
            _TERMINAL_TOMBSTONE_SCRIPT,
            1,
            self._key(child_session_id),
            child_session_id,
            str(self._lease_ttl_ms),
        )
        self._observed_monotonic.pop(child_session_id, None)
