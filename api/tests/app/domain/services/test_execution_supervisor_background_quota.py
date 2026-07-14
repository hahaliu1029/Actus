from __future__ import annotations

import asyncio
import fnmatch
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from app.domain.models.session import Session, SessionStatus
from app.domain.services import execution_supervisor as supervisor_module
from app.domain.services._lua_scripts import (
    LUA_ADMIT_SHA,
    LUA_ADMIT_SOURCE,
    LUA_GC_BACKGROUND_RECONCILE_MARKER_SHA,
    LUA_MARK_BACKGROUND_RECONCILE_HELD_SHA,
    LUA_RELEASE_HELD_BACKGROUND_RECONCILE_SHA,
    LUA_RESTORE_BACKGROUND_FROM_MARKER_SHA,
    LUA_REVOKE_SHA,
    LUA_REVOKE_SOURCE,
    LUA_SYNC_BACKGROUND_EXPIRY_SHA,
)
from app.domain.services.execution_supervisor import (
    ExecutionSupervisor,
    ModeTransitionFenceLostError,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Redis:
    def __init__(self) -> None:
        self.values = {
            "supervisor:system:bg_count": b"3",
        }
        self.hash_lengths = {
            "supervisor:user:u1": 2,
        }

    async def get(self, key: str) -> bytes | str | None:
        return self.values.get(key)

    async def hlen(self, key: str) -> int:
        return self.hash_lengths.get(key, 0)


async def test_get_background_quota_reads_redis_counts_and_limits() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=_Redis(),
        session_repository=object(),
        max_system_bg=7,
        max_user_bg=5,
    )

    quota = await supervisor.get_background_quota("u1")

    assert quota == {
        "system_used": 3,
        "system_limit": 7,
        "user_used": 2,
        "user_limit": 5,
    }


async def test_get_background_quota_treats_missing_counts_as_zero() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=_Redis(),
        session_repository=object(),
        max_system_bg=7,
        max_user_bg=5,
    )
    supervisor._redis.values.clear()

    quota = await supervisor.get_background_quota("new-user")

    assert quota["system_used"] == 0
    assert quota["user_used"] == 0


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc)

    def advance(self, delta: timedelta) -> None:
        self.now += delta


class _LeaseRepo:
    def __init__(
        self,
        session: Session,
        *,
        renew_result: bool = True,
        renew_error: Exception | None = None,
    ) -> None:
        self.session = session
        self.renew_result = renew_result
        self.renew_error = renew_error
        self.renew_calls: list[tuple[str, datetime]] = []

    async def get_by_id(self, session_id: str) -> Session | None:
        return self.session if self.session.id == session_id else None

    async def promote_foreground_to_background(
        self,
        session_id: str,
        *,
        expires_at: datetime,
        retry_budget_remaining: int,
        expected_execution_revision: int,
        background_reason: str,
        pending_event,
    ) -> int | None:
        if (
            self.session.id != session_id
            or self.session.status != SessionStatus.RUNNING
            or self.session.execution_mode != "foreground"
            or self.session.execution_revision != expected_execution_revision
        ):
            return None
        self.session.execution_mode = "background"
        self.session.background_reason = background_reason
        self.session.expires_at = expires_at
        self.session.execution_phase = "running"
        self.session.retry_budget_remaining = retry_budget_remaining
        self.session.execution_revision += 1
        self.session.pending_execution_event = pending_event
        return self.session.execution_revision

    async def renew_auto_degrade_expiry_if_running(
        self,
        session_id: str,
        *,
        expires_at: datetime,
    ) -> tuple[datetime, int] | None:
        self.renew_calls.append((session_id, expires_at))
        if self.renew_error is not None:
            raise self.renew_error
        if self.renew_result:
            self.session.expires_at = expires_at
            return (expires_at, self.session.execution_revision)
        return None

    async def resume_auto_degrade_to_foreground_if_running(
        self,
        session_id: str,
        *,
        expected_execution_revision: int,
        pending_event,
    ) -> int | None:
        if (
            self.session.id != session_id
            or self.session.status != SessionStatus.RUNNING
            or self.session.execution_mode != "background"
            or self.session.execution_phase != "running"
            or self.session.background_reason != "auto_degrade"
            or self.session.execution_revision != expected_execution_revision
        ):
            return None
        self.session.execution_mode = "foreground"
        self.session.background_reason = None
        self.session.expires_at = None
        self.session.execution_revision += 1
        self.session.pending_execution_event = pending_event
        return self.session.execution_revision

    async def clear_pending_execution_event(
        self, session_id: str, *, execution_revision: int
    ) -> bool:
        if (
            self.session.id != session_id
            or self.session.execution_revision != execution_revision
            or self.session.pending_execution_event is None
        ):
            return False
        self.session.pending_execution_event = None
        return True

    async def update_to_terminal_if_background_expired(
        self,
        session_id: str,
        status: SessionStatus,
        terminal_reason: str,
        *,
        expires_at_lte: datetime,
        expected_execution_revision: int | None = None,
        pending_event=None,
    ) -> int | None:
        expiry = self.session.expires_at
        if (
            self.session.id != session_id
            or self.session.status != SessionStatus.RUNNING
            or self.session.execution_mode != "background"
            or self.session.execution_phase not in ("running", "suspended")
            or expiry is None
            or expiry > expires_at_lte
            or (
                expected_execution_revision is not None
                and self.session.execution_revision != expected_execution_revision
            )
        ):
            return None
        self.session.status = status
        self.session.execution_phase = "terminated"
        self.session.terminal_reason = terminal_reason
        self.session.execution_revision += 1
        self.session.pending_execution_event = pending_event
        return self.session.execution_revision


class _LeaseRedis:
    def __init__(self) -> None:
        self.hset_calls: list[tuple[str, str, str]] = []
        self.zadd_calls: list[tuple[str, dict[str, float]]] = []
        self.expire_calls: list[tuple[str, int]] = []
        self.hashes: dict[tuple[str, str], str] = {
            ("supervisor:user:user-lease", "session-lease"): "0.000000",
            ("supervisor:bg-generation:user-lease", "session-lease"): "0",
            ("supervisor:system:bg-members", "session-lease"): "0",
        }
        self.scores: dict[tuple[str, str], float] = {
            ("supervisor:bg:user-lease", "session-lease"): 0.0,
        }
        self.ttls: dict[str, int] = {
            "supervisor:user:user-lease": 86400,
            "supervisor:bg:user-lease": 86400,
            "supervisor:bg-generation:user-lease": 86400,
        }
        self.evalsha_calls: list[tuple[str, int, tuple[object, ...]]] = []
        self.system_count = 1
        self.owner_values: dict[str, str] = {}

    @staticmethod
    def _membership_generation(value: str | None) -> int | None:
        if value is None:
            return None
        if value.startswith("v1|"):
            return int(value.split("|", 2)[1])
        return int(value)

    def _extend_ttl(self, key: str, ttl: int) -> None:
        self.ttls[key] = max(self.ttls.get(key, -1), ttl)

    def _clear_marker_through(self, session_id: str, generation: int) -> None:
        marker_key = "supervisor:system:bg-reconcile-pending"
        marker = self.hashes.get((marker_key, session_id))
        if marker is None:
            return
        parts = marker.split("|", 7)
        if len(parts) == 8 and int(parts[2]) <= generation:
            self.hashes.pop((marker_key, session_id), None)
            self.scores.pop(("supervisor:system:bg-reconcile-due", session_id), None)

    async def get(self, key: str) -> str | None:
        if key == "supervisor:system:bg_count":
            return str(self.system_count)
        return None

    async def hlen(self, key: str) -> int:
        return sum(1 for hash_key, _field in self.hashes if hash_key == key)

    async def hscan(
        self,
        key: str,
        *,
        cursor: int | str | bytes,
        count: int,
    ) -> tuple[int, dict[str, str]]:
        return 0, {
            field: value
            for (hash_key, field), value in self.hashes.items()
            if hash_key == key
        }

    async def scan_iter(self, *, match: str):
        keys = {key for key, _field in self.hashes}
        keys.update(key for key, _member in self.scores)
        for key in sorted(keys):
            if fnmatch.fnmatch(key, match):
                yield key

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool,
        ex: int,
    ) -> bool:
        if nx and key in self.owner_values:
            return False
        self.owner_values[key] = value
        return True

    async def eval(
        self,
        script: str,
        numkeys: int,
        key: str,
        token: str,
        *args: object,
    ) -> int:
        if self.owner_values.get(key) != token:
            return 0
        if "DEL" in script:
            self.owner_values.pop(key, None)
        return 1

    async def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
        self.evalsha_calls.append((sha, numkeys, args))
        keys = [str(value) for value in args[:numkeys]]
        argv = args[numkeys:]
        session_id = str(argv[0])
        if sha == LUA_GC_BACKGROUND_RECONCILE_MARKER_SHA:
            expected_marker = str(argv[1])
            if self.hashes.get((keys[0], session_id)) != expected_marker:
                return 0
            self.hashes.pop((keys[0], session_id), None)
            self.scores.pop((keys[1], session_id), None)
            return 1
        if sha == LUA_MARK_BACKGROUND_RECONCILE_HELD_SHA:
            expected_generation = int(argv[1])
            marker_value = str(argv[2])
            marker_due = float(argv[3])
            if self._membership_generation(
                self.hashes.get((keys[0], session_id))
            ) != expected_generation:
                return 0
            if (keys[1], session_id) in self.hashes:
                return 2
            self.hashes[(keys[1], session_id)] = marker_value
            self.scores[(keys[2], session_id)] = marker_due
            return 1
        if numkeys == 1:
            sweep_now = float(argv[0])
            expired = [
                member
                for (key, member), score in self.scores.items()
                if key == keys[0] and score <= sweep_now
            ]
            return [member.encode() for member in expired]
        if numkeys == 2:
            expected_generation = int(argv[1])
            member_generation = self._membership_generation(
                self.hashes.get((keys[1], session_id))
            )
            if member_generation != expected_generation:
                return 0
            self.hashes.pop((keys[1], session_id), None)
            if self.system_count > 0:
                self.system_count -= 1
            return 1
        if sha == LUA_RELEASE_HELD_BACKGROUND_RECONCILE_SHA:
            expected_generation = int(argv[1])
            expected_marker = str(argv[2])
            released_marker = str(argv[3])
            released_due = float(argv[4])
            full_revoke = int(argv[5])
            require_generation = int(argv[6])
            allow_missing_membership = int(argv[7])
            if self.hashes.get((keys[5], session_id)) != expected_marker:
                return 0
            raw_membership = self.hashes.get((keys[4], session_id))
            member_generation = self._membership_generation(raw_membership)
            if member_generation is None and not allow_missing_membership:
                return -1
            if (
                member_generation is not None
                and member_generation != expected_generation
            ):
                return -2
            if full_revoke:
                current_generation = self.hashes.get((keys[3], session_id))
                if current_generation is None and require_generation:
                    return -3
                if (
                    current_generation is not None
                    and int(current_generation) != expected_generation
                ):
                    return -3
                self.hashes.pop((keys[1], session_id), None)
                self.scores.pop((keys[2], session_id), None)
                self.hashes.pop((keys[3], session_id), None)
            member_existed = int(
                self.hashes.pop((keys[4], session_id), None) is not None
            )
            if member_existed and self.system_count > 0:
                self.system_count -= 1
            self.hashes[(keys[5], session_id)] = released_marker
            self.scores[(keys[6], session_id)] = released_due
            return 1
        if sha == LUA_RESTORE_BACKGROUND_FROM_MARKER_SHA:
            expires_at = float(argv[1])
            ttl = int(argv[2])
            generation = int(argv[3])
            membership_value = str(argv[4])
            expected_marker = str(argv[5])
            if self.hashes.get((keys[5], session_id)) != expected_marker:
                return 0
            marker_parts = expected_marker.split("|", 7)
            if len(marker_parts) != 8 or int(marker_parts[2]) > generation:
                return -1
            current_generation = self.hashes.get((keys[3], session_id))
            if current_generation is not None and int(current_generation) > generation:
                return -1
            member_generation = self._membership_generation(
                self.hashes.get((keys[4], session_id))
            )
            if member_generation is not None and member_generation > generation:
                return -1
            created_reservation = member_generation is None
            if created_reservation:
                self.system_count += 1
            self.hashes[(keys[1], session_id)] = f"{expires_at:.6f}"
            self.scores[(keys[2], session_id)] = expires_at
            self.hashes[(keys[3], session_id)] = str(generation)
            self.hashes[(keys[4], session_id)] = membership_value
            self._extend_ttl(keys[1], ttl)
            self._extend_ttl(keys[2], ttl)
            self._extend_ttl(keys[3], ttl)
            self.hashes.pop((keys[5], session_id), None)
            self.scores.pop((keys[6], session_id), None)
            return 1 if created_reservation else 2
        if sha == LUA_SYNC_BACKGROUND_EXPIRY_SHA:
            expires_at = float(argv[1])
            ttl = int(argv[2])
            generation = int(argv[3])
            membership_value = str(argv[4])
            current_generation = self.hashes.get((keys[2], session_id))
            if current_generation is not None and int(current_generation) > generation:
                return 0
            member_generation = self.hashes.get((keys[3], session_id))
            if member_generation is None:
                return -1
            parsed_member_generation = self._membership_generation(member_generation)
            if parsed_member_generation is not None and parsed_member_generation > generation:
                return 0
            if parsed_member_generation is not None and parsed_member_generation < generation:
                return -2
            if current_generation is not None and int(current_generation) < generation:
                return -2
            self.hashes[(keys[0], session_id)] = f"{expires_at:.6f}"
            self.scores[(keys[1], session_id)] = expires_at
            self.hashes[(keys[2], session_id)] = str(generation)
            self.hashes[(keys[3], session_id)] = membership_value
            self._extend_ttl(keys[0], ttl)
            self._extend_ttl(keys[1], ttl)
            self._extend_ttl(keys[2], ttl)
            self._clear_marker_through(session_id, generation)
            return 1
        if sha == LUA_ADMIT_SHA:
            expires_at = float(argv[1])
            max_system = int(argv[2])
            max_user = int(argv[3])
            generation = int(argv[5])
            authoritative = int(argv[6])
            membership_value = str(argv[7])
            ttl = int(argv[8])
            user_exists = (keys[1], session_id) in self.hashes
            current_generation = self.hashes.get((keys[4], session_id))
            member_generation = self.hashes.get((keys[5], session_id))
            parsed_member_generation = self._membership_generation(member_generation)
            marker = self.hashes.get((keys[6], session_id))
            marker_parts = marker.split("|", 7) if marker is not None else []
            membership_user = membership_value.split("|", 2)[2]
            released_marker_matches = (
                len(marker_parts) == 8
                and marker_parts[1] == "released"
                and int(marker_parts[2]) <= generation
                and (not marker_parts[7] or marker_parts[7] == membership_user)
            )

            def admit_released_residue() -> int:
                other_user_count = sum(
                    1 for key, _field in self.hashes if key == keys[1]
                ) - 1
                if not authoritative:
                    if other_user_count >= max_user:
                        return 2
                    if self.system_count >= max_system:
                        return 1
                self.system_count += 1
                self.hashes[(keys[1], session_id)] = f"{expires_at:.6f}"
                self.scores[(keys[3], session_id)] = expires_at
                self.hashes[(keys[4], session_id)] = str(generation)
                self.hashes[(keys[5], session_id)] = membership_value
                self._extend_ttl(keys[1], ttl)
                self._extend_ttl(keys[3], ttl)
                self._extend_ttl(keys[4], ttl)
                self._clear_marker_through(session_id, generation)
                return 0

            if user_exists:
                if current_generation is None:
                    if parsed_member_generation is None and released_marker_matches:
                        return admit_released_residue()
                    if not authoritative:
                        return 6
                    if (
                        parsed_member_generation is not None
                        and parsed_member_generation > generation
                    ):
                        return 4
                    self.hashes[(keys[1], session_id)] = f"{expires_at:.6f}"
                    self.scores[(keys[3], session_id)] = expires_at
                    self.hashes[(keys[4], session_id)] = str(generation)
                    self.hashes[(keys[5], session_id)] = membership_value
                    self._extend_ttl(keys[1], ttl)
                    self._extend_ttl(keys[3], ttl)
                    self._extend_ttl(keys[4], ttl)
                    self._clear_marker_through(session_id, generation)
                    return 3
                if int(current_generation) > generation:
                    return 4
                if parsed_member_generation is not None and parsed_member_generation > generation:
                    return 4
                if parsed_member_generation is None and released_marker_matches:
                    return admit_released_residue()
                if int(current_generation) == generation:
                    if parsed_member_generation is None or parsed_member_generation < generation:
                        if not authoritative:
                            return 7
                        self.hashes[(keys[5], session_id)] = membership_value
                    self._clear_marker_through(session_id, generation)
                    return 3
                self.hashes[(keys[1], session_id)] = f"{expires_at:.6f}"
                self.scores[(keys[3], session_id)] = expires_at
                self.hashes[(keys[4], session_id)] = str(generation)
                self.hashes[(keys[5], session_id)] = membership_value
                self._extend_ttl(keys[1], ttl)
                self._extend_ttl(keys[3], ttl)
                self._extend_ttl(keys[4], ttl)
                self._clear_marker_through(session_id, generation)
                return 5
            if member_generation is None and authoritative:
                return 8
            user_count = sum(1 for key, _field in self.hashes if key == keys[1])
            if user_count >= max_user:
                return 2
            if member_generation is None and self.system_count >= max_system:
                return 1
            if member_generation is None:
                self.system_count += 1
            self.hashes[(keys[1], session_id)] = f"{expires_at:.6f}"
            self.scores[(keys[3], session_id)] = expires_at
            self.hashes[(keys[4], session_id)] = str(generation)
            self.hashes[(keys[5], session_id)] = membership_value
            self._extend_ttl(keys[1], ttl)
            self._extend_ttl(keys[3], ttl)
            self._extend_ttl(keys[4], ttl)
            self._clear_marker_through(session_id, generation)
            return 0
        assert sha == LUA_REVOKE_SHA
        expected_generation = int(argv[1])
        allow_legacy = int(argv[2])
        expected_marker = str(argv[3])
        released_marker = str(argv[4])
        released_due = float(argv[5]) if str(argv[5]) else None
        raw_marker = self.hashes.get((keys[5], session_id))
        marker_phase = (
            raw_marker.split("|", 7)[1]
            if raw_marker is not None and raw_marker.startswith("v1|")
            else None
        )
        if marker_phase == "held" and (
            raw_marker != expected_marker
            or not released_marker
            or released_due is None
        ):
            return -2
        current_generation = self.hashes.get((keys[3], session_id))
        if current_generation is None and not allow_legacy:
            return 0
        if current_generation is not None and int(current_generation) != expected_generation:
            return 0
        member_generation = self.hashes.get((keys[4], session_id))
        if member_generation is None and current_generation is not None and not allow_legacy:
            return -1
        if member_generation is not None:
            parsed_member_generation = self._membership_generation(member_generation)
            if parsed_member_generation is not None and parsed_member_generation > expected_generation:
                return 0
            if parsed_member_generation is not None and parsed_member_generation < expected_generation and not allow_legacy:
                return -1
        existed = int((keys[1], session_id) in self.hashes)
        self.hashes.pop((keys[1], session_id), None)
        self.scores.pop((keys[2], session_id), None)
        self.hashes.pop((keys[3], session_id), None)
        member_existed = int(self.hashes.pop((keys[4], session_id), None) is not None)
        if (
            member_existed
            or (marker_phase != "released" and allow_legacy and existed)
        ) and self.system_count > 0:
            self.system_count -= 1
        if marker_phase == "held" and raw_marker == expected_marker:
            self.hashes[(keys[5], session_id)] = released_marker
            assert released_due is not None
            self.scores[(keys[6], session_id)] = released_due
        return existed

    async def hset(self, key: str, field: str, value: str) -> None:
        self.hset_calls.append((key, field, value))
        self.hashes[(key, field)] = value

    async def hget(self, key: str, field: str) -> str | None:
        return self.hashes.get((key, field))

    async def hdel(self, key: str, field: str) -> int:
        return int(self.hashes.pop((key, field), None) is not None)

    async def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self.zadd_calls.append((key, mapping))
        for member, score in mapping.items():
            self.scores[(key, member)] = score

    async def zscore(self, key: str, member: str) -> float | None:
        return self.scores.get((key, member))

    async def expire(self, key: str, seconds: int) -> None:
        self.expire_calls.append((key, seconds))
        self.ttls[key] = seconds

    async def ttl(self, key: str) -> int:
        return self.ttls.get(key, -2)


def _auto_degrade_session(
    clock: _Clock,
    *,
    status: SessionStatus = SessionStatus.RUNNING,
    execution_mode: str = "background",
    background_reason: str | None = "auto_degrade",
    execution_phase: str = "running",
    expires_in: timedelta = timedelta(minutes=5),
) -> Session:
    return Session(
        id="session-lease",
        user_id="user-lease",
        status=status,
        execution_mode=execution_mode,
        background_reason=background_reason,
        execution_phase=execution_phase,
        expires_at=clock.now + expires_in,
    )


async def test_auto_degrade_renew_extends_pg_before_redis_and_throttles_until_window(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    repo = _LeaseRepo(session)
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        utcnow=lambda: clock.now,
    )

    renewed = await supervisor.renew_auto_degrade_expiry_if_running(
        session_id=session.id,
    )

    assert renewed is True
    assert repo.renew_calls == [
        (session.id, clock.now + timedelta(hours=2))
    ]
    expected_expiry = (clock.now + timedelta(hours=2)).timestamp()
    assert len(redis.evalsha_calls) == 1
    assert redis.hashes[("supervisor:user:user-lease", session.id)] == (
        f"{expected_expiry:.6f}"
    )
    assert redis.scores[("supervisor:bg:user-lease", session.id)] == expected_expiry
    assert redis.ttls == {
        "supervisor:user:user-lease": 93600,
        "supervisor:bg:user-lease": 93600,
        "supervisor:bg-generation:user-lease": 93600,
    }

    # The first renewal moved expiry two hours out. The next heartbeat tick
    # must be throttled instead of writing PG/Redis every 15 seconds.
    assert await supervisor.renew_auto_degrade_expiry_if_running(
        session_id=session.id,
    ) is False
    assert len(repo.renew_calls) == 1
    assert len(redis.evalsha_calls) == 1


async def test_auto_degrade_renew_keeps_extending_for_more_than_two_hours() -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    repo = _LeaseRepo(session)
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        utcnow=lambda: clock.now,
    )

    for _ in range(4):
        assert await supervisor.renew_auto_degrade_expiry_if_running(
            session_id=session.id,
        ) is True
        clock.advance(timedelta(minutes=100))

    assert clock.now - datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc) > timedelta(
        hours=5
    )
    assert len(repo.renew_calls) == 4
    assert session.expires_at == (
        datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc)
        + timedelta(minutes=300, hours=2)
    )


@pytest.mark.parametrize(
    ("overrides",),
    [
        ({"status": SessionStatus.COMPLETED},),
        ({"execution_mode": "foreground"},),
        ({"background_reason": "explicit"},),
        ({"execution_phase": "suspended"},),
        ({"expires_in": timedelta(minutes=45)},),
    ],
)
async def test_auto_degrade_renew_skips_noneligible_or_not_yet_due_session(
    overrides: dict[str, object],
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, **overrides)
    repo = _LeaseRepo(session)
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        utcnow=lambda: clock.now,
    )
    if overrides.get("expires_in") == timedelta(minutes=45):
        assert session.expires_at is not None
        redis.hashes[("supervisor:user:user-lease", session.id)] = (
            f"{session.expires_at.timestamp():.6f}"
        )
        redis.scores[("supervisor:bg:user-lease", session.id)] = (
            session.expires_at.timestamp()
        )
        redis.hashes[("supervisor:system:bg-members", session.id)] = (
            "v1|0|user-lease"
        )
        redis.ttls["supervisor:user:user-lease"] = 90000
        redis.ttls["supervisor:bg:user-lease"] = 90000
        redis.ttls["supervisor:bg-generation:user-lease"] = 90000

    assert await supervisor.renew_auto_degrade_expiry_if_running(
        session_id=session.id,
    ) is False
    assert repo.renew_calls == []
    assert redis.hset_calls == []
    assert redis.zadd_calls == []


@pytest.mark.parametrize("renew_error", [None, RuntimeError("pg unavailable")])
async def test_auto_degrade_renew_never_extends_redis_when_pg_cas_does_not_succeed(
    renew_error: Exception | None,
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    repo = _LeaseRepo(
        session,
        renew_result=False,
        renew_error=renew_error,
    )
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        utcnow=lambda: clock.now,
    )

    if renew_error is None:
        assert await supervisor.renew_auto_degrade_expiry_if_running(
            session_id=session.id,
        ) is False
    else:
        with pytest.raises(RuntimeError, match="pg unavailable"):
            await supervisor.renew_auto_degrade_expiry_if_running(
                session_id=session.id,
            )

    assert redis.hset_calls == []
    assert redis.zadd_calls == []
    assert redis.expire_calls == []


class _FailFirstZaddRedis(_LeaseRedis):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def evalsha(self, sha: str, numkeys: int, *args: object) -> int:
        if sha == LUA_SYNC_BACKGROUND_EXPIRY_SHA and not self.failed:
            self.failed = True
            raise RuntimeError("redis zadd unavailable")
        return await super().evalsha(sha, numkeys, *args)


async def test_pg_success_redis_failure_retries_sync_despite_renew_throttle() -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    repo = _LeaseRepo(session)
    redis = _FailFirstZaddRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        utcnow=lambda: clock.now,
    )

    with pytest.raises(RuntimeError, match="redis zadd unavailable"):
        await supervisor.renew_auto_degrade_expiry_if_running(
            session_id=session.id,
        )

    # PG now holds a far-future expiry. The next tick must repair Redis from
    # that durable value rather than treating the PG window as fully renewed.
    assert await supervisor.renew_auto_degrade_expiry_if_running(
        session_id=session.id,
    ) is True
    assert len(repo.renew_calls) == 1
    assert redis.scores[
        ("supervisor:bg:user-lease", session.id)
    ] == session.expires_at.timestamp()


class _OrderedLeaseRedis(_LeaseRedis):
    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.order = order

    async def evalsha(self, sha: str, numkeys: int, *args: object) -> int:
        self.order.append("redis_projection")
        return await super().evalsha(sha, numkeys, *args)


class _LeaseUow:
    def __init__(
        self,
        repo: _LeaseRepo,
        order: list[str],
        *,
        commit_error: Exception | None = None,
    ) -> None:
        self.session = repo
        self.db_session = self
        self.order = order
        self.commit_error = commit_error

    async def commit(self) -> None:
        self.order.append("pg_commit")
        if self.commit_error is not None:
            raise self.commit_error

    async def __aenter__(self) -> _LeaseUow:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


@pytest.mark.parametrize("commit_error", [None, RuntimeError("commit failed")])
async def test_auto_degrade_renew_commits_pg_before_any_redis_write(
    commit_error: Exception | None,
) -> None:
    clock = _Clock()
    repo = _LeaseRepo(_auto_degrade_session(clock))
    order: list[str] = []
    redis = _OrderedLeaseRedis(order)
    uow = _LeaseUow(repo, order, commit_error=commit_error)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        uow_factory=lambda: uow,
        utcnow=lambda: clock.now,
    )

    if commit_error is None:
        assert await supervisor.renew_auto_degrade_expiry_if_running(
            session_id=repo.session.id,
        ) is True
        assert order[:2] == ["pg_commit", "redis_projection"]
    else:
        with pytest.raises(RuntimeError, match="commit failed"):
            await supervisor.renew_auto_degrade_expiry_if_running(
                session_id=repo.session.id,
            )
        assert order == ["pg_commit"]
        assert redis.hset_calls == []


class _TransactionalPromotionState:
    def __init__(self, session: Session) -> None:
        self.session = session


class _TransactionalPromotionRepo:
    def __init__(self, state: _TransactionalPromotionState) -> None:
        self.state = state
        self.staged: dict[str, object] | None = None

    async def get_by_id(self, session_id: str) -> Session | None:
        return self.state.session if self.state.session.id == session_id else None

    async def promote_foreground_to_background(
        self,
        session_id: str,
        **fields: object,
    ) -> int | None:
        current = self.state.session
        expected_revision = int(fields["expected_execution_revision"])
        if (
            current.id != session_id
            or current.status != SessionStatus.RUNNING
            or current.execution_mode != "foreground"
            or current.execution_revision != expected_revision
        ):
            return None
        self.staged = dict(fields)
        return expected_revision + 1

    def commit_staged(self) -> None:
        if self.staged is None:
            return
        current = self.state.session
        current.execution_mode = "background"
        current.execution_phase = "running"
        current.background_reason = str(self.staged["background_reason"])
        current.expires_at = self.staged["expires_at"]
        current.retry_budget_remaining = int(self.staged["retry_budget_remaining"])
        current.execution_revision += 1
        current.pending_execution_event = self.staged["pending_event"]
        self.staged = None

    def rollback_staged(self) -> None:
        self.staged = None


class _ProductionLikePromotionUow:
    def __init__(
        self,
        state: _TransactionalPromotionState,
        *,
        commit_error: BaseException | None = None,
        commit_entered: asyncio.Event | None = None,
    ) -> None:
        self.session = _TransactionalPromotionRepo(state)
        self.db_session = self
        self.commit_error = commit_error
        self.commit_entered = commit_entered

    async def commit(self) -> None:
        if self.session.staged is None:
            return
        if self.commit_entered is not None:
            self.commit_entered.set()
            await asyncio.Event().wait()
        if self.commit_error is not None:
            raise self.commit_error
        self.session.commit_staged()

    async def rollback(self) -> None:
        self.session.rollback_staged()

    async def __aenter__(self) -> _ProductionLikePromotionUow:
        return self

    async def __aexit__(self, exc_type, _exc, _tb) -> None:
        if exc_type is not None:
            await self.rollback()
            return None
        try:
            await self.commit()
        except asyncio.CancelledError:
            await self.rollback()
            return None
        except Exception:
            await self.rollback()
            return None


class _ProductionLikePromotionUowFactory:
    def __init__(
        self,
        state: _TransactionalPromotionState,
        *,
        commit_error: BaseException | None = None,
        commit_entered: asyncio.Event | None = None,
    ) -> None:
        self.state = state
        self.commit_error = commit_error
        self.commit_entered = commit_entered

    def __call__(self) -> _ProductionLikePromotionUow:
        return _ProductionLikePromotionUow(
            self.state,
            commit_error=self.commit_error,
            commit_entered=self.commit_entered,
        )


@pytest.mark.parametrize("operation", ["promote", "admit"])
async def test_background_transition_uow_commit_failure_propagates_before_postcondition(
    operation: str,
) -> None:
    clock = _Clock()
    durable = _auto_degrade_session(
        clock,
        execution_mode="foreground",
        background_reason=None,
    )
    state = _TransactionalPromotionState(durable)
    redis = _ExpiringFenceQuotaRedis()
    _clear_counted_projection(redis)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        uow_factory=_ProductionLikePromotionUowFactory(
            state,
            commit_error=RuntimeError("transition commit failed"),
        ),
    )
    expiry = clock.now + timedelta(hours=2)

    with pytest.raises(RuntimeError, match="transition commit failed"):
        if operation == "promote":
            await supervisor.promote(
                session_id=durable.id,
                user_id="user-lease",
                expires_at=expiry,
            )
        else:
            await supervisor.admit(
                session_id=durable.id,
                user_id="user-lease",
                execution_mode="background",
                background_reason="explicit",
                expires_at=expiry,
            )

    assert durable.execution_mode == "foreground"
    assert durable.execution_revision == 0
    assert redis.system_count == 0
    assert ("supervisor:bg-generation:user-lease", durable.id) not in redis.hashes
    assert ("supervisor:system:bg-members", durable.id) not in redis.hashes


class _ExpiredCasStateMachine:
    async def terminate_expired_background(self, session_id, to, terminal_reason, *,
                                           expires_at_lte, session_repo,
                                           expected_execution_revision=None,
                                           pending_event=None):
        return await session_repo.update_to_terminal_if_background_expired(
            session_id,
            to,
            terminal_reason,
            expires_at_lte=expires_at_lte,
            expected_execution_revision=expected_execution_revision,
            pending_event=pending_event,
        )


async def test_stale_redis_expiry_cannot_terminal_pg_renewed_session_and_repairs(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, expires_in=timedelta(hours=1))
    repo = _LeaseRepo(session)
    redis = _LeaseRedis()
    old_expiry = clock.now - timedelta(seconds=1)
    redis.hashes[("supervisor:user:user-lease", session.id)] = (
        f"{old_expiry.timestamp():.6f}"
    )
    redis.scores[("supervisor:bg:user-lease", session.id)] = old_expiry.timestamp()
    redis.ttls["supervisor:user:user-lease"] = 86400
    redis.ttls["supervisor:bg:user-lease"] = 86400
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        session_state_machine=_ExpiredCasStateMachine(),
        utcnow=lambda: clock.now,
    )
    assert await supervisor.sweep_expired(
        user_id="user-lease",
        sweep_now=clock.now,
    ) == [session.id]
    assert ("supervisor:bg:user-lease", session.id) in redis.scores

    assert await supervisor.terminate_expired_background(
        session_id=session.id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is False
    assert session.status == SessionStatus.RUNNING
    assert redis.scores[("supervisor:bg:user-lease", session.id)] == (
        session.expires_at.timestamp()
    )

    clock.advance(timedelta(hours=1, seconds=1))
    assert await supervisor.sweep_expired(
        user_id="user-lease",
        sweep_now=clock.now,
    ) == [session.id]
    assert await supervisor.terminate_expired_background(
        session_id=session.id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is True
    assert session.status == SessionStatus.TIMED_OUT
    assert ("supervisor:bg:user-lease", session.id) not in redis.scores
    assert ("supervisor:user:user-lease", session.id) not in redis.hashes


class _FailFirstRevokeLeaseRedis(_LeaseRedis):
    def __init__(self) -> None:
        super().__init__()
        self.revoke_failures = 1

    async def evalsha(self, sha: str, numkeys: int, *args: object) -> object:
        if sha == LUA_REVOKE_SHA and self.revoke_failures:
            self.revoke_failures -= 1
            raise RuntimeError("redis revoke failed")
        return await super().evalsha(sha, numkeys, *args)


async def test_terminal_revoke_failure_keeps_expired_candidate_for_next_sweep(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, expires_in=timedelta(seconds=-1))
    repo = _LeaseRepo(session)
    redis = _FailFirstRevokeLeaseRedis()
    assert session.expires_at is not None
    redis.hashes[("supervisor:user:user-lease", session.id)] = (
        f"{session.expires_at.timestamp():.6f}"
    )
    redis.scores[("supervisor:bg:user-lease", session.id)] = (
        session.expires_at.timestamp()
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        session_state_machine=_ExpiredCasStateMachine(),
        utcnow=lambda: clock.now,
    )

    assert await supervisor.sweep_expired(
        user_id="user-lease", sweep_now=clock.now,
    ) == [session.id]
    with pytest.raises(RuntimeError, match="redis revoke failed"):
        await supervisor.terminate_expired_background(
            session_id=session.id,
            user_id="user-lease",
            sweep_now=clock.now,
        )
    assert session.status == SessionStatus.TIMED_OUT
    assert redis.scores[("supervisor:bg:user-lease", session.id)] == (
        session.expires_at.timestamp()
    )

    assert await supervisor.sweep_expired(
        user_id="user-lease", sweep_now=clock.now,
    ) == [session.id]
    assert await supervisor.terminate_expired_background(
        session_id=session.id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is False
    assert redis.system_count == 0
    assert ("supervisor:bg:user-lease", session.id) not in redis.scores
    assert ("supervisor:user:user-lease", session.id) not in redis.hashes


async def test_watchdog_terminal_revision_clears_superseded_execution_outbox(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, expires_in=timedelta(seconds=-1))
    session.execution_revision = 1
    repo = _LeaseRepo(session)
    redis = _LeaseRedis()
    assert session.expires_at is not None
    redis.hashes[("supervisor:user:user-lease", session.id)] = (
        f"{session.expires_at.timestamp():.6f}"
    )
    redis.scores[("supervisor:bg:user-lease", session.id)] = (
        session.expires_at.timestamp()
    )
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "1"
    redis.hashes[("supervisor:system:bg-members", session.id)] = "1"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        session_state_machine=_ExpiredCasStateMachine(),
        utcnow=lambda: clock.now,
    )
    session.pending_execution_event = supervisor._pending_execution_event(
        execution_revision=1,
        execution_mode="background",
        execution_phase="running",
        transition_reason="auto_degrade_sse_disconnect",
        background_reason="auto_degrade",
        expires_at=session.expires_at,
        retry_budget_remaining=3,
    )

    assert await supervisor.terminate_expired_background(
        session_id=session.id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is True
    assert session.status == SessionStatus.TIMED_OUT
    assert session.execution_revision == 2
    assert session.pending_execution_event is None


class _MissingLeaseRepo:
    async def get_by_id(self, session_id: str) -> None:
        return None


class _MissingBootLeaseRepo(_MissingLeaseRepo):
    async def find_running_background(self) -> list:
        return []


async def test_global_reconcile_cleans_deleted_pg_row_after_delete_cleanup_failure(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes[("supervisor:bg-generation:user-lease", session_id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session_id)] = (
        "v1|7|user-lease"
    )
    redis.system_count = 1
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )

    summary = await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert summary == {"cleaned": 0, "repaired": 0, "skipped": 0}
    assert redis.system_count == 1

    clock.advance(timedelta(seconds=61))
    summary = await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert summary == {"cleaned": 1, "repaired": 0, "skipped": 0}
    assert redis.system_count == 0
    assert ("supervisor:user:user-lease", session_id) not in redis.hashes
    assert ("supervisor:bg-generation:user-lease", session_id) not in redis.hashes
    assert ("supervisor:system:bg-members", session_id) not in redis.hashes


async def test_boot_reconcile_cleans_global_only_legacy_membership_idempotently(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes.pop(("supervisor:user:user-lease", session_id), None)
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session_id), None)
    redis.scores.pop(("supervisor:bg:user-lease", session_id), None)
    redis.hashes[("supervisor:system:bg-members", session_id)] = "7"
    redis.system_count = 1
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingBootLeaseRepo(),
        utcnow=lambda: clock.now,
    )

    await supervisor.reconcile_running_background_at_boot()
    assert redis.system_count == 1
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_running_background_at_boot()
    await supervisor.reconcile_running_background_at_boot()

    assert redis.system_count == 0
    assert ("supervisor:system:bg-members", session_id) not in redis.hashes


async def test_global_only_structured_membership_is_cleaned_exactly_once() -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes.pop(("supervisor:user:user-lease", session_id), None)
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session_id), None)
    redis.scores.pop(("supervisor:bg:user-lease", session_id), None)
    redis.hashes[("supervisor:system:bg-members", session_id)] = (
        "v1|7|user-lease"
    )
    redis.system_count = 1
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )

    held = await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    clock.advance(timedelta(seconds=61))
    first = await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    second = await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert held == {"cleaned": 0, "repaired": 0, "skipped": 0}
    assert first == {"cleaned": 1, "repaired": 0, "skipped": 0}
    assert second == {"cleaned": 0, "repaired": 0, "skipped": 0}
    assert redis.system_count == 0


async def test_global_reconcile_repairs_live_background_projection_without_recount(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, expires_in=timedelta(hours=3))
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes.pop(("supervisor:user:user-lease", session.id), None)
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session.id), None)
    redis.scores.pop(("supervisor:bg:user-lease", session.id), None)
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    redis.system_count = 1
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )

    summary = await supervisor.reconcile_global_background_memberships()
    quota = await supervisor.get_background_quota("user-lease")

    assert summary == {"cleaned": 0, "repaired": 1, "skipped": 0}
    assert redis.system_count == 1
    assert quota["system_used"] == 1
    assert quota["user_used"] == 1
    assert redis.hashes[("supervisor:bg-generation:user-lease", session.id)] == "7"
    assert redis.scores[("supervisor:bg:user-lease", session.id)] == (
        session.expires_at.timestamp()
    )


async def test_global_reconcile_restores_expired_score_for_authoritative_cas(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, expires_in=timedelta(seconds=-1))
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes.pop(("supervisor:user:user-lease", session.id), None)
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session.id), None)
    redis.scores.pop(("supervisor:bg:user-lease", session.id), None)
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    redis.system_count = 1
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        session_state_machine=_ExpiredCasStateMachine(),
        utcnow=lambda: clock.now,
    )

    assert await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    ) == {"cleaned": 0, "repaired": 1, "skipped": 0}
    assert await supervisor.sweep_expired(
        user_id="user-lease",
        sweep_now=clock.now,
    ) == [session.id]
    assert await supervisor.terminate_expired_background(
        session_id=session.id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is True
    assert session.status == SessionStatus.TIMED_OUT
    assert redis.system_count == 0


async def test_global_reconcile_does_not_touch_newer_membership_generation(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|8|user-lease"
    )
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "8"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )

    before_hashes = dict(redis.hashes)
    before_scores = dict(redis.scores)

    summary = await supervisor.reconcile_global_background_memberships()

    assert summary == {"cleaned": 0, "repaired": 0, "skipped": 1}
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == (
        "v1|8|user-lease"
    )
    assert redis.hashes == before_hashes
    assert redis.scores == before_scores
    assert redis.system_count == 1


async def test_global_cleanup_scan_snapshot_cannot_delete_newer_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _LeaseRedis()
    redis.hashes.pop(("supervisor:user:user-lease", "session-lease"), None)
    redis.hashes.pop(("supervisor:bg-generation:user-lease", "session-lease"), None)
    redis.scores.pop(("supervisor:bg:user-lease", "session-lease"), None)
    redis.hashes[("supervisor:system:bg-members", "session-lease")] = (
        "v1|8|user-lease"
    )
    redis.system_count = 1
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
    )

    async def stale_scan() -> list[tuple[str, str | None, int]]:
        return [("session-lease", "user-lease", 7)]

    monkeypatch.setattr(
        supervisor,
        "_scan_global_background_memberships",
        stale_scan,
    )

    summary = await supervisor.reconcile_global_background_memberships()

    assert summary == {"cleaned": 0, "repaired": 0, "skipped": 1}
    assert redis.hashes[("supervisor:system:bg-members", "session-lease")] == (
        "v1|8|user-lease"
    )
    assert redis.system_count == 1


async def test_global_reconcile_waits_for_inflight_mode_transition_fence() -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    redis.hashes.pop(("supervisor:user:user-lease", "session-lease"), None)
    redis.hashes.pop(("supervisor:bg-generation:user-lease", "session-lease"), None)
    redis.scores.pop(("supervisor:bg:user-lease", "session-lease"), None)
    redis.hashes[("supervisor:system:bg-members", "session-lease")] = (
        "v1|7|user-lease"
    )
    redis.system_count = 1
    owner = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )
    reconciler = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )

    async with owner.mode_transition_fence(session_id="session-lease"):
        task = asyncio.create_task(
            reconciler.reconcile_global_background_memberships()
        )
        await asyncio.sleep(0.02)
        assert task.done() is False
        assert redis.system_count == 1

    assert await task == {"cleaned": 0, "repaired": 0, "skipped": 0}
    assert redis.system_count == 1
    clock.advance(timedelta(seconds=61))
    assert await reconciler.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    ) == {"cleaned": 1, "repaired": 0, "skipped": 0}
    assert redis.system_count == 0


async def test_global_reconcile_restores_pg_commit_during_held_quarantine() -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 1
    repo = _LeaseRepo(session)
    redis = _LeaseRedis()
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|1|user-lease"
    )
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "1"
    redis.system_count = 1
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        utcnow=lambda: clock.now,
    )
    assert await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    ) == {"cleaned": 0, "repaired": 0, "skipped": 0}
    assert redis.system_count == 1

    session.execution_mode = "background"
    session.execution_phase = "running"
    session.execution_revision = 1
    session.expires_at = clock.now + timedelta(hours=2)

    summary = await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert summary == {"cleaned": 0, "repaired": 1, "skipped": 0}
    assert redis.system_count == 1
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == (
        "v1|1|user-lease"
    )


async def test_reconcile_marker_repairs_pg_commit_after_post_revoke_read(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )

    assert await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    ) == {"cleaned": 0, "repaired": 0, "skipped": 0}
    assert redis.system_count == 1
    assert ("supervisor:system:bg-members", session.id) in redis.hashes
    assert (
        "supervisor:system:bg-reconcile-pending",
        session.id,
    ) in redis.hashes

    clock.advance(timedelta(seconds=61))
    assert await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    ) == {"cleaned": 1, "repaired": 0, "skipped": 0}
    assert redis.system_count == 0
    assert ("supervisor:system:bg-members", session.id) not in redis.hashes

    # The ambiguous transaction becomes visible only after the reconciler's
    # post-revoke PG read has already returned.
    session.execution_mode = "background"
    session.execution_phase = "running"
    session.expires_at = clock.now + timedelta(hours=2)

    restarted = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )
    assert await restarted.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    ) == {"cleaned": 0, "repaired": 1, "skipped": 0}
    assert redis.system_count == 1
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == (
        "v1|7|user-lease"
    )
    assert (
        "supervisor:system:bg-reconcile-pending",
        session.id,
    ) not in redis.hashes


async def test_reconcile_marker_is_idempotent_and_expires_after_cleanup_grace(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", "session-lease")] = "7"
    redis.hashes[("supervisor:system:bg-members", "session-lease")] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )
    marker = ("supervisor:system:bg-reconcile-pending", "session-lease")

    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    first_marker = redis.hashes[marker]
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert redis.hashes[marker] == first_marker
    assert redis.system_count == 1

    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    released_marker = redis.hashes[marker]
    assert released_marker != first_marker
    assert "|released|" in released_marker
    assert redis.system_count == 0

    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    assert redis.hashes[marker] == released_marker
    assert redis.system_count == 0

    clock.advance(timedelta(hours=25))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert marker not in redis.hashes
    assert redis.system_count == 0


async def test_old_reconcile_marker_does_not_touch_live_newer_generation(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )

    session.execution_mode = "foreground"
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    session.execution_mode = "background"
    session.execution_phase = "running"
    session.execution_revision = 8
    session.expires_at = clock.now + timedelta(hours=2)
    assert session.expires_at is not None
    redis.hashes[("supervisor:user:user-lease", session.id)] = (
        f"{session.expires_at.timestamp():.6f}"
    )
    redis.scores[("supervisor:bg:user-lease", session.id)] = (
        session.expires_at.timestamp()
    )
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "8"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|8|user-lease"
    )
    redis.system_count = 1

    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert redis.system_count == 1
    assert redis.hashes[("supervisor:bg-generation:user-lease", session.id)] == "8"
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == (
        "v1|8|user-lease"
    )
    assert (
        "supervisor:system:bg-reconcile-pending",
        session.id,
    ) not in redis.hashes


async def test_numeric_legacy_with_live_per_user_keys_uses_full_revoke() -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes[("supervisor:bg-generation:user-lease", session_id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session_id)] = "7"
    redis.system_count = 1
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )

    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    assert redis.system_count == 1

    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert redis.system_count == 0
    assert ("supervisor:user:user-lease", session_id) not in redis.hashes
    assert ("supervisor:bg-generation:user-lease", session_id) not in redis.hashes
    assert ("supervisor:bg:user-lease", session_id) not in redis.scores
    assert ("supervisor:system:bg-members", session_id) not in redis.hashes


async def test_normal_revoke_atomically_transitions_held_marker_to_released(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )

    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    assert redis.system_count == 1

    assert await supervisor._lua_revoke(
        session_id=session.id,
        user_id="user-lease",
        reason="normal_terminal_during_quarantine",
        expected_generation=7,
        allow_legacy=True,
    ) == 1

    marker = redis.hashes[
        ("supervisor:system:bg-reconcile-pending", session.id)
    ]
    assert "|released|" in marker
    assert redis.system_count == 0
    assert ("supervisor:system:bg-members", session.id) not in redis.hashes


async def test_released_marker_prevents_legacy_user_cleanup_double_decrement(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    assert redis.system_count == 0

    # A legacy per-user residue reappears while another real session owns the
    # sole system count. The released marker proves this session's count was
    # already decremented by quarantine release.
    redis.system_count = 1
    redis.hashes[("supervisor:user:user-lease", session.id)] = "0.000000"
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.scores[("supervisor:bg:user-lease", session.id)] = 0.0

    assert await supervisor._lua_revoke(
        session_id=session.id,
        user_id="user-lease",
        reason="released_marker_legacy_residue",
        expected_generation=7,
        allow_legacy=True,
    ) == 1

    assert redis.system_count == 1
    assert ("supervisor:user:user-lease", session.id) not in redis.hashes


async def test_successful_sync_clears_same_generation_reconcile_marker() -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    assert (
        "supervisor:system:bg-reconcile-pending",
        session.id,
    ) in redis.hashes

    assert await supervisor._sync_background_expiry_to_redis(
        session_id=session.id,
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2),
        execution_revision=7,
    ) == 1

    assert (
        "supervisor:system:bg-reconcile-pending",
        session.id,
    ) not in redis.hashes


async def test_successful_newer_admit_clears_released_reconcile_marker() -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert await supervisor._run_lua_admit(
        session_id=session.id,
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2),
        generation=8,
    ) == 0

    assert redis.system_count == 1
    assert (
        "supervisor:system:bg-reconcile-pending",
        session.id,
    ) not in redis.hashes


async def test_released_marker_restore_bypasses_full_cap_to_reflect_pg_truth(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        max_system_bg=1,
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    assert redis.system_count == 0

    # Another session fills the cap before the ambiguous PG transaction lands.
    redis.system_count = 1
    session.execution_mode = "background"
    session.execution_phase = "running"
    session.expires_at = clock.now + timedelta(hours=2)

    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert redis.system_count == 2
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == (
        "v1|7|user-lease"
    )


async def test_fresh_admit_with_released_marker_residue_still_respects_cap(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        max_system_bg=1,
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    marker_key = ("supervisor:system:bg-reconcile-pending", session.id)
    released_marker = redis.hashes[marker_key]

    # A stale per-user projection is not a counted reservation after release.
    # Another real session now owns the only available system slot.
    redis.hashes[("supervisor:user:user-lease", session.id)] = "0.000000"
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.scores[("supervisor:bg:user-lease", session.id)] = 0.0
    redis.system_count = 1

    assert await supervisor._run_lua_admit(
        session_id=session.id,
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2),
        generation=8,
    ) == 1
    assert redis.system_count == 1
    assert redis.hashes[marker_key] == released_marker
    assert ("supervisor:system:bg-members", session.id) not in redis.hashes


async def test_background_resume_restores_released_residue_as_counted_pg_truth(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 7
    repo = _LeaseRepo(session)
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        max_system_bg=1,
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    expires_at = clock.now + timedelta(hours=2)
    session.execution_mode = "background"
    session.execution_phase = "running"
    session.expires_at = expires_at
    redis.hashes[("supervisor:user:user-lease", session.id)] = "0.000000"
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.scores[("supervisor:bg:user-lease", session.id)] = 0.0
    # A different durable background session filled the cap before this
    # already-committed row became visible.
    redis.system_count = 1

    assert await supervisor.resume(
        session_id=session.id,
        user_id="user-lease",
        execution_mode="background",
        expires_at=expires_at,
        expected_execution_revision=7,
    ) == 0
    assert redis.system_count == 2
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == (
        "v1|7|user-lease"
    )
    assert (
        "supervisor:system:bg-reconcile-pending",
        session.id,
    ) not in redis.hashes


async def test_restore_marker_rejects_older_authoritative_generation(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 8
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "8"
    redis.hashes[("supervisor:system:bg-members", session.id)] = (
        "v1|8|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    marker_key = ("supervisor:system:bg-reconcile-pending", session.id)
    released_marker = redis.hashes[marker_key]
    before_count = redis.system_count

    assert await supervisor._restore_background_from_marker(
        session_id=session.id,
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2),
        generation=7,
        expected_marker=released_marker,
    ) == -1
    assert redis.system_count == before_count
    assert redis.hashes[marker_key] == released_marker
    assert ("supervisor:system:bg-members", session.id) not in redis.hashes


async def test_legacy_owner_generation_change_before_release_is_zero_write(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes[("supervisor:bg-generation:user-lease", session_id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session_id)] = "7"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    redis.hashes[("supervisor:bg-generation:user-lease", session_id)] = "8"
    before_hashes = dict(redis.hashes)
    before_scores = dict(redis.scores)
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert redis.system_count == 1
    assert redis.hashes == before_hashes
    assert redis.scores == before_scores


async def test_legacy_owner_disappearing_before_release_is_zero_write() -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes[("supervisor:bg-generation:user-lease", session_id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session_id)] = "7"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    redis.hashes.pop(("supervisor:bg-generation:user-lease", session_id), None)
    before_hashes = dict(redis.hashes)
    before_scores = dict(redis.scores)
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert redis.system_count == 1
    assert redis.hashes == before_hashes
    assert redis.scores == before_scores


async def test_ambiguous_numeric_legacy_owners_do_not_choose_one() -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes[("supervisor:bg-generation:user-lease", session_id)] = "7"
    redis.hashes[("supervisor:user:user-other", session_id)] = "0.000000"
    redis.hashes[("supervisor:bg-generation:user-other", session_id)] = "7"
    redis.scores[("supervisor:bg:user-other", session_id)] = 0.0
    redis.hashes[("supervisor:system:bg-members", session_id)] = "7"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )

    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    clock.advance(timedelta(seconds=61))
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )

    assert redis.system_count == 0
    assert ("supervisor:system:bg-members", session_id) not in redis.hashes
    assert ("supervisor:bg-generation:user-lease", session_id) in redis.hashes
    assert ("supervisor:bg-generation:user-other", session_id) in redis.hashes


async def test_reconcile_marker_token_aba_makes_old_gc_and_release_noop() -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", "session-lease")] = "7"
    redis.hashes[("supervisor:system:bg-members", "session-lease")] = (
        "v1|7|user-lease"
    )
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
        utcnow=lambda: clock.now,
    )
    await supervisor.reconcile_global_background_memberships(
        reconcile_now=clock.now,
    )
    marker_key = "supervisor:system:bg-reconcile-pending"
    old_raw = redis.hashes[(marker_key, "session-lease")]
    old_marker = supervisor._decode_background_reconcile_marker(old_raw)
    assert old_marker is not None
    new_raw = old_raw.replace(old_marker.token, "replacement-token")
    redis.hashes[(marker_key, "session-lease")] = new_raw

    assert await supervisor._gc_background_reconcile_marker(
        session_id="session-lease",
        expected_marker=old_raw,
    ) == 0
    assert await supervisor._release_held_background_reconcile(
        session_id="session-lease",
        marker=old_marker,
        expected_marker=old_raw,
        now=clock.now + timedelta(seconds=61),
        allow_missing_membership=False,
    ) == 0

    assert redis.hashes[(marker_key, "session-lease")] == new_raw
    assert redis.system_count == 1


async def test_missing_pg_row_terminal_cleanup_revokes_fallback_projection_once(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes[("supervisor:bg-generation:user-lease", session_id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session_id)] = "7"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
    )

    assert await supervisor.terminate_expired_background(
        session_id=session_id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is False
    assert redis.system_count == 0
    assert ("supervisor:user:user-lease", session_id) not in redis.hashes
    assert ("supervisor:bg-generation:user-lease", session_id) not in redis.hashes
    assert ("supervisor:system:bg-members", session_id) not in redis.hashes

    assert await supervisor.terminate_expired_background(
        session_id=session_id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is False
    assert redis.system_count == 0


async def test_missing_pg_row_uses_system_membership_generation_fallback() -> None:
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session_id), None)
    redis.hashes[("supervisor:system:bg-members", session_id)] = "7"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
    )

    assert await supervisor.terminate_expired_background(
        session_id=session_id,
        user_id="user-lease",
        sweep_now=_Clock().now,
    ) is False
    assert redis.system_count == 0
    assert ("supervisor:user:user-lease", session_id) not in redis.hashes
    assert ("supervisor:system:bg-members", session_id) not in redis.hashes


async def test_missing_pg_row_sweep_repair_revokes_fallback_projection_once(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes[("supervisor:bg-generation:user-lease", session_id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session_id)] = "7"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_MissingLeaseRepo(),
    )

    await supervisor._repair_swept_background_projection(
        session_id=session_id,
        fallback_user_id="user-lease",
        sweep_now=clock.now,
    )
    assert redis.system_count == 0
    assert ("supervisor:user:user-lease", session_id) not in redis.hashes
    assert ("supervisor:bg-generation:user-lease", session_id) not in redis.hashes
    assert ("supervisor:system:bg-members", session_id) not in redis.hashes

    await supervisor._repair_swept_background_projection(
        session_id=session_id,
        fallback_user_id="user-lease",
        sweep_now=clock.now,
    )
    assert redis.system_count == 0


class _DeleteDuringExpiredCasRepo(_LeaseRepo):
    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self.deleted = False

    async def get_by_id(self, session_id: str) -> Session | None:
        if self.deleted:
            return None
        return await super().get_by_id(session_id)

    async def update_to_terminal_if_background_expired(
        self,
        session_id: str,
        status: SessionStatus,
        terminal_reason: str,
        **kwargs,
    ) -> int | None:
        self.deleted = True
        return None


async def test_pg_row_deleted_during_expiry_cas_still_revokes_projection_once(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(
        clock,
        expires_in=timedelta(seconds=-1),
    )
    session.execution_revision = 7
    repo = _DeleteDuringExpiredCasRepo(session)
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = "7"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        session_state_machine=_ExpiredCasStateMachine(),
        utcnow=lambda: clock.now,
    )

    assert await supervisor.terminate_expired_background(
        session_id=session.id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is False
    assert redis.system_count == 0
    assert ("supervisor:bg-generation:user-lease", session.id) not in redis.hashes
    assert ("supervisor:system:bg-members", session.id) not in redis.hashes

    assert await supervisor.terminate_expired_background(
        session_id=session.id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is False
    assert redis.system_count == 0


@pytest.mark.parametrize(
    "status,execution_mode",
    [
        (SessionStatus.RUNNING, "foreground"),
        (SessionStatus.COMPLETED, "background"),
    ],
)
async def test_failed_expiry_cas_does_not_reproject_foreground_or_terminal_session(
    status: SessionStatus,
    execution_mode: str,
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(
        clock,
        status=status,
        execution_mode=execution_mode,
        expires_in=timedelta(hours=1),
    )
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        session_state_machine=_ExpiredCasStateMachine(),
        utcnow=lambda: clock.now,
    )

    assert await supervisor.terminate_expired_background(
        session_id=session.id,
        user_id="user-lease",
        sweep_now=clock.now,
    ) is False
    assert len(redis.evalsha_calls) == 1
    assert redis.evalsha_calls[0][1] == 7
    assert ("supervisor:user:user-lease", session.id) not in redis.hashes
    assert ("supervisor:bg:user-lease", session.id) not in redis.scores


async def test_redis_expiry_projection_is_one_atomic_lua_operation() -> None:
    clock = _Clock()
    repo = _LeaseRepo(_auto_degrade_session(clock))
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=repo,
        utcnow=lambda: clock.now,
    )

    assert await supervisor.renew_auto_degrade_expiry_if_running(
        session_id=repo.session.id,
    ) is True

    assert len(redis.evalsha_calls) == 1
    assert redis.evalsha_calls[0][1] == 6
    assert redis.hset_calls == []
    assert redis.zadd_calls == []
    assert redis.expire_calls == []


def test_released_residue_lua_declares_expiry_score_before_closure() -> None:
    assert LUA_ADMIT_SOURCE.index("local expires_score = tonumber(expires_at)") < (
        LUA_ADMIT_SOURCE.index("local function admit_released_residue()")
    )


@pytest.mark.parametrize("reconnect_between_commit_and_project", [False, True])
async def test_renew_reconnect_interleavings_never_resurrect_background_slot(
    reconnect_between_commit_and_project: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    repo = _LeaseRepo(_auto_degrade_session(clock))
    redis_slot = {"present": False}
    projection_entered = asyncio.Event()
    allow_projection = asyncio.Event()
    redis = _LeaseRedis()
    renewer = ExecutionSupervisor(
        redis_client=redis, session_repository=repo, utcnow=lambda: clock.now,
    )
    reconnect = ExecutionSupervisor(redis_client=redis, session_repository=repo)

    async def _project(**kwargs) -> int:
        projection_entered.set()
        if reconnect_between_commit_and_project:
            await allow_projection.wait()
        redis_slot["present"] = True
        return 1

    async def _revoke(**kwargs) -> int:
        redis_slot["present"] = False
        return 1

    monkeypatch.setattr(renewer, "_sync_background_expiry_to_redis", _project)
    monkeypatch.setattr(renewer, "_redis_expiry_matches", AsyncMock(return_value=False))
    monkeypatch.setattr(renewer, "_lua_revoke", _revoke)
    monkeypatch.setattr(reconnect, "_lua_revoke", _revoke)

    if reconnect_between_commit_and_project:
        renew_task = asyncio.create_task(
            renewer.renew_auto_degrade_expiry_if_running(session_id=repo.session.id)
        )
        await asyncio.wait_for(projection_entered.wait(), timeout=1)
        await reconnect.resume(
            session_id=repo.session.id,
            user_id="user-lease",
            execution_mode="foreground",
        )
        allow_projection.set()
        await asyncio.wait_for(renew_task, timeout=1)
    else:
        await renewer.renew_auto_degrade_expiry_if_running(
            session_id=repo.session.id,
        )
        await reconnect.resume(
            session_id=repo.session.id,
            user_id="user-lease",
            execution_mode="foreground",
        )

    assert repo.session.execution_mode == "foreground"
    assert redis_slot["present"] is False


def _clear_counted_projection(redis: _LeaseRedis) -> None:
    session_id = "session-lease"
    redis.hashes.pop(("supervisor:user:user-lease", session_id), None)
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session_id), None)
    redis.hashes.pop(("supervisor:system:bg-members", session_id), None)
    redis.scores.pop(("supervisor:bg:user-lease", session_id), None)
    redis.system_count = 0


async def test_explicit_long_deadline_ttl_covers_expiry_plus_cleanup_grace(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    _clear_counted_projection(redis)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
        utcnow=lambda: clock.now,
    )
    expiry = clock.now + timedelta(hours=72)

    assert await supervisor._run_lua_admit(
        session_id="session-lease",
        user_id="user-lease",
        expires_at=expiry,
        generation=1,
    ) == 0
    quota = await supervisor.get_background_quota("user-lease")

    minimum_ttl = int((expiry - clock.now).total_seconds()) + 86400
    assert redis.ttls["supervisor:user:user-lease"] >= minimum_ttl
    assert redis.ttls["supervisor:bg:user-lease"] >= minimum_ttl
    assert redis.ttls["supervisor:bg-generation:user-lease"] >= minimum_ttl
    assert quota["user_used"] == 1


async def test_shorter_same_user_admission_cannot_shrink_long_slot_key_ttls(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    _clear_counted_projection(redis)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
        utcnow=lambda: clock.now,
    )
    long_expiry = clock.now + timedelta(hours=72)
    short_expiry = clock.now + timedelta(hours=2)

    assert await supervisor._run_lua_admit(
        session_id="session-long",
        user_id="user-lease",
        expires_at=long_expiry,
        generation=1,
    ) == 0
    long_ttls = dict(redis.ttls)
    assert await supervisor._run_lua_admit(
        session_id="session-short",
        user_id="user-lease",
        expires_at=short_expiry,
        generation=1,
    ) == 0
    assert await supervisor._sync_background_expiry_to_redis(
        session_id="session-short",
        user_id="user-lease",
        expires_at=short_expiry,
        execution_revision=1,
    ) == 1

    for key in (
        "supervisor:user:user-lease",
        "supervisor:bg:user-lease",
        "supervisor:bg-generation:user-lease",
    ):
        assert redis.ttls[key] >= long_ttls[key]


async def test_live_legacy_numeric_membership_is_migrated_to_structured_value(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, expires_in=timedelta(minutes=45))
    session.execution_revision = 7
    assert session.expires_at is not None
    redis = _LeaseRedis()
    redis.hashes[("supervisor:user:user-lease", session.id)] = (
        f"{session.expires_at.timestamp():.6f}"
    )
    redis.scores[("supervisor:bg:user-lease", session.id)] = (
        session.expires_at.timestamp()
    )
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = "7"
    for key in (
        "supervisor:user:user-lease",
        "supervisor:bg:user-lease",
        "supervisor:bg-generation:user-lease",
    ):
        redis.ttls[key] = 90000
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        utcnow=lambda: clock.now,
    )

    assert await supervisor._redis_expiry_matches(
        session_id=session.id,
        user_id="user-lease",
        expires_at=session.expires_at,
        execution_revision=7,
    ) is False
    assert await supervisor.renew_auto_degrade_expiry_if_running(
        session_id=session.id,
    ) is True
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == (
        "v1|7|user-lease"
    )


async def test_crash_after_redis_reserve_before_pg_is_swept_without_count_leak(
) -> None:
    clock = _Clock()
    repo = _LeaseRepo(_auto_degrade_session(clock, execution_mode="foreground"))
    redis = _LeaseRedis()
    _clear_counted_projection(redis)
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=repo)
    expiry = clock.now + timedelta(hours=2)

    assert await supervisor._run_lua_admit(
        session_id=repo.session.id,
        user_id="user-lease",
        expires_at=expiry,
        generation=1,
    ) == 0
    assert redis.system_count == 1

    await supervisor._revoke_authoritative_slot(
        session_id=repo.session.id,
        user_id="user-lease",
        reason="crash_before_pg",
    )

    assert repo.session.execution_mode == "foreground"
    assert redis.system_count == 0
    assert ("supervisor:system:bg-members", repo.session.id) not in redis.hashes


@pytest.mark.parametrize("promotion_wins_before_revoke", [False, True])
async def test_cleanup_racing_same_generation_promotion_converges_one_counted_slot(
    promotion_wins_before_revoke: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    repo = _LeaseRepo(_auto_degrade_session(clock, execution_mode="foreground"))
    redis = _LeaseRedis()
    _clear_counted_projection(redis)
    cleanup = ExecutionSupervisor(redis_client=redis, session_repository=repo)
    promotion = ExecutionSupervisor(redis_client=redis, session_repository=repo)
    expiry = clock.now + timedelta(hours=2)
    assert await cleanup._run_lua_admit(
        session_id=repo.session.id, user_id="user-lease",
        expires_at=expiry, generation=1,
    ) == 0

    if promotion_wins_before_revoke:
        assert await promotion.promote(
            session_id=repo.session.id,
            user_id="user-lease",
            expires_at=expiry,
        ) == 3
        await cleanup._revoke_authoritative_slot(
            session_id=repo.session.id,
            user_id="user-lease",
            reason="cleanup_after_pg",
            restore_if_background=True,
        )
    else:
        removed = asyncio.Event()
        allow_cleanup_reread = asyncio.Event()
        original_revoke = cleanup._lua_revoke

        async def revoke_then_pause(**kwargs) -> int:
            rc = await original_revoke(**kwargs)
            removed.set()
            await allow_cleanup_reread.wait()
            return rc

        monkeypatch.setattr(cleanup, "_lua_revoke", revoke_then_pause)
        cleanup_task = asyncio.create_task(cleanup._revoke_authoritative_slot(
            session_id=repo.session.id,
            user_id="user-lease",
            reason="cleanup_before_pg",
            restore_if_background=True,
        ))
        await removed.wait()
        assert await promotion.promote(
            session_id=repo.session.id,
            user_id="user-lease",
            expires_at=expiry,
        ) == 3
        allow_cleanup_reread.set()
        await cleanup_task

    assert repo.session.execution_mode == "background"
    assert repo.session.execution_revision == 1
    assert redis.system_count == 1
    assert redis.hashes[("supervisor:system:bg-members", repo.session.id)] == (
        "v1|1|user-lease"
    )
    assert redis.hashes[("supervisor:bg-generation:user-lease", repo.session.id)] == "1"


class _DeleteProjectionAfterPromoteRepo(_LeaseRepo):
    def __init__(self, session: Session, redis: _LeaseRedis) -> None:
        super().__init__(session)
        self.redis = redis

    async def promote_foreground_to_background(self, *args, **kwargs) -> int | None:
        revision = await super().promote_foreground_to_background(*args, **kwargs)
        _clear_counted_projection(self.redis)
        return revision


async def test_promotion_postcondition_restores_concurrently_deleted_slot_once(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    _clear_counted_projection(redis)
    repo = _DeleteProjectionAfterPromoteRepo(
        _auto_degrade_session(clock, execution_mode="foreground"), redis
    )
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=repo)

    assert await supervisor.promote(
        session_id=repo.session.id,
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2),
    ) == 3
    assert repo.session.execution_revision == 1
    assert redis.system_count == 1
    assert redis.hashes[("supervisor:system:bg-members", repo.session.id)] == (
        "v1|1|user-lease"
    )


async def test_projection_repair_counts_complete_loss_but_not_partial_membership_loss(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    session.execution_revision = 4
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=_LeaseRepo(session))
    expiry = session.expires_at
    assert expiry is not None

    _clear_counted_projection(redis)
    await supervisor._ensure_authoritative_background_projection(
        session_id=session.id, user_id="user-lease",
        expires_at=expiry, generation=4,
    )
    assert redis.system_count == 1

    redis.hashes.pop(("supervisor:system:bg-members", session.id), None)
    await supervisor._ensure_authoritative_background_projection(
        session_id=session.id, user_id="user-lease",
        expires_at=expiry + timedelta(minutes=1), generation=4,
    )
    assert redis.system_count == 1
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == (
        "v1|4|user-lease"
    )


async def test_authoritative_repair_upgrades_generationless_counted_projection(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    session.execution_revision = 4
    redis = _LeaseRedis()
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session.id), None)
    redis.hashes.pop(("supervisor:system:bg-members", session.id), None)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        max_system_bg=1,
        max_user_bg=1,
    )
    expiry = clock.now + timedelta(hours=2)

    await supervisor._ensure_authoritative_background_projection(
        session_id=session.id,
        user_id="user-lease",
        expires_at=expiry,
        generation=4,
    )

    assert redis.system_count == 1
    assert redis.hashes[("supervisor:bg-generation:user-lease", session.id)] == "4"
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == (
        "v1|4|user-lease"
    )
    assert redis.hashes[("supervisor:user:user-lease", session.id)] == (
        f"{expiry.timestamp():.6f}"
    )

    existing_branch = LUA_ADMIT_SOURCE.split(
        "if redis.call('HEXISTS', user_key, session_id) == 1 then", 1
    )[1].split("if generation < current_generation", 1)[0]
    assert "authoritative_repair ~= 1" in existing_branch
    assert existing_branch.index("member_generation > generation") < (
        existing_branch.index("redis.call('HSET', generation_key")
    )


async def test_new_generation_wins_before_legacy_repair_without_old_overwrite(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session_id), None)
    redis.hashes.pop(("supervisor:system:bg-members", session_id), None)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
    )
    winner_expiry = clock.now + timedelta(hours=3)

    assert await supervisor._run_lua_admit(
        session_id=session_id,
        user_id="user-lease",
        expires_at=winner_expiry,
        generation=5,
        authoritative_repair=True,
    ) == 3
    winner_projection = dict(redis.hashes)
    winner_scores = dict(redis.scores)

    assert await supervisor._run_lua_admit(
        session_id=session_id,
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2),
        generation=4,
        authoritative_repair=True,
    ) == 4
    assert redis.hashes == winner_projection
    assert redis.scores == winner_scores
    assert redis.system_count == 1


async def test_new_generation_replaces_completed_legacy_repair_without_recount(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session_id), None)
    redis.hashes.pop(("supervisor:system:bg-members", session_id), None)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
    )

    assert await supervisor._run_lua_admit(
        session_id=session_id,
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2),
        generation=4,
        authoritative_repair=True,
    ) == 3
    assert await supervisor._run_lua_admit(
        session_id=session_id,
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=3),
        generation=5,
    ) == 5
    assert redis.hashes[("supervisor:bg-generation:user-lease", session_id)] == "5"
    assert redis.hashes[("supervisor:system:bg-members", session_id)] == (
        "v1|5|user-lease"
    )
    assert redis.system_count == 1


async def test_generationless_authoritative_repair_is_exactly_once(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    session_id = "session-lease"
    redis.hashes.pop(("supervisor:bg-generation:user-lease", session_id), None)
    redis.hashes.pop(("supervisor:system:bg-members", session_id), None)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
    )
    expiry = clock.now + timedelta(hours=2)

    assert await supervisor._run_lua_admit(
        session_id=session_id,
        user_id="user-lease",
        expires_at=expiry,
        generation=4,
        authoritative_repair=True,
    ) == 3
    first_projection = dict(redis.hashes)
    assert await supervisor._run_lua_admit(
        session_id=session_id,
        user_id="user-lease",
        expires_at=expiry,
        generation=4,
        authoritative_repair=True,
    ) == 3
    assert redis.hashes == first_projection
    assert redis.system_count == 1


async def test_complete_projection_loss_respects_cap_and_newer_generation_rejects_old_repair(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    session.execution_revision = 1
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
        max_system_bg=1,
    )
    expiry = session.expires_at
    assert expiry is not None

    _clear_counted_projection(redis)
    redis.system_count = 1
    with pytest.raises(RuntimeError, match="cannot safely repair"):
        await supervisor._ensure_authoritative_background_projection(
            session_id=session.id, user_id="user-lease",
            expires_at=expiry, generation=1,
        )
    assert ("supervisor:user:user-lease", session.id) not in redis.hashes

    redis.system_count = 1
    redis.hashes[("supervisor:user:user-lease", session.id)] = "999.000000"
    redis.scores[("supervisor:bg:user-lease", session.id)] = 999.0
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "2"
    redis.hashes[("supervisor:system:bg-members", session.id)] = "2"
    with pytest.raises(ModeTransitionFenceLostError):
        await supervisor._ensure_authoritative_background_projection(
            session_id=session.id, user_id="user-lease",
            expires_at=expiry, generation=1,
        )
    assert redis.hashes[("supervisor:user:user-lease", session.id)] == "999.000000"


async def test_generation_admit_stale_is_zero_write_and_newer_replaces_without_double_count(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
    )
    original_count = redis.system_count
    original_expiry = redis.hashes[("supervisor:user:user-lease", "session-lease")]

    assert await supervisor._run_lua_admit(
        session_id="session-lease", user_id="user-lease",
        expires_at=clock.now, generation=-1,
    ) == 4
    assert redis.hashes[("supervisor:user:user-lease", "session-lease")] == original_expiry
    assert redis.system_count == original_count

    assert await supervisor._run_lua_admit(
        session_id="session-lease", user_id="user-lease",
        expires_at=clock.now + timedelta(hours=1), generation=1,
    ) == 5
    assert redis.system_count == original_count
    same_expiry = redis.hashes[("supervisor:user:user-lease", "session-lease")]
    assert await supervisor._run_lua_admit(
        session_id="session-lease", user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2), generation=1,
    ) == 3
    assert redis.hashes[("supervisor:user:user-lease", "session-lease")] == same_expiry


async def test_stale_authoritative_repair_with_missing_membership_is_zero_write(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", "session-lease")] = "2"
    redis.hashes.pop(("supervisor:system:bg-members", "session-lease"), None)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
    )
    before = dict(redis.hashes)

    assert await supervisor._run_lua_admit(
        session_id="session-lease",
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2),
        generation=1,
        authoritative_repair=True,
    ) == 4
    assert redis.hashes == before
    assert redis.system_count == 1

    existing_branch = LUA_ADMIT_SOURCE.split(
        "if redis.call('HEXISTS', user_key, session_id) == 1 then", 1
    )[1].split("local user_count", 1)[0]
    assert existing_branch.index("generation < current_generation") < (
        existing_branch.index("if generation == current_generation")
    )


async def test_authoritative_repair_upgrades_lower_membership_without_recount(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", "session-lease")] = "2"
    redis.hashes[("supervisor:system:bg-members", "session-lease")] = "1"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
    )
    expiry = clock.now + timedelta(hours=2)

    assert await supervisor._run_lua_admit(
        session_id="session-lease",
        user_id="user-lease",
        expires_at=expiry,
        generation=2,
    ) == 7
    assert redis.hashes[("supervisor:system:bg-members", "session-lease")] == "1"
    assert await supervisor._run_lua_admit(
        session_id="session-lease",
        user_id="user-lease",
        expires_at=expiry,
        generation=2,
        authoritative_repair=True,
    ) == 3
    assert redis.hashes[("supervisor:system:bg-members", "session-lease")] == (
        "v1|2|user-lease"
    )
    assert redis.system_count == 1
    assert await supervisor._sync_background_expiry_to_redis(
        session_id="session-lease",
        user_id="user-lease",
        expires_at=expiry,
        execution_revision=2,
    ) == 1


async def test_authoritative_repair_never_overwrites_higher_membership_generation(
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", "session-lease")] = "2"
    redis.hashes[("supervisor:system:bg-members", "session-lease")] = "3"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
    )
    before = dict(redis.hashes)

    assert await supervisor._run_lua_admit(
        session_id="session-lease",
        user_id="user-lease",
        expires_at=clock.now + timedelta(hours=2),
        generation=2,
        authoritative_repair=True,
    ) == 4
    assert redis.hashes == before
    assert redis.system_count == 1


@pytest.mark.parametrize("reason", ["watchdog_timeout", "t7_reconnect"])
async def test_authoritative_cleanup_repairs_missing_membership_exactly_once(
    reason: str,
) -> None:
    clock = _Clock()
    redis = _LeaseRedis()
    redis.hashes.pop(("supervisor:system:bg-members", "session-lease"), None)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(_auto_degrade_session(clock)),
    )

    assert await supervisor._lua_revoke(
        session_id="session-lease",
        user_id="user-lease",
        reason=reason,
        expected_generation=0,
        allow_legacy=True,
    ) == 1
    assert redis.system_count == 0
    assert await supervisor._lua_revoke(
        session_id="session-lease",
        user_id="user-lease",
        reason=reason,
        expected_generation=0,
        allow_legacy=True,
    ) == 0
    assert redis.system_count == 0


async def test_rc3_retry_rollback_is_noop_after_reconnect_revoked_generation(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 8
    redis = _LeaseRedis()
    _clear_counted_projection(redis)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
    )

    await supervisor.rollback_background_resume_admission(
        session_id=session.id,
        user_id="user-lease",
        admission_rc=3,
        previous_expires_at=clock.now + timedelta(minutes=5),
        expected_execution_revision=7,
    )

    assert redis.system_count == 0
    assert ("supervisor:user:user-lease", session.id) not in redis.hashes
    assert ("supervisor:bg:user-lease", session.id) not in redis.scores
    assert ("supervisor:bg-generation:user-lease", session.id) not in redis.hashes
    assert ("supervisor:system:bg-members", session.id) not in redis.hashes


async def test_rc3_retry_rollback_restores_expiry_with_generation_sync_lua(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_phase="suspended")
    session.execution_revision = 7
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "7"
    redis.hashes[("supervisor:system:bg-members", session.id)] = "7"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
    )
    previous_expiry = clock.now + timedelta(minutes=5)
    before_calls = len(redis.evalsha_calls)

    await supervisor.rollback_background_resume_admission(
        session_id=session.id,
        user_id="user-lease",
        admission_rc=3,
        previous_expires_at=previous_expiry,
        expected_execution_revision=7,
    )

    calls = redis.evalsha_calls[before_calls:]
    assert len(calls) == 1
    assert calls[0][1] == 6
    assert redis.hashes[("supervisor:user:user-lease", session.id)] == (
        f"{previous_expiry.timestamp():.6f}"
    )
    assert redis.scores[("supervisor:bg:user-lease", session.id)] == (
        previous_expiry.timestamp()
    )
    assert redis.system_count == 1


async def test_ambiguous_retry_admit_cancel_does_not_touch_newer_redis_owner(
    monkeypatch,
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    session.execution_revision = 7
    redis = _LeaseRedis()
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
    )
    newer_expiry = clock.now + timedelta(hours=3)

    async def ambiguous_admit(**kwargs) -> int:
        redis.hashes[("supervisor:user:user-lease", session.id)] = (
            f"{newer_expiry.timestamp():.6f}"
        )
        redis.scores[("supervisor:bg:user-lease", session.id)] = (
            newer_expiry.timestamp()
        )
        redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "8"
        redis.hashes[("supervisor:system:bg-members", session.id)] = "8"
        raise asyncio.CancelledError("ambiguous Lua result")

    monkeypatch.setattr(supervisor, "_run_lua_admit", ambiguous_admit)

    with pytest.raises(asyncio.CancelledError, match="ambiguous Lua result"):
        await supervisor.resume(
            session_id=session.id,
            user_id="user-lease",
            execution_mode="background",
            expires_at=clock.now + timedelta(hours=2),
            previous_expires_at=clock.now + timedelta(minutes=5),
            retry_budget_remaining=1,
            expected_execution_revision=7,
        )

    assert len(redis.evalsha_calls) == 1
    assert redis.evalsha_calls[0][1] == 6
    assert redis.hashes[("supervisor:user:user-lease", session.id)] == (
        f"{newer_expiry.timestamp():.6f}"
    )
    assert redis.scores[("supervisor:bg:user-lease", session.id)] == (
        newer_expiry.timestamp()
    )
    assert redis.hashes[("supervisor:bg-generation:user-lease", session.id)] == "8"
    assert redis.hashes[("supervisor:system:bg-members", session.id)] == "8"
    assert redis.system_count == 1


async def test_authoritative_revoke_cleans_lower_membership_but_rejects_higher(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    session.execution_revision = 2
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "2"
    redis.hashes[("supervisor:system:bg-members", session.id)] = "1"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
    )

    assert await supervisor._lua_revoke(
        session_id=session.id,
        user_id="user-lease",
        reason="lower_membership_cleanup",
        expected_generation=2,
        allow_legacy=True,
    ) == 1
    assert redis.system_count == 0

    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "2"
    redis.hashes[("supervisor:system:bg-members", session.id)] = "3"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
    )
    before = dict(redis.hashes)
    assert await supervisor._lua_revoke(
        session_id=session.id,
        user_id="user-lease",
        reason="higher_membership_fence",
        expected_generation=2,
        allow_legacy=True,
    ) == 0
    assert redis.hashes == before
    assert redis.system_count == 1
    assert "member_generation_number > expected_generation" in LUA_REVOKE_SOURCE


async def test_authoritative_slot_cleanup_handles_missing_membership_via_pg_proof(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock, execution_mode="foreground")
    session.execution_revision = 1
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "0"
    redis.hashes.pop(("supervisor:system:bg-members", session.id), None)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
    )

    assert await supervisor._revoke_authoritative_slot(
        session_id=session.id,
        user_id="user-lease",
        reason="pg_foreground_cleanup",
    ) == 1
    assert redis.system_count == 0
    assert ("supervisor:user:user-lease", session.id) not in redis.hashes


async def test_final_cleanup_does_not_recreate_still_visible_background_row(
) -> None:
    clock = _Clock()
    session = _auto_degrade_session(clock)
    session.execution_revision = 1
    redis = _LeaseRedis()
    redis.hashes[("supervisor:bg-generation:user-lease", session.id)] = "1"
    redis.hashes[("supervisor:system:bg-members", session.id)] = "1"
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        session_repository=_LeaseRepo(session),
    )

    await supervisor.cleanup_background_slot(
        session_id=session.id,
        user_id="user-lease",
        reason="session_delete",
    )

    # SessionService deletes the PG row only after this cleanup call.  The
    # still-visible background snapshot must not be interpreted as a repair.
    assert session.execution_mode == "background"
    assert redis.system_count == 0
    assert ("supervisor:user:user-lease", session.id) not in redis.hashes
    assert ("supervisor:bg-generation:user-lease", session.id) not in redis.hashes


async def test_foreground_reconnect_retries_failed_redis_revoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    repo = _LeaseRepo(_auto_degrade_session(clock))
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)
    revoke_attempts = 0

    async def _revoke(**kwargs) -> int:
        nonlocal revoke_attempts
        revoke_attempts += 1
        if revoke_attempts == 1:
            raise RuntimeError("redis revoke failed")
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", _revoke)

    with pytest.raises(RuntimeError, match="redis revoke failed"):
        await supervisor.resume(
            session_id=repo.session.id,
            user_id="user-lease",
            execution_mode="foreground",
        )
    assert repo.session.execution_mode == "foreground"

    assert await supervisor.resume(
        session_id=repo.session.id,
        user_id="user-lease",
        execution_mode="foreground",
    ) is None
    assert revoke_attempts == 2


async def test_foreground_reconcile_never_revokes_legal_explicit_background_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    repo = _LeaseRepo(_auto_degrade_session(clock, background_reason="explicit"))
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)
    revoke = AsyncMock(return_value=1)
    monkeypatch.setattr(supervisor, "_lua_revoke", revoke)

    assert await supervisor.resume(
        session_id=repo.session.id,
        user_id="user-lease",
        execution_mode="foreground",
    ) is None
    revoke.assert_not_awaited()


class _FenceRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.set_ttls: list[int] = []

    async def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool:
        self.set_ttls.append(ex)
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def eval(self, script: str, numkeys: int, key: str, token: str, *args) -> int:
        if self.values.get(key) != token:
            return 0
        if args:
            return 1
        del self.values[key]
        return 1


class _ExpiringFenceQuotaRedis(_LeaseRedis):
    def __init__(self) -> None:
        super().__init__()
        self.fence_values: dict[str, str] = {}

    async def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool:
        if nx and key in self.fence_values:
            return False
        self.fence_values[key] = value
        return True

    async def eval(self, script: str, numkeys: int, key: str, token: str, *args) -> int:
        if self.fence_values.get(key) != token:
            return 0
        if args:
            return 1
        del self.fence_values[key]
        return 1

    def expire_fence_key(self, session_id: str) -> None:
        self.fence_values.pop(f"supervisor:mode-transition:{session_id}", None)


class _CommitRenewFailureQuotaRedis(_ExpiringFenceQuotaRedis):
    def __init__(self) -> None:
        super().__init__()
        self.renew_entered = asyncio.Event()
        self.fail_renew = asyncio.Event()

    async def eval(
        self,
        script: str,
        numkeys: int,
        key: str,
        token: str,
        *args,
    ) -> int:
        if args:
            self.renew_entered.set()
            await self.fail_renew.wait()
            return 0
        return await super().eval(script, numkeys, key, token, *args)


async def test_fence_renew_loss_during_production_uow_commit_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervisor_module,
        "_MODE_TRANSITION_FENCE_RENEW_SECONDS",
        0,
    )
    clock = _Clock()
    durable = _auto_degrade_session(
        clock,
        execution_mode="foreground",
        background_reason=None,
    )
    state = _TransactionalPromotionState(durable)
    commit_entered = asyncio.Event()
    redis = _CommitRenewFailureQuotaRedis()
    _clear_counted_projection(redis)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        uow_factory=_ProductionLikePromotionUowFactory(
            state,
            commit_entered=commit_entered,
        ),
    )

    task = asyncio.create_task(
        supervisor.admit(
            session_id=durable.id,
            user_id="user-lease",
            execution_mode="background",
            background_reason="explicit",
            expires_at=clock.now + timedelta(hours=2),
        )
    )
    await asyncio.wait_for(commit_entered.wait(), timeout=1)
    await asyncio.wait_for(redis.renew_entered.wait(), timeout=1)
    redis.fail_renew.set()
    result = (await asyncio.gather(task, return_exceptions=True))[0]

    assert isinstance(result, ModeTransitionFenceLostError)
    assert durable.execution_mode == "foreground"
    assert durable.execution_revision == 0
    assert redis.system_count == 0
    assert ("supervisor:bg-generation:user-lease", durable.id) not in redis.hashes
    assert ("supervisor:system:bg-members", durable.id) not in redis.hashes


async def test_external_cancel_during_production_uow_commit_propagates_and_cleans_slot(
) -> None:
    clock = _Clock()
    durable = _auto_degrade_session(
        clock,
        execution_mode="foreground",
        background_reason=None,
    )
    state = _TransactionalPromotionState(durable)
    commit_entered = asyncio.Event()
    redis = _ExpiringFenceQuotaRedis()
    _clear_counted_projection(redis)
    supervisor = ExecutionSupervisor(
        redis_client=redis,
        uow_factory=_ProductionLikePromotionUowFactory(
            state,
            commit_entered=commit_entered,
        ),
    )

    task = asyncio.create_task(
        supervisor.admit(
            session_id=durable.id,
            user_id="user-lease",
            execution_mode="background",
            background_reason="explicit",
            expires_at=clock.now + timedelta(hours=2),
        )
    )
    await asyncio.wait_for(commit_entered.wait(), timeout=1)
    task.cancel("external-cancel-at-commit")
    result = (await asyncio.gather(task, return_exceptions=True))[0]

    assert isinstance(result, asyncio.CancelledError)
    assert durable.execution_mode == "foreground"
    assert durable.execution_revision == 0
    assert redis.system_count == 0
    assert ("supervisor:bg-generation:user-lease", durable.id) not in redis.hashes
    assert ("supervisor:system:bg-members", durable.id) not in redis.hashes


async def test_expired_fence_allows_new_owner_but_old_body_cannot_retransition(
) -> None:
    clock = _Clock()
    redis = _ExpiringFenceQuotaRedis()
    _clear_counted_projection(redis)
    repo = _LeaseRepo(_auto_degrade_session(clock, execution_mode="foreground"))
    old_owner = ExecutionSupervisor(redis_client=redis, session_repository=repo)
    new_owner = ExecutionSupervisor(redis_client=redis, session_repository=repo)
    old_entered = asyncio.Event()
    resume_old = asyncio.Event()
    expiry = clock.now + timedelta(hours=2)

    async def old_body() -> int | None:
        async with old_owner.mode_transition_fence(session_id=repo.session.id):
            old_entered.set()
            await resume_old.wait()
            return await old_owner.promote(
                session_id=repo.session.id,
                user_id="user-lease",
                expires_at=expiry,
            )

    old_task = asyncio.create_task(old_body())
    await old_entered.wait()
    redis.expire_fence_key(repo.session.id)
    async with new_owner.mode_transition_fence(session_id=repo.session.id):
        assert await new_owner.promote(
            session_id=repo.session.id,
            user_id="user-lease",
            expires_at=expiry,
        ) == 3
    winning_pending = repo.session.pending_execution_event
    assert winning_pending is not None
    resume_old.set()

    assert await old_task is None
    assert repo.session.execution_revision == 1
    assert repo.session.execution_mode == "background"
    assert redis.system_count == 1
    assert redis.hashes[("supervisor:system:bg-members", repo.session.id)] == (
        "v1|1|user-lease"
    )
    assert repo.session.pending_execution_event == winning_pending
    assert winning_pending.payload.execution_revision == 1


class _ReconnectReadBarrierRepo(_LeaseRepo):
    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self.reads = 0
        self.reconnect_second_read = asyncio.Event()
        self.allow_reconnect_second_read = asyncio.Event()

    async def get_by_id(self, session_id: str) -> Session | None:
        self.reads += 1
        if self.reads == 2:
            self.reconnect_second_read.set()
            await self.allow_reconnect_second_read.wait()
        return await super().get_by_id(session_id)


async def test_explicit_admission_winning_foreground_reconnect_read_is_not_revoked(
) -> None:
    clock = _Clock()
    redis = _ExpiringFenceQuotaRedis()
    _clear_counted_projection(redis)
    repo = _ReconnectReadBarrierRepo(
        _auto_degrade_session(clock, execution_mode="foreground")
    )
    reconnect = ExecutionSupervisor(redis_client=redis, session_repository=repo)
    explicit = ExecutionSupervisor(redis_client=redis, session_repository=repo)

    reconnect_task = asyncio.create_task(reconnect.resume(
        session_id=repo.session.id,
        user_id="user-lease",
        execution_mode="foreground",
    ))
    await repo.reconnect_second_read.wait()
    await explicit.admit(
        session_id=repo.session.id,
        user_id="user-lease",
        execution_mode="background",
        background_reason="explicit",
        expires_at=clock.now + timedelta(hours=2),
    )
    repo.allow_reconnect_second_read.set()

    assert await reconnect_task is None
    assert repo.session.execution_mode == "background"
    assert repo.session.background_reason == "explicit"
    assert repo.session.execution_revision == 1
    assert redis.system_count == 1
    assert redis.hashes[("supervisor:bg-generation:user-lease", repo.session.id)] == "1"


async def test_mode_transition_fence_serializes_two_supervisor_instances() -> None:
    redis = _FenceRedis()
    pod_a = ExecutionSupervisor(redis_client=redis, session_repository=object())
    pod_b = ExecutionSupervisor(redis_client=redis, session_repository=object())
    assert hasattr(pod_a, "mode_transition_fence")
    a_entered = asyncio.Event()
    release_a = asyncio.Event()
    order: list[str] = []

    async def _hold_a() -> None:
        async with pod_a.mode_transition_fence(session_id="session-lease"):
            order.append("a")
            a_entered.set()
            await release_a.wait()

    async def _enter_b() -> None:
        await a_entered.wait()
        async with pod_b.mode_transition_fence(session_id="session-lease"):
            order.append("b")

    tasks = [asyncio.create_task(_hold_a()), asyncio.create_task(_enter_b())]
    await a_entered.wait()
    await asyncio.sleep(0.02)
    assert order == ["a"]
    release_a.set()
    await asyncio.gather(*tasks)
    assert order == ["a", "b"]
    assert redis.values == {}


async def test_shutdown_drain_cancels_and_consumes_mode_fence_cleanup_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()

    async def _blocked_cleanup() -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(supervisor_module, "_MODE_FENCE_SHUTDOWN_WAIT_SECONDS", 0.0)
    task = asyncio.create_task(
        _blocked_cleanup(),
        name="test-mode-fence-shutdown-drain",
    )
    supervisor_module._PENDING_MODE_FENCE_CLEANUPS.add(task)
    task.add_done_callback(supervisor_module._PENDING_MODE_FENCE_CLEANUPS.discard)
    await started.wait()

    await supervisor_module.drain_mode_transition_fence_cleanups()
    await asyncio.sleep(0)

    assert task.done()
    assert task.cancelled()
    assert task not in supervisor_module._PENDING_MODE_FENCE_CLEANUPS


async def test_shutdown_drain_consumes_already_failed_mode_fence_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _failed_cleanup() -> None:
        raise RuntimeError("fence cleanup failed")

    task = asyncio.create_task(
        _failed_cleanup(),
        name="test-failed-mode-fence-shutdown-drain",
    )
    await asyncio.sleep(0)
    assert task.done()
    supervisor_module._PENDING_MODE_FENCE_CLEANUPS.add(task)
    try:
        await supervisor_module.drain_mode_transition_fence_cleanups()
    finally:
        supervisor_module._PENDING_MODE_FENCE_CLEANUPS.discard(task)

    assert "mode transition fence cleanup failed during shutdown" in caplog.text


class _CancelAfterSetFenceRedis(_FenceRedis):
    def __init__(self) -> None:
        super().__init__()
        self.written = asyncio.Event()
        self.allow_return = asyncio.Event()

    async def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool:
        acquired = await super().set(key, value, nx=nx, ex=ex)
        if acquired:
            self.written.set()
            await self.allow_return.wait()
        return acquired


async def test_mode_transition_fence_cancel_after_set_is_ttl_bounded() -> None:
    redis = _CancelAfterSetFenceRedis()
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=object())

    async def _enter() -> None:
        async with supervisor.mode_transition_fence(session_id="session-lease"):
            pytest.fail("cancelled acquisition must not enter the fence")

    task = asyncio.create_task(_enter())
    await redis.written.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert redis.values
    assert redis.set_ttls == [15]


async def test_mode_transition_fence_releases_after_renewal_failure() -> None:
    redis = _FenceRedis()
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=object())
    key = "supervisor:mode-transition:session-lease"
    token = "owner-token"
    redis.values[key] = token

    async def _failed_renewal() -> None:
        raise RuntimeError("redis renew failed")

    renew_task = asyncio.create_task(_failed_renewal())
    await asyncio.sleep(0)

    await supervisor._cleanup_mode_transition_fence(
        key=key,
        token=token,
        renew_task=renew_task,
    )

    assert redis.values == {}


class _RenewFailureFenceRedis(_FenceRedis):
    def __init__(self, failure_mode: str) -> None:
        super().__init__()
        self.failure_mode = failure_mode
        self.renew_entered = asyncio.Event()
        self.fail_renew = asyncio.Event()
        self.release_completed = asyncio.Event()
        self.failures_remaining = 1

    async def eval(
        self,
        script: str,
        numkeys: int,
        key: str,
        token: str,
        *args,
    ) -> int:
        if args and self.failures_remaining:
            self.renew_entered.set()
            await self.fail_renew.wait()
            self.failures_remaining -= 1
            if self.failure_mode == "exception":
                raise RuntimeError("redis renew failed")
            return 0
        result = await super().eval(script, numkeys, key, token, *args)
        if not args:
            self.release_completed.set()
        return result


@pytest.mark.parametrize("failure_mode", ["exception", "token_lost"])
async def test_mode_transition_fence_renew_failure_stops_old_body_before_new_owner(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervisor_module,
        "_MODE_TRANSITION_FENCE_RENEW_SECONDS",
        0,
    )
    redis = _RenewFailureFenceRedis(failure_mode)
    pod_a = ExecutionSupervisor(redis_client=redis, session_repository=object())
    pod_b = ExecutionSupervisor(redis_client=redis, session_repository=object())
    old_entered = asyncio.Event()
    new_entered = asyncio.Event()
    never_release_old = asyncio.Event()
    active_bodies: set[str] = set()
    order: list[str] = []

    async def _old_owner() -> None:
        async with pod_a.mode_transition_fence(session_id="session-lease"):
            active_bodies.add("old")
            order.append("old_enter")
            old_entered.set()
            try:
                await never_release_old.wait()
            finally:
                active_bodies.remove("old")
                order.append("old_stopped")

    async def _new_owner() -> None:
        async with pod_b.mode_transition_fence(session_id="session-lease"):
            assert active_bodies == set()
            assert redis.release_completed.is_set()
            active_bodies.add("new")
            order.append("new_enter")
            new_entered.set()
            active_bodies.remove("new")

    old_task = asyncio.create_task(_old_owner())
    await old_entered.wait()
    await redis.renew_entered.wait()
    new_task = asyncio.create_task(_new_owner())
    await asyncio.sleep(0.02)
    assert new_entered.is_set() is False

    redis.fail_renew.set()
    old_result = (
        await asyncio.wait_for(
            asyncio.gather(old_task, return_exceptions=True),
            timeout=0.5,
        )
    )[0]
    await asyncio.wait_for(new_task, timeout=1)

    assert isinstance(old_result, RuntimeError)
    assert "lease lost" in str(old_result)
    assert order == ["old_enter", "old_stopped", "new_enter"]
    assert active_bodies == set()


async def test_mode_transition_fence_external_cancel_preserves_cancelled_error(
) -> None:
    redis = _FenceRedis()
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=object())
    entered = asyncio.Event()
    block = asyncio.Event()

    async def _owner() -> None:
        async with supervisor.mode_transition_fence(session_id="session-lease"):
            entered.set()
            await block.wait()

    task = asyncio.create_task(_owner())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert redis.values == {}
