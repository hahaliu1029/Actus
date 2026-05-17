"""Tests for ConfirmationQueue (formerly ConfirmationManager, moved in PE-0)."""
import asyncio
import secrets
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.domain.services.permission.confirmation_queue import (
    _CAS_LUA,
    ConfirmationDetail,
    ConfirmationQueue,
)


def _run(coro):
    return asyncio.run(coro)


def _make_detail(**overrides) -> ConfirmationDetail:
    """Test helper — supplies all 11 required fields with sensible defaults
    (real dataclass at api/app/domain/services/permission/confirmation_queue.py
    has no field-level defaults except status/claim_nonce/processing_started_at,
    so omitting any required field raises TypeError)."""
    base = dict(
        session_id="s_test",
        tool_call_id="tc_test",
        user_id="u_test",
        tool_name="file_write",
        tool_args={"path": "/x"},
        risk_level="medium",
        arg_digest="ad_test",
        primary_arg="",
        dir_arg=None,
        matched_patterns=[],
        deadline_ts=9999999999.0,
    )
    base.update(overrides)
    return ConfirmationDetail(**base)


class TestConfirmationQueue:
    def setup_method(self):
        self.redis = AsyncMock()
        self.mgr = ConfirmationQueue(redis=self.redis, timeout_seconds=300)

    def test_store(self):
        detail = ConfirmationDetail(
            session_id="s1", tool_call_id="tc1", user_id="u1",
            tool_name="shell_execute", tool_args={"command": "ls"},
            risk_level="high", arg_digest="abc123",
            primary_arg="ls", dir_arg="/app", matched_patterns=[],
            deadline_ts=time.time() + 300,
        )
        _run(self.mgr.store(detail))
        self.redis.zadd.assert_called_once()
        self.redis.hset.assert_called_once()

    def test_read_returns_detail(self):
        self.redis.hgetall.return_value = {
            "session_id": "s1", "tool_call_id": "tc1", "user_id": "u1",
            "tool_name": "shell_execute", "tool_args_json": '{"command": "ls"}',
            "risk_level": "high", "arg_digest": "abc123",
            "primary_arg": "ls", "dir_arg": "/app",
            "matched_patterns_json": "[]", "deadline_ts": "1234567890",
            "status": "pending",
        }
        detail = _run(self.mgr.read("s1", "tc1"))
        assert detail is not None
        assert detail.tool_name == "shell_execute"
        assert detail.status == "pending"
        # New PE-0 fields default to None when absent from hash
        assert detail.claim_nonce is None
        assert detail.processing_started_at is None

    def test_read_missing(self):
        self.redis.hgetall.return_value = {}
        assert _run(self.mgr.read("s1", "tc1")) is None

    def test_mark_processing(self):
        _run(self.mgr.mark_processing("s1", "tc1"))
        self.redis.hset.assert_called_once()

    def test_cleanup(self):
        _run(self.mgr.cleanup("s1", "tc1"))
        self.redis.zrem.assert_called_once()
        self.redis.delete.assert_called_once()

    def test_mark_processing_if_pending_cas_true(self):
        """Lua CAS 赢：返 1 → True；入参 KEYS/ARGV 按 'processing'/'pending' 传。"""
        self.redis.eval = AsyncMock(return_value=1)
        result = _run(self.mgr.mark_processing_if_pending("s1", "tc1"))
        assert result is True
        self.redis.eval.assert_awaited_once()
        args, _ = self.redis.eval.call_args
        # args = (script, numkeys, key, new_status, expected_status, nonce, proc_ts)
        assert args[0] is _CAS_LUA  # catches accidental Lua source changes
        assert args[1] == 1
        assert args[2] == "confirmation_detail:s1:tc1"
        assert args[3] == "processing"
        assert args[4] == "pending"
        # No claim_nonce/processing_started_at → empty strings
        assert args[5] == ""
        assert args[6] == ""

    def test_mark_processing_if_pending_cas_false_on_mismatch(self):
        """对手已把状态推成 processing → Lua 返 0 → False（losing 并发 resume）。"""
        self.redis.eval = AsyncMock(return_value=0)
        assert _run(self.mgr.mark_processing_if_pending("s1", "tc1")) is False

    def test_mark_processing_if_pending_cas_false_on_missing_key(self):
        """key 不存在（confirmation 已被 cleanup）→ Lua 返 None/0 → False。"""
        self.redis.eval = AsyncMock(return_value=None)
        assert _run(self.mgr.mark_processing_if_pending("s1", "tc1")) is False


# ---------------------------------------------------------------------------
# Task 2.2: ConfirmationDetail PE-0 fields
# ---------------------------------------------------------------------------

def test_confirmation_detail_has_optional_claim_nonce_and_processing_started_at():
    """ConfirmationDetail accepts new PE-0 fields with None defaults."""
    detail = _make_detail(session_id="s1", tool_call_id="tc1")
    assert detail.claim_nonce is None
    assert detail.processing_started_at is None

    now = datetime.now(timezone.utc)
    detail2 = _make_detail(
        session_id="s1",
        tool_call_id="tc2",
        claim_nonce="deadbeef" * 4,
        processing_started_at=now,
    )
    assert detail2.claim_nonce == "deadbeef" * 4
    assert detail2.processing_started_at == now


def test_store_and_read_round_trip_claim_nonce_and_processing_started_at():
    """store() writes PE-0 fields to Redis hash; read() hydrates them correctly."""
    redis = AsyncMock()
    queue = ConfirmationQueue(redis=redis, timeout_seconds=300)

    now = datetime.now(timezone.utc)
    nonce = secrets.token_hex(16)
    detail = _make_detail(
        session_id="s1",
        tool_call_id="tc1",
        claim_nonce=nonce,
        processing_started_at=now,
    )
    _run(queue.store(detail))

    # Verify hset was called with the two new fields in the mapping
    _, kwargs = redis.hset.call_args
    mapping = kwargs.get("mapping", {})
    assert mapping["claim_nonce"] == nonce
    assert mapping["processing_started_at"] == now.isoformat()

    # Now simulate read from Redis that returns those same values
    redis.hgetall.return_value = {
        "session_id": "s1",
        "tool_call_id": "tc1",
        "user_id": "u_test",
        "tool_name": "file_write",
        "tool_args_json": '{"path": "/x"}',
        "risk_level": "medium",
        "arg_digest": "ad_test",
        "primary_arg": "",
        "dir_arg": "",
        "matched_patterns_json": "[]",
        "deadline_ts": "9999999999.0",
        "status": "processing",
        "claim_nonce": nonce,
        "processing_started_at": now.isoformat(),
    }
    fetched = _run(queue.read("s1", "tc1"))
    assert fetched is not None
    assert fetched.claim_nonce == nonce
    assert fetched.processing_started_at is not None
    # Fromisoformat round-trip should preserve timezone-aware datetime
    assert abs((fetched.processing_started_at - now).total_seconds()) < 1.0


def test_store_and_read_none_fields_serialize_as_empty_strings():
    """None claim_nonce/processing_started_at → empty string in Redis → None on read."""
    redis = AsyncMock()
    queue = ConfirmationQueue(redis=redis, timeout_seconds=300)

    detail = _make_detail(session_id="s2", tool_call_id="tc2")
    _run(queue.store(detail))

    _, kwargs = redis.hset.call_args
    mapping = kwargs.get("mapping", {})
    assert mapping["claim_nonce"] == ""
    assert mapping["processing_started_at"] == ""

    redis.hgetall.return_value = {
        "session_id": "s2",
        "tool_call_id": "tc2",
        "user_id": "u_test",
        "tool_name": "file_write",
        "tool_args_json": "{}",
        "risk_level": "medium",
        "arg_digest": "ad_test",
        "primary_arg": "",
        "dir_arg": "",
        "matched_patterns_json": "[]",
        "deadline_ts": "9999999999.0",
        "status": "pending",
        "claim_nonce": "",
        "processing_started_at": "",
    }
    fetched = _run(queue.read("s2", "tc2"))
    assert fetched is not None
    assert fetched.claim_nonce is None
    assert fetched.processing_started_at is None


# ---------------------------------------------------------------------------
# Task 2.3: mark_processing_if_pending with claim_nonce + processing_started_at
# (Mock-based — real Redis Lua semantics are Redis-layer; we verify correct args)
# ---------------------------------------------------------------------------

def test_mark_processing_if_pending_records_claim_nonce_atomically():
    """mark_processing_if_pending passes claim_nonce and processing_started_at to Lua eval."""
    redis = AsyncMock()
    redis.eval = AsyncMock(return_value=1)
    queue = ConfirmationQueue(redis=redis, timeout_seconds=300)

    nonce = secrets.token_hex(16)
    started = datetime.now(timezone.utc)

    ok = _run(queue.mark_processing_if_pending(
        "s_nonce", "tc_nonce",
        claim_nonce=nonce,
        processing_started_at=started,
    ))
    assert ok is True

    redis.eval.assert_awaited_once()
    args, _ = redis.eval.call_args
    # args: (script, numkeys, hash_key, "processing", "pending", nonce_arg, proc_arg)
    assert args[3] == "processing"
    assert args[4] == "pending"
    assert args[5] == nonce
    assert args[6] == started.isoformat()


def test_mark_processing_if_pending_second_call_returns_false():
    """First caller wins (Lua returns 0 for second claim attempt)."""
    redis = AsyncMock()
    # First call wins (1), second loses (0)
    redis.eval = AsyncMock(side_effect=[1, 0])
    queue = ConfirmationQueue(redis=redis, timeout_seconds=300)

    now = datetime.now(timezone.utc)
    n1 = secrets.token_hex(16)
    n2 = secrets.token_hex(16)

    a = _run(queue.mark_processing_if_pending("s_a", "tc_a", claim_nonce=n1, processing_started_at=now))
    b = _run(queue.mark_processing_if_pending("s_a", "tc_a", claim_nonce=n2, processing_started_at=now))
    assert a is True
    assert b is False


def test_mark_processing_if_pending_no_kwargs_passes_empty_strings():
    """Legacy call without claim_nonce/processing_started_at passes empty strings."""
    redis = AsyncMock()
    redis.eval = AsyncMock(return_value=1)
    queue = ConfirmationQueue(redis=redis, timeout_seconds=300)

    _run(queue.mark_processing_if_pending("s1", "tc1"))

    args, _ = redis.eval.call_args
    assert args[5] == ""
    assert args[6] == ""


def test_mark_pending_clears_claim_fields():
    """mark_pending clears claim_nonce and processing_started_at atomically."""
    redis = AsyncMock()
    queue = ConfirmationQueue(redis=redis, timeout_seconds=300)

    _run(queue.mark_pending("s1", "tc1"))

    _, kwargs = redis.hset.call_args
    mapping = kwargs.get("mapping", {})
    assert mapping["status"] == "pending"
    assert mapping["claim_nonce"] == ""
    assert mapping["processing_started_at"] == ""


# ---------------------------------------------------------------------------
# Task 2.4: find_orphaned_processing sweeper rescue
# ---------------------------------------------------------------------------

def test_find_orphaned_processing_returns_only_stale_processing_entries():
    """Only entries with status='processing' AND processing_started_at older than
    threshold are returned; pending entries and recent processing entries are excluded."""
    redis = AsyncMock()
    queue = ConfirmationQueue(redis=redis, timeout_seconds=300)

    now = datetime.now(timezone.utc)
    old_ts = now.timestamp() - 120  # 2 minutes ago

    # Member names in the ZSET
    redis.zrange = AsyncMock(return_value=[b"s:A", b"s:B", b"s:C"])

    # Build fake hash data for each member
    def make_hash(tool_call_id, status, proc_ts=None):
        return {
            "session_id": "s",
            "tool_call_id": tool_call_id,
            "user_id": "u",
            "tool_name": "x",
            "tool_args_json": "{}",
            "risk_level": "medium",
            "arg_digest": "d",
            "primary_arg": "",
            "dir_arg": "",
            "matched_patterns_json": "[]",
            "deadline_ts": "9999999999.0",
            "status": status,
            "claim_nonce": "x" * 32 if status == "processing" else "",
            "processing_started_at": proc_ts or "",
        }

    # A: pending → must NOT be returned
    hash_A = make_hash("A", "pending")
    # B: processing, recent (<60s) → must NOT be returned
    hash_B = make_hash("B", "processing", proc_ts=now.isoformat())
    # C: processing, stale (>60s) → MUST be returned
    hash_C = make_hash("C", "processing", proc_ts=datetime.fromtimestamp(old_ts, tz=timezone.utc).isoformat())

    redis.hgetall = AsyncMock(side_effect=[hash_A, hash_B, hash_C])

    orphans = _run(queue.find_orphaned_processing(processing_age_threshold_seconds=60))
    orphan_ids = {d.tool_call_id for d in orphans}

    assert "C" in orphan_ids
    assert "A" not in orphan_ids
    assert "B" not in orphan_ids


def test_find_orphaned_processing_skips_entries_without_processing_started_at():
    """Entries in processing state with no processing_started_at are skipped (safe)."""
    redis = AsyncMock()
    queue = ConfirmationQueue(redis=redis, timeout_seconds=300)

    redis.zrange = AsyncMock(return_value=[b"s:X"])
    redis.hgetall = AsyncMock(return_value={
        "session_id": "s",
        "tool_call_id": "X",
        "user_id": "u",
        "tool_name": "x",
        "tool_args_json": "{}",
        "risk_level": "medium",
        "arg_digest": "d",
        "primary_arg": "",
        "dir_arg": "",
        "matched_patterns_json": "[]",
        "deadline_ts": "9999999999.0",
        "status": "processing",
        "claim_nonce": "y" * 32,
        "processing_started_at": "",  # absent
    })

    orphans = _run(queue.find_orphaned_processing(processing_age_threshold_seconds=60))
    assert orphans == []


def test_find_orphaned_processing_skips_none_hgetall():
    """If a ZSET member has no hash (already cleaned up), it is skipped gracefully."""
    redis = AsyncMock()
    queue = ConfirmationQueue(redis=redis, timeout_seconds=300)

    redis.zrange = AsyncMock(return_value=[b"s:Y"])
    redis.hgetall = AsyncMock(return_value={})  # missing → read returns None

    orphans = _run(queue.find_orphaned_processing(processing_age_threshold_seconds=60))
    assert orphans == []
