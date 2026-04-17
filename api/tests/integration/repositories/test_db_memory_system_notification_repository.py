"""Integration tests for DBMemorySystemNotificationRepository.

Requires: SQLALCHEMY_DATABASE_URL env var pointing to Postgres.
Run: cd api && uv run pytest tests/integration/repositories/test_db_memory_system_notification_repository.py -v

FK: memory_system_notifications.user_id → users.id (CASCADE).
每条测试必须先 INSERT 父 user 行（memory 相关表都走这个模式，见
tests/integration/test_db_memory_chunk_repository_integration.py）。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from app.infrastructure.repositories.db_memory_system_notification_repository import (
    DBMemorySystemNotificationRepository,
)

pytestmark = pytest.mark.anyio


async def _ensure_user(db_session, user_id: str) -> None:
    await db_session.execute(
        text("INSERT INTO users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
        {"uid": user_id},
    )


class TestCreateAndListUnread:

    async def test_create_returns_domain_with_defaults(self, db_session) -> None:
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        repo = DBMemorySystemNotificationRepository(db_session)

        n = await repo.create(
            notification_id=f"n-{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            event_type="memory_gate_paused",
            payload={"consecutive_failures": 3},
        )
        assert n.user_id == user_id
        assert n.event_type == "memory_gate_paused"
        assert n.payload == {"consecutive_failures": 3}
        assert n.read_at is None
        # Server-default expires_at ≈ now + 30d
        delta = n.expires_at - n.created_at
        assert timedelta(days=29, hours=23) <= delta <= timedelta(days=30, hours=1)

    async def test_list_unread_filters_by_user(self, db_session) -> None:
        uid_a = str(uuid.uuid4())
        uid_b = str(uuid.uuid4())
        await _ensure_user(db_session, uid_a)
        await _ensure_user(db_session, uid_b)
        repo = DBMemorySystemNotificationRepository(db_session)

        await repo.create(
            notification_id=f"n-{uuid.uuid4().hex[:8]}",
            user_id=uid_a,
            event_type="memory_gate_paused",
            payload={},
        )
        await repo.create(
            notification_id=f"n-{uuid.uuid4().hex[:8]}",
            user_id=uid_b,
            event_type="quota_exceeded",
            payload={},
        )
        items_a = await repo.list_unread(uid_a)
        items_b = await repo.list_unread(uid_b)
        assert [i.user_id for i in items_a] == [uid_a]
        assert [i.user_id for i in items_b] == [uid_b]

    async def test_list_unread_excludes_read_rows(self, db_session) -> None:
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        repo = DBMemorySystemNotificationRepository(db_session)
        nid = f"n-{uuid.uuid4().hex[:8]}"

        await repo.create(
            notification_id=nid,
            user_id=user_id,
            event_type="memory_gate_paused",
            payload={},
        )
        assert len(await repo.list_unread(user_id)) == 1
        assert await repo.mark_read(notification_id=nid, user_id=user_id)
        assert await repo.list_unread(user_id) == []

    async def test_list_unread_excludes_expired_rows(self, db_session) -> None:
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        repo = DBMemorySystemNotificationRepository(db_session)
        past = datetime.now(tz=timezone.utc) - timedelta(days=1)

        await repo.create(
            notification_id=f"n-{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            event_type="memory_gate_paused",
            payload={},
            created_at=past - timedelta(days=40),
            expires_at=past,
        )
        assert await repo.list_unread(user_id) == []

    async def test_list_unread_orders_created_at_desc(self, db_session) -> None:
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        repo = DBMemorySystemNotificationRepository(db_session)
        now = datetime.now(tz=timezone.utc)

        await repo.create(
            notification_id="n-old",
            user_id=user_id,
            event_type="memory_gate_paused",
            payload={},
            created_at=now - timedelta(hours=2),
            expires_at=now + timedelta(days=30),
        )
        await repo.create(
            notification_id="n-new",
            user_id=user_id,
            event_type="quota_exceeded",
            payload={},
            created_at=now - timedelta(minutes=1),
            expires_at=now + timedelta(days=30),
        )
        items = await repo.list_unread(user_id)
        assert [i.id for i in items] == ["n-new", "n-old"]


class TestCountUnread:

    async def test_count_matches_unfiltered_unread_total(self, db_session) -> None:
        """count_unread must equal the number of unread-and-unexpired
        rows — same WHERE as list_unread, no limit truncation."""
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        repo = DBMemorySystemNotificationRepository(db_session)
        for i in range(75):
            await repo.create(
                notification_id=f"n-{i}-{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                event_type="memory_gate_paused",
                payload={},
            )
        assert await repo.count_unread(user_id) == 75
        # list_unread default limit=50 — prove they diverge when they should
        assert len(await repo.list_unread(user_id)) == 50

    async def test_count_excludes_read_and_expired(self, db_session) -> None:
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        repo = DBMemorySystemNotificationRepository(db_session)
        now = datetime.now(tz=timezone.utc)

        # Unread + valid
        await repo.create(
            notification_id="n-unread",
            user_id=user_id,
            event_type="memory_gate_paused",
            payload={},
        )
        # Unread but expired
        await repo.create(
            notification_id="n-expired",
            user_id=user_id,
            event_type="quota_exceeded",
            payload={},
            created_at=now - timedelta(days=40),
            expires_at=now - timedelta(days=1),
        )
        # Read and valid
        nid_read = f"n-read-{uuid.uuid4().hex[:8]}"
        await repo.create(
            notification_id=nid_read,
            user_id=user_id,
            event_type="memory_gate_paused",
            payload={},
        )
        await repo.mark_read(notification_id=nid_read, user_id=user_id)

        assert await repo.count_unread(user_id) == 1

    async def test_count_isolated_per_user(self, db_session) -> None:
        uid_a = str(uuid.uuid4())
        uid_b = str(uuid.uuid4())
        await _ensure_user(db_session, uid_a)
        await _ensure_user(db_session, uid_b)
        repo = DBMemorySystemNotificationRepository(db_session)

        await repo.create(
            notification_id=f"n-a-{uuid.uuid4().hex[:8]}",
            user_id=uid_a, event_type="memory_gate_paused", payload={},
        )
        assert await repo.count_unread(uid_a) == 1
        assert await repo.count_unread(uid_b) == 0


class TestMarkRead:

    async def test_mark_read_twice_returns_false_second_time(
        self, db_session
    ) -> None:
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        repo = DBMemorySystemNotificationRepository(db_session)
        nid = f"n-{uuid.uuid4().hex[:8]}"
        await repo.create(
            notification_id=nid,
            user_id=user_id,
            event_type="memory_gate_paused",
            payload={},
        )
        assert await repo.mark_read(notification_id=nid, user_id=user_id)
        assert not await repo.mark_read(notification_id=nid, user_id=user_id)

    async def test_mark_read_wrong_user_returns_false(self, db_session) -> None:
        uid_a = str(uuid.uuid4())
        uid_b = str(uuid.uuid4())
        await _ensure_user(db_session, uid_a)
        await _ensure_user(db_session, uid_b)
        repo = DBMemorySystemNotificationRepository(db_session)
        nid = f"n-{uuid.uuid4().hex[:8]}"
        await repo.create(
            notification_id=nid,
            user_id=uid_a,
            event_type="memory_gate_paused",
            payload={},
        )
        # B 试图标记 A 的通知 —— 无声 false，让 "越权" 与 "不存在"
        # 对外语义一致，避免通过 200/404 差异枚举用户空间
        assert not await repo.mark_read(
            notification_id=nid, user_id=uid_b
        )

    async def test_mark_read_missing_id_returns_false(self, db_session) -> None:
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        repo = DBMemorySystemNotificationRepository(db_session)
        assert not await repo.mark_read(
            notification_id="nonexistent-id", user_id=user_id
        )


class TestPurgeExpired:

    async def test_purge_deletes_only_expired(self, db_session) -> None:
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        repo = DBMemorySystemNotificationRepository(db_session)
        now = datetime.now(tz=timezone.utc)

        await repo.create(
            notification_id="n-expired",
            user_id=user_id,
            event_type="memory_gate_paused",
            payload={},
            created_at=now - timedelta(days=40),
            expires_at=now - timedelta(days=1),
        )
        await repo.create(
            notification_id="n-valid",
            user_id=user_id,
            event_type="quota_exceeded",
            payload={},
            created_at=now - timedelta(days=1),
            expires_at=now + timedelta(days=29),
        )
        deleted = await repo.purge_expired()
        assert deleted == 1
        remaining = await repo.list_unread(user_id)
        assert [i.id for i in remaining] == ["n-valid"]
