"""Unit tests for the MemorySystemNotification domain dataclass."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from app.domain.models.memory_system_notification import (
    MemorySystemNotification,
)

from tests.conftest import TEST_USER_ID_FIXED


class TestMemorySystemNotification:
    def test_default_read_at_is_none(self) -> None:
        """新建通知 read_at 默认为 None——未读态。"""
        now = datetime.now(tz=timezone.utc)
        exp = datetime.now(tz=timezone.utc)
        n = MemorySystemNotification(
            id="n-1",
            user_id=str(TEST_USER_ID_FIXED),
            event_type="memory_gate_paused",
            payload={"consecutive_failures": 3},
            created_at=now,
            expires_at=exp,
        )
        assert n.read_at is None
        assert n.payload == {"consecutive_failures": 3}

    def test_frozen_instance(self) -> None:
        """frozen=True——字段不可变，防止上层误改内存态 payload。"""
        n = MemorySystemNotification(
            id="n-1",
            user_id=str(TEST_USER_ID_FIXED),
            event_type="quota_exceeded",
            payload={},
            created_at=datetime.now(tz=timezone.utc),
            expires_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(FrozenInstanceError):
            n.read_at = datetime.now(tz=timezone.utc)  # type: ignore[misc]
