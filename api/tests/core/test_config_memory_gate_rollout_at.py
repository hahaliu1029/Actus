"""Settings 级 regression：``memory_gate_rollout_at`` 必须是 timezone-aware。

codex fix P1 round-2：rollout_at 是 destructive delete（DELETE /legacy）的
时间边界；裸 naive datetime 会按宿主机 local tz 解释，跨时区部署节点产出
不同的 cutoff，删错行。本测试钉死 Settings(memory_gate_rollout_at=...)
构造阶段就拒绝 naive 输入——守护发生在配置加载时而非 SQL 构造时，
单点失败、不依赖调用链每层都记得校验。
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from core.config import Settings


def _required_env() -> dict[str, str]:
    """构造最小的 .env 替代值，让 Settings() 不因其它必填字段 crash。

    - ``POSTGRES_PASSWORD`` / ``jwt_secret_key`` 有 validator 要求非默认
    - 其它字段依赖默认值
    """
    return {
        "postgres_password": "ci-dummy",
        "jwt_secret_key": "ci-dummy-secret-key-32-chars-long",
        "env": "test",
    }


def _make_settings(**overrides: Any) -> Settings:
    """Pydantic v2 BaseSettings 支持 kwargs 初始化，绕开 .env 依赖。"""
    return Settings(**_required_env(), **overrides)


class TestMemoryGateRolloutAtTZRequired:
    def test_naive_datetime_rejected(self) -> None:
        """``"2026-04-01T00:00:00"``（无 tz）必须被 Pydantic 拒绝。"""
        naive = datetime(2026, 4, 1, 0, 0)  # tzinfo=None
        assert naive.tzinfo is None  # sanity check
        with pytest.raises(ValidationError) as exc_info:
            _make_settings(memory_gate_rollout_at=naive)
        # 错误消息应明确提到 timezone（AwareDatetime validator 语义）
        assert "aware" in str(exc_info.value).lower() or "tz" in str(
            exc_info.value
        ).lower()

    def test_naive_iso_string_rejected(self) -> None:
        """字符串形式的 naive 输入也必须被拒——用户常走环境变量路径。"""
        with pytest.raises(ValidationError):
            _make_settings(memory_gate_rollout_at="2026-04-01T00:00:00")

    def test_aware_utc_accepted(self) -> None:
        """``"2026-04-01T00:00:00Z"`` 合法。"""
        cutoff = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
        settings = _make_settings(memory_gate_rollout_at=cutoff)
        assert settings.memory_gate_rollout_at == cutoff
        assert settings.memory_gate_rollout_at.tzinfo is not None

    def test_aware_iso_string_accepted(self) -> None:
        """ISO 8601 + Z 后缀 / +08:00 偏移都应被接受。"""
        for s in (
            "2026-04-01T00:00:00Z",
            "2026-04-01T08:00:00+08:00",
            "2026-04-01T00:00:00+00:00",
        ):
            settings = _make_settings(memory_gate_rollout_at=s)
            assert settings.memory_gate_rollout_at is not None
            assert settings.memory_gate_rollout_at.tzinfo is not None

    def test_none_still_allowed(self) -> None:
        """默认 None 表示"未配置，前端走警告路径"，不能误伤。"""
        settings = _make_settings(memory_gate_rollout_at=None)
        assert settings.memory_gate_rollout_at is None
