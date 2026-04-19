"""Settings 级 regression：memory_root_host 必须是宿主机绝对路径。

本组测试覆盖 round-4 的配置收紧：
- ``memory_root_host`` 禁止 ``~``，避免在 api 容器内被误展开成 ``/root/...``
- ``memory_root_container`` 也要求绝对路径，避免相对目录被解释错
"""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from core.config import Settings


def _required_env() -> dict[str, str]:
    return {
        "postgres_password": "ci-dummy",
        "jwt_secret_key": "ci-dummy-secret-key-32-chars-long",
        "env": "test",
    }


def _make_settings(**overrides: Any) -> Settings:
    return Settings(**_required_env(), **overrides)


class TestMemoryRootPaths:
    def test_memory_root_host_rejects_tilde(self) -> None:
        with pytest.raises(ValidationError):
            _make_settings(memory_root_host="~/.actus/memory")

    def test_memory_root_host_rejects_relative_path(self) -> None:
        with pytest.raises(ValidationError):
            _make_settings(memory_root_host="tmp/actus-memory")

    def test_memory_root_container_rejects_relative_path(self) -> None:
        with pytest.raises(ValidationError):
            _make_settings(memory_root_container="app/data/memory")

    def test_absolute_paths_still_allowed(self) -> None:
        settings = _make_settings(
            memory_root_host="/tmp/actus-memory",
            memory_root_container="/app/data/memory",
        )
        assert settings.memory_root_host == "/tmp/actus-memory"
        assert settings.memory_root_container == "/app/data/memory"
