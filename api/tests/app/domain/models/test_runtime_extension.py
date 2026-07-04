"""B9 domain 聚合 DTO + flag 默认值（spec §2 R9#4 / §9）。"""
import dataclasses
from datetime import datetime, timezone

from app.domain.models.app_config import ToolRuntimeConfig
from app.domain.models.runtime_extension import (
    ExtensionConfigInfo,
    ExtensionHealthInfo,
    ExtensionItemInfo,
    ExtensionLivenessInfo,
    ExtensionStatsInfo,
    LivenessSnapshot,
    RuntimeExtensionsSnapshot,
)


def test_flags_default_off():
    cfg = ToolRuntimeConfig()
    assert cfg.extension_probe_enabled is False
    assert cfg.extension_stats_enabled is False


def test_item_info_is_frozen_dataclass():
    health = ExtensionHealthInfo(kind="probe", state="unknown")
    assert health.relative_file is None       # probe ⇒ relative_file 恒 None（R12#8）
    assert dataclasses.is_dataclass(ExtensionItemInfo)
    assert ExtensionItemInfo.__dataclass_params__.frozen


def test_snapshot_shape():
    snap = RuntimeExtensionsSnapshot(
        items=(), snapshot_at=datetime.now(timezone.utc),
        probe_enabled=False, stats_enabled=False,
    )
    assert snap.items == ()


def test_liveness_snapshot_shape():
    ls = LivenessSnapshot(active={("mcp", "srv"): frozenset({"run1"})}, degraded=False)
    assert ls.active[("mcp", "srv")] == frozenset({"run1"})


def test_domain_module_is_pure():
    """domain 纯净：不 import pydantic/fastapi/sqlalchemy/redis/httpx。

    只扫 import 语句行——`redis_unavailable` 是 StatsUnavailableReason 的
    合法 wire 词表值（P-2 钉子冻结），不是依赖导入，故裸 substring 扫描会误报。
    """
    import app.domain.models.runtime_extension as mod
    src = open(mod.__file__, encoding="utf-8").read()
    import_lines = [
        line for line in src.splitlines()
        if line.startswith(("import ", "from "))
    ]
    for banned in ("fastapi", "sqlalchemy", "redis", "httpx", "pydantic"):
        for line in import_lines:
            assert banned not in line, f"domain 模块不得 import {banned}"
