"""D1a Task 13 — D1-0 ③层：off-mode 字节 golden（治理关闭时 wire 响应零漂移）。

INV-D1-0 三层证明的第③层（行为半）：mode=off 下 `GET /v1/runtime/extensions`
的响应体**逐字节**等于 D1a 落地前的基线 golden。①层=注入拓扑 AST 门
（test_d1a_injection_topology.py）；②层=lifespan off gate 构造器零调用
（test_d1a_lifespan_off_gate.py）；本文件=③层 wire 面。

golden 生成惯例照抄 tests/golden/test_r2_golden_matrix.py：缺失即首次写盘并失败
提示 regenerate，稳定后提交 `.bin`。断言用原始字节比对（**不**用 resp.json() 或
同 serializer 生成 expected，R4-05——否则 serializer 自证同义反复）。

fixture 照抄 test_runtime_extension_routes.py 的 client_admin（admin 用户 + 固定
fake 聚合快照 + 固定时间戳），并把三治理 port 显式置 None（off）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.application.services.runtime_extension_service import (
    RuntimeExtensionService,
)
from app.domain.models.runtime_extension import (
    ExtensionConfigInfo,
    ExtensionHealthInfo,
    ExtensionItemInfo,
    ExtensionLivenessInfo,
    ExtensionStatsInfo,
    RuntimeExtensionsSnapshot,
)
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_write
from app.interfaces.service_dependencies import get_runtime_extension_service
from app.main import app

pytestmark = pytest.mark.anyio

# tests/app/interfaces/endpoints/ → parents[3] = tests/ → tests/golden/（既有布局）
GOLDEN = Path(__file__).resolve().parents[3] / "golden" / "d1a_off_extensions_response.bin"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _admin_user() -> User:
    return User(
        id="u1",
        username="admin",
        role=UserRole.SUPER_ADMIN,
        status=UserStatus.ACTIVE,
    )


def _admin_snapshot() -> RuntimeExtensionsSnapshot:
    """固定 fake 聚合快照（照抄 test_runtime_extension_routes._admin_snapshot——
    固定时间戳 + 固定字段 → wire 响应字节确定性）。"""
    item = ExtensionItemInfo(
        kind="mcp",
        id="server-a",
        name="server-a",
        description="a mcp server",
        config=ExtensionConfigInfo(
            enabled_global=True,
            enabled_user=True,
            effective_enabled=True,
            reason_code="enabled",
        ),
        health=ExtensionHealthInfo(kind="probe", state="unknown"),
        liveness=ExtensionLivenessInfo(state="unknown", active_run_count=0),
        stats=ExtensionStatsInfo(available=False, unavailable_reason="disabled"),
        details={"transport": "stdio", "tool_count": None},
    )
    return RuntimeExtensionsSnapshot(
        items=(item,),
        snapshot_at=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
        probe_enabled=False,
        stats_enabled=False,
    )


async def _noop_rate_limit() -> None:
    return None


@pytest.fixture
def client_admin():
    fake_service = MagicMock(spec=RuntimeExtensionService)
    fake_service.get_extensions = AsyncMock(return_value=_admin_snapshot())
    app.dependency_overrides[get_current_user] = _admin_user
    app.dependency_overrides[get_runtime_extension_service] = lambda: fake_service
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    # off：三治理 port 显式 None（INV-D1-0——GET 聚合器当前不读 port，置 None 是
    # 对 T25 治理投影落地后仍须零漂移的前瞻锁）。
    _prev = {}
    _had = {}
    for name in (
        "extension_admission_port",
        "extension_registry_write_port",
        "extension_registry_read_port",
    ):
        _had[name] = hasattr(app.state, name)
        _prev[name] = getattr(app.state, name, None)
        setattr(app.state, name, None)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        for dep in (get_current_user, get_runtime_extension_service, rate_limit_write):
            app.dependency_overrides.pop(dep, None)
        for name in (
            "extension_admission_port",
            "extension_registry_write_port",
            "extension_registry_read_port",
        ):
            if _had[name]:
                setattr(app.state, name, _prev[name])
            elif hasattr(app.state, name):
                delattr(app.state, name)


async def test_off_mode_response_byte_identical(client_admin):
    # R1#10a：真实前缀=/api/v1（main.py include；对齐 test_runtime_extension_routes.py:194）
    resp = await client_admin.get("/api/v1/runtime/extensions")
    assert resp.status_code == 200
    # off 下 governance 键不得以任何（含 null）形态出现——T25 加 serializer 后本断言仍须绿。
    assert b'"governance"' not in resp.content
    if not GOLDEN.exists():
        GOLDEN.write_bytes(resp.content)
        raise AssertionError("golden 首次生成——重跑本测试确认稳定后提交该文件")
    # 原始字节比对（不经 resp.json()/同 serializer 生成 expected，R4-05）。
    assert resp.content == GOLDEN.read_bytes()
