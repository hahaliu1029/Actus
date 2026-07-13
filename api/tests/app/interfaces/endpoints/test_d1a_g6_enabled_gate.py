"""D1a Task 13 — G6：`/enabled` façade 409 前置门（§4.2）。

quarantined 行的 enable → 409（enable **不**解除隔离；reapprove 是独立行政动作，
§9.2）。off（read_port=None）/ 未注册（get_row→None）/ 非 quarantined 行 → 现状
行为直通（INV-D1-0：治理关闭时装配咽喉恒等直通）。

端点测试只验 HTTP 面 + G6 门语义 + 写委托是否被短路（spy set_mcp_server_enabled）；
fixture 照抄 test_runtime_extension_routes.py 的 façade app/client 构造，用 DI
override 替换 read_port 为 fake（`get_extension_registry_read_port` 是本 task 新增
provider——handler 经 Depends 注入，off 时该 provider 返回 None）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.application.services.app_config_service import AppConfigService
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
from app.interfaces.service_dependencies import (
    get_app_config_service,
    get_extension_registry_read_port,
    get_runtime_extension_service,
)
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeReadPort:
    """治理只读 port fake：``status is None`` → get_row 返回 None（未注册行）。"""

    def __init__(self, status):
        self._status = status

    async def get_row(self, kind, ext_id):
        if self._status is None:
            return None
        return SimpleNamespace(status=self._status)


def _admin_user() -> User:
    return User(
        id="u1",
        username="admin",
        role=UserRole.SUPER_ADMIN,
        status=UserStatus.ACTIVE,
    )


def _item(kind: str, ext_id: str) -> ExtensionItemInfo:
    return ExtensionItemInfo(
        kind=kind,  # type: ignore[arg-type]
        id=ext_id,
        name=ext_id,
        description=None,
        config=ExtensionConfigInfo(
            enabled_global=True,
            enabled_user=True,
            effective_enabled=True,
            reason_code="enabled",
        ),
        health=ExtensionHealthInfo(kind="probe", state="unknown"),
        liveness=ExtensionLivenessInfo(state="unknown", active_run_count=0),
        stats=ExtensionStatsInfo(available=False, unavailable_reason="disabled"),
        details={},
    )


def _snapshot() -> RuntimeExtensionsSnapshot:
    return RuntimeExtensionsSnapshot(
        items=(_item("mcp", "srv-a"),),
        snapshot_at=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
        probe_enabled=False,
        stats_enabled=False,
    )


async def _noop_rate_limit() -> None:
    return None


def _make_client(read_port):
    """安装 façade + G6 overrides；返回 (client, fake_app_config_service, teardown)。

    ``read_port`` 经 ``get_extension_registry_read_port`` DI override 注入
    （None → off 直通；FakeReadPort → G6 门生效）。
    """
    fake_app_config = MagicMock(spec=AppConfigService)
    fake_app_config.set_mcp_server_enabled = AsyncMock(return_value=None)
    fake_app_config.set_a2a_server_enabled = AsyncMock(return_value=None)

    fake_runtime = MagicMock(spec=RuntimeExtensionService)
    fake_runtime.get_extensions = AsyncMock(return_value=_snapshot())

    app.dependency_overrides[get_current_user] = _admin_user
    app.dependency_overrides[get_app_config_service] = lambda: fake_app_config
    app.dependency_overrides[get_runtime_extension_service] = lambda: fake_runtime
    app.dependency_overrides[get_extension_registry_read_port] = lambda: read_port
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit

    # probe service 挂 app.state（handler 用 request.app.state 读，非 DI）——
    # G6 门在写委托之前，probe invalidate 走不到；但直通用例会走到，给个 no-op。
    had_probe = hasattr(app.state, "extension_probe_service")
    prev_probe = getattr(app.state, "extension_probe_service", None)
    probe = MagicMock()
    probe.invalidate = MagicMock(return_value=None)
    app.state.extension_probe_service = probe

    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    def teardown() -> None:
        for dep in (
            get_current_user,
            get_app_config_service,
            get_runtime_extension_service,
            get_extension_registry_read_port,
            rate_limit_write,
        ):
            app.dependency_overrides.pop(dep, None)
        if had_probe:
            app.state.extension_probe_service = prev_probe
        else:
            delattr(app.state, "extension_probe_service")

    return client, fake_app_config, teardown


async def test_enable_quarantined_409():
    """quarantined 行 enable=True → 409 extension_quarantined；写委托零调用（G6 前置短路）。"""
    client, fake_app_config, teardown = _make_client(FakeReadPort("quarantined"))
    try:
        resp = await client.post(
            "/api/v1/runtime/extensions/mcp/srv-a/enabled", json={"enabled": True}
        )
        assert resp.status_code == 409
        assert "extension_quarantined" in resp.text
        assert resp.json()["data"]["reason"] == "extension_quarantined"
        # G6 门在写委托之前——set_mcp_server_enabled 绝不被调用（enable 不解除隔离）。
        fake_app_config.set_mcp_server_enabled.assert_not_called()
    finally:
        teardown()


async def test_enable_active_passthrough():
    """active 行 enable=True → 现状行为（200 + 写委托一次）。"""
    client, fake_app_config, teardown = _make_client(FakeReadPort("active"))
    try:
        resp = await client.post(
            "/api/v1/runtime/extensions/mcp/srv-a/enabled", json={"enabled": True}
        )
        assert resp.status_code == 200
        fake_app_config.set_mcp_server_enabled.assert_awaited_once_with("srv-a", True)
    finally:
        teardown()


async def test_off_mode_port_none_passthrough():
    """read_port=None（mode=off，INV-D1-0）→ G6 门跳过，现状行为直通。"""
    client, fake_app_config, teardown = _make_client(None)
    try:
        resp = await client.post(
            "/api/v1/runtime/extensions/mcp/srv-a/enabled", json={"enabled": True}
        )
        assert resp.status_code == 200
        fake_app_config.set_mcp_server_enabled.assert_awaited_once_with("srv-a", True)
    finally:
        teardown()


async def test_unknown_row_passthrough():
    """get_row→None（未注册扩展）→ 不因治理缺行被拒，现状行为直通。"""
    client, fake_app_config, teardown = _make_client(FakeReadPort(None))
    try:
        resp = await client.post(
            "/api/v1/runtime/extensions/mcp/srv-a/enabled", json={"enabled": True}
        )
        assert resp.status_code == 200
        fake_app_config.set_mcp_server_enabled.assert_awaited_once_with("srv-a", True)
    finally:
        teardown()


async def test_quarantined_disable_not_gated():
    """G6 只门 enable=True；quarantined 行 disable=False 仍直通（收敛=独立行政动作）。"""
    client, fake_app_config, teardown = _make_client(FakeReadPort("quarantined"))
    try:
        resp = await client.post(
            "/api/v1/runtime/extensions/mcp/srv-a/enabled", json={"enabled": False}
        )
        assert resp.status_code == 200
        fake_app_config.set_mcp_server_enabled.assert_awaited_once_with("srv-a", False)
    finally:
        teardown()
