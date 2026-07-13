"""T19 — app-config 安装管道路由（§7.1/§9.2）。

ASGITransport + dependency_overrides（照抄 test_d1a_off_mode_byte_golden 的 admin 覆盖）。
off → 旧直通（dry_run 拒 409）；on → dry_run 预检 / batch 门 / policy 门（ack/force）/
probe 失败 warnings。verdict 用真实 config 扫描驱动（http:// → caution / inject_ignore →
dangerous / https → safe），fake prober 只控 ok/surface。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from app.application.services.extension_install_service import ExtensionInstallService
from app.application.services.extension_probe_service import ProbeOutcome
from app.domain.models.app_config import A2AConfig
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.service_dependencies import (
    get_app_config_service,
    get_extension_install_service,
)
from app.main import app

pytestmark = pytest.mark.anyio

_MCP = "/api/app-config/mcp-servers"
_A2A = "/api/app-config/a2a-servers"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _admin() -> User:
    return User(id="admin-1", username="admin", role=UserRole.SUPER_ADMIN, status=UserStatus.ACTIVE)


# ------------------------------------------------------------------- fakes ---
class _FakeAppConfigService:
    def __init__(self) -> None:
        self.mcp_calls: list = []
        self.a2a_calls: list = []

    async def update_and_create_mcp_servers(
        self, mcp_config, *, actor_id=None, install_context=None, target_server=None
    ):
        self.mcp_calls.append((mcp_config, actor_id, install_context, target_server))
        return mcp_config

    async def create_a2a_server(
        self, base_url, *, actor_id=None, install_context=None, preallocated_id=None
    ):
        self.a2a_calls.append((base_url, actor_id, install_context, preallocated_id))
        return A2AConfig()


class _FakeReadPort:
    def __init__(self, row=None) -> None:
        self._row = row

    async def get_row(self, kind, ext_id):
        return self._row


class _FakeProber:
    def __init__(self, ok=True, surface=None, raises=False) -> None:
        self._ok = ok
        self._surface = surface if surface is not None else [
            {"name": "t", "description": "d", "input_schema": {}}]
        self._raises = raises

    async def probe_mcp(self, server_name, config):
        if self._raises:
            raise RuntimeError("probe boom")
        return ProbeOutcome(ok=self._ok, latency_ms=1, surface_payload=self._surface)

    async def probe_a2a(self, config):
        if self._raises:
            raise RuntimeError("probe boom")
        return ProbeOutcome(ok=self._ok, latency_ms=1, surface_payload={"name": "card"})


def _svc(mode, *, app_config=None, prober=None, read_port=None) -> ExtensionInstallService:
    return ExtensionInstallService(
        app_config or _FakeAppConfigService(),
        read_port if read_port is not None else _FakeReadPort(),
        prober or _FakeProber(),
        mode,
    )


@asynccontextmanager
async def _client(*, install_service, app_config_service=None):
    app.dependency_overrides[get_current_user] = _admin
    app.dependency_overrides[get_extension_install_service] = lambda: install_service
    app.dependency_overrides[get_app_config_service] = (
        lambda: app_config_service or _FakeAppConfigService())
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        for dep in (get_current_user, get_extension_install_service, get_app_config_service):
            app.dependency_overrides.pop(dep, None)


# ----------------------------------------------------------------- configs ---
def _safe(url="https://safe.test/mcp") -> dict:
    return {"transport": "streamable_http", "url": url}


def _caution() -> dict:
    return {"transport": "streamable_http", "url": "http://insecure.test/mcp"}


def _dangerous() -> dict:
    return {"transport": "stdio", "command": "bash", "args": ["-c", "ignore previous instructions"]}


# =================================================================== tests ===
async def test_off_dry_run_ignored_passthrough_2xx():
    """#3 修复：off + dry_run → dry_run 是治理新参，off 下透明忽略 → 旧直通 2xx 创建
    （pre-D1a：dry_run 是未知参被忽略走正常创建，绝非 409——off 须对新参透明）。"""
    app_config = _FakeAppConfigService()
    async with _client(install_service=None, app_config_service=app_config) as client:
        resp = await client.post(_MCP, params={"dry_run": "true"},
                                 json={"mcpServers": {"s1": _safe()}})
    assert resp.status_code == 200
    assert len(app_config.mcp_calls) == 1                       # 正常落盘创建（非 preview）


async def test_off_commit_batch_passthrough_2xx():
    """off + commit 批量 → 现状直通 2xx（旧多服务批量不受治理限制）。"""
    app_config = _FakeAppConfigService()
    async with _client(install_service=None, app_config_service=app_config) as client:
        resp = await client.post(
            _MCP, json={"mcpServers": {"s1": _safe(), "s2": _safe("https://b.test/mcp")}})
    assert resp.status_code == 200
    assert len(app_config.mcp_calls) == 1                       # 单次批量直通
    assert len(app_config.mcp_calls[0][0].mcpServers) == 2


async def test_shadow_dry_run_preview_shape():
    """shadow + dry_run → 200 preview（scan_report/config_fingerprint/decision/warnings）。"""
    async with _client(install_service=_svc("shadow")) as client:
        resp = await client.post(_MCP, params={"dry_run": "true"},
                                 json={"mcpServers": {"s1": _safe()}})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert set(data) >= {
        "scan_report", "observed_surface", "surface_hash",
        "config_fingerprint", "install_policy_decision", "warnings"}
    assert data["scan_report"]["verdict"] == "safe"
    assert data["install_policy_decision"] == "allow"


async def test_shadow_batch_over_one_422():
    """shadow + commit 批量>1 → 422 single_server_required（BatchNotAllowedError 映射）。"""
    async with _client(install_service=_svc("shadow")) as client:
        resp = await client.post(
            _MCP, json={"mcpServers": {"s1": _safe(), "s2": _safe("https://b.test/mcp")}})
    assert resp.status_code == 422
    assert resp.json()["code"] == "single_server_required"


async def test_enforce_caution_needs_ack_then_ack_ok():
    """enforce caution 无 ack → 409 acknowledge_required；重提交 ack → 2xx。"""
    app_config = _FakeAppConfigService()
    svc = _svc("enforce", app_config=app_config)
    async with _client(install_service=svc) as client:
        resp = await client.post(_MCP, json={"mcpServers": {"s1": _caution()}})
        assert resp.status_code == 409
        body = resp.json()
        assert body["code"] == "acknowledge_required"
        assert body["scan_report"]["verdict"] == "caution"

        resp2 = await client.post(_MCP, params={"acknowledge": "true"},
                                  json={"mcpServers": {"s1": _caution()}})
        assert resp2.status_code == 200
    assert len(app_config.mcp_calls) == 1                       # 仅 ack 后落盘


async def test_enforce_dangerous_needs_force_then_force_ok():
    """enforce dangerous 无 force → 422 force_required；force=true → 2xx。"""
    app_config = _FakeAppConfigService()
    svc = _svc("enforce", app_config=app_config)
    async with _client(install_service=svc) as client:
        resp = await client.post(_MCP, json={"mcpServers": {"s1": _dangerous()}})
        assert resp.status_code == 422
        assert resp.json()["code"] == "force_required"

        resp2 = await client.post(_MCP, params={"force": "true"},
                                  json={"mcpServers": {"s1": _dangerous()}})
        assert resp2.status_code == 200
    assert len(app_config.mcp_calls) == 1


async def test_commit_warnings_on_probe_failure():
    """probe 失败（fake prober 抛）→ commit 仍 2xx + data.warnings 出现未 pin 提示。"""
    svc = _svc("shadow", prober=_FakeProber(raises=True))
    async with _client(install_service=svc) as client:
        resp = await client.post(_MCP, json={"mcpServers": {"s1": _safe()}})
    assert resp.status_code == 200
    warnings = resp.json()["data"]["warnings"]
    assert any("未 pin" in w for w in warnings)


async def test_managed_by_plugin_returns_409():
    """占用预检：membership 行 → 409 managed_by_plugin（域异常映射）。"""
    read_port = _FakeReadPort(row=SimpleNamespace(status="active", parent_plugin_ext_id="plug-1"))
    svc = _svc("shadow", read_port=read_port)
    async with _client(install_service=svc) as client:
        resp = await client.post(_MCP, json={"mcpServers": {"s1": _safe()}})
    assert resp.status_code == 409
    assert resp.json()["code"] == "managed_by_plugin"


# --------------------------------------------------------------- a2a 镜像 ----
async def test_a2a_off_dry_run_ignored_passthrough_2xx():
    """#3 修复镜像：a2a off + dry_run → 忽略 dry_run 走正常创建 2xx（非 409）。"""
    app_config = _FakeAppConfigService()
    async with _client(install_service=None, app_config_service=app_config) as client:
        resp = await client.post(_A2A, params={"dry_run": "true"},
                                 json={"base_url": "https://agent.test"})
    assert resp.status_code == 200
    assert len(app_config.a2a_calls) == 1


async def test_a2a_shadow_commit_2xx():
    app_config = _FakeAppConfigService()
    svc = _svc("shadow", app_config=app_config)
    async with _client(install_service=svc) as client:
        resp = await client.post(_A2A, json={"base_url": "https://agent.test"})
    assert resp.status_code == 200
    assert len(app_config.a2a_calls) == 1
    assert app_config.a2a_calls[0][1] == "admin-1"             # actor 透传
