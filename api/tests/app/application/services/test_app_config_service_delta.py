"""T16 — AppConfigService 治理 delta 挂钩单测（D1a §6.1）。

fake repo=内存 AppConfig；FakeReconciler 记录调用；FakeReadPort 供成员守卫。
断言：深拷贝 old（F6 merge 原地）、post-save delta 顺序、context/target 透传、
standalone uninstall ctx 构造、missing_ok 幂等、preallocated_id、reconciler=None
向后兼容、enabled 翻转不抛、成员守卫 + off 跳过。
"""
from __future__ import annotations

import uuid

import pytest

from app.application.errors.exceptions import NotFoundError
from app.application.services.app_config_service import AppConfigService
from app.domain.external.extension_admission import UninstallContext
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MCPConfig,
    MCPServerConfig,
)
from app.domain.models.extension_governance import ManagedByPluginError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------- fakes ----
def _mcp_server(url: str = "http://svc") -> MCPServerConfig:
    return MCPServerConfig(url=url)


def _make_app_config(
    mcp: dict[str, MCPServerConfig] | None = None,
    a2a: list[A2AServerConfig] | None = None,
) -> AppConfig:
    return AppConfig(
        llm_config=LLMConfig(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(mcpServers=mcp or {}),
        a2a_config=A2AConfig(a2a_servers=a2a or []),
    )


class _FakeRepo:
    def __init__(self, app_config: AppConfig, events: list[str] | None = None) -> None:
        self._app_config = app_config
        self.save_count = 0
        self._events = events

    def load(self) -> AppConfig:
        return self._app_config

    def save(self, app_config: AppConfig) -> None:
        self.save_count += 1
        self._app_config = app_config
        if self._events is not None:
            self._events.append("save")


class _FakeReconciler:
    def __init__(self, events: list[str] | None = None) -> None:
        self.mcp_calls: list[dict] = []
        self.a2a_calls: list[dict] = []
        self._events = events

    async def reconcile_mcp_delta(self, old, new, **kwargs) -> None:
        self.mcp_calls.append({"old": old, "new": new, **kwargs})
        if self._events is not None:
            self._events.append("delta")

    async def reconcile_a2a_delta(self, old, new, **kwargs) -> None:
        self.a2a_calls.append({"old": old, "new": new, **kwargs})
        if self._events is not None:
            self._events.append("delta")


class _FakeRow:
    def __init__(self, parent_plugin_ext_id: str | None = None, status: str = "active") -> None:
        self.parent_plugin_ext_id = parent_plugin_ext_id
        self.status = status


class _FakeReadPort:
    def __init__(self, row: _FakeRow | None = None) -> None:
        self._row = row
        self.calls: list[tuple[str, str]] = []

    async def get_row(self, kind: str, ext_id: str):
        self.calls.append((kind, ext_id))
        return self._row


# ---------------------------------------------------------------- cases ----
async def test_mcp_update_deepcopies_old() -> None:
    cfg = _make_app_config(mcp={"s1": _mcp_server()})
    recon = _FakeReconciler()
    svc = AppConfigService(_FakeRepo(cfg), reconciler=recon)

    await svc.update_and_create_mcp_servers(MCPConfig(mcpServers={"s2": _mcp_server()}))

    call = recon.mcp_calls[0]
    assert call["old"] is not call["new"]
    assert "s2" not in call["old"].mcpServers       # 深拷贝——old 不含新 server
    assert "s2" in call["new"].mcpServers


async def test_delta_called_after_save() -> None:
    events: list[str] = []
    cfg = _make_app_config(mcp={"s1": _mcp_server()})
    repo = _FakeRepo(cfg, events=events)
    recon = _FakeReconciler(events=events)
    svc = AppConfigService(repo, reconciler=recon)

    await svc.update_and_create_mcp_servers(MCPConfig(mcpServers={"s2": _mcp_server()}))
    assert events == ["save", "delta"]


async def test_install_context_and_target_passthrough() -> None:
    cfg = _make_app_config(mcp={})
    recon = _FakeReconciler()
    svc = AppConfigService(_FakeRepo(cfg), reconciler=recon)
    sentinel = object()

    await svc.update_and_create_mcp_servers(
        MCPConfig(mcpServers={"s1": _mcp_server()}),
        install_context=sentinel,
        target_server="s1",
    )
    call = recon.mcp_calls[0]
    assert call["install_context"] is sentinel
    assert call["target_ext_id"] == "s1"


async def test_delete_builds_standalone_uninstall_context() -> None:
    cfg = _make_app_config(mcp={"s1": _mcp_server()})
    recon = _FakeReconciler()
    svc = AppConfigService(_FakeRepo(cfg), reconciler=recon)

    await svc.delete_mcp_server("s1", actor_id="admin-1")
    ctx = recon.mcp_calls[0]["uninstall_context"]
    assert isinstance(ctx, UninstallContext)
    assert ctx.correlation_id is None
    assert ctx.actor_user_id == "admin-1"


async def test_delete_missing_ok() -> None:
    cfg = _make_app_config(mcp={})
    recon = _FakeReconciler()
    repo = _FakeRepo(cfg)
    svc = AppConfigService(repo, reconciler=recon)

    # missing_ok=True → 无抛、零 save、零 delta
    await svc.delete_mcp_server("nope", actor_id="a", missing_ok=True)
    assert repo.save_count == 0
    assert recon.mcp_calls == []

    # missing_ok=False → NotFoundError
    with pytest.raises(NotFoundError):
        await svc.delete_mcp_server("nope", actor_id="a")


async def test_a2a_create_preallocated_id() -> None:
    cfg = _make_app_config(a2a=[])
    svc = AppConfigService(_FakeRepo(cfg))
    result = await svc.create_a2a_server("http://a", preallocated_id="fixed-uuid")
    assert result.a2a_servers[-1].id == "fixed-uuid"

    cfg2 = _make_app_config(a2a=[])
    svc2 = AppConfigService(_FakeRepo(cfg2))
    result2 = await svc2.create_a2a_server("http://a")
    assert result2.a2a_servers[-1].id != "fixed-uuid"
    uuid.UUID(result2.a2a_servers[-1].id)   # 合法 uuid4


async def test_reconciler_none_backward_compat() -> None:
    cfg = _make_app_config(mcp={"s1": _mcp_server()}, a2a=[A2AServerConfig(id="a1", base_url="http://a")])
    repo = _FakeRepo(cfg)
    svc = AppConfigService(repo)   # reconciler=None

    await svc.update_and_create_mcp_servers(MCPConfig(mcpServers={"s2": _mcp_server()}))
    await svc.set_mcp_server_enabled("s1", False)
    await svc.delete_mcp_server("s1")
    await svc.create_a2a_server("http://b")
    await svc.set_a2a_server_enabled("a1", False)
    await svc.delete_a2a_server("a1")
    # 无治理副作用、无异常——现状行为
    assert repo.save_count == 6


async def test_enabled_flip_delta_noop() -> None:
    cfg = _make_app_config(mcp={"s1": _mcp_server()})
    recon = _FakeReconciler()
    svc = AppConfigService(_FakeRepo(cfg), reconciler=recon)

    await svc.set_mcp_server_enabled("s1", False)
    assert len(recon.mcp_calls) == 1
    call = recon.mcp_calls[0]
    # enabled 翻转不带 install/uninstall context（fingerprint 排除 enabled → T15 自然空转）
    assert call.get("install_context") is None
    assert call.get("uninstall_context") is None


async def test_delete_member_guard_mcp_a2a() -> None:
    # MCP standalone delete，行带 parent → ManagedByPluginError + 零 save
    cfg = _make_app_config(mcp={"s1": _mcp_server()})
    repo = _FakeRepo(cfg)
    read_port = _FakeReadPort(_FakeRow(parent_plugin_ext_id="plugin-x"))
    svc = AppConfigService(repo, reconciler=_FakeReconciler(), registry_read_port=read_port)

    with pytest.raises(ManagedByPluginError):
        await svc.delete_mcp_server("s1", actor_id="admin-1")
    assert repo.save_count == 0

    # A2A standalone delete，同守卫
    cfg_a = _make_app_config(a2a=[A2AServerConfig(id="a1", base_url="http://a")])
    repo_a = _FakeRepo(cfg_a)
    read_port_a = _FakeReadPort(_FakeRow(parent_plugin_ext_id="plugin-x"))
    svc_a = AppConfigService(repo_a, reconciler=_FakeReconciler(), registry_read_port=read_port_a)
    with pytest.raises(ManagedByPluginError):
        await svc_a.delete_a2a_server("a1", actor_id="admin-1")
    assert repo_a.save_count == 0

    # saga 路（correlation 非 None）→ 守卫跳过、放行删除
    cfg2 = _make_app_config(mcp={"s1": _mcp_server()})
    repo2 = _FakeRepo(cfg2)
    read_port2 = _FakeReadPort(_FakeRow(parent_plugin_ext_id="plugin-x"))
    svc2 = AppConfigService(repo2, reconciler=_FakeReconciler(), registry_read_port=read_port2)
    saga_ctx = UninstallContext(correlation_id=uuid.uuid4(), actor_user_id="admin-1")
    await svc2.delete_mcp_server("s1", uninstall_context=saga_ctx)
    assert repo2.save_count == 1
    assert read_port2.calls == []     # saga 路不查 read_port


async def test_delete_guard_skipped_when_read_port_none() -> None:
    cfg = _make_app_config(mcp={"s1": _mcp_server()})
    repo = _FakeRepo(cfg)
    svc = AppConfigService(repo, reconciler=_FakeReconciler())   # read_port=None
    await svc.delete_mcp_server("s1", actor_id="admin-1")
    assert repo.save_count == 1
