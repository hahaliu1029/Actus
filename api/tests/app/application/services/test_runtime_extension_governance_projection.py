"""D1a Task 25：B9 聚合治理投影单测（spec §9.1/§10）。

覆盖：第五段 plugin 源 + governance block（Admin-only）+ None-omit wrap 序列化
+ 非 Admin 三层剥离（本文件锁第①层 service 投影 + 第②层 wire dump）+ 阻断语义
（quarantined/disabled/parent_blocked → governance_blocked）+ ConfigReasonCode 八值
snapshot + INV-D1-5 词表不扩 + plugin 展示名 helper best-effort + provider read_port 注入。
"""
from __future__ import annotations

import json
from typing import get_args
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.application.services.runtime_extension_service import RuntimeExtensionService
from app.domain.external.extension_admission import GovernanceRowSnapshot
from app.domain.models.app_config import (
    A2AConfig,
    AppConfig,
    MCPConfig,
    MCPServerConfig,
    MCPTransport,
)
from app.domain.models.extension_governance import HASH_SCHEMA_VERSION
from app.domain.models.runtime_extension import (
    ConfigReasonCode,
    ExtensionConfigInfo,
    ExtensionGovernanceInfo,
    ExtensionHealthInfo,
    ExtensionItemInfo,
    ExtensionLivenessInfo,
    ExtensionStatsInfo,
)
from app.domain.models.skill_diagnostic import SkillDiagnostic
from app.infrastructure.external.governance.plugin_display_name import (
    read_plugin_display_name,
)
from app.interfaces.schemas.runtime_extensions import to_wire_item


# --- fixtures / builders -------------------------------------------------- #


def _row(**over) -> GovernanceRowSnapshot:
    base = dict(
        id=uuid4(),
        kind="mcp",
        ext_id="srv-on",
        status="active",
        quarantine_reason=None,
        trust_origin="user_installed",
        source_type="config",
        source_ref=None,
        version="1.0.0",
        artifact_hash=None,
        surface_hash="sfc",
        config_fingerprint="cfg",
        hash_schema_version=HASH_SCHEMA_VERSION,
        observed_surface_hash="sfc",
        observed_artifact_hash=None,
        observed_config_fingerprint="cfg",
        observed_hash_schema_version=HASH_SCHEMA_VERSION,
        last_observed_at=None,
        last_verified_at=None,
        last_mismatch_at=None,
        pinned_at=None,
        pinned_by=None,
        scan_verdict=None,
        scan_report=None,
        source_missing_at=None,
        installed_by="admin",
        row_revision=1,
        deleted_at=None,
        parent_plugin_ext_id=None,
    )
    base.update(over)
    return GovernanceRowSnapshot(**base)


def _read_port(rows):
    port = MagicMock()
    port.list_live_rows = AsyncMock(return_value=list(rows))
    return port


def _app_config() -> AppConfig:
    return AppConfig.model_construct(
        llm_config=MagicMock(),
        agent_config=MagicMock(),
        mcp_config=MCPConfig(
            mcpServers={
                "srv-on": MCPServerConfig(
                    transport=MCPTransport.STDIO, command="npx", enabled=True
                ),
            }
        ),
        a2a_config=A2AConfig(a2a_servers=[]),
    )


def _good_skill(skill_id: str):
    s = MagicMock()
    s.id = skill_id
    s.name = skill_id
    s.description = "d"
    s.enabled = True
    s.runtime_type = MagicMock(value="native")
    s.source_type = MagicMock(value="local")
    s.manifest = {"bundle_file_count": 1}
    return s


def _service(*, rows=None, resolver=None, diags=None) -> RuntimeExtensionService:
    skill_repo = MagicMock()
    skill_repo.list_with_diagnostics = AsyncMock(return_value=diags or [])
    enablement = MagicMock()
    enablement.list_user_enablements = AsyncMock(return_value=[])
    return RuntimeExtensionService(
        config_provider=_app_config,
        skill_repository=skill_repo,
        enablement_service=enablement,
        registry_read_port=_read_port(rows) if rows is not None else None,
        plugin_name_resolver=resolver,
    )


def _item_info(**over) -> ExtensionItemInfo:
    base = dict(
        kind="mcp",
        id="srv-a",
        name="srv-a",
        description=None,
        config=ExtensionConfigInfo(
            enabled_global=True,
            enabled_user=True,
            effective_enabled=True,
            reason_code="enabled",
        ),
        health=ExtensionHealthInfo(kind="probe", state="unknown"),
        liveness=ExtensionLivenessInfo(state="unknown"),
        stats=ExtensionStatsInfo(available=False, unavailable_reason="disabled"),
        details={"transport": "stdio"},
    )
    base.update(over)
    return ExtensionItemInfo(**base)


# --- test 1: governance block Admin-only 三层 ----------------------------- #


@pytest.mark.anyio
async def test_governance_block_admin_only_three_layers():
    rows = [_row(kind="mcp", ext_id="srv-on", status="active")]
    svc = _service(rows=rows)

    admin = await svc.get_extensions(user_id="u1", is_admin=True)
    a_items = {(i.kind, i.id): i for i in admin.items}
    gov = a_items[("mcp", "srv-on")].governance
    assert gov is not None
    assert gov.status == "active"
    assert gov.trust_origin == "user_installed"
    # mcp 必需 {surface, config_fingerprint} 全 pinned → pinned=True
    assert gov.pinned is True and gov.unpinned is False and gov.pin_stale is False
    # 第②层：Admin wire dump 含 governance 键
    dumped = to_wire_item(a_items[("mcp", "srv-on")]).model_dump()
    assert "governance" in dumped and dumped["governance"]["status"] == "active"

    # 第①层：非 Admin service 投影层 governance=None
    user = await svc.get_extensions(user_id="u1", is_admin=False)
    u_items = {(i.kind, i.id): i for i in user.items}
    assert u_items[("mcp", "srv-on")].governance is None
    # 第②层：非 Admin wire dump 无 governance 键（任何形态）
    dumped_u = to_wire_item(u_items[("mcp", "srv-on")]).model_dump()
    assert "governance" not in dumped_u
    dumped_u_json = json.loads(to_wire_item(u_items[("mcp", "srv-on")]).model_dump_json())
    assert "governance" not in dumped_u_json


# --- test 2: 阻断条目 用户可见语义 --------------------------------------- #


@pytest.mark.anyio
async def test_blocked_item_user_visible_semantics():
    # quarantined / disabled：own status 阻断
    for blocking_status in ("quarantined", "disabled"):
        rows = [_row(kind="mcp", ext_id="srv-on", status=blocking_status)]
        svc = _service(rows=rows)
        snap = await svc.get_extensions(user_id="u1", is_admin=True)
        it = {(i.kind, i.id): i for i in snap.items}[("mcp", "srv-on")]
        assert it.config.effective_enabled is False
        assert it.config.reason_code == "governance_blocked"
        # 用户偏好写不受阻断影响：enabled_global 仍映射 config（srv-on enabled=True）
        assert it.config.enabled_global is True

    # parent_blocked：成员 skill 经 parent_plugin_ext_id → 父 plugin quarantined
    diags = [SkillDiagnostic(skill_key="skill-mem", ok=True, skill=_good_skill("skill-mem"))]
    rows = [
        _row(
            kind="skill",
            ext_id="skill-mem",
            status="active",
            parent_plugin_ext_id="plug-1",
            artifact_hash="art",
            observed_artifact_hash="art",
        ),
        _row(
            kind="plugin",
            ext_id="plug-1",
            status="quarantined",
            artifact_hash="art",
            observed_artifact_hash="art",
        ),
    ]
    svc = _service(rows=rows, diags=diags)
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    it = {(i.kind, i.id): i for i in snap.items}[("skill", "skill-mem")]
    assert it.config.effective_enabled is False
    assert it.config.reason_code == "governance_blocked"


# --- test 3: plugin 第五段 ----------------------------------------------- #


@pytest.mark.anyio
async def test_plugin_fifth_segment():
    rows = [
        _row(
            kind="plugin",
            ext_id="plug-1",
            status="active",
            version="2.1.0",
            artifact_hash="art",
            observed_artifact_hash="art",
        ),
        _row(
            kind="skill",
            ext_id="mem-a",
            status="active",
            parent_plugin_ext_id="plug-1",
            artifact_hash="a1",
            observed_artifact_hash="a1",
        ),
        _row(kind="mcp", ext_id="mem-b", status="active", parent_plugin_ext_id="plug-1"),
    ]
    svc = _service(
        rows=rows, resolver=lambda ext_id, version: f"Display {ext_id} {version}"
    )
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    # plugin 条目在序尾（mcp→a2a→skill→plugin）
    plug = snap.items[-1]
    assert plug.kind == "plugin"
    assert plug.id == "plug-1"
    assert plug.name == "Display plug-1 2.1.0"  # resolver 经线程池注入
    assert plug.details == {"member_count": 2, "plugin_version": "2.1.0"}
    assert plug.health.kind == "integrity"
    assert plug.liveness.state == "not_applicable"
    assert plug.stats.unavailable_reason == "unsupported"
    assert plug.config.reason_code == "not_applicable_plugin"


@pytest.mark.anyio
async def test_plugin_name_fallback_when_resolver_none():
    rows = [_row(kind="plugin", ext_id="plug-x", status="active", version="9.9.9",
                 artifact_hash="art", observed_artifact_hash="art")]
    svc = _service(rows=rows, resolver=None)
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    plug = snap.items[-1]
    assert plug.kind == "plugin" and plug.name == "plug-x"  # fallback ext_id


# --- #5 修复：plugin enabled_global 映射治理启用态（非恒 True）----------- #


@pytest.mark.anyio
@pytest.mark.parametrize("blocking_status", ["disabled", "quarantined"])
async def test_plugin_enabled_global_reflects_governance_status(blocking_status):
    """disabled/quarantined plugin → enabled_global=False（FE Switch 绑定
    config.enabled_global，须映射真实治理启用态，不得恒 True 显示 ON）；effective_enabled
    亦 False + reason_code=governance_blocked（_attach_governance 阻断路径）。"""
    rows = [_row(kind="plugin", ext_id="plug-x", status=blocking_status,
                 artifact_hash="art", observed_artifact_hash="art")]
    svc = _service(rows=rows)
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    plug = {(i.kind, i.id): i for i in snap.items}[("plugin", "plug-x")]
    assert plug.config.enabled_global is False
    assert plug.config.effective_enabled is False
    assert plug.config.reason_code == "governance_blocked"


@pytest.mark.anyio
async def test_plugin_active_enabled_global_true():
    """回归：active plugin → enabled_global=True + reason_code=not_applicable_plugin。"""
    rows = [_row(kind="plugin", ext_id="plug-ok", status="active",
                 artifact_hash="art", observed_artifact_hash="art")]
    svc = _service(rows=rows)
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    plug = {(i.kind, i.id): i for i in snap.items}[("plugin", "plug-ok")]
    assert plug.config.enabled_global is True
    assert plug.config.effective_enabled is True
    assert plug.config.reason_code == "not_applicable_plugin"


# --- test 4: mode off 字节等价（read_port=None → 零 governance/零 plugin）- #


@pytest.mark.anyio
async def test_mode_off_zero_governance_zero_plugin():
    svc = _service(rows=None)  # read_port=None
    snap = await svc.get_extensions(user_id="u1", is_admin=True)
    # 无 plugin 条目
    assert all(i.kind != "plugin" for i in snap.items)
    # 零 governance 键（wire dump）
    for i in snap.items:
        assert i.governance is None
        assert "governance" not in to_wire_item(i).model_dump()


# --- test 5: None-omit（键不存在 ≠ null）--------------------------------- #


def test_governance_key_omitted_not_null():
    dumped = to_wire_item(_item_info(governance=None)).model_dump()
    assert "governance" not in dumped
    dumped_json = json.loads(to_wire_item(_item_info(governance=None)).model_dump_json())
    assert "governance" not in dumped_json

    gov = ExtensionGovernanceInfo(
        status="active",
        trust_origin="builtin",
        pinned=True,
        unpinned=False,
        pin_stale=False,
        scan_verdict=None,
        quarantine_reason=None,
        last_mismatch_at=None,
        last_verified_at=None,
        row_revision=1,
        observed_surface_hash=None,
        observed_artifact_hash=None,
        observed_config_fingerprint=None,
        pinned_at=None,
        pinned_by=None,
        installed_by=None,
        source_type="config",
        source_ref=None,
        version=None,
        source_missing_at=None,
        parent_plugin_ext_id=None,
    )
    dumped2 = to_wire_item(_item_info(governance=gov)).model_dump()
    assert "governance" in dumped2 and dumped2["governance"]["status"] == "active"


# --- test 6b: INV-D1-5 词表不扩（plugin 不进执行面）--------------------- #


def test_tooltype_vocab_not_extended():
    from app.domain.models.approval_grant import ToolSource
    from app.domain.models.user_tool_enablement import ToolType

    tt = {t.value for t in ToolType}
    assert tt == {"mcp", "a2a", "skill"}
    assert "plugin" not in tt
    ts = set(get_args(ToolSource))
    assert "plugin" not in ts


# --- test 6c: ConfigReasonCode 八值 snapshot ----------------------------- #


def test_config_reason_code_vocab_snapshot():
    assert set(get_args(ConfigReasonCode)) == {
        "enabled",
        "disabled_global",
        "disabled_user",
        "disabled_both",
        "user_enablement_unknown",
        "config_unreadable",
        "not_applicable_plugin",
        "governance_blocked",
    }


# --- provider 注入回归：read_port 透传 ----------------------------------- #


@pytest.mark.anyio
async def test_provider_injects_read_port_passthrough():
    from app.interfaces.service_dependencies import get_runtime_extension_service

    fake_request = MagicMock()
    sentinel = object()
    fake_request.app.state.extension_registry_read_port = sentinel
    svc = get_runtime_extension_service(request=fake_request, db_session=MagicMock())
    assert svc._registry_read_port is sentinel
    # 展示名 resolver 由 provider 注入 helper 偏函数
    assert svc._plugin_name_resolver is not None


# --- plugin 展示名 helper（纯同步 best-effort）-------------------------- #


def test_read_plugin_display_name_reads_name(tmp_path):
    d = tmp_path / "plug-1" / "1.0.0"
    d.mkdir(parents=True)
    (d / "plugin.json").write_text(json.dumps({"name": "My Plugin"}), encoding="utf-8")
    assert read_plugin_display_name(tmp_path, "plug-1", "1.0.0") == "My Plugin"


def test_read_plugin_display_name_fallback_missing(tmp_path):
    assert read_plugin_display_name(tmp_path, "nope", "1.0.0") == "nope"


def test_read_plugin_display_name_fallback_no_version(tmp_path):
    assert read_plugin_display_name(tmp_path, "plug-1", None) == "plug-1"


def test_read_plugin_display_name_fallback_bad_json(tmp_path):
    d = tmp_path / "plug-1" / "1.0.0"
    d.mkdir(parents=True)
    (d / "plugin.json").write_text("not json{", encoding="utf-8")
    assert read_plugin_display_name(tmp_path, "plug-1", "1.0.0") == "plug-1"


def test_read_plugin_display_name_fallback_oversized(tmp_path):
    d = tmp_path / "plug-1" / "1.0.0"
    d.mkdir(parents=True)
    (d / "plugin.json").write_text(
        '{"name":"x","pad":"' + "a" * (70 * 1024) + '"}', encoding="utf-8"
    )
    assert read_plugin_display_name(tmp_path, "plug-1", "1.0.0") == "plug-1"
