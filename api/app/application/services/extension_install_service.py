"""D1a §7.1/§9.2 MCP/A2A 两阶段安装管道（preview 零写 / commit 建 pin）。

- ``preview_*``：probe → scan → policy 组装 ``ExtensionInstallPreview`` DTO——
  **零写零预检**（dry_run 严格不触碰 registry / config）。
- ``commit_*``：per-identity 锁内 probe → scan → policy 门 → 占用预检 → pin 写
  （经 ``app_config_service`` + reconciler 落 registry；本服务不直触 WritePort，
  记账由 ``AppConfigService.update_and_create_mcp_servers`` / ``create_a2a_server``
  的 install_context 触发）。commit 恒**自跑**观测——不复用 preview 结果（§7.1 防
  preview↔commit 表面篡改）。
- ``ensure_single_mcp``：R4-02 批量门（治理模式逐个安装）——路由消费，>1 server 抛
  ``BatchNotAllowedError``。

三个安装异常（Acknowledge/Force/BatchNotAllowed）在本模块定义，interfaces
exception_handlers 映射 HTTP（AcknowledgeRequired→409 acknowledge_required(+findings) /
ForceRequired→422 force_required(+findings) / BatchNotAllowed→422 single_server_required）。

分层：application 层——**禁 import FastAPI/SQLAlchemy**（identity 锁工厂取用式统一
``self._identity_locks or get_identity_locks()`` 复用 T16 进程单例）。
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from app.application.services.extension_identity_locks import get_identity_locks
from app.domain.external.extension_admission import InstallContext
from app.domain.models.app_config import (
    A2AServerConfig,
    MCPConfig,
    MCPServerConfig,
)
from app.domain.models.extension_governance import (
    HASH_SCHEMA_VERSION,
    GovernanceScanSummary,
    InvalidStateTransitionError,
    ManagedByPluginError,
)
from app.domain.services.extension_hashing import (
    a2a_config_fingerprint,
    a2a_surface_hash,
    mcp_config_fingerprint,
    mcp_surface_hash,
)
from app.domain.services.extension_scan import (
    evaluate_install_policy,
    scan_a2a_entry,
    scan_mcp_entry,
)
from app.interfaces.schemas.extension_governance import ExtensionInstallPreview

logger = logging.getLogger(__name__)

# observed_surface 描述摘要截断（脱敏摘要，非全文——防大 schema/长文本进 wire）。
_SURFACE_DESC_MAX = 200
_PROBE_FAILED_WARNING = (
    "probe 失败：surface 未 pin（enforce 下该扩展在 approve 前不可装配）"
)
_STDIO_ENV_WARNING = (
    "stdio server 继承 API 进程完整环境变量（已知限制，不承诺 secret isolation）"
)


# --------------------------------------------------------------- exceptions --
class AcknowledgeRequiredError(Exception):
    """caution-tier 需显式 acknowledge（enforce）——interfaces 映射 409
    acknowledge_required(+findings)。"""

    def __init__(self, summary: GovernanceScanSummary | None = None) -> None:
        super().__init__("acknowledgement required to install a caution-tier extension")
        self.summary = summary


class ForceRequiredError(Exception):
    """dangerous-tier 需 force（enforce）——interfaces 映射 422 force_required(+findings)。"""

    def __init__(self, summary: GovernanceScanSummary | None = None) -> None:
        super().__init__("force required to install a dangerous-tier extension")
        self.summary = summary


class BatchNotAllowedError(Exception):
    """治理模式下逐个安装 MCP——interfaces 映射 422 single_server_required。"""

    def __init__(
        self,
        message: str = "治理模式下逐个安装 MCP 服务",
        summary: GovernanceScanSummary | None = None,
    ) -> None:
        super().__init__(message)
        self.summary = summary


# ------------------------------------------------------------------ service --
class ExtensionInstallService:
    """MCP/A2A 两阶段安装管道（preview/commit）。"""

    def __init__(
        self,
        app_config_service: Any,
        read_port: Any,
        prober: Any,
        mode: str,
        identity_locks: Any = None,
    ) -> None:
        self._app_config_service = app_config_service
        self._read_port = read_port
        self._prober = prober
        self._mode = mode
        # R4#2 统一 fallback 合同：测试注入 fake、生产零注入 → get_identity_locks() 进程单例
        self._identity_locks = identity_locks

    # ---- 批量门（R4-02；路由在 mode≠off 分支消费）----
    def ensure_single_mcp(
        self, mcp_config: MCPConfig
    ) -> tuple[str, MCPServerConfig]:
        """治理模式下 MCP 逐个安装：>1 server → ``BatchNotAllowedError``。返回单项 (name, cfg)。"""
        if self._mode != "off" and len(mcp_config.mcpServers) != 1:
            raise BatchNotAllowedError()
        return next(iter(mcp_config.mcpServers.items()))

    # ---- preview（零写零预检）----
    async def preview_mcp(
        self, server_name: str, config: MCPServerConfig
    ) -> ExtensionInstallPreview:
        outcome = await self._probe_safe("mcp", server_name, config)
        payload = outcome.surface_payload if outcome else None
        scan = scan_mcp_entry(server_name, config, payload if isinstance(payload, list) else None)
        decision = evaluate_install_policy(
            self._mode, scan.verdict, acknowledged=False, forced=False)
        surface_hash = self._mcp_surface_hash(outcome)
        return ExtensionInstallPreview(
            scan_report=scan,
            observed_surface=self._redacted_mcp_surface(payload),
            surface_hash=surface_hash,
            config_fingerprint=mcp_config_fingerprint(config),
            install_policy_decision=decision,
            warnings=self._mcp_warnings(config, surface_hash),
        )

    async def preview_a2a(self, base_url: str) -> ExtensionInstallPreview:
        cfg = A2AServerConfig(base_url=base_url)
        outcome = await self._probe_safe("a2a", cfg.id, cfg)
        card = outcome.surface_payload if outcome else None
        scan = scan_a2a_entry(base_url, card if isinstance(card, dict) else None)
        decision = evaluate_install_policy(
            self._mode, scan.verdict, acknowledged=False, forced=False)
        surface_hash = self._a2a_surface_hash(outcome)
        return ExtensionInstallPreview(
            scan_report=scan,
            observed_surface=self._redacted_a2a_surface(card),
            surface_hash=surface_hash,
            config_fingerprint=a2a_config_fingerprint(base_url),
            install_policy_decision=decision,
            warnings=self._a2a_warnings(surface_hash),
        )

    # ---- commit（锁内 probe→scan→policy→precheck→pin 写）----
    async def commit_mcp(
        self, server_name: str, config: MCPServerConfig, *,
        actor_id: str, acknowledged: bool, forced: bool,
    ):
        locks = self._identity_locks or get_identity_locks()   # R4#2 统一取用式（T16 模块工厂）
        async with locks.acquire_all([("mcp", server_name)]):   # §3.6 per-identity 互斥
            outcome = await self._probe_safe("mcp", server_name, config)
            payload = outcome.surface_payload if outcome else None
            scan = scan_mcp_entry(
                server_name, config, payload if isinstance(payload, list) else None)
            self._enforce_policy(scan, acknowledged, forced)
            await self._occupancy_precheck("mcp", server_name)         # R32#4 + R44#3
            surface_hash = self._mcp_surface_hash(outcome)
            warnings = self._mcp_warnings(config, surface_hash)
            ctx = InstallContext(
                actor_user_id=actor_id, correlation_id=None, source_type="config",
                source_ref=None, version=None, trust_origin="user_installed",
                artifact_hash=None, surface_hash=surface_hash,
                config_fingerprint=mcp_config_fingerprint(config),
                hash_schema_version=HASH_SCHEMA_VERSION, scan=scan,
                probe_failed=surface_hash is None,
                acknowledged=acknowledged, forced=forced)
            new_cfg = await self._app_config_service.update_and_create_mcp_servers(
                MCPConfig(mcpServers={server_name: config}), actor_id=actor_id,
                install_context=ctx, target_server=server_name)
            return new_cfg, warnings

    async def commit_a2a(
        self, base_url: str, *, actor_id: str, acknowledged: bool, forced: bool,
    ):
        # a2a id 恒新建——预分配后作为锁 key / 占用预检 ext_id / InstallContext 目标同源
        a2a_id = str(uuid.uuid4())
        cfg = A2AServerConfig(id=a2a_id, base_url=base_url)
        locks = self._identity_locks or get_identity_locks()
        async with locks.acquire_all([("a2a", a2a_id)]):
            outcome = await self._probe_safe("a2a", a2a_id, cfg)
            card = outcome.surface_payload if outcome else None
            scan = scan_a2a_entry(base_url, card if isinstance(card, dict) else None)
            self._enforce_policy(scan, acknowledged, forced)
            await self._occupancy_precheck("a2a", a2a_id)
            surface_hash = self._a2a_surface_hash(outcome)
            warnings = self._a2a_warnings(surface_hash)
            ctx = InstallContext(
                actor_user_id=actor_id, correlation_id=None, source_type="config",
                source_ref=None, version=None, trust_origin="user_installed",
                artifact_hash=None, surface_hash=surface_hash,
                config_fingerprint=a2a_config_fingerprint(base_url),
                hash_schema_version=HASH_SCHEMA_VERSION, scan=scan,
                probe_failed=surface_hash is None,
                acknowledged=acknowledged, forced=forced)
            new_cfg = await self._app_config_service.create_a2a_server(
                base_url, actor_id=actor_id, install_context=ctx, preallocated_id=a2a_id)
            return new_cfg, warnings

    # -------------------------------------------------------------- helpers --
    def _enforce_policy(
        self, scan: GovernanceScanSummary, acknowledged: bool, forced: bool
    ) -> None:
        decision = evaluate_install_policy(
            self._mode, scan.verdict, acknowledged=acknowledged, forced=forced)
        if decision == "need_acknowledge":
            raise AcknowledgeRequiredError(scan)
        if decision == "need_force":
            raise ForceRequiredError(scan)

    async def _probe_safe(self, kind: str, ext_id: str, config: Any):
        """§7.1：probe 失败不阻塞——try/except 返 None（20s 预算复用 prober 内化超时）。"""
        try:
            if kind == "mcp":
                return await self._prober.probe_mcp(ext_id, config)
            return await self._prober.probe_a2a(config)
        except Exception:  # noqa: BLE001 - probe 失败 → 装未 pin（§7.1），不外泄
            logger.warning(
                "D1a install-time probe 失败（非致命，surface 不 pin）: %s/%s",
                kind, ext_id, exc_info=True)
            return None

    async def _occupancy_precheck(self, kind: str, ext_id: str) -> None:
        """R32#4 + R44#3：membership→ManagedByPlugin；quarantined/disabled→InvalidState。
        active/无行 → 放行（update 语义）。off（read_port=None）→ 零预检。"""
        if self._read_port is None:
            return
        row = await self._read_port.get_row(kind, ext_id)
        if row is None:
            return
        if row.parent_plugin_ext_id is not None:
            raise ManagedByPluginError(
                f"{kind}[{ext_id}] 由 plugin 管理，禁 standalone 安装（§7.1）")
        if row.status in ("quarantined", "disabled"):
            raise InvalidStateTransitionError(
                f"{kind}[{ext_id}] 处于 {row.status}——安装不是恢复动作"
                "（先 reapprove/enable 或 delete）")

    @staticmethod
    def _mcp_surface_hash(outcome: Any) -> str | None:
        payload = outcome.surface_payload if outcome else None
        if outcome and outcome.ok and isinstance(payload, list) and payload:
            return mcp_surface_hash(payload)
        return None

    @staticmethod
    def _a2a_surface_hash(outcome: Any) -> str | None:
        card = outcome.surface_payload if outcome else None
        if outcome and outcome.ok and isinstance(card, dict) and card:
            return a2a_surface_hash(card)
        return None

    def _mcp_warnings(self, config: MCPServerConfig, surface_hash: str | None) -> list[str]:
        warnings = [] if surface_hash else [_PROBE_FAILED_WARNING]
        transport = getattr(config.transport, "value", config.transport)
        if transport == "stdio":                                    # F26/R3#9
            warnings.append(_STDIO_ENV_WARNING)
        return warnings

    @staticmethod
    def _a2a_warnings(surface_hash: str | None) -> list[str]:
        return [] if surface_hash else [_PROBE_FAILED_WARNING]

    def _redacted_mcp_surface(self, payload: Any) -> list[dict] | None:
        if not isinstance(payload, list) or not payload:
            return None
        return [
            {"name": t.get("name"), "description": _truncate(t.get("description"))}
            for t in payload if isinstance(t, dict)
        ]

    def _redacted_a2a_surface(self, card: Any) -> dict | None:
        if not isinstance(card, dict):
            return None
        skills = card.get("skills")
        skill_names = (
            [s.get("name") for s in skills if isinstance(s, dict)]
            if isinstance(skills, list) else None
        )
        return {
            "name": card.get("name"),
            "description": _truncate(card.get("description")),
            "skills": skill_names,
        }


def _truncate(text: Any) -> str | None:
    if text is None:
        return None
    return str(text)[:_SURFACE_DESC_MAX]
