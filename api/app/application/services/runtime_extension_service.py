"""B9 运行时扩展聚合服务（application 层，P-4 钉子）。

`RuntimeExtensionService.get_extensions()` 把四路独立数据源（config /
skill diagnostics / user enablement / stats）汇聚成 `RuntimeExtensionsSnapshot`
中间 DTO（domain），供 Task 7 的 GET 端点做 wire 平移。**本路径纯读、零网络
I/O**（INV-B9-4）：MCP/A2A 探测数据由注入的 `probe_view`/`liveness_view`/
`stats_reader` 快照提供；PR-1 阶段三者全传 None → 走 stub 语义（health=unknown/
skipped、liveness=unknown/not_applicable、stats.available=False）。

四路降级（R3#2 / spec §12）：
- config 加载失败 → 让异常冒泡（config 是根基，无 config 无响应）
- skill diagnostics 失败 → skill 条目缺失，mcp/a2a 照常 + warn
- user enablement DB 失败 → enabled_user=None + reason_code=user_enablement_unknown
  + effective 按 global 回退
- stats reader 失败/None → available=False + reason（优先级见下）

分层纪律：本模块**禁止** import `app.interfaces.*` 与
`app.infrastructure.runtime_stats`；config 经 `config_provider` callable 注入。
PR-2/3 接入真实 probe/liveness/stats view 后本文件**不再改语义**，仅替换注入物
（Task 17/20 验证）。
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Protocol

from app.domain.external.extension_stats import ExtensionStatsReader
from app.domain.models.extension_governance import HASH_SCHEMA_VERSION
from app.domain.models.runtime_extension import (
    ExtensionConfigInfo,
    ExtensionGovernanceInfo,
    ExtensionHealthInfo,
    ExtensionItemInfo,
    ExtensionLivenessInfo,
    ExtensionStatsInfo,
    RuntimeExtensionsSnapshot,
)
from app.domain.models.user_tool_enablement import ToolType
from app.domain.services.extension_admission_logic import (
    REQUIRED_OBSERVED_CATEGORIES,
    pin_presence,
)

if TYPE_CHECKING:  # 仅类型注解，避免 domain/infra 具体实现在运行时被强绑
    from app.domain.external.extension_admission import (
        ExtensionRegistryReadPort,
        GovernanceRowSnapshot,
    )
    from app.domain.models.app_config import (
        AppConfig,
        A2AServerConfig,
        MCPServerConfig,
    )
    from app.domain.models.runtime_extension import LivenessSnapshot
    from app.domain.models.skill_diagnostic import SkillDiagnostic
    from app.domain.repositories.skill_repository import SkillRepository
    from app.application.services.user_tool_enablement_service import (
        UserToolEnablementService,
    )

# §5.2 pin 类别 → GovernanceRowSnapshot pin 列（镜像 infrastructure `_PIN_FOR_CATEGORY`；
# application 不 import infrastructure，此处按 §5.1 冻结类别本地复刻，snapshot 词表锁保护）。
_PIN_COLUMN_FOR_CATEGORY = {
    "surface": "surface_hash",
    "artifact": "artifact_hash",
    "config_fingerprint": "config_fingerprint",
}

_BLOCKING_STATUSES = frozenset({"quarantined", "disabled"})


def _derive_pin_flags(row: "GovernanceRowSnapshot") -> tuple[bool, bool, bool]:
    """对该 kind 必需 pin 写集聚合派生（全 pinned→pinned；任一 unpinned→unpinned；
    任一 stale→pin_stale）——与 governance_counters 同源 pin_presence 规则。"""
    presences = [
        pin_presence(
            getattr(row, _PIN_COLUMN_FOR_CATEGORY[c]),
            row.hash_schema_version,
            HASH_SCHEMA_VERSION,
        )
        for c in REQUIRED_OBSERVED_CATEGORIES[row.kind]
    ]
    pinned = bool(presences) and all(p == "pinned" for p in presences)
    unpinned = any(p == "unpinned" for p in presences)
    pin_stale = any(p == "pin_stale" for p in presences)
    return pinned, unpinned, pin_stale


def _governance_info_from_row(row: "GovernanceRowSnapshot") -> ExtensionGovernanceInfo:
    """GovernanceRowSnapshot → ExtensionGovernanceInfo（§9.1 全字段平移 + pin 派生）。"""
    pinned, unpinned, pin_stale = _derive_pin_flags(row)
    return ExtensionGovernanceInfo(
        status=row.status,
        trust_origin=row.trust_origin,
        pinned=pinned,
        unpinned=unpinned,
        pin_stale=pin_stale,
        scan_verdict=row.scan_verdict,
        quarantine_reason=row.quarantine_reason,
        last_mismatch_at=row.last_mismatch_at,
        last_verified_at=row.last_verified_at,
        row_revision=row.row_revision,
        observed_surface_hash=row.observed_surface_hash,
        observed_artifact_hash=row.observed_artifact_hash,
        observed_config_fingerprint=row.observed_config_fingerprint,
        pinned_at=row.pinned_at,
        pinned_by=row.pinned_by,
        installed_by=row.installed_by,
        source_type=row.source_type,
        source_ref=row.source_ref,
        version=row.version,
        source_missing_at=row.source_missing_at,
        parent_plugin_ext_id=row.parent_plugin_ext_id,
    )

logger = logging.getLogger(__name__)

# P-5 冻结值镜像（PROBE_STALE_AFTER_SECONDS=120）——权威定义在
# extension_probe_service.py，但 INV-B9-4 零网络门禁止 GET 聚合器运行期 import
# 该模块（会传递引入 httpx/mcp 网络栈）。此处以纯常数镜像，语义与 P-5 逐字一致。
_STALE_AFTER_SECONDS = 120


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_log_id(value: str) -> str:
    """日志注入防御：仅保留可打印字符并截断（对齐审计路径 id_display 语义）。"""
    return "".join(ch for ch in str(value) if ch.isprintable())[:64]


class ProbeView(Protocol):
    """探测视图端口（Task 12 的 ExtensionProbeService 结构化满足）。

    GET 聚合只读 `snapshot()`；`reconcile()` 由后台循环消费（本 task 不调用，
    仅为契约完整性声明）。PR-1 阶段注入 None → 全 stub。
    """

    def snapshot(self) -> dict[tuple[str, str], Any]: ...

    def reconcile(
        self,
        live_keys: set[tuple[str, str]],
        fingerprints: dict[tuple[str, str], str],
    ) -> list[tuple[str, str]]: ...


class LivenessView(Protocol):
    """liveness 视图端口（Task 10 的 RuntimeLivenessRegistry 满足）。PR-1 注入 None。"""

    def snapshot(self) -> "LivenessSnapshot": ...


class RuntimeExtensionService:
    """聚合运行时扩展清单 + 角色投影（P-4 钉子）。"""

    def __init__(
        self,
        config_provider: Callable[[], "AppConfig"],
        skill_repository: "SkillRepository",
        enablement_service: "UserToolEnablementService",
        probe_view: "ProbeView | None" = None,
        liveness_view: "LivenessView | None" = None,
        stats_reader: "ExtensionStatsReader | None" = None,
        probe_enabled_provider: Callable[[], bool] = lambda: False,
        stats_enabled_provider: Callable[[], bool] = lambda: False,
        clock: Callable[[], datetime] = _utcnow,
        registry_read_port: "ExtensionRegistryReadPort | None" = None,
        plugin_name_resolver: Callable[[str, str | None], str] | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._skill_repository = skill_repository
        self._enablement_service = enablement_service
        self._probe_view = probe_view
        self._liveness_view = liveness_view
        self._stats_reader = stats_reader
        self._probe_enabled_provider = probe_enabled_provider
        self._stats_enabled_provider = stats_enabled_provider
        # 测试注入 fake clock；生产默认 UTC now。stale 计算与 snapshot_at 复用同一时钟源。
        self._clock = clock
        # D1a Task 25：治理只读 port（None=mode off，INV-D1-0 零行为）+ plugin 展示名 resolver
        # （None → fallback ext_id；provider 注入 read_plugin_display_name 偏函数）。
        self._registry_read_port = registry_read_port
        self._plugin_name_resolver = plugin_name_resolver

    async def get_extensions(
        self, *, user_id: str, is_admin: bool
    ) -> RuntimeExtensionsSnapshot:
        now = self._clock()  # 单一时钟源：stale 计算 + snapshot_at 复用
        app_config = self._config_provider()  # 失败冒泡（根基数据源，INV：无 config 无响应）
        enablement_map = await self._load_enablement_map(user_id)  # None = user 级降级
        diags = await self._load_skill_diagnostics()  # None = skill 条目缺失（降级）
        probe_enabled = self._probe_enabled_provider()

        # 惰性 reconcile（INV-B9-4 白名单内的本地副作用——纯 evict/reset 簿记，绝不探测）：
        # 从 config 清单构造 mcp/a2a live_keys + fingerprints，喂给 probe_view.reconcile；
        # 被剔除 keys 转发 stats DEL（PR-3 起生效；stats_reader None 时忽略）。
        self._reconcile_probe_view(app_config)

        probe_snapshot = self._probe_view.snapshot() if self._probe_view else {}
        liveness = self._liveness_view.snapshot() if self._liveness_view else None

        items: list[ExtensionItemInfo] = []
        # 条目顺序：mcp（配置序）→ a2a（配置序）→ skill（diagnostics 序）
        for server_name, server_cfg in app_config.mcp_config.mcpServers.items():
            items.append(
                self._build_mcp_item(
                    server_name, server_cfg, enablement_map, probe_snapshot,
                    liveness, now, probe_enabled,
                )
            )
        for a2a_cfg in app_config.a2a_config.a2a_servers:
            items.append(
                self._build_a2a_item(
                    a2a_cfg, enablement_map, probe_snapshot, liveness,
                    now, probe_enabled,
                )
            )
        if diags is not None:
            for diag in diags:
                items.append(self._build_skill_item(diag, enablement_map))

        # D1a Task 25：治理投影——拉全部 live 行 map（read_port=None → mode off，零行为
        # INV-D1-0）。读失败降级：零 governance/零 plugin（R3#2，config/skill 照常）。
        governance_map: dict[tuple[str, str], "GovernanceRowSnapshot"] = {}
        if self._registry_read_port is not None:
            try:
                governance_map = {
                    (r.kind, r.ext_id): r
                    for r in await self._registry_read_port.list_live_rows()
                }
            except Exception:  # noqa: BLE001 — 治理读故障不 5xx（降级：零 governance）
                logger.warning(
                    "governance projection degraded (read failed)", exc_info=True
                )
        # 第五段（序尾）：plugin 元容器条目
        items.extend(await self._build_plugin_items(governance_map))
        # 逐条 attach 治理块 + 阻断投影（在 stats/角色投影之前）
        items = [self._attach_governance(info, governance_map) for info in items]

        stats_map = await self._load_stats({(i.kind, i.id) for i in items})
        items = [
            self._attach_stats_and_project(info, stats_map, is_admin) for info in items
        ]
        return RuntimeExtensionsSnapshot(
            items=tuple(items),
            snapshot_at=now,
            probe_enabled=probe_enabled,
            stats_enabled=self._stats_enabled_provider(),
        )

    # ------------------------------------------------------------------ #
    # 惰性 reconcile（INV-B9-4 本地簿记副作用）
    # ------------------------------------------------------------------ #
    def _reconcile_probe_view(self, app_config: "AppConfig") -> None:
        """从 config 清单构造 mcp/a2a live_keys + fingerprints → probe_view.reconcile；
        被剔除的 (kind, id) 转发 stats_reader.delete_key（PR-3 起生效；None 忽略）。

        skill 不进 probe reconcile（skill 无网络探测语义，健康走 repo 直读，spec-R9#1）。
        probe_view=None（PR-1）时整段短路。**纯簿记（evict/reset）——绝不触发探测/网络
        （INV-B9-4）。**
        """
        if self._probe_view is None:
            return
        live_keys: set[tuple[str, str]] = set()
        fingerprints: dict[tuple[str, str], str] = {}
        for server_name, server_cfg in app_config.mcp_config.mcpServers.items():
            key = ("mcp", server_name)
            live_keys.add(key)
            fingerprints[key] = self._fingerprint(server_cfg)
        for a2a_cfg in app_config.a2a_config.a2a_servers:
            key = ("a2a", a2a_cfg.id)
            live_keys.add(key)
            fingerprints[key] = self._fingerprint(a2a_cfg)

        try:
            evicted = self._probe_view.reconcile(live_keys, fingerprints)
        except Exception:  # noqa: BLE001 — reconcile 簿记失败不阻断 GET（R3#2 降级）
            logger.warning("probe view reconcile failed; skipping eviction", exc_info=True)
            return

        if self._stats_reader is None or not evicted:
            return
        for kind, ext_id in evicted:
            try:
                self._stats_reader.delete_key(kind, ext_id)  # fire-and-forget
            except Exception as exc:  # noqa: BLE001 — DEL 入队失败不影响响应
                # ext_id 攻击者可控（config 条目 id）——过脱敏防日志注入。
                # 不带 exc_info：异常文本可能内嵌 raw key（如下游把 ext_id 拼进
                # message），traceback 渲染会绕过 _safe_log_id；只记异常类型名。
                logger.warning(
                    "stats delete_key failed for evicted %s/%s (%s)",
                    kind, _safe_log_id(ext_id), type(exc).__name__,
                )

    def _fingerprint(self, config_obj: Any) -> str:
        """与 ExtensionProbeService._fingerprint 同规则（model_dump(mode=json)+sha256），
        保证 reconcile 指纹比对同源——纯读，无网络。"""
        return hashlib.sha256(
            json.dumps(
                config_obj.model_dump(mode="json"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------ #
    # 数据源加载（各自降级）
    # ------------------------------------------------------------------ #
    async def _load_enablement_map(
        self, user_id: str
    ) -> dict[tuple[str, str], bool] | None:
        """{(kind, tool_id): enabled}；DB 失败返回 None（→ user_enablement_unknown）。"""
        try:
            rows = await self._enablement_service.list_user_enablements(user_id)
        except Exception:  # noqa: BLE001 — 降级：任何下游 DB 故障不 5xx（R5#2）
            logger.warning(
                "user enablement lookup failed; degrading to user_enablement_unknown",
                exc_info=True,
            )
            return None
        result: dict[tuple[str, str], bool] = {}
        for row in rows:
            tool_type = row.tool_type
            kind = tool_type.value if isinstance(tool_type, ToolType) else str(tool_type)
            result[(kind, row.tool_id)] = bool(row.enabled)
        return result

    async def _load_skill_diagnostics(self) -> list["SkillDiagnostic"] | None:
        """diagnostics 列表；失败返回 None（→ skill 条目缺失 + warn，mcp/a2a 照常）。"""
        try:
            return await self._skill_repository.list_with_diagnostics()
        except Exception:  # noqa: BLE001 — 降级：skill 扫描故障不影响 mcp/a2a（R3#2）
            logger.warning(
                "skill diagnostics scan failed; omitting skill items", exc_info=True
            )
            return None

    async def _load_stats(
        self, keys: set[tuple[str, str]]
    ) -> dict[tuple[str, str], Any] | None:
        """统计快照；reader=None（PR-1）或失败返回 None（→ redis_unavailable 兜底）。"""
        if self._stats_reader is None or not self._stats_enabled_provider():
            return None
        try:
            return await self._stats_reader.read_many(list(keys))
        except Exception:  # noqa: BLE001 — 降级：stats 读故障不 5xx（R3#2）
            logger.warning("extension stats read failed; degrading", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # 条目组装（PR-1 stub 语义）
    # ------------------------------------------------------------------ #
    def _resolve_config(
        self,
        kind: str,
        ext_id: str,
        enabled_global: bool,
        enablement_map: dict[tuple[str, str], bool] | None,
    ) -> ExtensionConfigInfo:
        """global/user → effective + reason_code（R8#4 六值矩阵 + R5#2 降级）。"""
        if enablement_map is None:
            # user 级 DB 降级：enabled_user 未知，effective 按 global 回退
            return ExtensionConfigInfo(
                enabled_global=enabled_global,
                enabled_user=None,
                effective_enabled=enabled_global,
                reason_code="user_enablement_unknown",
            )
        # 无记录默认启用（对齐 UserToolEnablementService.is_tool_enabled_for_user）
        enabled_user = enablement_map.get((kind, ext_id), True)
        effective = enabled_global and enabled_user
        if enabled_global and enabled_user:
            reason_code = "enabled"
        elif enabled_global and not enabled_user:
            reason_code = "disabled_user"
        elif not enabled_global and enabled_user:
            reason_code = "disabled_global"
        else:
            reason_code = "disabled_both"
        return ExtensionConfigInfo(
            enabled_global=enabled_global,
            enabled_user=enabled_user,
            effective_enabled=effective,
            reason_code=reason_code,
        )

    def _probe_health(
        self,
        enabled_global: bool,
        record: Any | None,
        now: datetime,
        probe_enabled: bool,
    ) -> ExtensionHealthInfo:
        """probe kind health（R11#2 / R12#3 / F10）：

        - global disabled → skipped（压过一切；health 只看 global，user 级 disable 不影响）
        - flag-off（probe_enabled=False）→ unknown/None/stale=False（内部 record 不清，仅投影清空）
        - 有 record → 按 record 落字段，stale = last_checked_at is not None and
          now - last_checked_at > 120s
        - 无 record → unknown（无探测数据）
        """
        if not enabled_global:
            return ExtensionHealthInfo(kind="probe", state="skipped")
        if not probe_enabled or record is None:
            # flag-off 投影清空 / 无记录 → unknown（时间戳全清、stale=False）
            return ExtensionHealthInfo(kind="probe", state="unknown")
        last_checked = getattr(record, "last_checked_at", None)
        stale = (
            last_checked is not None
            and (now - last_checked) > timedelta(seconds=_STALE_AFTER_SECONDS)
        )
        return ExtensionHealthInfo(
            kind="probe",
            state=getattr(record, "state", "unknown"),
            last_checked_at=last_checked,
            latency_ms=getattr(record, "latency_ms", None),
            error_code=getattr(record, "error_code", None),
            error_message=getattr(record, "error_message", None),
            stale=stale,
            consecutive_failures=getattr(record, "consecutive_failures", 0),
            next_probe_at=getattr(record, "next_probe_at", None),
        )

    def _probe_liveness(
        self,
        kind: str,
        ext_id: str,
        liveness: "LivenessSnapshot | None",
    ) -> ExtensionLivenessInfo:
        """mcp/a2a liveness（R11#3 / R12#2）：

        - liveness_view=None → unknown（PR-1 stub）
        - degraded → unknown（权威降级）
        - active[(kind,id)] 非空 → (in_use, len(runs))；空 → (idle, 0)
        """
        if liveness is None:
            return ExtensionLivenessInfo(state="unknown", active_run_count=0)
        if liveness.degraded:
            return ExtensionLivenessInfo(state="unknown", active_run_count=0)
        runs = liveness.active.get((kind, ext_id))
        if runs:
            return ExtensionLivenessInfo(state="in_use", active_run_count=len(runs))
        return ExtensionLivenessInfo(state="idle", active_run_count=0)

    def _build_mcp_item(
        self,
        server_name: str,
        server_cfg: "MCPServerConfig",
        enablement_map: dict[tuple[str, str], bool] | None,
        probe_snapshot: dict[tuple[str, str], Any],
        liveness: "LivenessSnapshot | None",
        now: datetime,
        probe_enabled: bool,
    ) -> ExtensionItemInfo:
        enabled_global = bool(server_cfg.enabled)
        config = self._resolve_config("mcp", server_name, enabled_global, enablement_map)
        transport = (
            server_cfg.transport.value
            if hasattr(server_cfg.transport, "value")
            else str(server_cfg.transport)
        )
        record = probe_snapshot.get(("mcp", server_name))
        # Admin details.tool_count：flag-on 且有 record 才取；flag-off/无 record → None
        # （flag-off 投影清空，与 health 时间戳清空同门控 R12#3）。
        tool_count = (
            getattr(record, "tool_count", None)
            if (probe_enabled and record is not None)
            else None
        )
        details: dict[str, Any] = {"transport": transport, "tool_count": tool_count}
        return ExtensionItemInfo(
            kind="mcp",
            id=server_name,
            name=server_name,
            description=server_cfg.description,
            config=config,
            health=self._probe_health(enabled_global, record, now, probe_enabled),
            liveness=self._probe_liveness("mcp", server_name, liveness),
            stats=ExtensionStatsInfo(available=False),  # reason 在投影阶段裁定
            details=details,
        )

    def _build_a2a_item(
        self,
        a2a_cfg: "A2AServerConfig",
        enablement_map: dict[tuple[str, str], bool] | None,
        probe_snapshot: dict[tuple[str, str], Any],
        liveness: "LivenessSnapshot | None",
        now: datetime,
        probe_enabled: bool,
    ) -> ExtensionItemInfo:
        ext_id = a2a_cfg.id
        enabled_global = bool(a2a_cfg.enabled)
        config = self._resolve_config("a2a", ext_id, enabled_global, enablement_map)
        # A2A 身份（R9#3 / R1#5）：record.display_name（agent_card.name）优先，
        # 无探测数据（probe_view=None → record 不存在）→ 前缀 fallback。description
        # 严禁 fallback 到 base_url（恒 None）。身份是长期属性，不随 flag-off 清空
        # （成功探测过的 display_name 保留展示——只有 health/tool_count 时间戳类字段受
        # flag-off 投影门控）。
        record = probe_snapshot.get(("a2a", ext_id))
        display_name = getattr(record, "display_name", None) if record else None
        name = display_name or f"A2A {ext_id[:8]}"
        # Admin details：{streaming, base_url}；streaming stub=None（探测不承载 streaming）
        details: dict[str, Any] = {"streaming": None, "base_url": a2a_cfg.base_url}
        return ExtensionItemInfo(
            kind="a2a",
            id=ext_id,
            name=name,
            description=None,
            config=config,
            health=self._probe_health(enabled_global, record, now, probe_enabled),
            liveness=self._probe_liveness("a2a", ext_id, liveness),
            stats=ExtensionStatsInfo(available=False),
            details=details,
        )

    def _build_skill_item(
        self,
        diag: "SkillDiagnostic",
        enablement_map: dict[tuple[str, str], bool] | None,
    ) -> ExtensionItemInfo:
        if not diag.ok:
            return self._build_broken_skill_item(diag)
        skill = diag.skill
        assert skill is not None  # ok=True 契约保证（P-1）
        ext_id = skill.id
        enabled_global = bool(skill.enabled)
        config = self._resolve_config("skill", ext_id, enabled_global, enablement_map)
        runtime_type = (
            skill.runtime_type.value
            if hasattr(skill.runtime_type, "value")
            else str(skill.runtime_type)
        )
        source_type = (
            skill.source_type.value
            if hasattr(skill.source_type, "value")
            else str(skill.source_type)
        )
        manifest = skill.manifest if isinstance(skill.manifest, dict) else {}
        bundle_file_count = manifest.get("bundle_file_count")
        # skill health（integrity kind）：好条目 ok；disabled(global) → skipped
        # （per health 优先级：global_disabled → skipped 压过一切配置可读条目）；kind 不变。
        # 仅看 global（enabled_user 不影响共享 health）。
        health = ExtensionHealthInfo(
            kind="integrity", state="ok" if enabled_global else "skipped"
        )
        return ExtensionItemInfo(
            kind="skill",
            id=ext_id,
            name=skill.name,
            description=skill.description,
            config=config,
            health=health,
            liveness=ExtensionLivenessInfo(state="not_applicable", active_run_count=0),
            stats=ExtensionStatsInfo(available=False),
            details={
                "runtime_type": runtime_type,
                "source_type": source_type,
                "bundle_file_count": bundle_file_count,
            },
        )

    def _build_broken_skill_item(self, diag: "SkillDiagnostic") -> ExtensionItemInfo:
        """坏 skill fallback（R7#2 / R1#9 / R8#3）：id=name=目录名；config_unreadable；
        health=integrity error（**恒 error 不落 skipped**），error_code/relative_file
        直通 diagnostic 原值。"""
        key = diag.skill_key
        return ExtensionItemInfo(
            kind="skill",
            id=key,
            name=key,
            description=None,
            config=ExtensionConfigInfo(
                enabled_global=False,
                enabled_user=None,
                effective_enabled=False,
                reason_code="config_unreadable",
            ),
            health=ExtensionHealthInfo(
                kind="integrity",
                state="error",
                error_code=diag.error_code,
                relative_file=diag.relative_file,
            ),
            liveness=ExtensionLivenessInfo(state="not_applicable", active_run_count=0),
            stats=ExtensionStatsInfo(available=False),
            details={
                "runtime_type": "unknown",
                "source_type": "unknown",
                "bundle_file_count": None,
            },
        )

    # ------------------------------------------------------------------ #
    # D1a Task 25：plugin 第五段 + 治理 attach
    # ------------------------------------------------------------------ #
    async def _build_plugin_items(
        self, governance_map: dict[tuple[str, str], "GovernanceRowSnapshot"]
    ) -> list[ExtensionItemInfo]:
        """plugin 元容器条目（第五段，序尾）：来源=governance_map 中 kind=plugin 行（按
        ext_id 稳定序）。details={member_count, plugin_version}；member_count=governance_map
        中 parent_plugin_ext_id 命中该 plugin 的成员行数。展示名经 resolver（线程池，
        registry 无 name 列）——None → fallback ext_id。config=not_applicable_plugin（阻断
        投影在 _attach_governance 统一处理）；health=integrity、liveness=not_applicable、
        stats reason=unsupported（在投影阶段裁定）。"""
        plugin_rows = sorted(
            (r for r in governance_map.values() if r.kind == "plugin"),
            key=lambda r: r.ext_id,
        )
        items: list[ExtensionItemInfo] = []
        for row in plugin_rows:
            member_count = sum(
                1
                for r in governance_map.values()
                if r.parent_plugin_ext_id == row.ext_id
            )
            if self._plugin_name_resolver is not None:
                # 仓库全异步硬约束：同步小文件读走线程池（helper 本体保持纯同步便于单测）
                name = await asyncio.to_thread(
                    self._plugin_name_resolver, row.ext_id, row.version
                )
            else:
                name = row.ext_id
            # #5：plugin 无独立 config 级用户偏好——其全局启用态即治理状态。enabled_global
            # 须映射真实治理启用（active=True / quarantined|disabled=False），使 FE Switch
            # 不对已停用/隔离 plugin 恒显 ON。阻断细化（effective_enabled=False +
            # reason_code=governance_blocked）仍由 _attach_governance 统一裁定。
            governance_enabled = row.status not in _BLOCKING_STATUSES
            items.append(
                ExtensionItemInfo(
                    kind="plugin",
                    id=row.ext_id,
                    name=name,
                    description=None,
                    config=ExtensionConfigInfo(
                        enabled_global=governance_enabled,
                        enabled_user=None,
                        effective_enabled=governance_enabled,
                        reason_code="not_applicable_plugin",
                    ),
                    health=ExtensionHealthInfo(kind="integrity", state="ok"),
                    liveness=ExtensionLivenessInfo(
                        state="not_applicable", active_run_count=0
                    ),
                    stats=ExtensionStatsInfo(available=False),
                    details={"member_count": member_count, "plugin_version": row.version},
                )
            )
        return items

    def _attach_governance(
        self,
        info: ExtensionItemInfo,
        governance_map: dict[tuple[str, str], "GovernanceRowSnapshot"],
    ) -> ExtensionItemInfo:
        """纯 helper：命中治理行 → attach ExtensionGovernanceInfo（Admin 裁剪在
        _project_item）；命中且行阻断（own status ∈ {quarantined, disabled}）或父 plugin
        阻断（成员行经 parent_plugin_ext_id → 父行 status 查 governance_map）→ config
        投影替换 effective_enabled=False + reason_code=governance_blocked（普通用户可见的
        唯一治理泄漏）。用户启停偏好（enabled_global/enabled_user）原样保留。"""
        if not governance_map:
            return info
        row = governance_map.get((info.kind, info.id))
        if row is None:
            return info
        info = dataclasses.replace(info, governance=_governance_info_from_row(row))
        blocked = row.status in _BLOCKING_STATUSES
        if not blocked and row.parent_plugin_ext_id is not None:
            parent = governance_map.get(("plugin", row.parent_plugin_ext_id))
            blocked = parent is not None and parent.status in _BLOCKING_STATUSES
        if blocked:
            info = dataclasses.replace(
                info,
                config=dataclasses.replace(
                    info.config,
                    effective_enabled=False,
                    reason_code="governance_blocked",
                ),
            )
        return info

    # ------------------------------------------------------------------ #
    # 统计裁定 + 角色投影
    # ------------------------------------------------------------------ #
    def _stats_reason(
        self, kind: str, is_admin: bool
    ) -> "str":
        """stats.unavailable_reason 优先级（R5#4）：
        admin_only ＞ unsupported ＞ disabled ＞ redis_unavailable。

        PR-1 stub（stats_reader=None）：非 Admin → admin_only；A2A/plugin → unsupported；
        mcp/skill → disabled。真实 reader（PR-3）读取失败才 redis_unavailable。
        """
        if not is_admin:
            return "admin_only"
        if kind in ("a2a", "plugin"):
            return "unsupported"  # A2A（v1 无归因）+ plugin（元容器不采集执行统计）
        # disabled = 有效开关 false（extension_stats_enabled=false，含启动失败——DI
        # provider 已封装 started ∧ flag）；压过 redis_unavailable。
        if not self._stats_enabled_provider():
            return "disabled"
        # flag on 但无数据：reader None 或 PR-1 stub 阶段未读 → redis_unavailable
        # （PR-3 接入真实数据路径后细化）。
        return "redis_unavailable"

    def _attach_stats_and_project(
        self,
        info: ExtensionItemInfo,
        stats_map: dict[tuple[str, str], Any] | None,
        is_admin: bool,
    ) -> ExtensionItemInfo:
        # 先算 unavailable_reason（优先级 admin_only ＞ unsupported ＞ disabled ＞
        # redis_unavailable）。只有当该 reason 是 redis_unavailable（= Admin + 非
        # A2A + flag on）且 stats_map 里有该条目的真实数据时，才平移为 available=True。
        # 其余情形保持 available=False + reason（PR-1 stub / A2A unsupported /
        # 非 Admin admin_only / flag off disabled / flag on 但无数据）。
        reason = self._stats_reason(info.kind, is_admin)
        stats = ExtensionStatsInfo(available=False, unavailable_reason=reason)
        if reason == "redis_unavailable" and stats_map is not None:
            data = stats_map.get((info.kind, info.id))
            if data is not None:
                # ExtensionStatsData → ExtensionStatsInfo 逐字段平移（Task 19 → 20）。
                stats = ExtensionStatsInfo(
                    available=True,
                    unavailable_reason=None,
                    call_count=data.call_count,
                    success_count=data.success_count,
                    failure_count=data.failure_count,
                    last_active_at=data.last_active_at,
                    last_success_at=data.last_success_at,
                    last_failure_at=data.last_failure_at,
                )
        info = dataclasses.replace(info, stats=stats)
        return self._project_item(info, is_admin=is_admin)

    def _project_item(
        self, info: ExtensionItemInfo, *, is_admin: bool
    ) -> ExtensionItemInfo:
        """角色投影（纯函数）：Admin 全量；非 Admin 最小投影（R4#8/R8#1/R16#4/R17#1）。

        非 Admin 裁剪：health 隐藏 latency_ms/error_code/error_message/relative_file/
        next_probe_at + consecutive_failures 置 0；liveness mcp/a2a=("unknown",0)、
        skill=not_applicable；details 精确 shape（mcp={transport}、a2a={streaming}、
        skill={runtime_type}）；description 严禁 fallback 到 base_url。
        """
        if is_admin:
            return info
        # 非 Admin：治理块整体剥离（Admin-only 三层剥离第①层——service 投影层，R7#7）。
        info = dataclasses.replace(info, governance=None)
        health = dataclasses.replace(
            info.health,
            latency_ms=None,
            error_code=None,
            error_message=None,
            relative_file=None,
            consecutive_failures=0,
            next_probe_at=None,
        )
        if info.kind == "skill":
            liveness = ExtensionLivenessInfo(state="not_applicable", active_run_count=0)
            details = {"runtime_type": info.details.get("runtime_type", "unknown")}
        elif info.kind == "mcp":
            liveness = ExtensionLivenessInfo(state="unknown", active_run_count=0)
            details = {"transport": info.details.get("transport")}
        elif info.kind == "plugin":
            # plugin 显式分支（防误落 a2a else，R2#6）；member_count/plugin_version 非敏感
            # 元数据，非 Admin 保留（治理块已剥离）。
            liveness = ExtensionLivenessInfo(state="not_applicable", active_run_count=0)
            details = {
                "member_count": info.details.get("member_count"),
                "plugin_version": info.details.get("plugin_version"),
            }
        else:  # a2a
            liveness = ExtensionLivenessInfo(state="unknown", active_run_count=0)
            details = {"streaming": info.details.get("streaming")}
        return dataclasses.replace(
            info, health=health, liveness=liveness, details=details
        )
