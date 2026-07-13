"""B9 运行时扩展聚合 wire schema（interfaces）——spec §2 状态契约的逐字镜像。

设计原则：共享 base + 按 kind 判别载荷（codex Q-A 裁决），杜绝 "connected" 谎言
（INV-B9-1）。`ExtensionKind` 与聚合中间 DTO 的权威定义在 domain
（`app.domain.models.runtime_extension`）；本模块复用 domain 的 Literal 并把
domain DTO（`RuntimeExtensionsSnapshot` 等）平移为 Pydantic wire model，由 endpoint
消费。domain 层禁止 import 本模块（interfaces → domain 单向）。

wire 格式约定（R6#1/#8）：所有 ``datetime`` 序列化为 ISO 8601 UTC 字符串
（如 ``2026-07-04T12:00:00Z``），前端按 ``string`` 建模。
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, field_serializer, model_serializer

from app.domain.models.runtime_extension import (
    ExtensionGovernanceInfo,
    ExtensionItemInfo,
    ExtensionKind,
    RuntimeExtensionsSnapshot,
)


def _iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class ExtensionConfigStatus(BaseModel):
    enabled_global: bool                    # F8 配置层
    enabled_user: bool | None = None        # F10 用户级（当前请求者视角；无记录=True）
    effective_enabled: bool                 # global AND user
    reason_code: str                        # openclaw 式可解释启停：
                                            # "enabled" | "disabled_global" | "disabled_user" | "disabled_both"
                                            # | "user_enablement_unknown"（R5#2：user 级 DB 查询降级时，
                                            #   enabled_user=None + effective 按 global 回退 + 前端
                                            #   用户级 Switch 置灰并提示"用户偏好暂不可用"）
                                            # | "config_unreadable"（R7#2：坏 skill 条目 fallback，
                                            #   两 Switch 均禁用，见 §8）


class ExtensionHealth(BaseModel):
    kind: Literal["probe", "integrity"]     # mcp/a2a=probe；skill=integrity
    state: Literal[
        "reachable", "unreachable",         # probe 专用
        "ok", "error",                      # integrity 专用
        "unknown",                          # 未探测过 / flag off
        "skipped",                          # disabled 扩展不探测
    ]
    last_checked_at: datetime | None = None
    latency_ms: int | None = None           # 全握手耗时（hermes mcp test 模式）
    error_code: str | None = None           # 结构化（R10#1 补全）：
                                            # probe 专用: timeout | connect_failed | auth_failed |
                                            #             protocol_error | spawn_failed
                                            # integrity 专用: parse_error | missing_meta | missing_manifest
                                            # ——与 error_message 同级：仅 Admin 投影下发（R8#2/R7#1）
    error_message: str | None = None        # 脱敏短消息；仅 Admin 投影可见（§6 投影规则）
    relative_file: str | None = None        # integrity 专用（R11#1，P1 修）：坏 skill 的
                                            # "meta.json"|"manifest.json"；仅 Admin 投影下发
                                            # （与 error_code 同级），来源=SkillDiagnostic.relative_file
                                            # 不变式（R12#8）：kind="probe" ⇒ relative_file 恒 None
    stale: bool = False                     # now - last_checked_at > STALE_AFTER；
                                            # last_checked_at=None 时恒 false（R11#2：
                                            # "未探测"由 state=unknown 表达，不重复标 stale）
    consecutive_failures: int = 0
    next_probe_at: datetime | None = None   # 退避可视化（后台探测启用时）

    @field_serializer("last_checked_at", "next_probe_at")
    def _ser_dt(self, v: datetime | None) -> str | None:
        return _iso_utc(v)


class ExtensionLiveness(BaseModel):
    state: Literal["in_use", "idle", "unknown", "not_applicable"]
                                            # 权威转换规则（R11#3）：
                                            # kind=skill → 恒 not_applicable；
                                            # 旁路采集异常（fail-open 吞掉后）→ unknown（R4#5，
                                            #   绝不假报 idle）；
                                            # 否则 active_run_count>0 → in_use；==0 → idle
    active_run_count: int = 0               # 当前持有该扩展 live session 的 task 数


class ExtensionStats(BaseModel):
    available: bool                         # False 时看 unavailable_reason
    unavailable_reason: Literal[
        "disabled",                         # extension_stats_enabled=false
        "redis_unavailable",                # Redis 读失败
        "unsupported",                      # A2A（v1 无归因，§5.1/D9）
        "admin_only",                       # 非 Admin 投影隐藏（§6）
    ] | None = None                         # R4#7：available=false 必带 reason，前端按 reason 渲染
                                            # R5#4 优先级（多因子同时成立取最高）：
                                            # admin_only（非 Admin 一票否决）＞ unsupported（A2A）
                                            # ＞ disabled（flag off/启动失败）＞ redis_unavailable
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_active_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None

    @field_serializer("last_active_at", "last_success_at", "last_failure_at")
    def _ser_dt(self, v: datetime | None) -> str | None:
        return _iso_utc(v)


class GovernanceBlock(BaseModel):
    """D1a §9.1 R46#8：registry 治理块（Admin-only 投影；ExtensionGovernanceInfo 的
    wire 镜像）。None-omit：mode=off / 非 Admin 时整块不出现（见 ExtensionItem 序列化器）。
    kind=plugin 时 parent_plugin_ext_id 恒 None（顶层容器）；成员条目携父 plugin ext_id。"""
    status: str
    trust_origin: str
    pinned: bool
    unpinned: bool
    pin_stale: bool
    scan_verdict: str | None = None
    quarantine_reason: str | None = None
    last_mismatch_at: datetime | None = None
    last_verified_at: datetime | None = None
    row_revision: int
    observed_surface_hash: str | None = None
    observed_artifact_hash: str | None = None
    observed_config_fingerprint: str | None = None
    pinned_at: datetime | None = None
    pinned_by: str | None = None
    installed_by: str | None = None
    source_type: str
    source_ref: str | None = None
    version: str | None = None
    source_missing_at: datetime | None = None
    parent_plugin_ext_id: str | None = None

    @field_serializer(
        "last_mismatch_at", "last_verified_at", "pinned_at", "source_missing_at"
    )
    def _ser_dt(self, v: datetime | None) -> str | None:
        return _iso_utc(v)


class ExtensionItem(BaseModel):
    kind: ExtensionKind
    id: str                                 # mcp=server_name；a2a=config uuid；skill=skill.id
    name: str
    description: str | None = None
    config: ExtensionConfigStatus
    health: ExtensionHealth
    liveness: ExtensionLiveness
    stats: ExtensionStats
    details: dict[str, Any] = {}            # 按 kind 判别载荷（R6#1：字段集冻结，TS 侧以 kind 为
                                            # 判别键写 discriminated union）：
                                            # kind=mcp:   {transport: str, tool_count: int|None}
                                            # kind=a2a:   {streaming: bool|None, base_url: str|None}
                                            #             （base_url 仅 Admin 投影存在，§6）
                                            # kind=skill: {runtime_type: str, source_type: str,
                                            #              bundle_file_count: int|None}
                                            # kind=plugin:{member_count: int, plugin_version: str|None}
    governance: GovernanceBlock | None = None  # D1a：Admin-only 治理块；None 时序列化器 pop（F24）

    @model_serializer(mode="wrap")
    def _omit_none_governance(self, handler: Any) -> dict:
        """None-omit governance（F24/R2#1）：值为 None 时键**不存在**（非 null）——保证
        mode=off 字节 golden（T13 D1-0 ③层）零漂移。局部 wrap（**禁**全局 exclude_none），
        照抄 domain 侧 None-omit wrap serializer 先例；handler(self) 保留嵌套 serializer。"""
        data: dict = handler(self)
        if data.get("governance") is None:
            data.pop("governance", None)
        return data


class CatalogItem(BaseModel):               # R12#5：结构显式冻结（§7 语义详述）
    id: str                                 # 推荐默认 server_name（R1#7）
    name: str
    description: str
    transport: Literal["stdio", "sse", "streamable_http"]
    config_template: dict[str, Any]         # 完整 MCPServerConfig 形态（R8#5），含 transport
    homepage: str
    tags: list[str] = []
    source: str                             # 来源仓库 URL
    reviewed_at: str                        # ISO 日期 YYYY-MM-DD


class RuntimeExtensionCatalogResponse(BaseModel):
    items: list[CatalogItem]                # R11#8：catalog 端点响应外形冻结（非裸 list）


class RuntimeExtensionsResponse(BaseModel):
    items: list[ExtensionItem]
    snapshot_at: datetime                   # 快照组装时刻；消费口径（R11#4）：仅调试/日志用途，
                                            # 前端不消费（竞态判定用 request token，非时间戳）
    probe_enabled: bool                     # R5#3：语义 = 有效能力（flag on 且后台循环成功启动），
    stats_enabled: bool                     # 非 raw 配置值；启动 fail-open 后为 False。
                                            # UI 权威判据始终是 per-item 的 health.state /
                                            # stats.unavailable_reason，顶层两位仅作面板级横幅提示

    @field_serializer("snapshot_at")
    def _ser_dt(self, v: datetime) -> str | None:
        return _iso_utc(v)


def _to_governance_block(
    info: ExtensionGovernanceInfo | None,
) -> GovernanceBlock | None:
    """domain ExtensionGovernanceInfo → wire GovernanceBlock（None 透传 → 序列化器 pop）。"""
    if info is None:
        return None
    return GovernanceBlock(**dataclasses.asdict(info))


def to_wire_item(info: ExtensionItemInfo) -> ExtensionItem:
    return ExtensionItem(
        kind=info.kind, id=info.id, name=info.name, description=info.description,
        config=ExtensionConfigStatus(**dataclasses.asdict(info.config)),
        health=ExtensionHealth(**dataclasses.asdict(info.health)),
        liveness=ExtensionLiveness(**dataclasses.asdict(info.liveness)),
        stats=ExtensionStats(**dataclasses.asdict(info.stats)),
        details=dict(info.details),
        governance=_to_governance_block(info.governance),  # R1#16：唯一 domain→wire 平移点
    )


def to_wire_response(snap: RuntimeExtensionsSnapshot) -> RuntimeExtensionsResponse:
    return RuntimeExtensionsResponse(
        items=[to_wire_item(info) for info in snap.items],
        snapshot_at=snap.snapshot_at,
        probe_enabled=snap.probe_enabled,
        stats_enabled=snap.stats_enabled,
    )
