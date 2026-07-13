"""D1a §9.2 治理 wire DTO（T19 起步 ExtensionInstallPreview；T20/T24 续用同文件）。

仅 pydantic + domain 模型——**禁 import FastAPI/SQLAlchemy**（application 层的
``ExtensionInstallService`` 返回本 DTO，镜像 ``app_config_service`` 复用 interfaces
schema 的既有务实耦合；DTO 不得引入框架依赖以免污染 application 纯度）。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.extension_governance import (
    GovernanceScanSummary,
    GovernedExtensionKind,
)


class ExtensionInstallPreview(BaseModel):
    """两阶段安装的 preview（dry_run）响应体——零写观测投影 + policy 决策。

    - ``scan_report``：§7.3 扫描摘要（唯一对外 scan 形态，禁原始 match 文本）。
    - ``observed_surface``：probe 观测到的表面脱敏摘要（mcp=工具名+描述截断列表 /
      a2a=卡片名+描述+skills 名；probe 失败/空=None）。
    - ``surface_hash`` / ``config_fingerprint``：§5.1 canonicalizer pin 值（surface
      probe 失败=None）。
    - ``install_policy_decision``：§7.2 决策（allow / allow_with_warnings /
      need_acknowledge / need_force）——preview 只报告不拦。
    - ``warnings``：probe 失败 / stdio 环境继承等提示。
    """

    model_config = ConfigDict(extra="forbid")

    scan_report: GovernanceScanSummary
    observed_surface: list[dict] | dict | None = None
    surface_hash: str | None = None
    config_fingerprint: str
    install_policy_decision: str
    warnings: list[str] = Field(default_factory=list)


# ---- T20 §9.2 治理路由请求体（CAS 强制非可选，R4-01） --------------------


class GovernanceCASBody(BaseModel):
    """reapprove / governance-enable / governance-disable：仅 expected_row_revision。"""

    model_config = ConfigDict(extra="forbid")

    expected_row_revision: int


class GovernanceQuarantineBody(BaseModel):
    """quarantine：expected_row_revision 必填 + 可选 note。"""

    model_config = ConfigDict(extra="forbid")

    expected_row_revision: int
    note: str | None = None


class RefreshBatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: GovernedExtensionKind   # 闭词表：非法 kind → 422 parse-time（与单项路由 _validate_kind 一致）
    ext_id: str


class RefreshBatchBody(BaseModel):
    """POST /refresh-observations：{"all": true} | {"items": [{kind, ext_id}]}。"""

    model_config = ConfigDict(extra="forbid")

    all: bool = False
    items: list[RefreshBatchItem] | None = None


class ApprovePinItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: GovernedExtensionKind   # 闭词表：非法 kind → 422 parse-time（与单项路由 _validate_kind 一致）
    ext_id: str
    expected_row_revision: int | None = None   # R36#7：可空 → 回退 fetched row_revision


class ApprovePinsBody(BaseModel):
    """POST /approve-pins：{"items": [{kind, ext_id, expected_row_revision?}]} | {"all": true}。"""

    model_config = ConfigDict(extra="forbid")

    all: bool = False
    items: list[ApprovePinItem] | None = None


# ---- T24 §9.2 Plugin 路由请求/响应 DTO（尾三条 + install + list） -------------


class PluginInstallRequest(BaseModel):
    """POST /v2/plugins/install：全 body（R13#8）——source + dry_run/force/acknowledge。

    ``source_type`` 为字符串（``local`` / ``github`` / ...），route 侧转 ``SkillSourceType``
    枚举（非法值 → 422）。``dry_run=True`` → 零写 preview；``force`` / ``acknowledge`` 承载
    §7.2 policy 门放行意图（safe 成员恒不消费）。"""

    model_config = ConfigDict(extra="forbid")

    source_type: str
    source_ref: str
    dry_run: bool = False
    force: bool = False
    acknowledge: bool = False


class PluginUninstallBody(BaseModel):
    """DELETE /v2/plugins/{plugin_ext_id}：``expected_row_revision`` 必填（缺 → 422，CAS 强制）。"""

    model_config = ConfigDict(extra="forbid")

    expected_row_revision: int


class PluginEnabledBody(BaseModel):
    """POST /v2/plugins/{plugin_ext_id}/enabled：``enabled`` + ``expected_row_revision``。

    enabled=true/false 均复用 T20 ``ExtensionGovernanceService.set_enabled``（同一迁移服务 +
    前置——非终态 operation → 409 operation_pending，防专用端点绕过，R47#1）。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    expected_row_revision: int


class PluginLastOperation(BaseModel):
    """PluginDetail 内嵌最新 saga operation 摘要（membership/operations 声明读者）。"""

    model_config = ConfigDict(extra="forbid")

    type: str
    state: str
    error: str | None = None
    updated_at: str | None = None


class PluginMemberDetail(BaseModel):
    """PluginDetail 内嵌成员摘要（membership join 子行 registry 状态 + pin/scan 声明读者）。"""

    model_config = ConfigDict(extra="forbid")

    declared_component_id: str
    kind: str
    ext_id: str
    expected_hash: str | None = None
    installed_version: str | None = None
    managed_by_plugin: bool
    status: str
    scan_verdict: str | None = None
    scan_report: dict | None = None


class PluginDetail(BaseModel):
    """GET /v2/plugins 单条 + install completed 摘要（R6#9 显式 DTO）。

    ``name`` 无 registry 列（manifest name 不落库）→ 恒 None（保留字段供 D1b 演进）。"""

    model_config = ConfigDict(extra="forbid")

    ext_id: str
    name: str | None = None
    version: str | None = None
    status: str
    artifact_hash: str | None = None
    row_revision: int
    last_operation: PluginLastOperation | None = None
    members: list[PluginMemberDetail] = Field(default_factory=list)
