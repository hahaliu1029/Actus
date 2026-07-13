"""D1a §8.3 Plugin 元容器安装管道——T21 载入 + preflight + preview（saga 执行 T22/T23 续写）。

本任务（T21）实现 §8.3-1 **全内存 preflight**：载入 bundle → 构建不可变 ``PluginBundle``
快照 → manifest 校验 → 逐成员解析+scan（skill 全扫 / mcp,a2a §7.3）→ mcp/a2a probe →
expected_hash 强制核对 → 聚合 verdict → 身份碰撞查（read_port，锁外）→ INSTALL_POLICY gate →
组装不可变 ``PluginInstallContext``；``dry_run`` 分支产 ``PluginInstallPreview``（脱敏，零写）。

**写入例外（R12#2）**：真实 install（非 dry_run）被 preflight 拒绝时，允许且仅允许写一条
自持 identity 的 ``install_rejected`` audit（extension_id NULL）——preflight 唯一 DB 写；
``dry_run=True`` 时连这条也没有。

分层：application 层——**禁 import FastAPI/SQLAlchemy**。frozen dataclass/pydantic DTO 产物
供 T22 逐字消费（不可变，写入阶段不重算/重观测/重生成 id）。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from app.application.errors.exceptions import ConflictError, ValidationError
from app.application.services.extension_identity_locks import get_identity_locks
from app.application.services.extension_install_service import (
    AcknowledgeRequiredError,
    ForceRequiredError,
)
from app.domain.external.extension_admission import InstallContext, UninstallContext
from app.domain.models.app_config import A2AServerConfig, MCPConfig
from app.domain.models.extension_governance import (
    HASH_SCHEMA_VERSION,
    GovernanceScanFinding,
    GovernanceScanSummary,
)
from app.domain.models.plugin_manifest import (
    PluginManifest,
    UnsupportedManifestVersionError,
    detect_secret_literals,
)
from app.domain.models.skill import SkillRuntimeType, SkillSourceType, build_skill_key
from app.domain.services.extension_hashing import (
    a2a_config_fingerprint,
    a2a_surface_hash,
    entry_content_hash,
    mcp_config_fingerprint,
    mcp_surface_hash,
)
from app.domain.services.extension_scan import (
    canonicalize_source_ref,
    evaluate_install_policy,
    scan_a2a_entry,
    scan_mcp_entry,
)
from app.domain.services.skills_guard import SkillsGuard
from app.domain.services.trust_matrix import get_install_decision, scan_skill_source

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.application.services.skill_source_loader import SkillBundle

logger = logging.getLogger(__name__)

_GOV_FINDING_FIELD_MAX = 256
_VERDICT_ORDER = {"safe": 0, "caution": 1, "dangerous": 2}
_COMPRESSED_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".bz2", ".xz")
_PROBE_FAILED_WARNING = (
    "probe 失败：surface 未 pin（enforce 下该成员在 approve 前不可装配）"
)

# ------------------------------------------------------------ saga fs 布局（§8.3-3）--
# staging 与 skills/plugins store 同 volume 同 fs（rename 进位原子，R50#3）；staging 显式
# **不在 skill 根目录树下**（R51#4——避免 FileSkillRepository 把 .plugin-staging 投影成虚假
# broken skill）。模块级常量便于测试 monkeypatch 到 tmp（生产 = /app/data 下）。
PLUGIN_STAGING_ROOT = Path("/app/data/.plugin-staging")
PLUGIN_STORE_ROOT = Path("/app/data/plugins")
SKILL_STORE_ROOT = Path("/app/data/skills")

# rollback 三分中「内容缺失」哨兵 → 幂等成功（§3.4）
_MEMBER_TARGET_TYPES = ("skill_dir", "mcp_config", "a2a_config")
_CONFIG_TARGET_TYPES = ("mcp_config", "a2a_config")

# §3.4 saga 终态（staging 收尾 sweep 判定——非终态 in_progress 保留 staging）
_TERMINAL_OP_STATES = ("completed", "compensated", "failed")


# ------------------------------------------------------------------ exceptions --
class BundleContainmentError(Exception):
    """bundle 内 path 逃逸（绝对路径/`..`/symlink）——§8.2 containment 违约（映射 422）。"""


class PluginIdentityCollisionError(ConflictError):
    """父 (plugin, id) 或成员 (kind, ext_id) 已存活——v1 一律拒（升级=先卸后装，R13#6）→ 409。"""


class PluginExpectedHashMismatchError(ValidationError):
    """成员声明 expected_hash 与实测不符——防「声明 A 装 B」替换攻击（R1#8）→ 422。"""


class PluginMemberProbeFailedError(ValidationError):
    """mcp/a2a 成员 preflight probe 失败且未 force——默认整体拒装（R5#3）→ 422。"""


class _SagaAbort(Exception):
    """§8.3-3 内容写入前占用检出（目标已存在=不归本 operation 所有）——该 step 标 collided +
    中止转补偿（content_write_collision 语义，本模块内部异常）。"""

    def __init__(self, failed_seq: int, details: Mapping[str, Any]) -> None:
        super().__init__(f"saga abort at seq={failed_seq}")
        self.failed_seq = failed_seq
        self.details = dict(details)


class ReverifyError(Exception):
    """§8.3-4 发布前复验失败（磁盘 artifact / config fingerprint ≠ ctx pin）——转补偿；
    携带 stage=publish_reverify（补偿前写 install_rejected，extension_id=父行）。"""

    def __init__(self, member: str) -> None:
        super().__init__(f"publish reverify failed: {member}")
        self.member = member


# 一切 preflight 拒绝异常（真实 install 触发唯一 install_rejected audit + 外层重抛）
_REJECT_EXCEPTIONS = (
    ValidationError,
    ConflictError,
    AcknowledgeRequiredError,
    ForceRequiredError,
    UnsupportedManifestVersionError,
    BundleContainmentError,
    PydanticValidationError,
)


class PluginRejectAuditSink(Protocol):
    """preflight 拒绝时写自持 install_rejected audit（extension_id NULL）的注入面。

    生产实现（T24 布线）经 session 落 ``insert_audit(event="install_rejected",
    extension_id=None, ext_id=None, ...)``；测试注入 spy。``dry_run`` 时不调用。
    """

    async def record_install_rejected(
        self, *, source_type: str, source_ref: str | None,
        details: Mapping[str, Any] | None = None,
    ) -> None: ...


# ------------------------------------------------------------------- products --
@dataclass(frozen=True)
class Provenance:
    """安装 provenance 四字段（父=请求来源 / 成员=plugin 来源，R26#3/R27#2）。"""
    source_type: str
    source_ref: str | None
    version: str | None
    trust_origin: str = "user_installed"


@dataclass(frozen=True)
class PluginBundle:
    """§8.2 不可变内存字节快照（preflight 扫描 / expected_hash 核对 / 写入落盘共用同一快照——
    本地目录源的 preflight↔写入 TOCTOU 由此关闭）。"""

    normalized_source_ref: str
    files: dict[str, bytes]  # relpath → bytes 快照

    @classmethod
    def from_loader_bundle(cls, bundle: "SkillBundle") -> "PluginBundle":
        """变换 ``SkillBundle`` 产物为不可变 ``{relpath: bytes}`` 快照 + 逐 path containment
        断言（绝对路径/`..` → ``BundleContainmentError``；loader 已拒 symlink=F26 继承，本层
        为防御纵深重校）。"""
        files: dict[str, bytes] = {}
        for rel, entry in bundle.files.items():
            pure = PurePosixPath(rel)
            if pure.is_absolute() or rel.startswith("/") or ".." in pure.parts:
                raise BundleContainmentError(f"bundle path 逃逸: {rel}")
            files[rel] = entry.content
        return cls(normalized_source_ref=bundle.normalized_source_ref, files=files)


@dataclass(frozen=True)
class MemberPlan:
    """单成员 preflight 计划（不可变；T22 逐字消费落 pin/scan/provenance/audit 事实）。"""
    kind: str                                    # skill / mcp / a2a
    declared_component_id: str                   # manifest 内声明名
    ext_id: str                                  # 落地 registry ext_id
    scan: GovernanceScanSummary
    observed_surface_hash: str | None            # mcp/a2a probe 表面 pin（skill=None）
    observed_config_fingerprint: str | None      # mcp/a2a config 指纹（skill=None）
    observed_artifact_hash: str | None           # skill 子树 artifact pin（mcp/a2a=None）
    expected_hash_declared: str | None           # manifest 声明（None=实测即 pin）
    provenance: Provenance
    probe_failed: bool
    acknowledged: bool                           # 成员级 policy 事实（safe 成员恒 False，R2#F1）
    forced: bool
    payload: Any                                 # skill=子树快照 dict / mcp=MCPServerConfig / a2a=base_url
    entry_dump: dict[str, Any] | None            # mcp/a2a 完整 dump 含 secrets（T5 entry_content_hash 用）


@dataclass(frozen=True)
class PluginInstallContext:
    """§8.3-1 不可变安装上下文（写入阶段全部消费它，不重算/重观测/重生成 id）。"""
    plugin_ext_id: str
    name: str
    version: str
    parent_bundle_hash: str
    parent_provenance: Provenance
    members: tuple[MemberPlan, ...]
    preallocated_a2a_ids: dict[str, str]         # declared_id → uuid（R19#F1）
    aggregate_verdict: str
    forced: bool
    acknowledged: bool
    initiated_by: str
    # §8.2 不可变字节快照——preflight↔写入共用同一份（本地目录源的 TOCTOU 由此关闭，T22 落盘）；
    # 内部写上下文（非脱敏 preview），含 secrets 的 payload/entry_dump 已在 members 内同源。
    parent_bundle_files: dict[str, bytes] = field(default_factory=dict)


class PluginMemberPreview(BaseModel):
    """dry_run 单成员摘要（脱敏——仅 hash + scan 摘要，无 secrets/entry_dump）。"""

    model_config = ConfigDict(extra="forbid")

    kind: str
    declared_component_id: str
    ext_id: str
    scan_report: GovernanceScanSummary
    surface_hash: str | None = None
    artifact_hash: str | None = None
    config_fingerprint: str | None = None
    probe_failed: bool = False
    warnings: list[str] = Field(default_factory=list)


class PluginInstallPreview(BaseModel):
    """§8.3-1 dry_run 响应 DTO（成员清单 + scan + policy + probe 摘要，脱敏）。"""

    model_config = ConfigDict(extra="forbid")

    plugin_id: str
    name: str
    version: str
    aggregate_verdict: str
    install_policy_decision: str
    members: list[PluginMemberPreview] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class PreflightResult:
    """``preflight()`` 统一返回类型（R2#F8）：dry_run 消费 ``.preview``、install 消费 ``.context``。"""
    context: PluginInstallContext
    preview: PluginInstallPreview


@dataclass(frozen=True)
class InstallResult:
    """§8.3 安装终态 API 合同（R3#2 + R4#3——补偿/失败不得伪装 2xx）。自 operation 行读取：
    ``status``=state、``error``=error 列、``collided_targets``=steps 中 collided 的
    ``type:key`` 列表。T24 路由映射：completed→200 / compensated→422 / failed→500。"""
    status: Literal["completed", "compensated", "failed"]
    operation_id: uuid.UUID
    plugin_ext_id: str
    error: str | None
    collided_targets: list[str]


# --------------------------------------------------------------------- service --
class PluginInstallService:
    """Plugin 元容器安装服务（T21：载入 + preflight + preview）。"""

    def __init__(
        self,
        *,
        loader: Any,
        prober: Any,
        read_port: Any,
        mode: str,
        reject_audit: Any = None,
        identity_locks: Any = None,
        store: Any = None,
        skill_service: Any = None,
        app_config_service: Any = None,
        write_port: Any = None,
        config_loader: Any = None,
    ) -> None:
        self._loader = loader
        self._prober = prober
        self._read_port = read_port
        self._mode = mode
        self._reject_audit = reject_audit
        # T22 install saga 锁取用式复用（R4#2）；T21 preflight 无锁（碰撞查锁外）。
        self._identity_locks = identity_locks
        # T22 saga 协作者（T24 布线注入；T21 preflight-only 构造恒 None——不影响 preflight）。
        self._store = store
        self._skill_service = skill_service
        self._app_config_service = app_config_service
        self._write_port = write_port
        self._config_loader = config_loader

    def _locks(self) -> Any:
        """§3.6 identity 锁取用式（R4#2 统一 fallback）——T21 preflight 碰撞查在**锁外**
        （含远程 probe 不宜持锁）；T22 install saga 经本访问器持 {父 plugin}∪{全部成员} 排序
        锁。生产零注入 → ``get_identity_locks()`` 进程单例（非 app.state，R4#1）。"""
        return self._identity_locks or get_identity_locks()

    # ---- §8.3-1 preflight（全内存）----
    async def preflight(
        self,
        source_type: SkillSourceType,
        source_ref: str,
        *,
        actor_id: str,
        force: bool,
        acknowledge: bool,
        dry_run: bool,
    ) -> PreflightResult:
        canonical_ref = canonicalize_source_ref(source_ref)
        try:
            _reject_compressed_source(source_ref)                          # R3#8 zip → 422
            bundle = await self._loader.load(
                source_type, source_ref, require_skill_md=False)           # R8#1 plugin 根无 SKILL.md
            plugin_bundle = PluginBundle.from_loader_bundle(bundle)        # §8.2 containment
            manifest = _parse_manifest(plugin_bundle)                      # 缺 plugin.json / 前向门 / charset
            members, parent_bundle_hash, preallocated = await self._assemble_members(
                plugin_bundle, manifest, acknowledge=acknowledge, force=force)
            aggregate = _aggregate_verdict([m.scan.verdict for m in members])
            policy_decision = evaluate_install_policy(
                self._mode, aggregate, acknowledged=acknowledge, forced=force)
            preview = _build_preview(manifest, members, aggregate, policy_decision)

            if not dry_run:
                # 真实 install 门（全内存判定；任一拒绝 → 外层写唯一 install_rejected audit）
                self._gate_probe_failures(members, force)                  # §8.3-1 probe 失败默认拒
                _gate_expected_hash(members)                               # R1#8 replacement 防御
                await self._gate_collisions(manifest, members)             # §8.4/R13#6 身份碰撞
                _gate_policy(self._mode, aggregate, acknowledge, force)    # §7.2 三态门

            ctx = _build_context(
                manifest, members, parent_bundle_hash, preallocated,
                source_type=source_type, canonical_ref=canonical_ref,
                aggregate=aggregate, force=force, acknowledge=acknowledge, actor_id=actor_id,
                bundle_files=plugin_bundle.files)
            return PreflightResult(context=ctx, preview=preview)
        except _REJECT_EXCEPTIONS as exc:
            if not dry_run:
                await self._write_reject_audit(
                    source_type=source_type, source_ref=canonical_ref, exc=exc)
            raise

    # ---- 成员解析（skill sync 全扫 / mcp,a2a async probe）----
    async def _assemble_members(
        self,
        plugin_bundle: PluginBundle,
        manifest: PluginManifest,
        *,
        acknowledge: bool,
        force: bool,
    ) -> tuple[list[MemberPlan], str, dict[str, str]]:
        member_prov = Provenance(
            source_type="plugin", source_ref=manifest.id,
            version=manifest.version, trust_origin="user_installed")
        members: list[MemberPlan] = []
        preallocated: dict[str, str] = {}

        # --- skill 成员 + parent bundle hash（materialize 一次；skill 全扫 sync）---
        skill_specs: list[tuple[Any, str, dict[str, bytes], str, GovernanceScanSummary]] = []
        with tempfile.TemporaryDirectory(prefix="actus-plugin-preflight-") as tmp:
            root = Path(tmp)
            _materialize_bundle(plugin_bundle.files, root)
            # skill/plugin artifact_hash 复用 SkillsGuard.compute_content_hash（§5.1；tmpdir 只读中转）
            parent_bundle_hash = SkillsGuard.compute_content_hash(root)
            for comp in manifest.components.skills:
                ext_id = build_skill_key(
                    comp.id, SkillSourceType.LOCAL, f"plugin:{manifest.id}")   # Deviation #5
                member_dir = root / comp.path
                artifact_hash = SkillsGuard.compute_content_hash(member_dir)
                scan_summary = _project_scan_report(
                    scan_skill_source(SkillRuntimeType.NATIVE, member_dir))     # skill 全扫
                subtree = _extract_subtree(plugin_bundle.files, comp.path)
                skill_specs.append((comp, ext_id, subtree, artifact_hash, scan_summary))

        for comp, ext_id, subtree, artifact_hash, scan_summary in skill_specs:
            ack, frc = self._member_flags(scan_summary.verdict, acknowledge, force)
            members.append(MemberPlan(
                kind="skill", declared_component_id=comp.id, ext_id=ext_id, scan=scan_summary,
                observed_surface_hash=None, observed_config_fingerprint=None,
                observed_artifact_hash=artifact_hash, expected_hash_declared=comp.expected_hash,
                provenance=member_prov, probe_failed=False, acknowledged=ack, forced=frc,
                payload=subtree, entry_dump=None))

        # --- mcp 成员（async probe + §7.3 scan + secret-literal 合并）---
        for comp in manifest.components.mcp_servers:
            config = comp.config
            outcome = await self._probe_safe("mcp", comp.id, config)
            surface = outcome.surface_payload if outcome else None
            scan = _merge_secret_findings(
                scan_mcp_entry(comp.id, config, surface if isinstance(surface, list) else None),
                detect_secret_literals(config))
            surface_hash = _surface_hash_mcp(outcome)
            ack, frc = self._member_flags(scan.verdict, acknowledge, force)
            members.append(MemberPlan(
                kind="mcp", declared_component_id=comp.id, ext_id=comp.id, scan=scan,
                observed_surface_hash=surface_hash,
                observed_config_fingerprint=mcp_config_fingerprint(config),
                observed_artifact_hash=None, expected_hash_declared=comp.expected_hash,
                provenance=member_prov, probe_failed=not (outcome and outcome.ok),
                acknowledged=ack, forced=frc, payload=config,
                entry_dump=config.model_dump(mode="json")))

        # --- a2a 成员（preflight 预分配 uuid；async probe + §7.3 scan）---
        for comp in manifest.components.a2a_agents:
            a2a_id = str(uuid.uuid4())
            preallocated[comp.id] = a2a_id
            cfg = A2AServerConfig(id=a2a_id, base_url=comp.base_url)
            outcome = await self._probe_safe("a2a", a2a_id, cfg)
            card = outcome.surface_payload if outcome else None
            scan = scan_a2a_entry(comp.base_url, card if isinstance(card, Mapping) else None)
            surface_hash = _surface_hash_a2a(outcome)
            ack, frc = self._member_flags(scan.verdict, acknowledge, force)
            members.append(MemberPlan(
                kind="a2a", declared_component_id=comp.id, ext_id=a2a_id, scan=scan,
                observed_surface_hash=surface_hash,
                observed_config_fingerprint=a2a_config_fingerprint(comp.base_url),
                observed_artifact_hash=None, expected_hash_declared=comp.expected_hash,
                provenance=member_prov, probe_failed=not (outcome and outcome.ok),
                acknowledged=ack, forced=frc, payload=comp.base_url,
                entry_dump=cfg.model_dump(mode="json")))

        return members, parent_bundle_hash, preallocated

    # ---- 门（真实 install 才调用；dry_run 报告不拦）----
    def _gate_probe_failures(self, members: list[MemberPlan], force: bool) -> None:
        if force:
            return
        for m in members:
            if m.kind in ("mcp", "a2a") and m.probe_failed:
                raise PluginMemberProbeFailedError(
                    msg=f"{m.kind}[{m.declared_component_id}] preflight probe 失败"
                        "（force=true 可以 unpinned 成员继续）")

    async def _gate_collisions(
        self, manifest: PluginManifest, members: list[MemberPlan]
    ) -> None:
        if self._read_port is None:                                       # off → 零 registry 交互
            return
        if await self._read_port.get_row("plugin", manifest.id) is not None:
            raise PluginIdentityCollisionError(
                msg=f"plugin[{manifest.id}] 已存活（v1 升级=先卸后装）")
        for m in members:
            if await self._read_port.get_row(m.kind, m.ext_id) is not None:
                raise PluginIdentityCollisionError(
                    msg=f"{m.kind}[{m.ext_id}] 已被占用（成员身份碰撞）")

    def _member_flags(
        self, verdict: str, acknowledge: bool, force: bool
    ) -> tuple[bool, bool]:
        """per-member §7.2 评估 → 成员级 acknowledged/forced（R2#F1；safe 成员/非 enforce 恒 False）。"""
        if self._mode != "enforce":
            return False, False
        tier = get_install_decision("user_installed", verdict)            # allow / warn / block
        ack = tier == "warn" and acknowledge
        frc = tier == "block" and force
        return ack, frc

    # ---- probe 兜底（§7.1：失败不阻塞，返 None）----
    async def _probe_safe(self, kind: str, ext_id: str, config: Any):
        try:
            if kind == "mcp":
                return await self._prober.probe_mcp(ext_id, config)
            return await self._prober.probe_a2a(config)
        except Exception:  # noqa: BLE001 - probe 失败 → 成员未 pin（§7.1），不外泄
            logger.warning(
                "D1a plugin preflight probe 失败（非致命，成员 surface 不 pin）: %s/%s",
                kind, ext_id, exc_info=True)
            return None

    # ---- 拒绝 audit（真实 install 唯一 DB 写；extension_id NULL 自持）----
    async def _write_reject_audit(
        self, *, source_type: SkillSourceType, source_ref: str | None, exc: Exception
    ) -> None:
        if self._reject_audit is None:
            return
        await self._reject_audit.record_install_rejected(
            source_type=_source_type_str(source_type),
            source_ref=source_ref,
            details={"category": _reject_category(exc)})

    # ================================================================ §8.3 saga ==
    async def install(
        self,
        source_type: SkillSourceType,
        source_ref: str,
        *,
        actor_id: str,
        dry_run: bool = False,
        force: bool = False,
        acknowledge: bool = False,
    ) -> "InstallResult | PluginInstallPreview":
        """§8.3 安装 saga（对外承诺 all-or-nothing）——严格按 §8.3-1..5 顺序。"""
        result = await self.preflight(
            source_type, source_ref, actor_id=actor_id,
            force=force, acknowledge=acknowledge, dry_run=dry_run)
        if dry_run:
            return result.preview                                  # §8.3-1 严格零写（R2#F8）
        ctx = result.context
        lock_keys = sorted(
            [("plugin", ctx.plugin_ext_id)]
            + [(m.kind, m.ext_id) for m in ctx.members])
        locks = self._locks()                                      # R4#2 统一取用式
        async with locks.acquire_all(lock_keys):                   # §3.6 R36#5
            await self._recheck_collisions_dual_source(ctx)        # §8.3-2 R46#3（409 内抛）
            skeleton = await self._store.create_install_skeleton(ctx)
            try:
                await self._write_parent_bundle(ctx, skeleton)     # §8.3-3 步 3 首段
                for member in ctx.members:
                    await self._write_member(ctx, skeleton, member)
                await self._publish_reverify(ctx)                  # §8.3-4（失败抛 ReverifyError）
                await self._store.complete_install(skeleton.operation_id)
            except Exception as exc:                               # R2#F9：一切失败即时补偿
                # _SagaAbort=content_write_collision（写时已 audit）；ReverifyError=publish_reverify；
                # 其余 I/O 异常=一般写入失败——三者统一走补偿（补偿自身失败 → op=failed）。
                await self._compensate_install(ctx, skeleton, exc)
            return await self._result_of(skeleton.operation_id)

    # ---- §8.3-2 锁内双源碰撞重检（骨架前；409 自持 install_rejected）----
    async def _recheck_collisions_dual_source(self, ctx: PluginInstallContext) -> None:
        """双源（R46#3 + R47#5）：(a) registry 存活行 或 (b) 原生 store 现值占用（config 同名键 /
        skill 目录 / 父 bundle 目录）——任一占用 → 409 + install_rejected（与 preflight 拒绝同
        语义，extension_id NULL 自持），骨架零建。"""
        collided: tuple[str, str] | None = None
        if (await self._registry_alive("plugin", ctx.plugin_ext_id)
                or _bundle_dir_exists(_bundle_key(ctx.plugin_ext_id, ctx.version))):
            collided = ("plugin", ctx.plugin_ext_id)
        if collided is None:
            for m in ctx.members:
                if (await self._registry_alive(m.kind, m.ext_id)
                        or self._native_occupied(m)):
                    collided = (m.kind, m.ext_id)
                    break
        if collided is not None:
            await self._write_reject_audit_recheck(ctx, collided)
            raise PluginIdentityCollisionError(
                msg=f"{collided[0]}[{collided[1]}] 锁内重检已被占用（双源，R46#3）")

    async def _registry_alive(self, kind: str, ext_id: str) -> bool:
        if self._read_port is None:
            return False
        return await self._read_port.get_row(kind, ext_id) is not None

    def _native_occupied(self, member: MemberPlan) -> bool:
        if member.kind == "skill":
            return _skill_dir_exists(member.ext_id)
        # mcp/a2a：config.yaml 同名键存在性（config_loader 注入面）
        return bool(self._config_loader
                    and self._config_loader.occupied(member.kind, member.ext_id))

    async def _write_reject_audit_recheck(
        self, ctx: PluginInstallContext, collided: tuple[str, str]
    ) -> None:
        if self._reject_audit is None:
            return
        await self._reject_audit.record_install_rejected(
            source_type=ctx.parent_provenance.source_type,
            source_ref=ctx.parent_provenance.source_ref,
            details={"category": "identity_collision",
                     "collided_targets": [f"{collided[0]}:{collided[1]}"]})

    # ---- §8.3-3 内容写入（父 bundle 首段 + 逐成员）----
    async def _write_parent_bundle(
        self, ctx: PluginInstallContext, skeleton: Any
    ) -> None:
        """父 bundle 快照 staging→rename 进位 + plugin 行 pin（record_install update-only 命中
        disabled 骨架行，status 不变，R44#3）+ 父行 installed/pin_established/scan_recorded。"""
        op_id = skeleton.operation_id
        bundle_key = _bundle_key(ctx.plugin_ext_id, ctx.version)
        await self._store.mark_step(op_id, 1, "attempting")        # 写前置标 R48#2
        if _bundle_dir_exists(bundle_key):                          # 写入前占用 → collided + abort
            await self._store.mark_step(op_id, 1, "collided")
            await self._store.record_content_collision_audit(
                op_id, stage="content_write_collision",
                target_key=f"plugin_bundle:{bundle_key}")
            raise _SagaAbort(1, {"target": f"plugin_bundle:{bundle_key}"})
        _stage_and_promote_bundle(op_id, bundle_key, ctx.parent_bundle_files)
        if self._write_port is not None:                            # off=None 时零记账
            await self._write_port.record_install(
                "plugin", ctx.plugin_ext_id, self._parent_install_ctx(ctx, op_id))
        await self._store.mark_step(op_id, 1, "done")

    async def _write_member(
        self, ctx: PluginInstallContext, skeleton: Any, member: MemberPlan
    ) -> None:
        """§8.3-3：mark attempting → 目标占用检查（存在→collided+abort）→ 内容写 → mark done。"""
        op_id = skeleton.operation_id
        seq = self._member_step_seq(ctx, member)
        await self._store.mark_step(op_id, seq, "attempting")       # 写前置标 R48#2
        if self._native_occupied(member):                          # create 语义强制（R48#3）
            await self._store.mark_step(op_id, seq, "collided")
            await self._store.record_content_collision_audit(
                op_id, stage="content_write_collision", member=member.declared_component_id,
                target_key=f"{_target_type(member.kind)}:{member.ext_id}")
            raise _SagaAbort(seq, {"target": f"{_target_type(member.kind)}:{member.ext_id}"})
        member_ctx = self._synthesize_member_ctx(ctx, member, op_id)
        if member.kind == "skill":
            staging = _stage_skill_subtree(op_id, member.ext_id, member.payload)
            await self._skill_service.install_skill(
                SkillSourceType.LOCAL, str(staging), {}, "", ctx.initiated_by,
                trust_origin="user_installed", force=member.forced,
                actor_id=ctx.initiated_by, governance_install_context=member_ctx)
        elif member.kind == "mcp":
            await self._app_config_service.update_and_create_mcp_servers(
                MCPConfig(mcpServers={member.ext_id: member.payload}),
                actor_id=ctx.initiated_by, install_context=member_ctx,
                target_server=member.ext_id)
        else:  # a2a
            await self._app_config_service.create_a2a_server(
                member.payload, actor_id=ctx.initiated_by,
                install_context=member_ctx, preallocated_id=member.ext_id)
        await self._store.mark_step(op_id, seq, "done")

    def _synthesize_member_ctx(
        self, ctx: PluginInstallContext, member: MemberPlan, op_id: uuid.UUID
    ) -> InstallContext:
        """§4.1 R25#2：per-member InstallContext = 成员观测/scan/provenance + actor=发起 Admin
        + correlation=op.id + acknowledged/forced/probe_failed 取 MemberPlan 成员级字段（R2#F1——
        safe 成员三者 False，不伪产 acknowledged/force_installed audit）。"""
        prov = member.provenance
        return InstallContext(
            actor_user_id=ctx.initiated_by, correlation_id=op_id,
            source_type=prov.source_type, source_ref=prov.source_ref, version=prov.version,
            trust_origin=prov.trust_origin,
            artifact_hash=member.observed_artifact_hash,
            surface_hash=member.observed_surface_hash,
            config_fingerprint=member.observed_config_fingerprint,
            hash_schema_version=HASH_SCHEMA_VERSION, scan=member.scan,
            probe_failed=member.probe_failed, acknowledged=member.acknowledged,
            forced=member.forced)

    def _parent_install_ctx(
        self, ctx: PluginInstallContext, op_id: uuid.UUID
    ) -> InstallContext:
        """父 plugin 行 pin 建立 ctx（artifact_hash=parent_bundle_hash；scan=聚合 verdict 投影；
        acknowledged/forced 恒 False——聚合 policy 事件由成员级承载，父行只 installed/pin/scan）。"""
        prov = ctx.parent_provenance
        return InstallContext(
            actor_user_id=ctx.initiated_by, correlation_id=op_id,
            source_type=prov.source_type, source_ref=prov.source_ref, version=ctx.version,
            trust_origin=prov.trust_origin,
            artifact_hash=ctx.parent_bundle_hash, surface_hash=None, config_fingerprint=None,
            hash_schema_version=HASH_SCHEMA_VERSION,
            scan=GovernanceScanSummary(
                verdict=ctx.aggregate_verdict, finding_count=0, findings=[]),
            probe_failed=False, acknowledged=False, forced=False)

    @staticmethod
    def _member_step_seq(ctx: PluginInstallContext, member: MemberPlan) -> int:
        """成员 step seq = bundle(1) 之后的成员序（build_saga_steps 同序）。"""
        return 2 + ctx.members.index(member)

    # ---- §8.3-4 发布前成员复验（本地 I/O，零远程；surface 不复验）----
    async def _publish_reverify(self, ctx: PluginInstallContext) -> None:
        """skill=重算磁盘 artifact hash；mcp/a2a=重算当前 config.yaml 条目 config_fingerprint
        （§5.1 同一纯函数）→ 任一 ≠ ctx pin → ReverifyError（转补偿 + publish_reverify audit）。"""
        for m in ctx.members:
            if m.kind == "skill":
                current = _skill_hash(m.ext_id)
                if current != m.observed_artifact_hash:
                    raise ReverifyError(m.declared_component_id)
            else:  # mcp / a2a：当前 config fingerprint
                current_fp = (self._config_loader.current_fingerprint(m.kind, m.ext_id)
                              if self._config_loader else None)
                if current_fp != m.observed_config_fingerprint:
                    raise ReverifyError(m.declared_component_id)

    # ---- §8.3-5 失败补偿（thin wrapper——委托共享内核 _rollback_steps，R2#F3）----
    async def _compensate_install(
        self, ctx: PluginInstallContext, skeleton: Any, exc: Exception
    ) -> "InstallResult | None":
        """读取当前 config entries 后委托 T23 共享内核 ``_rollback_steps``（运行期与崩溃恢复恒
        同一算法，不留双实现）。ReverifyError → 先写 publish_reverify audit（extension_id=父行）。

        **硬化 #1（INV-D1-8 崩溃一致性守卫）**：只补偿 ``in_progress`` 的 operation。若
        ``complete_install`` 已成功 COMMIT 但随后在 session teardown 抛错（COMMIT 后连接重置），
        外层 ``except`` 会携 **completed** op 进本方法——无守卫则补偿会拆掉一个已成功的安装
        （删成员 + bundle、软删父行），令成功安装静默消失。故 op 缺失或已终态
        （completed / compensated / failed）→ **不补偿**，返回反映当前终态的 ``InstallResult``。"""
        op = await self._store.load_operation(skeleton.operation_id)
        if op is None or op.state != "in_progress":
            return await self._result_of(skeleton.operation_id)
        if isinstance(exc, ReverifyError):
            await self._store.record_content_collision_audit(
                op.id, stage="publish_reverify", member=exc.member)
        config_entries = self._load_config_entries()
        await _rollback_steps(
            op, config_entries, self._store,
            app_config_service=self._app_config_service,
            skill_service=self._skill_service, write_port=self._write_port)

    def _load_config_entries(self) -> dict[str, Any] | None:
        """当前 config.yaml 条目快照（补偿三分 entry_content_hash 用）——容忍读失败返 None
        （R52#3 恢复保守分支复用；运行期恒可读）。"""
        if self._config_loader is None:
            return None
        try:
            return self._config_loader.load_entries()
        except Exception:  # noqa: BLE001 - 不可读 → 保守分支（config target 全 collided）
            logger.warning("D1a saga 补偿读 config 失败（转保守分支）", exc_info=True)
            return None

    # ---- §8.3 安装终态 API 合同 ----
    async def _result_of(self, operation_id: uuid.UUID) -> InstallResult:
        """自 operation 行读取三态（R4#3）——status=state / error=error / collided_targets=
        steps 中 collided 的 ``type:key`` 列表。

        **硬化 #2**：``load_operation`` 可能返 None（op 行缺失/被删）——不解引用 None 崩
        ``AttributeError``，返回合同内 ``failed`` 终态（plugin_ext_id 未知置空）。"""
        op = await self._store.load_operation(operation_id)
        if op is None:
            return InstallResult(
                status="failed", operation_id=operation_id, plugin_ext_id="",
                error="operation not found", collided_targets=[])
        collided = [
            f"{s['target']['type']}:{s['target']['key']}"
            for s in op.steps if s["state"] == "collided"
        ]
        return InstallResult(
            status=op.state, operation_id=op.id,
            plugin_ext_id=op.plugin_ext_id, error=op.error, collided_targets=collided)

    # ============================================================ §8.3-6 uninstall ==
    async def uninstall(
        self, plugin_ext_id: str, *, actor_id: str, expected_row_revision: int | None,
    ) -> None:
        """§8.3-6 卸载 saga（独立 forward 语义，非 install 补偿复用，R1#9）。首步单一事务已阻断
        父级——中途失败/崩溃 → op failed（整树保持阻断），Admin 重发 = 幂等收尾残留步骤（missing_ok
        + 缺源收尾）。删除不可逆，无逆补偿。"""
        locks = self._locks()                                      # R4#2 统一取用式
        async with locks.acquire_all([("plugin", plugin_ext_id)]):
            # ①首步单一事务（begin_uninstall 内聚全部分支：failed 复活 / 新建 / in_progress 409）
            op = await self._store.begin_uninstall(
                plugin_ext_id, initiated_by=actor_id,
                expected_row_revision=expected_row_revision)
            ctx = UninstallContext(correlation_id=op.id, actor_user_id=op.initiated_by)
            # ②逐成员 forward 删除（仅 managed_by_plugin=true 成员步；按 seq 跳过已 done——幂等重试）
            MEMBER_TARGET_TYPES = ("skill_dir", "mcp_config", "a2a_config")
            pending = [s for s in sorted(op.steps, key=lambda s: s["seq"])
                       if s["target"]["type"] in MEMBER_TARGET_TYPES and s["state"] != "done"]
            for step in pending:
                try:
                    await self._delete_member_content(step, ctx)
                    await self._store.mark_step(op.id, step["seq"], "done")
                except Exception:
                    await self._store.fail(op.id, error="uninstall step failed; retry")
                    raise                     # 首步已阻断父行——Admin 重发＝幂等收尾
            # ③membership 物理删 + plugin bundle 目录删 ④最终单事务：软删父行 + op=completed +
            #   旧 failed install → compensated（error 追加 cleaned_up_via）——均由 finalize_uninstall
            await self._store.finalize_uninstall(op.id)

    async def retry_failed_uninstall(
        self, plugin_ext_id: str, *, actor_id: str, expected_row_revision: int | None = None,
    ) -> None:
        """§3.4：failed uninstall 重试——经**同一** uninstall 入口，``begin_uninstall`` detect
        failed → CAS 复活原 operation（保 id/steps，correlation 连续）。expected_row_revision 在
        复活路不消费（父行首次已阻断），仅新建路守卫。"""
        await self.uninstall(
            plugin_ext_id, actor_id=actor_id, expected_row_revision=expected_row_revision)

    async def _delete_member_content(
        self, step: Mapping[str, Any], ctx: UninstallContext
    ) -> None:
        """§8.3-6 单成员内容删除：内容源在 → 经 service delete hooks 标准路径（delta 单写软删成员行 +
        ``uninstalled`` 双字段，R30#2）；内容源已缺 → WritePort 幂等缺源收尾（R22#F1——软删成员行 +
        audit；行已软删则零写零 audit）。二者互斥于内容存在性 → 每成员恰一条 ``uninstalled``。"""
        ttype = step["target"]["type"]
        key = step["target"]["key"]
        kind = _TARGET_KIND_FOR_TYPE[ttype]
        if self._member_content_present(kind, key):
            if kind == "skill":
                await self._skill_service.delete_skill(
                    key, uninstall_context=ctx, missing_ok=True)
            elif kind == "mcp":
                await self._app_config_service.delete_mcp_server(
                    key, uninstall_context=ctx, missing_ok=True)
            else:  # a2a
                await self._app_config_service.delete_a2a_server(
                    key, uninstall_context=ctx, missing_ok=True)
        elif self._write_port is not None:
            await self._write_port.record_delete(kind, key, uninstall_context=ctx)

    def _member_content_present(self, kind: str, key: str) -> bool:
        """成员内容源存在性（决定 delta 路 vs 缺源收尾路）：skill=本地盘目录；mcp/a2a=config 条目
        占用（config_loader）。无 config_loader → 保守视为存在（走 service delete missing_ok 兜底，
        不误判缺源跳过真实 config 删除）。"""
        if kind == "skill":
            return _skill_dir_exists(key)
        if self._config_loader is not None:
            return bool(self._config_loader.occupied(kind, key))
        return True


# ---------------------------------------------------------------- pure helpers --
def _reject_compressed_source(source_ref: str) -> None:
    low = (source_ref or "").strip().lower()
    if low.endswith(_COMPRESSED_SUFFIXES):
        raise ValidationError(
            msg="plugin 不支持压缩包输入（zip/tar），请用目录或 GitHub tree URL（R3#8）")


def _parse_manifest(plugin_bundle: PluginBundle) -> PluginManifest:
    raw = plugin_bundle.files.get("plugin.json")
    if raw is None:
        raise ValidationError(msg="plugin bundle 根目录缺少 plugin.json")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValidationError(msg="plugin.json 不是合法 UTF-8 JSON") from exc
    if not isinstance(data, Mapping):
        raise ValidationError(msg="plugin.json 顶层必须是对象")
    return PluginManifest.parse(data)


def _materialize_bundle(files: Mapping[str, bytes], root: Path) -> None:
    """快照落 tmpdir（containment 已保证 relpath 不逃逸；写入例外仅此计算性 tmp，不触生产 store）。"""
    for rel, content in files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)


def _extract_subtree(files: Mapping[str, bytes], comp_path: str) -> dict[str, bytes]:
    """抽取组件子树快照 → ``{组件内相对路径: bytes}``（skill 成员 payload，T22 落盘用）。"""
    prefix = comp_path.rstrip("/") + "/"
    out: dict[str, bytes] = {}
    for rel, content in files.items():
        if rel == comp_path:
            out[PurePosixPath(rel).name] = content
        elif rel.startswith(prefix):
            out[rel[len(prefix):]] = content
    return out


def _project_scan_report(scan_report: Any) -> GovernanceScanSummary:
    """内存 ``ScanReport`` → 持久化/DTO 唯一 scan 形态（§7.3；丢弃原始 match 文本，INV-D1-7）。"""
    findings = list(getattr(scan_report, "findings", None) or [])

    def _s(value: Any) -> str:
        return str(value or "")[:_GOV_FINDING_FIELD_MAX]

    projected = [
        GovernanceScanFinding(
            category=_s(getattr(f, "category", "")),
            severity=_s(getattr(f, "severity", "")),
            pattern_id=_s(getattr(f, "pattern_id", "")),
            path=_s(getattr(f, "file", "")),
            line=getattr(f, "line", None),
        )
        for f in findings[:50]
    ]
    return GovernanceScanSummary(
        verdict=getattr(scan_report, "verdict", "safe"),
        finding_count=len(findings),
        findings=projected,
    )


def _merge_secret_findings(
    summary: GovernanceScanSummary, extra: list[GovernanceScanFinding]
) -> GovernanceScanSummary:
    """合并 secret-literal findings（§8.1）→ verdict 升 dangerous（critical severity）。"""
    if not extra:
        return summary
    merged = list(summary.findings) + list(extra)
    return GovernanceScanSummary(
        verdict="dangerous",
        finding_count=summary.finding_count + len(extra),
        findings=merged[:50],
    )


def _aggregate_verdict(verdicts: list[str]) -> str:
    if not verdicts:
        return "safe"
    return max(verdicts, key=lambda v: _VERDICT_ORDER.get(v, 2))


def _gate_expected_hash(members: list[MemberPlan]) -> None:
    """R1#8：成员非空 expected_hash 与本次同算法实测值不等 → 整体拒装（防替换攻击）。"""
    for m in members:
        if m.expected_hash_declared is None:
            continue
        measured = m.observed_artifact_hash if m.kind == "skill" else m.observed_surface_hash
        if measured is None:
            # 无实测值（mcp/a2a probe 失败/未观测）→ 交由 probe 门 + force 处置，非 hash mismatch；
            # skill artifact 本地恒可测，此分支只对未 pin 的 mcp/a2a 成员成立。
            continue
        if measured != m.expected_hash_declared:
            raise PluginExpectedHashMismatchError(
                msg=f"{m.kind}[{m.declared_component_id}] expected_hash 与实测不符（替换攻击防御）")


def _gate_policy(mode: str, aggregate: str, acknowledge: bool, force: bool) -> None:
    """§7.2 三态门（聚合 verdict）：enforce caution 无 ack → 409；dangerous 无 force → 422。"""
    decision = evaluate_install_policy(mode, aggregate, acknowledged=acknowledge, forced=force)
    if decision == "need_acknowledge":
        raise AcknowledgeRequiredError()
    if decision == "need_force":
        raise ForceRequiredError()


def _build_preview(
    manifest: PluginManifest,
    members: list[MemberPlan],
    aggregate: str,
    policy_decision: str,
) -> PluginInstallPreview:
    member_previews = [
        PluginMemberPreview(
            kind=m.kind,
            declared_component_id=m.declared_component_id,
            ext_id=m.ext_id,
            scan_report=m.scan,
            surface_hash=m.observed_surface_hash,
            artifact_hash=m.observed_artifact_hash,
            config_fingerprint=m.observed_config_fingerprint,
            probe_failed=m.probe_failed,
            warnings=[_PROBE_FAILED_WARNING] if m.probe_failed else [],
        )
        for m in members
    ]
    return PluginInstallPreview(
        plugin_id=manifest.id,
        name=manifest.name,
        version=manifest.version,
        aggregate_verdict=aggregate,
        install_policy_decision=policy_decision,
        members=member_previews,
    )


def _build_context(
    manifest: PluginManifest,
    members: list[MemberPlan],
    parent_bundle_hash: str,
    preallocated: dict[str, str],
    *,
    source_type: SkillSourceType,
    canonical_ref: str | None,
    aggregate: str,
    force: bool,
    acknowledge: bool,
    actor_id: str,
    bundle_files: Mapping[str, bytes],
) -> PluginInstallContext:
    parent_prov = Provenance(
        source_type=_source_type_str(source_type),
        source_ref=canonical_ref,
        version=manifest.version,
        trust_origin="user_installed",
    )
    return PluginInstallContext(
        plugin_ext_id=manifest.id,
        name=manifest.name,
        version=manifest.version,
        parent_bundle_hash=parent_bundle_hash,
        parent_provenance=parent_prov,
        members=tuple(members),
        preallocated_a2a_ids=dict(preallocated),
        aggregate_verdict=aggregate,
        forced=force,
        acknowledged=acknowledge,
        initiated_by=actor_id,
        parent_bundle_files=dict(bundle_files),
    )


def _surface_hash_mcp(outcome: Any) -> str | None:
    payload = outcome.surface_payload if outcome else None
    if outcome and outcome.ok and isinstance(payload, list) and payload:
        return mcp_surface_hash(payload)
    return None


def _surface_hash_a2a(outcome: Any) -> str | None:
    card = outcome.surface_payload if outcome else None
    if outcome and outcome.ok and isinstance(card, Mapping) and card:
        return a2a_surface_hash(card)
    return None


def _source_type_str(source_type: Any) -> str:
    return getattr(source_type, "value", source_type)


def _reject_category(exc: Exception) -> str:
    if isinstance(exc, PluginIdentityCollisionError):
        return "identity_collision"
    if isinstance(exc, PluginExpectedHashMismatchError):
        return "expected_hash_mismatch"
    if isinstance(exc, PluginMemberProbeFailedError):
        return "probe_failed"
    if isinstance(exc, (AcknowledgeRequiredError, ForceRequiredError)):
        return "policy_gate"
    if isinstance(exc, UnsupportedManifestVersionError):
        return "unsupported_manifest_version"
    if isinstance(exc, BundleContainmentError):
        return "bundle_containment"
    return "manifest_invalid"


# ------------------------------------------------------------- saga fs helpers --
# 模块级路径解析——引用 module-global 常量（PLUGIN_*_ROOT），测试 monkeypatch 生效。

_TARGET_TYPE_FOR_KIND = {"skill": "skill_dir", "mcp": "mcp_config", "a2a": "a2a_config"}
_TARGET_KIND_FOR_TYPE = {"skill_dir": "skill", "mcp_config": "mcp", "a2a_config": "a2a"}


def _bundle_key(plugin_ext_id: str, version: str) -> str:
    """plugin_bundle target key = ``{plugin_ext_id}/{version}``（与 store.bundle_target_key 同源）。"""
    return f"{plugin_ext_id}/{version}"


def _target_type(kind: str) -> str:
    return _TARGET_TYPE_FOR_KIND[kind]


def _bundle_dir(bundle_key: str) -> Path:
    return PLUGIN_STORE_ROOT / bundle_key


def _skill_dir(ext_id: str) -> Path:
    return SKILL_STORE_ROOT / ext_id


def _bundle_dir_exists(bundle_key: str) -> bool:
    return _bundle_dir(bundle_key).is_dir()


def _skill_dir_exists(ext_id: str) -> bool:
    return _skill_dir(ext_id).is_dir()


def _bundle_hash(bundle_key: str) -> str | None:
    d = _bundle_dir(bundle_key)
    return SkillsGuard.compute_content_hash(d) if d.is_dir() else None


def _skill_hash(ext_id: str) -> str | None:
    d = _skill_dir(ext_id)
    return SkillsGuard.compute_content_hash(d) if d.is_dir() else None


def _write_tree(root: Path, files: Mapping[str, bytes]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)


def _stage_and_promote_bundle(
    op_id: uuid.UUID, bundle_key: str, files: Mapping[str, bytes]
) -> None:
    """父 bundle 快照先落 staging 再 ``os.replace`` rename 原子进位（R50#2/R50#3）。"""
    staging = PLUGIN_STAGING_ROOT / str(op_id) / "bundle"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    _write_tree(staging, files)
    final = _bundle_dir(bundle_key)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)


def _stage_skill_subtree(
    op_id: uuid.UUID, ext_id: str, files: Mapping[str, bytes]
) -> Path:
    """skill 成员子树落 staging（install_skill 从此 source_ref 载入 → repo 布局进位）。"""
    staging = PLUGIN_STAGING_ROOT / str(op_id) / "skill" / ext_id
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    _write_tree(staging, files)
    return staging


def _remove_bundle_dir(bundle_key: str) -> None:
    shutil.rmtree(_bundle_dir(bundle_key), ignore_errors=True)


def _current_target_hash(step: Mapping[str, Any], config_entries: Mapping[str, Any] | None) -> str | None:
    """恢复三分现值 hash（§3.4/§5.1 同算法）：bundle/skill=``compute_content_hash`` 目录；
    mcp/a2a=``entry_content_hash`` 当前 config 条目 dump（缺=None）。"""
    ttype = step["target"]["type"]
    key = step["target"]["key"]
    if ttype == "plugin_bundle":
        return _bundle_hash(key)
    if ttype == "skill_dir":
        return _skill_hash(key)
    entries = (config_entries or {}).get(_TARGET_KIND_FOR_TYPE.get(ttype, ""), {})
    dump = entries.get(key)
    return entry_content_hash(dump) if dump is not None else None


async def _delete_target(
    ttype: str,
    key: str,
    uninstall_ctx: UninstallContext,
    *,
    app_config_service: Any,
    skill_service: Any,
    write_port: Any,
) -> None:
    """§3.4 逆向删除——经 delete hooks 标准路径（R30#2：软删行 + ``uninstalled`` 双字段）；
    plugin_bundle 目录直删（父行软删由 compensate 事务承担，无 hook）。"""
    if ttype == "plugin_bundle":
        _remove_bundle_dir(key)
    elif ttype == "skill_dir":
        await skill_service.delete_skill(key, uninstall_context=uninstall_ctx, missing_ok=True)
    elif ttype == "mcp_config":
        await app_config_service.delete_mcp_server(
            key, uninstall_context=uninstall_ctx, missing_ok=True)
    elif ttype == "a2a_config":
        await app_config_service.delete_a2a_server(
            key, uninstall_context=uninstall_ctx, missing_ok=True)


async def _rollback_steps(
    op: Any,
    config_entries: Mapping[str, Any] | None,
    saga_store: Any,
    *,
    app_config_service: Any,
    skill_service: Any,
    write_port: Any,
) -> None:
    """T22 ``_compensate_install`` 与 T23 崩溃恢复的**共享内核**（运行期与恢复恒同一算法，
    不留双实现，R2#F3）。steps 逆序、state ∈ {attempting, done}（§3.4）：
    - done → 照删（写入已原子进位 + hash 校验）；
    - attempting → 三分：内容缺失→幂等成功 / hash==expected→删 / ≠expected→collided 不删
      + recovery_collision；
    - planned/collided → 恒不删。
    config 不可读（``config_entries is None``）→ config target 全 collided + ``fail``（非
    compensated，R53#1）；非 config target 照常。**补偿自身异常 → ``fail(op)``**（父行保持
    disabled=阻断态）——不外抛，令 ``_result_of`` 读出 failed 终态。"""
    uninstall_ctx = UninstallContext(correlation_id=op.id, actor_user_id=op.initiated_by)
    collided: list[str] = []
    config_unreadable = config_entries is None
    try:
        for step in sorted(op.steps, key=lambda s: s["seq"], reverse=True):
            state = step["state"]
            if state not in ("attempting", "done"):
                continue                                          # planned/collided 恒不删
            ttype = step["target"]["type"]
            key = step["target"]["key"]
            target_str = f"{ttype}:{key}"
            if config_unreadable and ttype in _CONFIG_TARGET_TYPES:
                collided.append(target_str)                       # 保守：不可读不删（R52#3/R53#1）
                continue
            if state == "done":
                await _delete_target(
                    ttype, key, uninstall_ctx, app_config_service=app_config_service,
                    skill_service=skill_service, write_port=write_port)
                continue
            current = _current_target_hash(step, config_entries)  # attempting 三分（R49#2）
            if current is None:
                continue                                          # 内容缺失→幂等成功
            if current == step["expected_hash"]:
                await _delete_target(
                    ttype, key, uninstall_ctx, app_config_service=app_config_service,
                    skill_service=skill_service, write_port=write_port)
            else:
                collided.append(target_str)                       # ≠expected→带外占用，不删
    except Exception:  # noqa: BLE001 - 补偿自身失败 → op=failed（父行阻断态，Admin 处置）
        logger.error("D1a saga 补偿失败 → operation failed（父行保持 disabled）", exc_info=True)
        await saga_store.fail(op.id, error="compensation failed; parent stays disabled")
        return
    if config_unreadable and any(
            t.split(":", 1)[0] in _CONFIG_TARGET_TYPES for t in collided):
        # config 不可读保守终态：op 转/保持 failed（修复后走既定 failed-install 恢复路，§3.4）
        await saga_store.record_content_collision_audit(
            op.id, stage="recovery_collision", collided_targets=collided)
        await saga_store.fail(op.id, error="config unreadable during recovery")
        return
    await saga_store.compensate(op.id, details={
        "failed_step": _failed_step(op),
        "compensated_targets": _compensated_targets(op),
        "collided_targets": collided})


def _failed_step(op: Any) -> int | None:
    for s in op.steps:
        if s["state"] == "collided":
            return s["seq"]
    return None


def _compensated_targets(op: Any) -> list[str]:
    return [f"{s['target']['type']}:{s['target']['key']}"
            for s in op.steps if s["state"] in ("attempting", "done")]


# ------------------------------------------------------ §8.3-2 startup ②段 saga 收尾 --
def _sweep_staging_dirs(op_states: Mapping[uuid.UUID, str]) -> None:
    """终态 op 的 ``PLUGIN_STAGING_ROOT/{op_id}/`` 整删（收尾，R50#3 断言③）；非终态保留。

    共享 fs 逻辑单源——真 store ``sweep_terminal_staging`` DB 查询后调用（staging 根常量归属本
    模块，store 侧惰性 import 复用，避免双份路径）。含 secrets 的 staging 不过夜（§8.3-3 R51#3）。"""
    for op_id, state in op_states.items():
        if state in _TERMINAL_OP_STATES:
            shutil.rmtree(PLUGIN_STAGING_ROOT / str(op_id), ignore_errors=True)


def _tolerant_load_config_entries(config_path: str) -> dict[str, Any] | None:
    """§8.3-2 startup ②段容忍读：加载 config.yaml 提取 mcp/a2a 条目 dump（``entry_content_hash``
    输入形态——恢复三分现值 hash 用）；任一失败（不可读/损坏/``ServerRequestsError``）→ ``None``
    （不抛，R52#3——config 不可读保守分支：config target 全 collided，非 config target 照常）。"""
    try:
        from app.infrastructure.repositories.file_app_config_repository import (
            FileAppConfigRepository,
        )
        app_config = FileAppConfigRepository(config_path).load()
        if app_config is None:
            return {"mcp": {}, "a2a": {}}
        mcp_servers = app_config.mcp_config.mcpServers if app_config.mcp_config else {}
        a2a_servers = app_config.a2a_config.a2a_servers if app_config.a2a_config else []
        return {
            "mcp": {name: cfg.model_dump(mode="json") for name, cfg in mcp_servers.items()},
            "a2a": {srv.id: srv.model_dump(mode="json") for srv in a2a_servers},
        }
    except Exception:  # noqa: BLE001 - 不可读/损坏 → 保守分支（config target 全 collided，R52#3）
        logger.warning(
            "D1a startup saga 收尾读 config 失败（转保守分支：config target 全 collided）",
            exc_info=True)
        return None


async def run_startup_saga_closure(
    *, saga_store: Any, app_config_service: Any, skill_service: Any,
    write_port: Any, config_path: str,
) -> None:
    """§8.3-2 startup 四段全序②段：容忍 config 不可读的 saga 收尾。**normal load（③段）之前**执行
    ——不可读场景下恢复必须仍可达（R52#2：普通 load 失败即抛 ``ServerRequestsError``，closure 排后
    = 不可读场景恢复不可达）。恢复分流（§3.4）：

    - ``plugin_uninstall`` 孤儿 → 标 ``failed``（父级首步已阻断——不逆补偿，R5#4）；
    - ``plugin_install`` 孤儿 → 逆序补偿（与 T22 运行期补偿共用 ``_rollback_steps``，不留双实现）。

    末尾整删终态 op 的 staging 目录。补偿走 T22 同一 ``_rollback_steps``（内容删除经 services 标准
    路径 + 缺源收尾经 write_port；config 不可读分支不触 config 依赖——config target 全 collided）。"""
    operations = await saga_store.load_inflight_operations()
    config_entries = _tolerant_load_config_entries(config_path)   # 失败 → None（不抛）
    for op in operations:
        # 单孤儿隔离（§8.3-3 R51#3 "secrets 不过夜"）：某 op 的终态迁移调用
        # （fail / _rollback_steps 内 compensate/fail/record_content_collision_audit）抛错
        # 不得中断其余孤儿处理，更不得跳过末尾 staging sweep（否则含 secrets 的 staging 多留一个
        # boot）。逐 op try/except 吞异常并记录——该 op 保持 in_progress，下次 boot 再试。
        try:
            if op.operation_type == "plugin_uninstall":
                await saga_store.fail(op.id, error="orphaned in_progress; retry via API")
                continue
            # plugin_install 孤儿：逆序补偿（与 T22 运行期补偿共用 _rollback_steps）
            await _rollback_steps(
                op, config_entries, saga_store,
                app_config_service=app_config_service,
                skill_service=skill_service, write_port=write_port)
        except Exception:  # noqa: BLE001 - 单 op 终态迁移失败隔离，保证 sweep 必达
            logger.error(
                "D1a startup saga 收尾：孤儿 op 处理失败（隔离，继续其余孤儿）op=%s type=%s",
                op.id, op.operation_type, exc_info=True)
    await saga_store.sweep_terminal_staging()
