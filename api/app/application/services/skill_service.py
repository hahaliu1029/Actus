"""Skill 服务"""

import json
import logging
import re
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.application.errors.exceptions import NotFoundError, ValidationError
from app.application.services.extension_identity_locks import get_identity_locks
from app.application.services.skill_source_loader import (
    SkillBundleFile,
    SkillSourceLoader,
    TEXT_INJECT_EXTENSIONS,
)
from app.domain.external.extension_admission import InstallContext, UninstallContext
from app.domain.models.extension_governance import (
    HASH_SCHEMA_VERSION,
    TRUST_ORIGINS,
    GovernanceScanFinding,
    GovernanceScanSummary,
    InvalidStateTransitionError,
    ManagedByPluginError,
)
from app.domain.models.skill import (
    Skill,
    SkillManifest,
    SkillSourceType,
    build_skill_key,
    normalize_skill_slug,
)
from app.domain.repositories.skill_repository import SkillRepository
from app.domain.services.extension_scan import canonicalize_source_ref
import yaml

logger = logging.getLogger(__name__)

DEFAULT_BLOCKED_NATIVE_COMMAND_PATTERNS = (
    r"\brm\s+-rf\b",
    r":\(\)\s*\{",
    r"\bmkfs\.",
    r"\bshutdown\b",
    r"\breboot\b",
)
MAX_CONTEXT_BLOB_CHARS = 12 * 1024
MAX_CONTEXT_REF_FILE_CHARS = 2 * 1024

_GOV_FINDING_FIELD_MAX = 256   # GovernanceScanFinding str 字段上限（防 pydantic 校验失败）


def _to_governance_scan_summary(scan_report: Any) -> GovernanceScanSummary | None:
    """§7.3：内存 ScanReport → 持久化/DTO 唯一 scan 形态。

    逐字段投影（category/severity/pattern_id/path/line）+ 截断 findings ≤50 +
    **丢弃原始 match 文本**（INV-D1-7 audit 禁 secrets）。
    """
    if scan_report is None:
        return None
    findings = list(getattr(scan_report, "findings", None) or [])

    def _s(value: Any) -> str:
        return str(value or "")[:_GOV_FINDING_FIELD_MAX]

    projected = [
        GovernanceScanFinding(
            category=_s(getattr(f, "category", "")),
            severity=_s(getattr(f, "severity", "")),
            pattern_id=_s(getattr(f, "pattern_id", "")),
            path=_s(getattr(f, "file", "")),   # ScanFinding.file → 受控 schema.path
            line=getattr(f, "line", None),
        )
        for f in findings[:50]
    ]
    return GovernanceScanSummary(
        verdict=getattr(scan_report, "verdict", "safe"),
        finding_count=len(findings),
        findings=projected,
    )


class SkillService:
    """Skill 生态管理服务"""

    def __init__(
        self,
        skill_repository: SkillRepository,
        source_loader: SkillSourceLoader | None = None,
        *,
        registry_write_port: Any = None,
        registry_read_port: Any = None,
        identity_locks: Any = None,
    ) -> None:
        self.skill_repository = skill_repository
        self._source_loader = source_loader or SkillSourceLoader()
        # D1a §6.1-3 治理挂钩——off（默认）=三 None → install/delete 行为 byte 级一致
        self._registry_write_port = registry_write_port
        self._registry_read_port = registry_read_port
        self._identity_locks = identity_locks    # R3#1：进程单例锁；取用式 self._x or get_identity_locks()

    async def list_skills(self) -> list[Skill]:
        return await self.skill_repository.list()

    async def list_enabled_skills(self) -> list[Skill]:
        return await self.skill_repository.list_enabled()

    async def get_skill(self, skill_id: str) -> Skill:
        skill = await self.skill_repository.get_by_id(skill_id)
        if not skill:
            raise NotFoundError(f"Skill[{skill_id}]不存在")
        return skill

    async def install_skill(
        self,
        source_type: SkillSourceType,
        source_ref: str,
        manifest: dict,
        skill_md: str,
        installed_by: str,
        *,
        trust_origin: str = "user_installed",
        force: bool = False,
        actor_id: str | None = None,
        governance_install_context: "InstallContext | None" = None,
    ) -> Skill:
        if source_type not in {SkillSourceType.LOCAL, SkillSourceType.GITHUB}:
            raise ValidationError(msg="source_type 仅支持 local 或 github")

        incoming_manifest = dict(manifest or {})
        override_skill_md = (skill_md or "").strip()
        normalized_source_ref = source_ref
        bundle_files: dict[str, SkillBundleFile] = {}

        try:
            bundle = await self._source_loader.load(source_type, source_ref)
            normalized_source_ref = bundle.normalized_source_ref
            bundle_files = bundle.files
            source_skill_md = bundle.skill_md
        except ValidationError:
            if not incoming_manifest and not override_skill_md:
                raise
            source_skill_md = ""

        effective_skill_md = override_skill_md or source_skill_md

        normalized_manifest = self._normalize_manifest_input(
            source_ref=normalized_source_ref,
            manifest=incoming_manifest,
            skill_md=effective_skill_md,
        )

        try:
            parsed_manifest = SkillManifest.model_validate(normalized_manifest)
        except Exception as e:
            raise ValidationError(msg=f"Manifest 校验失败: {e}") from e

        self._validate_native_command_policy(parsed_manifest.model_dump(mode="json"))
        context_blob, context_refs = self._build_context_blob(
            skill_md=effective_skill_md,
            bundle_files=bundle_files,
        )

        slug = normalize_skill_slug(parsed_manifest.slug or parsed_manifest.name)
        existed = await self.skill_repository.get_by_slug(slug)
        skill_key = build_skill_key(slug, source_type, normalized_source_ref)

        manifest_payload = dict(normalized_manifest)
        if effective_skill_md:
            manifest_payload["skill_md"] = effective_skill_md
        if context_blob:
            manifest_payload["context_blob"] = context_blob
        manifest_payload["context_refs"] = context_refs
        manifest_payload["context_ref_count"] = len(context_refs)
        manifest_payload["bundle_file_count"] = len(bundle_files)
        manifest_payload["last_sync_at"] = datetime.now().isoformat()
        if bundle_files:
            if override_skill_md:
                override_bytes = override_skill_md.encode("utf-8")
                bundle_files["SKILL.md"] = SkillBundleFile(
                    path="SKILL.md",
                    content=override_bytes,
                    size=len(override_bytes),
                    sha256="",
                    is_text=True,
                )
            manifest_payload["_bundle_files"] = {
                path: item.content for path, item in bundle_files.items()
            }

        skill_payload = {
            "slug": slug,
            "name": parsed_manifest.name,
            "description": parsed_manifest.description,
            "version": parsed_manifest.version,
            "source_type": source_type,
            "source_ref": normalized_source_ref,
            "runtime_type": parsed_manifest.runtime_type,
            "manifest": manifest_payload,
            "enabled": existed.enabled if existed else True,
            "installed_by": installed_by,
        }
        skill_payload["id"] = existed.id if existed else skill_key

        skill = Skill(**skill_payload)

        # ========== D1a §3.6 identity-locked governance envelope ==========
        # off (write_port=None) → nullcontext：零锁、零 factory、install 行为 byte-identical。
        _governed = self._registry_write_port is not None
        _lock_cm = (
            (self._identity_locks or get_identity_locks()).acquire_all([("skill", skill.id)])
            if _governed
            else nullcontext()
        )
        async with _lock_cm:
            # D1a standalone 占用预检（R33#4/R44#3）——saga 成员路（correlation 非 None）跳过：
            # membership/骨架行由 saga 事务先建，属设计内占用。
            _is_saga_member = (
                governance_install_context is not None
                and governance_install_context.correlation_id is not None
            )
            if self._registry_read_port is not None and not _is_saga_member:
                _row = await self._registry_read_port.get_row("skill", skill.id)
                if _row is not None:
                    if _row.parent_plugin_ext_id is not None:
                        raise ManagedByPluginError(
                            f"skill[{skill.id}] 由 plugin 管理，禁 standalone 重装")
                    if _row.status in ("quarantined", "disabled"):
                        raise InvalidStateTransitionError(
                            f"skill[{skill.id}] 处于 {_row.status}——重装不是恢复动作"
                            "（先 reapprove/enable 或 delete）")

            result, _scan_report = await self._run_scan_gate_and_persist(
                skill, trust_origin=trust_origin, force=force, installed_by=installed_by,
                bundle_files=bundle_files, effective_skill_md=effective_skill_md,
                normalized_manifest=normalized_manifest)

            # ---------- D1a §6.1-3 registry hook（R2#5 锚点：二次 upsert 之后、broad except 之外）----------
            if self._registry_write_port is not None:
                _final_hash = (result.scan_report or {}).get("content_hash")
                if governance_install_context is not None:
                    # R1#6：saga 成员安装——成员 InstallContext **逐字**使用（correlation=operation.id、
                    # provenance source_type="plugin"、pins/scan 取自 PluginInstallContext 不重算
                    # [spec §8.3-3]；盘上内容与 ctx 同源自同一 bundle 快照——偏差由发布前复验兜底
                    # 走补偿，不在此静默覆盖 pin）。
                    _ctx = governance_install_context
                else:
                    _ctx = InstallContext(
                        actor_user_id=actor_id or "system",
                        correlation_id=None,
                        source_type=getattr(result, "source_type", "local") or "local",
                        source_ref=canonicalize_source_ref(getattr(result, "source_ref", None)),
                        version=(result.manifest or {}).get("version"),
                        # R7#4/spec §2：registry 列值由治理写入方决定——未知 trust_origin 回退
                        # user_installed（与 INSTALL_POLICY 回退语义一致，F13）。
                        trust_origin=(result.trust_origin
                                      if result.trust_origin in TRUST_ORIGINS else "user_installed"),
                        artifact_hash=_final_hash,
                        surface_hash=None, config_fingerprint=None,
                        hash_schema_version=HASH_SCHEMA_VERSION,
                        scan=_to_governance_scan_summary(_scan_report),
                        forced=bool(force and _scan_report.verdict == "dangerous"),
                    )
                await self._registry_write_port.record_install("skill", result.id, _ctx)
            return result

    async def _run_scan_gate_and_persist(
        self,
        skill: Skill,
        *,
        trust_origin: str,
        force: bool,
        installed_by: str,
        bundle_files: dict[str, SkillBundleFile],
        effective_skill_md: str,
        normalized_manifest: dict,
    ) -> tuple[Skill, Any]:
        """R3 install-time scan gate + 二次 upsert（实际磁盘 content_hash 重算）。

        返回 (result, scan_report)——scan_report 供治理 hook 投影 GovernanceScanSummary。
        行为与治理接线前 byte 级一致（纯抽取，无逻辑改动）。
        """
        # ========== R3: Install-time scan gate ==========
        from app.domain.services.trust_matrix import (
            compute_base_floor,
            compute_final_risk,
            get_install_decision,
            scan_skill_source,
        )

        with tempfile.TemporaryDirectory(prefix="skill_scan_") as _scan_dir:
            _scan_path = Path(_scan_dir)
            if bundle_files:
                for _rel_path, _bf in bundle_files.items():
                    _target = _scan_path / _rel_path
                    _target.parent.mkdir(parents=True, exist_ok=True)
                    _content = _bf.content if hasattr(_bf, "content") else _bf
                    _target.write_bytes(
                        _content if isinstance(_content, bytes)
                        else str(_content).encode("utf-8")
                    )
            if effective_skill_md:
                (_scan_path / "SKILL.md").write_text(effective_skill_md, encoding="utf-8")
            if normalized_manifest:
                (_scan_path / "manifest.json").write_text(
                    json.dumps(normalized_manifest, ensure_ascii=False),
                    encoding="utf-8",
                )

            _scan_report = scan_skill_source(skill.runtime_type, _scan_path)

        _decision = get_install_decision(trust_origin, _scan_report.verdict)

        if _decision == "block" and not force:
            raise ValidationError(
                msg=f"Skill 安全扫描未通过 (verdict={_scan_report.verdict})",
                data={
                    "installed": False,
                    "verdict": _scan_report.verdict,
                    "findings": [
                        {"pattern_id": f.pattern_id, "category": f.category,
                         "severity": f.severity, "file": f.file,
                         "line": f.line, "match": f.match}
                        for f in _scan_report.findings[:20]
                    ],
                    "final_risk": compute_final_risk(
                        compute_base_floor(skill.runtime_type, trust_origin),
                        _scan_report.verdict, None,
                    ).name.lower(),
                    "trust_origin": trust_origin,
                },
            )

        if force and _scan_report.verdict == "dangerous":
            logger.warning(
                "force_install: user=%s skill=%s verdict=%s findings=%s",
                installed_by, skill.id, _scan_report.verdict,
                [f.pattern_id for f in _scan_report.findings[:10]],
            )

        _base = compute_base_floor(skill.runtime_type, trust_origin)
        _manifest_risk = (
            (normalized_manifest or {}).get("policy", {}).get("risk_level")
            if isinstance(normalized_manifest, dict) else None
        )
        _final_risk = compute_final_risk(_base, _scan_report.verdict, _manifest_risk)

        # force + dangerous → final_risk must be HIGH (spec invariant)
        if force and _scan_report.verdict == "dangerous":
            from app.domain.services.risk_assessor import RiskLevel
            assert _final_risk == RiskLevel.HIGH, (
                f"force+dangerous invariant violated: {_final_risk}"
            )

        skill.trust_origin = trust_origin
        skill.scan_report = _scan_report.to_dict()
        # ========== End R3 gate ==========

        result = await self.skill_repository.upsert(skill)

        # R3: Recompute content_hash on actual disk layout for consistency.
        # Install scans a tmpdir (flat), but repo writes bundle/ subdirectory.
        # compute_content_hash excludes meta.json/bundle_index.json to avoid
        # self-referential hash, so this is safe to do after upsert.
        try:
            from app.domain.services.skills_guard import SkillsGuard
            _get_dir = getattr(self.skill_repository, "get_skill_dir", None)
            _actual_dir = _get_dir(result.id) if _get_dir else None
            if _actual_dir and _actual_dir.exists():
                _actual_hash = SkillsGuard.compute_content_hash(_actual_dir)
                _stored = dict(result.scan_report or {})
                _stored["content_hash"] = _actual_hash
                result.scan_report = _stored

                # force + dangerous: persist force_approved_hash as its own
                # Skill field so it survives any scan_report overwrite.
                if force and _scan_report.verdict == "dangerous":
                    result.force_approved_hash = _actual_hash

                await self.skill_repository.upsert(result)
        except Exception:
            logger.warning("R3: failed to recompute content_hash post-upsert", exc_info=True)

        return result, _scan_report

    async def set_skill_enabled(self, skill_id: str, enabled: bool) -> Skill:
        skill = await self.skill_repository.get_by_id(skill_id)
        if not skill:
            raise NotFoundError(f"Skill[{skill_id}]不存在")

        skill.enabled = enabled
        return await self.skill_repository.upsert(skill)

    async def delete_skill(
        self,
        skill_id: str,
        *,
        actor_id: str | None = None,
        uninstall_context: "UninstallContext | None" = None,
        missing_ok: bool = False,
    ) -> None:
        # standalone 成员守卫（§8.4）：uninstall_context 为 None 或 correlation 为 None →
        # 查 read_port；行带 parent_plugin_ext_id 拒删。saga 路（correlation 非 None）跳过。
        if self._registry_read_port is not None and (
                uninstall_context is None or uninstall_context.correlation_id is None):
            _row = await self._registry_read_port.get_row("skill", skill_id)
            if _row is not None and _row.parent_plugin_ext_id is not None:
                raise ManagedByPluginError("managed by plugin——先卸 plugin（§8.4）")

        deleted = await self.skill_repository.delete(skill_id)
        if not deleted:
            if missing_ok:
                return                      # saga 幂等（R19#F3）
            raise NotFoundError(f"Skill[{skill_id}]不存在")

        # D1a §6.1 delete hook（standalone → 构造 UninstallContext；off=None 时零记账）
        if self._registry_write_port is not None:
            ctx = uninstall_context
            if ctx is None and actor_id is not None:
                ctx = UninstallContext(correlation_id=None, actor_user_id=actor_id)
            if ctx is not None:
                await self._registry_write_port.record_delete(
                    "skill", skill_id, uninstall_context=ctx)

    @classmethod
    def _build_context_blob(
        cls,
        skill_md: str,
        bundle_files: dict[str, SkillBundleFile],
    ) -> tuple[str, list[str]]:
        body = cls._strip_frontmatter(skill_md)
        if not body:
            return "", []

        sections = [body.strip()]
        context_refs: list[str] = []
        total_chars = len(sections[0])

        referenced_paths = cls._extract_referenced_paths(skill_md)
        for ref_path in referenced_paths:
            normalized_ref = cls._resolve_relative_reference(ref_path)
            if not normalized_ref:
                continue
            bundle_file = bundle_files.get(normalized_ref)
            if not bundle_file:
                continue
            ext = Path(normalized_ref).suffix.lower()
            if ext not in TEXT_INJECT_EXTENSIONS:
                continue
            try:
                ref_text = bundle_file.content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            ref_text = ref_text.strip()
            if not ref_text:
                continue
            if len(ref_text) > MAX_CONTEXT_REF_FILE_CHARS:
                ref_text = ref_text[:MAX_CONTEXT_REF_FILE_CHARS].rstrip() + "\n...(truncated)"

            section = f"## reference:{normalized_ref}\n{ref_text}"
            if total_chars + len(section) > MAX_CONTEXT_BLOB_CHARS:
                break
            sections.append(section)
            total_chars += len(section)
            context_refs.append(normalized_ref)

        blob = "\n\n".join(sections).strip()
        if len(blob) > MAX_CONTEXT_BLOB_CHARS:
            blob = blob[:MAX_CONTEXT_BLOB_CHARS].rstrip()
        return blob, context_refs

    @staticmethod
    def _validate_native_command_policy(manifest: dict) -> None:
        raw_runtime = manifest.get("runtime_type")
        runtime_type = (
            raw_runtime.value
            if hasattr(raw_runtime, "value")
            else str(raw_runtime or "").strip().lower()
        )
        if runtime_type != "native":
            return

        tools = manifest.get("tools")
        if not isinstance(tools, list):
            return

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            entry = tool.get("entry")
            if not isinstance(entry, dict):
                continue
            command = str(entry.get("command") or "").strip()
            if not command:
                continue
            for pattern in DEFAULT_BLOCKED_NATIVE_COMMAND_PATTERNS:
                if re.search(pattern, command, flags=re.IGNORECASE):
                    raise ValidationError(msg=f"native skill 命令包含高风险模式: {pattern}")

    @classmethod
    def _normalize_manifest_input(
        cls,
        source_ref: str,
        manifest: dict[str, Any] | None,
        skill_md: str,
    ) -> dict[str, Any]:
        incoming = dict(manifest or {})
        if incoming:
            return incoming
        if not skill_md.strip():
            raise ValidationError(msg="至少需要提供 SKILL.md 或 Manifest")

        return cls._build_manifest_from_skill_md(source_ref=source_ref, skill_md=skill_md)

    @classmethod
    def _build_manifest_from_skill_md(cls, source_ref: str, skill_md: str) -> dict[str, Any]:
        frontmatter = cls._extract_frontmatter(skill_md)
        title = cls._extract_title(skill_md)
        fallback_name = source_ref.split("/")[-1] or "skill"

        runtime_type = str(frontmatter.get("runtime_type") or "native").strip().lower()
        if runtime_type not in {"native", "mcp", "a2a"}:
            runtime_type = "native"

        tools_raw = frontmatter.get("tools")
        tools = tools_raw if isinstance(tools_raw, list) else []

        return {
            "name": str(frontmatter.get("name") or title or fallback_name),
            "description": str(frontmatter.get("description") or ""),
            "version": str(frontmatter.get("version") or "0.1.0"),
            "runtime_type": runtime_type,
            "tools": tools,
            "activation": frontmatter.get("activation", {}),
            "policy": frontmatter.get("policy", {}),
            "security": frontmatter.get("security", {}),
            "skill_md": skill_md,
        }

    @staticmethod
    def _extract_frontmatter(skill_md: str) -> dict[str, Any]:
        lines = skill_md.splitlines()
        if len(lines) < 3 or lines[0].strip() != "---":
            return {}

        end_idx = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_idx = idx
                break
        if end_idx is None:
            return {}

        block = "\n".join(lines[1:end_idx]).strip()
        if not block:
            return {}

        try:
            parsed = yaml.safe_load(block)
        except Exception:
            return {}

        if isinstance(parsed, dict):
            return parsed
        return {}

    @staticmethod
    def _extract_title(skill_md: str) -> str:
        for raw in skill_md.splitlines():
            line = raw.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip()
        return ""

    @staticmethod
    def _strip_frontmatter(skill_md: str) -> str:
        lines = skill_md.splitlines()
        if len(lines) < 3 or lines[0].strip() != "---":
            return skill_md.strip()

        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :]).strip()
        return skill_md.strip()

    @classmethod
    def _extract_referenced_paths(cls, skill_md: str) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()

        for match in re.findall(r"\[[^\]]+\]\(([^)]+)\)", skill_md or ""):
            candidate = str(match).strip()
            if candidate and candidate not in seen:
                refs.append(candidate)
                seen.add(candidate)

        for match in re.findall(
            r"(?:(?:^|[\s`'\"(]))((?:references|assets|scripts)/[^\s`'\"()]+)",
            skill_md or "",
        ):
            candidate = str(match).strip()
            if candidate and candidate not in seen:
                refs.append(candidate)
                seen.add(candidate)

        return refs

    @staticmethod
    def _resolve_relative_reference(raw_ref: str) -> str:
        candidate = (raw_ref or "").strip()
        if not candidate:
            return ""

        candidate = candidate.split("#", 1)[0].split("?", 1)[0].strip()
        if not candidate:
            return ""

        parsed = urlparse(candidate)
        if parsed.scheme or parsed.netloc:
            return ""

        normalized = candidate.replace("\\", "/")
        if normalized.startswith("/"):
            normalized = normalized.lstrip("/")

        parts: list[str] = []
        for part in normalized.split("/"):
            token = part.strip()
            if not token or token == ".":
                continue
            if token == "..":
                if not parts:
                    return ""
                parts.pop()
                continue
            parts.append(token)
        return "/".join(parts)
