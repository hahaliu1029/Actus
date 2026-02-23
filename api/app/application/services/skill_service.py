"""Skill 服务"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.application.errors.exceptions import NotFoundError, ValidationError
from app.application.services.skill_source_loader import (
    SkillBundleFile,
    SkillSourceLoader,
    TEXT_INJECT_EXTENSIONS,
)
from app.domain.models.skill import (
    Skill,
    SkillManifest,
    SkillSourceType,
    build_skill_key,
    normalize_skill_slug,
)
from app.domain.repositories.skill_repository import SkillRepository
import yaml

DEFAULT_BLOCKED_NATIVE_COMMAND_PATTERNS = (
    r"\brm\s+-rf\b",
    r":\(\)\s*\{",
    r"\bmkfs\.",
    r"\bshutdown\b",
    r"\breboot\b",
)
MAX_CONTEXT_BLOB_CHARS = 12 * 1024
MAX_CONTEXT_REF_FILE_CHARS = 2 * 1024


class SkillService:
    """Skill 生态管理服务"""

    def __init__(
        self,
        skill_repository: SkillRepository,
        source_loader: SkillSourceLoader | None = None,
    ) -> None:
        self.skill_repository = skill_repository
        self._source_loader = source_loader or SkillSourceLoader()

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

        return await self.skill_repository.upsert(skill)

    async def set_skill_enabled(self, skill_id: str, enabled: bool) -> Skill:
        skill = await self.skill_repository.get_by_id(skill_id)
        if not skill:
            raise NotFoundError(f"Skill[{skill_id}]不存在")

        skill.enabled = enabled
        return await self.skill_repository.upsert(skill)

    async def delete_skill(self, skill_id: str) -> None:
        deleted = await self.skill_repository.delete(skill_id)
        if not deleted:
            raise NotFoundError(f"Skill[{skill_id}]不存在")

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
