from __future__ import annotations

"""File-system based Skill repository."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.skill_diagnostic import SkillDiagnostic
from app.domain.repositories.skill_repository import SkillRepository

logger = logging.getLogger(__name__)


class FileSkillRepository(SkillRepository):
    """基于文件系统的 Skill 仓储实现。"""

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)

    async def list(self) -> list[Skill]:
        return await asyncio.to_thread(self._list_sync, enabled_only=False)

    async def list_enabled(self) -> list[Skill]:
        return await asyncio.to_thread(self._list_sync, enabled_only=True)

    async def get_by_id(self, skill_id: str) -> Skill | None:
        return await asyncio.to_thread(self._read_skill_sync, self._skill_dir(skill_id))

    async def get_by_slug(self, slug: str) -> Skill | None:
        skills = await self.list()
        for skill in skills:
            if skill.slug == slug:
                return skill
        return None

    async def upsert(self, skill: Skill) -> Skill:
        await asyncio.to_thread(self._upsert_sync, skill)
        loaded = await self.get_by_id(skill.id)
        return loaded or skill

    async def delete(self, skill_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, skill_id)

    async def list_with_diagnostics(self) -> list[SkillDiagnostic]:
        return await asyncio.to_thread(self._list_with_diagnostics_sync)

    def _list_with_diagnostics_sync(self) -> list[SkillDiagnostic]:
        self._ensure_root()
        return [
            self._diagnose_skill_sync(child)
            for child in sorted(self._root_dir.iterdir())
            if child.is_dir()
        ]

    def get_skill_dir(self, skill_id: str) -> Path:
        """Public accessor for the skill's filesystem directory."""
        return self._skill_dir(skill_id)

    def _ensure_root(self) -> None:
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def _skill_dir(self, skill_id: str) -> Path:
        return self._root_dir / skill_id

    def _list_sync(self, enabled_only: bool) -> list[Skill]:
        self._ensure_root()
        skills: list[Skill] = []
        for child in sorted(self._root_dir.iterdir()):
            if not child.is_dir():
                continue
            try:
                skill = self._read_skill_sync(child)
            except Exception:
                # B9 PR-0（spec §8）：单 skill 损坏不塌整列表；诊断走 list_with_diagnostics
                logger.warning("跳过损坏的 skill 目录: %s", child.name, exc_info=True)
                continue
            if not skill:
                continue
            if enabled_only and not skill.enabled:
                continue
            skills.append(skill)

        skills.sort(key=lambda item: item.created_at, reverse=True)
        return skills

    def _read_skill_sync(self, skill_dir: Path) -> Skill | None:
        meta_path = skill_dir / "meta.json"
        manifest_path = skill_dir / "manifest.json"
        if not meta_path.exists() or not manifest_path.exists():
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        skill_md_path = skill_dir / "SKILL.md"
        if skill_md_path.exists() and not manifest.get("skill_md"):
            manifest["skill_md"] = skill_md_path.read_text(encoding="utf-8")

        bundle_index_path = skill_dir / "bundle_index.json"
        if bundle_index_path.exists():
            try:
                bundle_index = json.loads(bundle_index_path.read_text(encoding="utf-8"))
            except Exception:
                bundle_index = []
            if isinstance(bundle_index, list):
                manifest.setdefault("bundle_file_count", len(bundle_index))

        return Skill(
            id=str(meta["id"]),
            slug=str(meta["slug"]),
            name=str(meta["name"]),
            description=str(meta.get("description") or ""),
            version=str(meta.get("version") or "0.1.0"),
            source_type=SkillSourceType(str(meta["source_type"])),
            source_ref=str(meta["source_ref"]),
            runtime_type=SkillRuntimeType(str(meta["runtime_type"])),
            manifest=manifest,
            enabled=bool(meta.get("enabled", True)),
            installed_by=meta.get("installed_by"),
            created_at=self._parse_datetime(meta.get("created_at")),
            updated_at=self._parse_datetime(meta.get("updated_at")),
            trust_origin=meta.get("trust_origin", "user_installed"),
            scan_report=meta.get("scan_report"),
            force_approved_hash=meta.get("force_approved_hash"),
        )

    def _diagnose_skill_sync(self, skill_dir: Path) -> SkillDiagnostic:
        key = skill_dir.name
        meta_path = skill_dir / "meta.json"
        manifest_path = skill_dir / "manifest.json"
        if not meta_path.exists():
            return SkillDiagnostic(skill_key=key, ok=False, error_code="missing_meta", relative_file="meta.json")
        if not manifest_path.exists():
            return SkillDiagnostic(skill_key=key, ok=False, error_code="missing_manifest", relative_file="manifest.json")
        # 区分 meta / manifest 哪个解析失败：逐个 json.loads + 顶层 shape 校验。
        # 顶层必须是 dict：非 dict（如 `[]`）虽是合法 JSON，但会让 _read_skill_sync
        # 里 `manifest.get(...)` / 把非 dict 传给 Skill 崩溃——归因必须指向坏文件本身。
        try:
            meta_parsed = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return SkillDiagnostic(skill_key=key, ok=False, error_code="parse_error", relative_file="meta.json")
        if not isinstance(meta_parsed, dict):
            return SkillDiagnostic(skill_key=key, ok=False, error_code="parse_error", relative_file="meta.json")
        try:
            manifest_parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return SkillDiagnostic(skill_key=key, ok=False, error_code="parse_error", relative_file="manifest.json")
        if not isinstance(manifest_parsed, dict):
            return SkillDiagnostic(skill_key=key, ok=False, error_code="parse_error", relative_file="manifest.json")
        try:
            skill = self._read_skill_sync(skill_dir)
        except Exception:
            # 归因近似：两文件 shape-check 为 dict 后，Skill 构造失败（非法枚举/缺必填键）
            # 源自 meta 字段——manifest 作为不透明 dict 原样透传。注意 _read_skill_sync
            # 还会读 SKILL.md（bundle_index.json 异常在其内部被吞），SKILL.md 的 IO 失败
            # 也走此 fallback；归 meta.json 是对罕见路径的保守近似（P-1 relative_file
            # 词表只有两个文件名）。
            return SkillDiagnostic(skill_key=key, ok=False, error_code="parse_error", relative_file="meta.json")
        if skill is None:
            return SkillDiagnostic(skill_key=key, ok=False, error_code="missing_meta", relative_file="meta.json")
        return SkillDiagnostic(skill_key=key, ok=True, skill=skill)

    def _upsert_sync(self, skill: Skill) -> None:
        self._ensure_root()
        skill_dir = self._skill_dir(skill.id)
        skill_dir.mkdir(parents=True, exist_ok=True)

        manifest = dict(skill.manifest or {})
        bundle_files_raw = manifest.pop("_bundle_files", None)
        skill_md = str(manifest.get("skill_md") or "")

        meta_payload: dict[str, Any] = {
            "id": skill.id,
            "slug": skill.slug,
            "name": skill.name,
            "description": skill.description,
            "version": skill.version,
            "source_type": skill.source_type.value,
            "source_ref": skill.source_ref,
            "runtime_type": skill.runtime_type.value,
            "enabled": skill.enabled,
            "installed_by": skill.installed_by,
            "created_at": skill.created_at.isoformat(),
            "updated_at": skill.updated_at.isoformat(),
            "trust_origin": skill.trust_origin,
            "scan_report": skill.scan_report,
            "force_approved_hash": skill.force_approved_hash,
        }

        # R3: Preserve last_rejected_sync if it exists in current meta.json
        meta_path = skill_dir / "meta.json"
        if meta_path.exists():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if "last_rejected_sync" in existing_meta:
                    meta_payload["last_rejected_sync"] = existing_meta["last_rejected_sync"]
            except (json.JSONDecodeError, OSError):
                pass

        meta_path.write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (skill_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

        if isinstance(bundle_files_raw, dict):
            bundle_dir = skill_dir / "bundle"
            self._clear_dir(bundle_dir)
            bundle_dir.mkdir(parents=True, exist_ok=True)

            bundle_index: list[dict[str, Any]] = []
            for raw_path, raw_content in sorted(bundle_files_raw.items()):
                normalized_path = self._normalize_bundle_path(str(raw_path))
                if not normalized_path:
                    continue

                content = (
                    raw_content
                    if isinstance(raw_content, bytes)
                    else str(raw_content).encode("utf-8")
                )
                target_path = bundle_dir / normalized_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(content)

                bundle_index.append(
                    {
                        "path": normalized_path,
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "is_text": Path(normalized_path).suffix.lower()
                        in {".md", ".txt", ".yaml", ".yml", ".json"},
                    }
                )

            (skill_dir / "bundle_index.json").write_text(
                json.dumps(bundle_index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _delete_sync(self, skill_id: str) -> bool:
        skill_dir = self._skill_dir(skill_id)
        if not skill_dir.exists() or not skill_dir.is_dir():
            return False

        for path in sorted(skill_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        skill_dir.rmdir()
        return True

    @staticmethod
    def _clear_dir(path: Path) -> None:
        if not path.exists() or not path.is_dir():
            return
        for item in sorted(path.rglob("*"), reverse=True):
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                item.rmdir()
        path.rmdir()

    @staticmethod
    def _normalize_bundle_path(raw_path: str) -> str:
        normalized = raw_path.replace("\\", "/").strip()
        if not normalized:
            return ""
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

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
        return datetime.now()
