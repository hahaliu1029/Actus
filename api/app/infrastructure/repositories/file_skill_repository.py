from __future__ import annotations

"""File-system based Skill repository."""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.repositories.skill_repository import SkillRepository


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
            skill = self._read_skill_sync(child)
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
        )

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
        }
        (skill_dir / "meta.json").write_text(
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
