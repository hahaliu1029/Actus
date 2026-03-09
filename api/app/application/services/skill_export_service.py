"""Skill 导出服务"""

from __future__ import annotations

import asyncio
import io
import json
import re
import zipfile
from pathlib import Path

from app.application.errors.exceptions import NotFoundError
from app.interfaces.schemas.skill import SkillExportFormat


class SkillExportService:
    """将 Skill 导出为 ZIP 包。"""

    _SCRIPT_EXTS = {".py", ".sh", ".js", ".ts"}
    _REFERENCE_EXTS = {".md", ".txt", ".json", ".yaml", ".yml"}

    def __init__(self, skills_root_dir: str | Path) -> None:
        self._root = Path(skills_root_dir)

    async def export_skill(
        self, skill_id: str, fmt: SkillExportFormat
    ) -> tuple[bytes, str]:
        """返回 (zip_bytes, filename)。"""
        skill_dir = self._root / skill_id
        if not skill_dir.resolve().is_relative_to(self._root.resolve()):
            raise NotFoundError(f"Skill 不存在: {skill_id}")
        if not skill_dir.is_dir() or not (skill_dir / "meta.json").exists():
            raise NotFoundError(f"Skill 不存在: {skill_id}")

        meta = json.loads((skill_dir / "meta.json").read_text(encoding="utf-8"))
        slug = str(meta.get("slug") or skill_id)

        if fmt == SkillExportFormat.ACTUS:
            zip_bytes = await asyncio.to_thread(self._pack_actus, skill_dir, slug)
        else:
            zip_bytes = await asyncio.to_thread(
                self._pack_agent_skills, skill_dir, slug, meta
            )

        filename = f"{slug}-{fmt.value}.zip"
        return zip_bytes, filename

    # ---- Actus native format ----

    def _pack_actus(self, skill_dir: Path, slug: str) -> bytes:
        """打包 Actus 原生格式：原样复制所有文件。"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(skill_dir.rglob("*")):
                if file_path.is_file():
                    arcname = f"{slug}/{file_path.relative_to(skill_dir)}"
                    zf.write(file_path, arcname)
        return buf.getvalue()

    # ---- Agent Skills standard format ----

    def _pack_agent_skills(
        self, skill_dir: Path, slug: str, meta: dict
    ) -> bytes:
        """打包为 Agent Skills 标准格式。"""
        manifest_path = skill_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )

        skill_md_content = self._build_agent_skills_md(slug, meta, manifest)

        runtime_type = str(meta.get("runtime_type") or "native")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{slug}/SKILL.md", skill_md_content)
            if runtime_type == "native":
                self._add_bundle_files(zf, skill_dir, slug)
        return buf.getvalue()

    def _build_agent_skills_md(
        self, slug: str, meta: dict, manifest: dict
    ) -> str:
        """构建符合 Agent Skills 标准的 SKILL.md。"""
        name = self._normalize_agent_skill_name(
            str(meta.get("slug") or meta.get("name") or slug)
        )
        description = str(
            meta.get("description") or manifest.get("description") or ""
        )[:1024]
        version = str(meta.get("version") or manifest.get("version") or "0.1.0")
        runtime_type = str(
            meta.get("runtime_type") or manifest.get("runtime_type") or "native"
        )
        risk_level = str((manifest.get("policy") or {}).get("risk_level") or "")

        lines = ["---"]
        lines.append(f"name: {name}")
        lines.append(f"description: {description or name}")

        compatibility = self._infer_compatibility(
            runtime_type, meta.get("source_ref", "")
        )
        if compatibility:
            lines.append(f"compatibility: {compatibility}")

        lines.append("metadata:")
        lines.append(f'  version: "{version}"')
        lines.append(f"  runtime-type: {runtime_type}")
        if risk_level:
            lines.append(f"  risk-level: {risk_level}")

        lines.append("---")

        raw_skill_md = str(manifest.get("skill_md") or "")
        body = self._extract_body(raw_skill_md)
        if not body.strip():
            body = f"\n# {meta.get('name') or name}\n\n{description}\n"

        lines.append(body)
        return "\n".join(lines)

    def _add_bundle_files(
        self, zf: zipfile.ZipFile, skill_dir: Path, slug: str
    ) -> None:
        """将 bundle/ 中的文件按类型分类到 scripts/、references/、assets/。"""
        bundle_dir = skill_dir / "bundle"
        if not bundle_dir.is_dir():
            return
        for file_path in sorted(bundle_dir.rglob("*")):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()
            rel = file_path.relative_to(bundle_dir)
            if ext in self._SCRIPT_EXTS:
                arcname = f"{slug}/scripts/{rel}"
            elif ext in self._REFERENCE_EXTS:
                arcname = f"{slug}/references/{rel}"
            else:
                arcname = f"{slug}/assets/{rel}"
            zf.write(file_path, arcname)

    @staticmethod
    def _normalize_agent_skill_name(raw: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
        normalized = re.sub(r"-{2,}", "-", normalized)
        return (normalized or "skill")[:64]

    @staticmethod
    def _extract_body(skill_md: str) -> str:
        if not skill_md.startswith("---"):
            return skill_md
        end = skill_md.find("---", 3)
        if end == -1:
            return skill_md
        return skill_md[end + 3 :]

    @staticmethod
    def _infer_compatibility(runtime_type: str, source_ref: str) -> str:
        if runtime_type == "mcp":
            return (
                f"Requires MCP server: {source_ref}"
                if source_ref
                else "Requires MCP server"
            )
        if runtime_type == "a2a":
            return (
                f"Requires remote agent: {source_ref}"
                if source_ref
                else "Requires remote agent"
            )
        if runtime_type == "native":
            return "Requires Python 3.12 sandbox environment"
        return ""
