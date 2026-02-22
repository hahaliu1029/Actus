"""Skill 服务"""

from app.application.errors.exceptions import NotFoundError, ValidationError
from app.domain.models.skill import (
    Skill,
    SkillDiscoveryItem,
    SkillManifest,
    SkillRuntimeType,
    SkillSourceType,
    normalize_skill_slug,
)
from app.domain.repositories.skill_repository import SkillRepository


class SkillService:
    """Skill 生态管理服务"""

    def __init__(self, skill_repository: SkillRepository) -> None:
        self.skill_repository = skill_repository

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
        try:
            parsed_manifest = SkillManifest.model_validate(manifest)
        except Exception as e:
            raise ValidationError(msg=f"Manifest 校验失败: {e}") from e

        if not parsed_manifest.tools:
            raise ValidationError(msg="Manifest 必须包含至少一个工具定义")

        slug = normalize_skill_slug(parsed_manifest.slug or parsed_manifest.name)
        existed = await self.skill_repository.get_by_slug(slug)

        manifest_payload = dict(manifest)
        if skill_md:
            manifest_payload["skill_md"] = skill_md

        skill_payload = {
            "slug": slug,
            "name": parsed_manifest.name,
            "description": parsed_manifest.description,
            "version": parsed_manifest.version,
            "source_type": source_type,
            "source_ref": source_ref,
            "runtime_type": parsed_manifest.runtime_type,
            "manifest": manifest_payload,
            "enabled": existed.enabled if existed else True,
            "installed_by": installed_by,
        }
        if existed:
            skill_payload["id"] = existed.id

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

    async def discover_mcp_skills(self) -> list[SkillDiscoveryItem]:
        return [
            SkillDiscoveryItem(
                source_type=SkillSourceType.MCP_REGISTRY,
                source_ref="mcp:filesystem-basic",
                name="Filesystem Basic",
                description="Read/write sandbox files via MCP provider",
                runtime_type=SkillRuntimeType.MCP,
            ),
            SkillDiscoveryItem(
                source_type=SkillSourceType.MCP_REGISTRY,
                source_ref="mcp:web-search-lite",
                name="Web Search Lite",
                description="Search web knowledge via MCP provider",
                runtime_type=SkillRuntimeType.MCP,
            ),
        ]

    async def discover_github_skills(self) -> list[SkillDiscoveryItem]:
        return [
            SkillDiscoveryItem(
                source_type=SkillSourceType.GITHUB,
                source_ref="github:openai/codex-skills-native-shell",
                name="Native Shell Skill",
                description="Execute curated shell workflow in strong sandbox",
                runtime_type=SkillRuntimeType.NATIVE,
            ),
            SkillDiscoveryItem(
                source_type=SkillSourceType.GITHUB,
                source_ref="github:openai/codex-skills-a2a-triage",
                name="A2A Triage Skill",
                description="Delegate triage tasks to remote agents",
                runtime_type=SkillRuntimeType.A2A,
            ),
        ]
