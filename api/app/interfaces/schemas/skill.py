"""Skill 接口 Schema"""

from typing import Any, Dict, List

from app.domain.models.skill import SkillRuntimeType, SkillSourceType
from pydantic import BaseModel, Field


class SkillInstallRequest(BaseModel):
    """安装 Skill 请求"""

    source_type: SkillSourceType = Field(..., description="来源类型")
    source_ref: str = Field(..., description="来源标识，如市场ID或仓库ID")
    manifest: Dict[str, Any] = Field(default_factory=dict, description="Skill Manifest")
    skill_md: str = Field(default="", description="SKILL.md 内容")


class SkillItem(BaseModel):
    """Skill 列表条目"""

    id: str
    slug: str
    name: str
    description: str
    version: str
    source_type: SkillSourceType
    source_ref: str
    runtime_type: SkillRuntimeType
    enabled: bool


class SkillListResponse(BaseModel):
    """Skill 列表响应"""

    skills: List[SkillItem] = Field(default_factory=list)


class SkillDiscoveryItem(BaseModel):
    """Skill 发现条目"""

    source_type: SkillSourceType
    source_ref: str
    name: str
    description: str
    runtime_type: SkillRuntimeType


class SkillDiscoveryResponse(BaseModel):
    """Skill 发现响应"""

    skills: List[SkillDiscoveryItem] = Field(default_factory=list)
