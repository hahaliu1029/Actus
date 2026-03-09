"""Skill 接口 Schema"""

from enum import Enum
from typing import Any, Dict, List

from app.domain.models.skill import SkillRuntimeType, SkillSourceType
from pydantic import BaseModel, Field


class SkillInstallRequest(BaseModel):
    """安装 Skill 请求"""

    source_type: SkillSourceType = Field(..., description="来源类型")
    source_ref: str = Field(
        ...,
        description="来源标识：github 支持仓库根 URL 或 tree URL；local 使用绝对路径或 local:/abs/path",
    )
    manifest: Dict[str, Any] = Field(
        default_factory=dict,
        description="可选兼容字段；默认由 SKILL.md frontmatter 自动生成",
    )
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
    installed_by: str | None = None
    created_at: str
    updated_at: str
    bundle_file_count: int = 0
    context_ref_count: int = 0
    last_sync_at: str | None = None


class SkillListResponse(BaseModel):
    """Skill 列表响应"""

    skills: List[SkillItem] = Field(default_factory=list)


class SkillRiskPolicyItem(BaseModel):
    """Skill 风险策略"""

    mode: str = Field(..., description="off | enforce_confirmation")


class SkillToolItem(BaseModel):
    """Skill Tool 声明"""

    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    entry: Dict[str, Any] | None = None


class BundleFileItem(BaseModel):
    """Bundle 文件条目"""

    path: str
    size: int
    sha256: str
    is_text: bool


class SkillDetailResponse(BaseModel):
    """Skill 详情响应"""

    id: str
    slug: str
    name: str
    description: str
    version: str
    source_type: SkillSourceType
    source_ref: str
    runtime_type: SkillRuntimeType
    enabled: bool
    installed_by: str | None = None
    created_at: str
    updated_at: str
    bundle_file_count: int = 0
    context_ref_count: int = 0
    last_sync_at: str | None = None
    tools: List[SkillToolItem] = Field(default_factory=list)
    skill_md: str = ""
    bundle_files: List[BundleFileItem] = Field(default_factory=list)
    activation: Dict[str, Any] = Field(default_factory=dict)
    policy: Dict[str, Any] = Field(default_factory=dict)
    security: Dict[str, Any] = Field(default_factory=dict)


class SkillExportFormat(str, Enum):
    """Skill 导出格式"""
    AGENT_SKILLS = "agent-skills"
    ACTUS = "actus"
