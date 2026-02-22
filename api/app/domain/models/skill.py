"""Skill 生态领域模型"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class SkillSourceType(str, Enum):
    """Skill 安装来源"""

    MCP_REGISTRY = "mcp_registry"
    GITHUB = "github"


class SkillRuntimeType(str, Enum):
    """Skill 运行时类型"""

    NATIVE = "native"
    MCP = "mcp"
    A2A = "a2a"


class SkillManifestTool(BaseModel):
    """Manifest 中的单个工具声明"""

    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    entry: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class SkillManifest(BaseModel):
    """Skill Manifest 结构"""

    name: str
    slug: Optional[str] = None
    version: str = "0.1.0"
    description: str = ""
    runtime_type: SkillRuntimeType
    tools: List[SkillManifestTool] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class Skill(BaseModel):
    """Skill 领域模型"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    slug: str
    name: str
    description: str = ""
    version: str = "0.1.0"
    source_type: SkillSourceType
    source_ref: str
    runtime_type: SkillRuntimeType
    manifest: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    installed_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(from_attributes=True)


class SkillDiscoveryItem(BaseModel):
    """Skill 发现结果条目"""

    source_type: SkillSourceType
    source_ref: str
    name: str
    description: str
    runtime_type: SkillRuntimeType


def normalize_skill_slug(raw: str) -> str:
    """规范化 skill slug。"""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", raw or "").strip("-").lower()
    return normalized or "skill"
