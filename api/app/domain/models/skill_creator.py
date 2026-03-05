"""Skill Creator 领域模型。"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator


class ToolParamDef(BaseModel):
    """工具参数定义。"""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False


class ToolDef(BaseModel):
    """工具定义。"""

    name: str
    description: str
    parameters: list[ToolParamDef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_parameters(cls, data: Any) -> Any:
        """兼容 parameters 为对象字典的场景。"""
        if not isinstance(data, dict):
            return data

        raw_parameters = data.get("parameters")
        if raw_parameters is None:
            normalized = dict(data)
            normalized["parameters"] = []
            return normalized

        if not isinstance(raw_parameters, dict):
            return data

        required_raw = data.get("required")
        required_names = (
            {str(item) for item in required_raw if isinstance(item, str)}
            if isinstance(required_raw, list)
            else set()
        )

        normalized_parameters: list[dict[str, Any]] = []
        for name, spec in raw_parameters.items():
            param_name = str(name).strip()
            if not param_name:
                continue
            if isinstance(spec, dict):
                normalized_parameters.append(
                    {
                        "name": param_name,
                        "type": str(spec.get("type") or "string"),
                        "description": str(spec.get("description") or ""),
                        "required": bool(
                            spec.get("required", param_name in required_names)
                        ),
                    }
                )
            else:
                normalized_parameters.append(
                    {
                        "name": param_name,
                        "type": "string",
                        "description": "",
                        "required": param_name in required_names,
                    }
                )

        normalized = dict(data)
        normalized["parameters"] = normalized_parameters
        return normalized


class SkillBlueprint(BaseModel):
    """由 LLM 从需求中提取的结构化蓝图。"""

    skill_name: str
    description: str
    tools: list[ToolDef] = Field(default_factory=list)
    search_keywords: list[str] = Field(default_factory=list)
    estimated_deps: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def normalized_slug(self) -> str:
        slug = self.skill_name.strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = re.sub(r"-{2,}", "-", slug).strip("-")
        return slug or "skill"


class ScriptFile(BaseModel):
    """生成的脚本文件。"""

    path: str
    content: str


class SkillGeneratedFiles(BaseModel):
    """生成阶段输出的 Skill 文件集合。"""

    skill_md: str
    manifest: dict[str, Any]
    scripts: list[ScriptFile] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class GitHubRepoInfo(BaseModel):
    """GitHub 仓库摘要。"""

    name: str
    full_name: str
    description: str = ""
    stars: int = 0
    url: str
    readme_summary: str = ""
    install_command: str = ""


class SkillCreationProgress(BaseModel):
    """Skill 创建过程中的进度事件。"""

    step: Literal["analyzing", "researching", "generating", "validating", "installing"]
    message: str
    detail: str | None = None
    references: list[GitHubRepoInfo] | None = None


class SkillCreationResult(BaseModel):
    """Skill 创建完成结果。"""

    skill_id: str
    skill_name: str
    tools: list[str]
    files_count: int
    summary: str
