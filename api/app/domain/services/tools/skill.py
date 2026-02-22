"""Skill 统一工具层"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from app.domain.external.sandbox import Sandbox
from app.domain.models.skill import Skill, SkillRuntimeType
from app.domain.models.tool_result import ToolResult

from .a2a import A2ATool
from .base import BaseTool
from .mcp import MCPTool

logger = logging.getLogger(__name__)
TOOL_NAME_MAX_LENGTH = 64


class SkillTool(BaseTool):
    """统一 Skill 工具层，支持 native/mcp/a2a 三类运行时"""

    name: str = "skill"

    def __init__(
        self,
        sandbox: Sandbox,
        mcp_tool: MCPTool,
        a2a_tool: A2ATool,
    ) -> None:
        super().__init__()
        self._sandbox = sandbox
        self._mcp_tool = mcp_tool
        self._a2a_tool = a2a_tool
        self._skills: list[Skill] = []
        self._tools: list[dict[str, Any]] = []
        self._tool_bindings: dict[str, dict[str, Any]] = {}
        self._tool_name_index: dict[str, int] = {}

    async def initialize(self, skills: list[Skill]) -> None:
        """初始化可用 Skill 列表并生成工具声明"""
        self._skills = [skill for skill in skills if skill.enabled]
        self._tools = []
        self._tool_bindings = {}
        self._tool_name_index = {}

        for skill in self._skills:
            runtime_type = skill.runtime_type
            manifest_tools = (skill.manifest or {}).get("tools", [])
            if not isinstance(manifest_tools, list):
                continue

            for manifest_tool in manifest_tools:
                if not isinstance(manifest_tool, dict):
                    continue

                raw_tool_name = str(manifest_tool.get("name") or "").strip()
                if not raw_tool_name:
                    continue

                function_name = self._build_function_name(skill.slug, raw_tool_name)
                parameters = manifest_tool.get("parameters")
                required = manifest_tool.get("required")
                description = self._build_tool_description(skill, manifest_tool)

                if not isinstance(parameters, dict):
                    parameters = {}
                if not isinstance(required, list):
                    required = []

                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": parameters,
                            "required": required,
                        },
                    },
                }
                self._tools.append(tool_schema)
                self._tool_bindings[function_name] = {
                    "skill": skill,
                    "runtime_type": runtime_type,
                    "manifest_tool": manifest_tool,
                }

        self._tools_cache = self._tools
        logger.info(
            "SkillTool 初始化完成: enabled_skills=%s, available_tools=%s",
            len(self._skills),
            [tool["function"]["name"] for tool in self._tools],
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return self._tools

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_bindings

    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        binding = self._tool_bindings.get(tool_name)
        if not binding:
            return ToolResult(success=False, message=f"Skill工具[{tool_name}]不存在")

        runtime_type: SkillRuntimeType = binding["runtime_type"]
        manifest_tool: dict[str, Any] = binding["manifest_tool"]

        if runtime_type == SkillRuntimeType.NATIVE:
            return await self._invoke_native(manifest_tool, kwargs)
        if runtime_type == SkillRuntimeType.MCP:
            return await self._invoke_mcp(manifest_tool, kwargs)
        if runtime_type == SkillRuntimeType.A2A:
            return await self._invoke_a2a(manifest_tool, kwargs)

        return ToolResult(success=False, message=f"暂不支持的Skill运行时: {runtime_type}")

    async def cleanup(self) -> None:
        self._skills = []
        self._tools = []
        self._tool_bindings = {}
        self._tools_cache = []

    async def _invoke_native(
        self,
        manifest_tool: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> ToolResult:
        entry = manifest_tool.get("entry")
        if not isinstance(entry, dict):
            return ToolResult(success=False, message="native skill 缺少 entry 配置")

        exec_dir = str(entry.get("exec_dir") or "/home/ubuntu/workspace")
        command = str(entry.get("command") or "").strip()
        if not command:
            return ToolResult(success=False, message="native skill 缺少 entry.command")

        payload = json.dumps(kwargs, ensure_ascii=False) if kwargs else "{}"
        full_command = f"{command} '{payload}'"
        session_id = f"skill-{uuid.uuid4()}"

        result = await self._sandbox.exec_command(session_id, exec_dir, full_command)
        if not result.success:
            return result

        output = await self._sandbox.read_shell_output(session_id)
        return output if output.success else result

    async def _invoke_mcp(
        self,
        manifest_tool: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> ToolResult:
        entry = manifest_tool.get("entry")
        if not isinstance(entry, dict):
            return ToolResult(success=False, message="mcp skill 缺少 entry 配置")

        actual_tool_name = str(entry.get("tool_name") or manifest_tool.get("name") or "").strip()
        if not actual_tool_name:
            return ToolResult(success=False, message="mcp skill 缺少 entry.tool_name")

        return await self._mcp_tool.invoke(actual_tool_name, **kwargs)

    async def _invoke_a2a(
        self,
        manifest_tool: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> ToolResult:
        entry = manifest_tool.get("entry")
        if not isinstance(entry, dict):
            return ToolResult(success=False, message="a2a skill 缺少 entry 配置")

        agent_id = str(entry.get("agent_id") or "").strip()
        if not agent_id:
            return ToolResult(success=False, message="a2a skill 缺少 entry.agent_id")

        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            query = json.dumps(kwargs, ensure_ascii=False)

        return await self._a2a_tool.call_remote_agent(id=agent_id, query=query)

    @classmethod
    def _normalize_function_part(cls, raw: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", raw or "").strip("_").lower()
        return normalized or "tool"

    def _build_function_name(self, skill_slug: str, tool_name: str) -> str:
        slug_part = self._normalize_function_part(skill_slug)
        tool_part = self._normalize_function_part(tool_name)
        base = f"skill_{slug_part}_{tool_part}"
        suffix_num = self._tool_name_index.get(base, 0)
        self._tool_name_index[base] = suffix_num + 1

        candidate = base if suffix_num == 0 else f"{base}_{suffix_num}"
        if len(candidate) <= TOOL_NAME_MAX_LENGTH:
            return candidate

        digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
        prefix = candidate[: TOOL_NAME_MAX_LENGTH - 9].rstrip("_")
        return f"{prefix}_{digest}"

    @classmethod
    def _extract_skill_md_summary(cls, skill_md: str) -> str:
        if not skill_md:
            return ""

        lines: list[str] = []
        for raw_line in skill_md.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            line = re.sub(r"`+", "", line)
            lines.append(line)
            if len(" ".join(lines)) >= 180:
                break

        summary = " ".join(lines).strip()
        if len(summary) > 200:
            summary = summary[:197].rstrip() + "..."
        return summary

    def _build_tool_description(self, skill: Skill, manifest_tool: dict[str, Any]) -> str:
        base_desc = str(manifest_tool.get("description") or "").strip()
        if not base_desc:
            base_desc = (skill.description or "").strip()

        skill_md_summary = self._extract_skill_md_summary(
            str((skill.manifest or {}).get("skill_md") or "")
        )

        parts = [f"[{skill.name}] ({skill.runtime_type.value})"]
        if base_desc:
            parts.append(base_desc)
        if skill_md_summary:
            parts.append(f"Skill guide: {skill_md_summary}")

        return " ".join(parts)[:512]
