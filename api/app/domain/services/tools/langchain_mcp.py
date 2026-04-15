"""Wrap existing MCPTool schemas as LangChain StructuredTool instances.

Instead of using langchain-mcp-adapters (which requires direct MCP server access),
we wrap our existing MCPTool.get_tools() schemas and MCPTool.invoke() dispatcher
into LangChain tools. This preserves the existing MCP client management.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Literal, Optional, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from app.domain.services.tools.mcp import MCPTool
from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)

logger = logging.getLogger(__name__)

# JSON Schema type → Python type 映射
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _sanitize_model_name(tool_name: str) -> str:
    """将 MCP 工具名转为合法的 Python 类名，确保每个工具 Model 名唯一。"""
    # mcp_notion_search → McpNotionSearch
    parts = re.split(r"[_\-. ]+", tool_name)
    return "".join(p.capitalize() for p in parts if p) + "Args"


def _resolve_type(prop_def: dict) -> Any:
    """递归解析 JSON Schema property 为 Python 类型注解。

    支持：基本类型、enum、嵌套 object、带 items 的 array。
    """
    # enum → Literal
    enum_values = prop_def.get("enum")
    if enum_values and all(isinstance(v, str) for v in enum_values):
        return Literal[tuple(enum_values)]

    json_type = prop_def.get("type", "")

    # 基本类型
    if json_type in _JSON_TYPE_MAP:
        return _JSON_TYPE_MAP[json_type]

    # array + items → list[item_type]
    if json_type == "array":
        items = prop_def.get("items")
        if items:
            item_type = _resolve_type(items)
            return list[item_type]
        return list

    # object + properties → 动态 Pydantic Model
    if json_type == "object":
        inner_props = prop_def.get("properties")
        if inner_props:
            return _build_pydantic_model("InlineObject", prop_def)
        return dict

    return Any


def _build_pydantic_model(model_name: str, schema: dict) -> type[BaseModel]:
    """从 JSON Schema 构建 Pydantic Model（支持嵌套）。"""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for prop_name, prop_def in properties.items():
        py_type = _resolve_type(prop_def)
        description = prop_def.get("description", "")

        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=description))
        else:
            fields[prop_name] = (
                Optional[py_type],
                Field(default=None, description=description),
            )

    return create_model(model_name, **fields)


def _json_schema_to_pydantic(
    tool_name: str, schema: dict
) -> type[BaseModel] | None:
    """将 MCP 工具的 JSON Schema 转换为 Pydantic Model，用作 args_schema。

    每个工具生成唯一的 Model 类名以避免 Pydantic registry 冲突。
    """
    properties = schema.get("properties", {})
    if not properties:
        return None

    model_name = _sanitize_model_name(tool_name)
    return _build_pydantic_model(model_name, schema)


_SANDBOX_PATH_PREFIX = "/home/ubuntu/"


async def _resolve_sandbox_paths(
    kwargs: dict[str, Any],
    url_map: dict[str, str],
    sandbox_file_uploader: Any | None = None,
) -> dict[str, Any]:
    """Replace sandbox file paths in MCP tool arguments with presigned URLs.

    Two resolution paths:
    1. Static: path found in url_map (user-uploaded files) → instant replacement
    2. Dynamic: path starts with /home/ubuntu/ but not in url_map (agent-generated
       files, e.g. extracted from zip) → download from sandbox, upload to storage,
       get presigned URL, cache in url_map for future calls
    """
    resolved = {}
    for key, value in kwargs.items():
        if not isinstance(value, str) or not value.startswith(_SANDBOX_PATH_PREFIX):
            resolved[key] = value
            continue

        # Path 1: already in URL map
        if value in url_map:
            resolved[key] = url_map[value]
            continue

        # Path 2: dynamic upload for agent-generated sandbox files
        if sandbox_file_uploader is not None:
            try:
                url = await sandbox_file_uploader(value)
                if url:
                    url_map[value] = url  # cache for future calls
                    resolved[key] = url
                    continue
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to upload sandbox file %s for MCP tool: %s", value, exc,
                )

        # Fallback: pass through unchanged (MCP tool will likely fail)
        resolved[key] = value
    return resolved


def _make_mcp_coroutine(
    mcp_tool: MCPTool,
    tool_name: str,
    url_map_ref: Any | None = None,
    sandbox_file_uploader: Any | None = None,
):
    """为每个 MCP tool 创建独立的协程，通过闭包绑定 tool_name。"""

    async def _invoke(**kwargs: Any) -> str:
        # Resolve sandbox paths → presigned URLs before calling MCP server
        if url_map_ref is not None:
            try:
                url_map = url_map_ref()
                kwargs = await _resolve_sandbox_paths(
                    kwargs, url_map, sandbox_file_uploader,
                )
            except Exception:
                pass  # Don't break tool call if resolution fails

        result = await mcp_tool.invoke(tool_name, **kwargs)
        # Raise on failure so the caller detects errors structurally
        if hasattr(result, "success") and not result.success:
            raise RuntimeError(getattr(result, "message", None) or str(result))
        if hasattr(result, "message") and result.message:
            return result.message
        if hasattr(result, "data") and result.data:
            # Return string data as-is; only JSON-encode non-string data
            # (dicts, lists) to avoid double-encoding strings with json.dumps
            return result.data if isinstance(result.data, str) else json.dumps(result.data)
        return str(result)

    return _invoke


def create_mcp_langchain_tools(
    mcp_tool: MCPTool,
    tool_names: set[str] | None = None,
    url_map_ref: Any | None = None,
    sandbox_file_uploader: Any | None = None,
) -> list[StructuredTool]:
    """Convert MCPTool's registered tools into LangChain StructuredTool instances.

    Parameters
    ----------
    tool_names : optional set of tool names to include.
        None means all tools (backward compat); empty set means no tools.
    url_map_ref : optional callable returning dict[str, str] mapping
        sandbox paths to presigned URLs, for automatic path resolution.
    sandbox_file_uploader : optional async callable(sandbox_path) -> presigned_url
        for dynamically uploading agent-generated sandbox files to storage.
    """
    tools: list[StructuredTool] = []

    for schema in mcp_tool.get_tools():
        fn_def = schema.get("function", {})
        name = fn_def.get("name", "")
        description = fn_def.get("description", "")
        parameters = fn_def.get("parameters", {})

        if not name:
            continue
        if tool_names is not None and name not in tool_names:
            continue

        try:
            args_schema = _json_schema_to_pydantic(name, parameters)

            tool = StructuredTool.from_function(
                coroutine=_make_mcp_coroutine(
                    mcp_tool, name,
                    url_map_ref=url_map_ref,
                    sandbox_file_uploader=sandbox_file_uploader,
                ),
                name=name,
                description=description,
                args_schema=args_schema,
            )
            tools.append(tool)
        except Exception:
            logger.warning(
                "Skipping malformed MCP tool schema: %s",
                name,
                exc_info=True,
            )
            continue

    for tool in tools:
        annotate_and_register_tool_source(tool, source="mcp", category="mcp")
    return tools
