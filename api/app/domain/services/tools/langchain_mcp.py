"""Wrap existing MCPTool schemas as LangChain StructuredTool instances.

Instead of using langchain-mcp-adapters (which requires direct MCP server access),
we wrap our existing MCPTool.get_tools() schemas and MCPTool.invoke() dispatcher
into LangChain tools. This preserves the existing MCP client management.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Literal, Optional, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from app.domain.services.tools.mcp import MCPTool

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


def _make_mcp_coroutine(mcp_tool: MCPTool, tool_name: str):
    """为每个 MCP tool 创建独立的协程，通过闭包绑定 tool_name。"""

    async def _invoke(**kwargs: Any) -> str:
        result = await mcp_tool.invoke(tool_name, **kwargs)
        if hasattr(result, "message") and result.message:
            return result.message
        if hasattr(result, "data") and result.data:
            return json.dumps(result.data)
        return str(result)

    return _invoke


def create_mcp_langchain_tools(mcp_tool: MCPTool) -> list[StructuredTool]:
    """Convert MCPTool's registered tools into LangChain StructuredTool instances."""
    tools: list[StructuredTool] = []

    for schema in mcp_tool.get_tools():
        fn_def = schema.get("function", {})
        name = fn_def.get("name", "")
        description = fn_def.get("description", "")
        parameters = fn_def.get("parameters", {})

        if not name:
            continue

        args_schema = _json_schema_to_pydantic(name, parameters)

        tool = StructuredTool.from_function(
            coroutine=_make_mcp_coroutine(mcp_tool, name),
            name=name,
            description=description,
            args_schema=args_schema,
        )
        tools.append(tool)

    return tools
