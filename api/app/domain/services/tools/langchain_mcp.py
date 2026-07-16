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

from app.domain.models.tool_result import AllowError, AllowSuccess, DecisionReason, ToolOutcome
from app.domain.services.tools.extension_attribution import register_extension_tool
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


def _sandbox_files_disabled_outcome(kwargs: dict[str, Any]) -> AllowError | None:
    """SPM Task 24：off 档探测 MCP 工具参数里的沙箱路径。

    任一 str 值以 ``/home/ubuntu/`` 开头 → 返回结构化 disabled ``AllowError``
    （「本部署未启用沙箱文件系统」，match module 的 outcome 约定），调用方短路：
    **不上传、不透传路径、不发 MCP invoke、不抛裸异常**。无沙箱路径参数 → None
    （纯远程 MCP 工具不受影响，正常执行）。
    """
    for value in kwargs.values():
        if isinstance(value, str) and value.startswith(_SANDBOX_PATH_PREFIX):
            message = (
                "本部署未启用沙箱文件系统（sandbox_provision_mode=off），"
                f"无法处理沙箱路径参数：{value}"
            )
            return AllowError(
                content=message,
                reason=DecisionReason(
                    type="exception",
                    code="sandbox_files_disabled",
                    message=message,
                ),
            )
    return None


async def _resolve_sandbox_paths(
    kwargs: dict[str, Any],
    url_map: dict[str, str],
    sandbox_file_uploader: Any | None = None,
    *,
    sandbox_files_enabled: bool = True,
) -> dict[str, Any]:
    """Replace sandbox file paths in MCP tool arguments with presigned URLs.

    Two resolution paths:
    1. Static: path found in url_map (user-uploaded files) → instant replacement
    2. Dynamic: path starts with /home/ubuntu/ but not in url_map (agent-generated
       files, e.g. extracted from zip) → download from sandbox, upload to storage,
       get presigned URL, cache in url_map for future calls

    ``sandbox_files_enabled``（SPM Task 24，R22-U4：显式 flag，**非** ``uploader is
    None`` 推断——always 无 uploader 时同样传 None，None 不可区分 off/always）：off
    档为 ``False``，此时**不触发动态上传**（沙箱 FS 未启用）。``_make_mcp_coroutine``
    已在上游对含沙箱路径的调用短路结构化拒绝，此处 flag 仅做防御性跳过 upload 分支
    （always ``True`` → 逐字节不变）。
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
        if sandbox_files_enabled and sandbox_file_uploader is not None:
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
    *,
    sandbox_files_enabled: bool = True,
):
    """为每个 MCP tool 创建独立的协程，通过闭包绑定 tool_name。

    ``sandbox_files_enabled``（SPM Task 24）：off 档为 ``False``——含 ``/home/ubuntu``
    路径参数的调用**在发 MCP invoke 之前**短路结构化拒绝。该检测独立于
    ``url_map_ref``（flow 独立装配链不传 url_map_ref，仍须能拒绝），确保两条装配链
    统一收口。always / on_demand（``True``）跳过该分支 → 逐字节不变。
    """

    async def _invoke(**kwargs: Any) -> tuple[str, ToolOutcome]:
        # SPM Task 24: off 档——沙箱文件系统未启用。任一沙箱路径参数直接结构化
        # 拒绝（不上传、不透传、不发 MCP invoke），独立于 url_map_ref 布线。
        if not sandbox_files_enabled:
            disabled = _sandbox_files_disabled_outcome(kwargs)
            if disabled is not None:
                return disabled.content, disabled
        # Resolve sandbox paths → presigned URLs before calling MCP server
        if url_map_ref is not None:
            try:
                url_map = url_map_ref()
                kwargs = await _resolve_sandbox_paths(
                    kwargs, url_map, sandbox_file_uploader,
                    sandbox_files_enabled=sandbox_files_enabled,
                )
            except Exception:
                pass  # Don't break tool call if resolution fails

        try:
            result = await mcp_tool.invoke(tool_name, **kwargs)
        except TimeoutError as exc:
            outcome = AllowError(
                content=f"MCP 工具 '{tool_name}' 超时: {exc}",
                reason=DecisionReason(
                    type="timeout",
                    code="mcp_client_timeout",
                    message=str(exc),
                ),
                retryable=True,
            )
            return outcome.content, outcome
        except (ConnectionError, OSError) as exc:
            outcome = AllowError(
                content=f"MCP 连接失败: {exc}",
                reason=DecisionReason(
                    type="exception",
                    code=type(exc).__name__,
                    message=str(exc),
                ),
                retryable=True,
            )
            return outcome.content, outcome
        except Exception as exc:
            outcome = AllowError(
                content=f"MCP 工具 '{tool_name}' 内部异常: {exc}",
                reason=DecisionReason(
                    type="exception",
                    code=type(exc).__name__,
                    message=str(exc),
                ),
            )
            return outcome.content, outcome

        if hasattr(result, "success") and not result.success:
            message = getattr(result, "message", None) or "MCP 工具执行失败"
            outcome = AllowError(
                content=message,
                reason=DecisionReason(
                    type="exception",
                    code="mcp_tool_error",
                    message=message,
                ),
            )
            return outcome.content, outcome

        if hasattr(result, "message") and result.message:
            content = result.message
        elif hasattr(result, "data") and result.data is not None:
            content = (
                result.data
                if isinstance(result.data, str)
                else json.dumps(result.data, ensure_ascii=False)
            )
        else:
            content = str(result)

        outcome = AllowSuccess(
            content=content,
            data=result.data if hasattr(result, "data") and isinstance(result.data, dict) else None,
        )
        return outcome.content, outcome

    return _invoke


def create_mcp_langchain_tools(
    mcp_tool: MCPTool,
    tool_names: set[str] | None = None,
    url_map_ref: Any | None = None,
    sandbox_file_uploader: Any | None = None,
    *,
    sandbox_files_enabled: bool = True,
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
    sandbox_files_enabled : SPM Task 24 explicit off-mode flag. ``False`` (off)
        makes every generated tool refuse ``/home/ubuntu`` path arguments with a
        structured disabled outcome (no upload / no passthrough). ``True`` (default,
        always / on_demand) is byte-identical to the pre-Task-24 behavior. BOTH
        assembly chains (runner ``_build_lc_tools_full`` + flow ``_collect_mcp_tools``)
        thread this flag.
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
                    sandbox_files_enabled=sandbox_files_enabled,
                ),
                name=name,
                description=description,
                args_schema=args_schema,
                response_format="content_and_artifact",
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

    # B9 归因注册（spec §5.1）：消费 tool_server_bindings()（Task 11），**不解析名字反推**。
    # fail-open——归因失败仅告警，绝不影响工具创建。
    try:
        bindings = mcp_tool.tool_server_bindings()
        for tool in tools:
            server_name = bindings.get(tool.name)
            if server_name:
                register_extension_tool(tool.name, "mcp", server_name)
    except Exception:
        logger.warning("MCP 归因注册失败（fail-open）", exc_info=True)

    return tools
