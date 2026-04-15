"""Wrap existing A2ATool as LangChain StructuredTool instances.

Similar to langchain_mcp.py, we wrap A2ATool's methods into LangChain tools,
preserving the existing A2A client management.
"""

from __future__ import annotations

import json

from langchain_core.tools import StructuredTool, tool as lc_tool

from app.domain.models.tool_result import AllowError, AllowSuccess, DecisionReason, ToolOutcome
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)


def create_a2a_langchain_tools(a2a_tool: A2ATool) -> list[StructuredTool]:
    """Convert A2ATool's methods into LangChain StructuredTool instances.

    Returns empty list if A2ATool has no initialized manager (not yet connected).
    """
    if not getattr(a2a_tool, "manager", None):
        return []

    @lc_tool(response_format="content_and_artifact")
    async def get_remote_agent_cards() -> tuple[str, ToolOutcome]:
        """获取可远程调用的Agent卡片信息, 包含Agent id、名称、描述、技能、请求端点等。"""
        try:
            result = await a2a_tool.get_remote_agent_cards()
        except TimeoutError as exc:
            outcome = AllowError(
                content=f"A2A 获取远程 Agent 超时: {exc}",
                reason=DecisionReason(type="timeout", code="a2a_timeout", message=str(exc)),
                retryable=True,
            )
            return outcome.content, outcome
        except Exception as exc:
            outcome = AllowError(
                content=f"A2A 获取远程 Agent 异常: {exc}",
                reason=DecisionReason(type="exception", code=type(exc).__name__, message=str(exc)),
                retryable=False,
            )
            return outcome.content, outcome

        if hasattr(result, "success") and not result.success:
            message = getattr(result, "message", None) or str(result)
            outcome = AllowError(
                content=message,
                reason=DecisionReason(type="exception", code="a2a_remote_error", message=message),
            )
            return outcome.content, outcome
        if hasattr(result, "data") and result.data is not None:
            content = json.dumps(result.data, ensure_ascii=False)
        elif hasattr(result, "message") and result.message:
            content = result.message
        else:
            content = str(result)
        outcome = AllowSuccess(
            content=content,
            data=result.data if hasattr(result, "data") and isinstance(result.data, dict) else None,
        )
        return outcome.content, outcome

    @lc_tool(response_format="content_and_artifact")
    async def call_remote_agent(id: str, query: str) -> tuple[str, ToolOutcome]:
        """根据传递的id+query(分配给远程Agent完成的任务query)调用远程Agent完成对应需求"""
        try:
            result = await a2a_tool.call_remote_agent(id=id, query=query)
        except TimeoutError as exc:
            outcome = AllowError(
                content=f"A2A 调用远程 Agent 超时: {exc}",
                reason=DecisionReason(type="timeout", code="a2a_timeout", message=str(exc)),
                retryable=True,
            )
            return outcome.content, outcome
        except Exception as exc:
            outcome = AllowError(
                content=f"A2A 调用远程 Agent 异常: {exc}",
                reason=DecisionReason(type="exception", code=type(exc).__name__, message=str(exc)),
            )
            return outcome.content, outcome

        if hasattr(result, "success") and not result.success:
            message = getattr(result, "message", None) or str(result)
            outcome = AllowError(
                content=message,
                reason=DecisionReason(type="exception", code="a2a_remote_error", message=message),
            )
            return outcome.content, outcome
        if hasattr(result, "data") and result.data is not None:
            content = json.dumps(result.data, ensure_ascii=False)
        elif hasattr(result, "message") and result.message:
            content = result.message
        else:
            content = str(result)
        outcome = AllowSuccess(
            content=content,
            data=result.data if hasattr(result, "data") and isinstance(result.data, dict) else None,
        )
        return outcome.content, outcome

    tools = [get_remote_agent_cards, call_remote_agent]
    for t in tools:
        annotate_and_register_tool_source(t, source="a2a", category="a2a")
    return tools
