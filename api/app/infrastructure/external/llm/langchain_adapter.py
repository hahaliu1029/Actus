"""Adapter: wraps Actus LLM Protocol as LangChain BaseChatModel.

This allows LangGraph nodes to call our existing LLM implementations
(OpenAI, Azure, etc.) through the standard LangChain interface without
replacing any infrastructure code.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from app.domain.external.llm import LLM


class LLMAdapter(BaseChatModel):
    """Wraps an Actus ``LLM`` Protocol object as a LangChain ``BaseChatModel``."""

    _actus_llm: LLM
    _tools: list | None = None

    def __init__(self, llm: LLM, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._actus_llm = llm

    # ---- Properties -------------------------------------------------------- #

    @property
    def _llm_type(self) -> str:
        return "actus-llm-adapter"

    @property
    def model_name(self) -> str:
        return self._actus_llm.model_name

    # ---- Message conversion ------------------------------------------------ #

    def _to_actus_messages(self, messages: List[BaseMessage]) -> list[dict]:
        """Convert LangChain messages → Actus dict format."""
        result: list[dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"])
                                if isinstance(tc["args"], dict)
                                else tc["args"],
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content or "",
                })
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _to_langchain_message(self, response: dict) -> AIMessage:
        """Convert Actus LLM response dict → LangChain AIMessage."""
        content = response.get("content", "") or ""
        tool_calls_raw = response.get("tool_calls") or []

        tool_calls = []
        for tc in tool_calls_raw:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "args": args,
            })

        return AIMessage(content=content, tool_calls=tool_calls)

    # ---- LangChain interface ----------------------------------------------- #

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use async interface (_agenerate)")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        actus_msgs = self._to_actus_messages(messages)

        invoke_kwargs: dict[str, Any] = {"messages": actus_msgs}
        if self._tools:
            invoke_kwargs["tools"] = self._tools
        if stop:
            invoke_kwargs["stop"] = stop
        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            invoke_kwargs["tool_choice"] = tool_choice

        response = await self._actus_llm.invoke(**invoke_kwargs)
        ai_msg = self._to_langchain_message(response)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def bind_tools(self, tools: list, **kwargs: Any) -> "LLMAdapter":
        """Return a copy with tool schemas bound for LLM calls."""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        new = LLMAdapter(llm=self._actus_llm)
        new._tools = [convert_to_openai_tool(t) for t in tools]
        return new
