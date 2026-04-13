"""ActusResponsesModel -- BaseChatModel wrapping OpenAI Responses API.

Direct LangChain BaseChatModel implementation that calls the AsyncOpenAI
Responses endpoint (client.responses.create). Replaces the old LLM Protocol +
OpenAIResponsesLLM indirection with a single unified class.

Key differences from ActusChatModel (Chat Completions):
- Uses ``responses.create()`` instead of ``chat.completions.create()``
- ``max_output_tokens`` instead of ``max_tokens``
- ``input`` param instead of ``messages``
- ``response_format`` -> ``text.format`` mapping
- Function_call items instead of tool_calls in messages
- Needs JSON schema sanitization (array items:{})

Implements:
- _generate: raises NotImplementedError (project is async-only)
- _agenerate: calls Responses API, returns ChatResult with AIMessage
- _astream: fallback to _agenerate (yields single ChatGenerationChunk)
- bind_tools: returns new instance with tools bound, tools converted via _convert_tools
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, List, Literal, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from openai import AsyncOpenAI

from app.application.errors.exceptions import ServerRequestsError
from app.infrastructure.external.llm._timeout_helpers import with_llm_timeout

logger = logging.getLogger(__name__)


class ActusResponsesModel(BaseChatModel):
    """BaseChatModel that wraps the OpenAI Responses API directly.

    Configuration fields (Pydantic):
        base_url: OpenAI-compatible API base URL.
        api_key: API key for authentication.
        model_name: Model identifier (e.g. "gpt-5.4-pro").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate (mapped to max_output_tokens internally).
        timeout_seconds: per-call hard timeout in seconds (D5.1). Default 120.
            0 disables the wrap. Wraps LLM client calls in asyncio.wait_for
            and translates timeouts to ServerRequestsError for LangGraph
            RetryPolicy.
    """

    # ---- Pydantic config fields ------------------------------------------ #

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-5.4-pro"
    temperature: float = 0.7
    max_tokens: int = 8192
    supports_vision: bool = True
    supports_pdf_input: bool = False
    # B5 C0a: provider identification for prompt rendering (system-reminder format etc.)
    # Currently all Actus LLM adapters target OpenAI-compatible endpoints; B5.1 may
    # introduce real Anthropic routing via LLMConfig.provider field.
    provider_name: Literal["openai", "anthropic"] = "openai"
    # D5.1: per-call hard timeout (seconds). 0 disables the wait_for wrap.
    # See docs/superpowers/specs/2026-04-13-per-operation-llm-timeout-design.md
    timeout_seconds: float = 120.0

    # Tools bound via bind_tools() -- None means no tools bound
    _bound_tools: Optional[list] = None
    # tool_choice bound via bind_tools() — critical for with_structured_output
    _bound_tool_choice: Optional[Any] = None
    # B5 C11: telemetry port attached via attach_telemetry(). None means
    # the _agenerate hook no-ops. Never serialized.
    _telemetry: Optional[Any] = None

    # ---- Properties ------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        return "actus-responses"

    # ---- B5 C11: telemetry hook ----------------------------------------- #

    def attach_telemetry(self, telemetry: Any, lang: str = "zh") -> None:
        """Attach a ``PromptTelemetryPort`` for LLM-invocation logging.

        Non-blocking — see ``ActusChatModel.attach_telemetry`` docstring.
        """
        from app.infrastructure.external.llm._telemetry_mixin import (
            attach_telemetry,
        )

        attach_telemetry(self, telemetry, lang=lang)

    # ---- Client factory -------------------------------------------------- #

    def _get_client(self) -> AsyncOpenAI:
        """Create AsyncOpenAI client. Extracted as method for testability.

        D5.1: ``max_retries=0`` disables SDK-level retry so the LangGraph
        ``RetryPolicy(max_attempts=3)`` at ``react_graph.llm_node`` and
        ``main_graph.planner_node`` is the single retry authority.
        """
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0,
        )

    # ------------------------------------------------------------------
    # JSON Schema sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _schema_declares_array(schema_type: Any) -> bool:
        """Check if a schema type declares an array (string or list form)."""
        if schema_type == "array":
            return True
        if isinstance(schema_type, list):
            return "array" in schema_type
        return False

    @classmethod
    def _sanitize_json_schema(cls, schema: Any, path: str = "root") -> Any:
        """Recursively fix incomplete JSON Schema for the stricter Responses API.

        Adds missing ``"items": {}`` for array-type properties.
        """
        if isinstance(schema, list):
            return [cls._sanitize_json_schema(item, f"{path}[]") for item in schema]

        if not isinstance(schema, dict):
            return schema

        sanitized = {
            key: cls._sanitize_json_schema(value, f"{path}.{key}")
            for key, value in schema.items()
        }

        if cls._schema_declares_array(sanitized.get("type")) and "items" not in sanitized:
            logger.warning("检测到数组 schema 缺少 items，已自动补齐: %s", path)
            sanitized["items"] = {}

        return sanitized

    # ------------------------------------------------------------------
    # Tool format conversion: Chat Completions -> Responses API
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Convert Chat Completions tool format to Responses API format.

        Chat Completions: {"type": "function", "function": {"name": ..., "parameters": ...}}
        Responses API:    {"type": "function", "name": ..., "parameters": ..., "strict": False}
        """
        converted: List[dict[str, Any]] = []
        for tool in tools:
            func = tool.get("function", {})
            parameters = ActusResponsesModel._sanitize_json_schema(func.get("parameters", {}))
            converted.append({
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": parameters,
                "strict": False,
            })
        return converted

    # ------------------------------------------------------------------
    # Message conversion: LangChain BaseMessage -> Responses API input
    # ------------------------------------------------------------------

    def _to_openai_messages(self, messages: List[BaseMessage]) -> list[dict]:
        """Convert LangChain BaseMessage list to OpenAI Chat dict format.

        Intermediate step before _convert_input_messages transforms them
        to Responses API input items.
        """
        result: list[dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                if isinstance(msg.content, list):
                    from app.infrastructure.external.llm.message_sanitizer import (
                        sanitize_multimodal_blocks,
                    )
                    content = sanitize_multimodal_blocks(
                        msg.content,
                        supports_vision=self.supports_vision,
                        supports_pdf_input=self.supports_pdf_input,
                    )
                    result.append({"role": "user", "content": content})
                else:
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

    @staticmethod
    def _convert_content_blocks_for_responses(content: list) -> list[dict[str, Any]]:
        """Convert Chat Completions multimodal content blocks to Responses API format.

        Chat Completions: {"type": "text", "text": "..."} / {"type": "image_url", "image_url": {"url": "..."}}
        Responses API:    {"type": "input_text", "text": "..."} / {"type": "input_image", "image_url": "..."}
        """
        converted: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type", "")
            if block_type == "text":
                converted.append({"type": "input_text", "text": block.get("text", "")})
            elif block_type == "image_url":
                image_url = block.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                converted.append({"type": "input_image", "image_url": url})
            elif block_type == "file":
                file_info = block.get("file", {})
                converted.append({
                    "type": "input_file",
                    "filename": file_info.get("filename", "document.pdf"),
                    "file_data": file_info.get("file_data", ""),
                })
            else:
                converted.append(block)
        return converted

    @staticmethod
    def _convert_input_messages_from_dicts(messages: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Convert Chat-style message dicts to Responses API input items.

        Handles:
        - role "tool" -> type "function_call_output"
        - assistant tool_calls -> "function_call" items
        - user messages with multimodal content (list) -> Responses API format
        - Other messages pass through unchanged
        """
        converted: List[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")

            if role == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": message.get("content", ""),
                })
                continue

            if role == "assistant" and message.get("tool_calls"):
                content = message.get("content")
                if content is not None and content != "":
                    converted.append({
                        "role": "assistant",
                        "content": content,
                    })

                for tool_call in message.get("tool_calls", []):
                    function = tool_call.get("function", {})
                    converted.append({
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    })
                continue

            # Convert multimodal content blocks for user messages
            if role == "user" and isinstance(message.get("content"), list):
                converted.append({
                    "role": "user",
                    "content": ActusResponsesModel._convert_content_blocks_for_responses(
                        message["content"]
                    ),
                })
                continue

            converted.append(message)

        return converted

    def _convert_input_messages(self, messages: List[BaseMessage]) -> List[dict[str, Any]]:
        """Convert LangChain BaseMessage list to Responses API input items.

        Two-step process:
        1. Convert BaseMessage -> Chat Completions dict format
        2. Convert Chat dicts -> Responses API input items
        """
        chat_dicts = self._to_openai_messages(messages)
        return self._convert_input_messages_from_dicts(chat_dicts)

    # ------------------------------------------------------------------
    # Response normalization: Responses API output -> message dict
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_response(response: Any) -> dict[str, Any]:
        """Normalize Responses API response to Chat Completions-compatible message dict.

        Responses API output array may contain:
        - type="message": text message (content contains type="output_text" items)
        - type="function_call": tool call
        """
        dumped = response.model_dump() if hasattr(response, "model_dump") else response
        if not isinstance(dumped, dict):
            dumped = {"output": []}
        output_items = dumped.get("output", [])

        content_text = ""
        tool_calls: List[dict[str, Any]] = []

        for item in output_items:
            item_type = item.get("type")

            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        content_text += part.get("text", "")

            elif item_type == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                })

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content_text or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return message

    # ------------------------------------------------------------------
    # Parse tool calls from normalized response into LangChain format
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[dict]:
        """Parse tool_calls from normalized response into LangChain format."""
        if not raw_tool_calls:
            return []

        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments", "{}")

            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}

            tool_calls.append({
                "id": tc.get("id", ""),
                "name": fn_name,
                "args": fn_args,
            })
        return tool_calls

    # ---- LangChain interface: _generate (sync) --------------------------- #

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use async interface (_agenerate). Project is async-only.")

    # ---- LangChain interface: _agenerate (async) ------------------------- #

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call AsyncOpenAI Responses API and return ChatResult."""
        client = self._get_client()
        input_items = self._convert_input_messages(messages)

        # Build request params
        params: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "input": input_items,
        }

        # response_format -> text.format mapping
        response_format = kwargs.get("response_format")
        if response_format is not None:
            params["text"] = {"format": response_format}

        # Merge tools: bound tools + per-call tools from kwargs
        all_tools = list(self._bound_tools or [])
        extra_tools = kwargs.get("tools")
        if extra_tools:
            # Per-call tools come in Chat Completions format, convert them
            all_tools.extend(self._convert_tools(extra_tools))
        if all_tools:
            params["tools"] = all_tools
            logger.info("调用Responses API并携带工具信息: %s", self.model_name)
        else:
            logger.info("调用Responses API未携带工具: %s", self.model_name)

        # B5 C11: emit telemetry (non-blocking — any failure is swallowed).
        # Note: Responses API tool format differs from Chat Completions —
        # _extract_tool_names handles the nested ``function.name`` shape
        # that both formats share.
        from app.infrastructure.external.llm._telemetry_mixin import (
            emit_invocation_telemetry,
        )

        emit_invocation_telemetry(self, messages, all_tools)

        # tool_choice: per-call kwarg > bound value from bind_tools
        # LangChain uses "any" internally (e.g. with_structured_output),
        # but OpenAI API expects "required" for the same semantics.
        tool_choice = kwargs.get("tool_choice") or self._bound_tool_choice
        if tool_choice == "any":
            tool_choice = "required"
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        logger.info("ActusResponsesModel._agenerate: model=%s, tools=%d",
                     self.model_name, len(all_tools))

        response = await with_llm_timeout(
            self, client.responses.create(**params)
        )

        # Validate response — proxies may return strings, ints, or other
        # non-object types instead of a proper Responses API object.
        if not hasattr(response, "model_dump") and not isinstance(response, dict):
            raw = str(response)[:200]
            raise ServerRequestsError(
                f"LLM ({self.model_name}) returned unexpected response "
                f"(type={type(response).__name__}): {raw}"
            )

        # Normalize Responses API output to Chat Completions-compatible dict
        normalized = self._normalize_response(response)
        content = normalized.get("content") or ""
        tool_calls = self._parse_tool_calls(normalized.get("tool_calls"))

        # Validate: entirely empty response is almost always a provider-side
        # error (e.g. 404 wrapped in 200, or empty output array).
        # Raise ServerRequestsError so RetryPolicy / fallback can act on it.
        if not content and not tool_calls:
            raise ServerRequestsError(
                f"LLM ({self.model_name}) returned empty response "
                f"(no content, no tool_calls)"
            )

        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ---- LangChain interface: _astream (async streaming) ----------------- #

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Streaming fallback: calls _agenerate and yields a single chunk.

        The Responses API does not use the same streaming interface as
        Chat Completions. This method provides compatibility by wrapping
        the non-streaming result as a single ChatGenerationChunk.
        """
        result = await self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        msg = result.generations[0].message

        ai_chunk = AIMessageChunk(
            content=msg.content,
            tool_call_chunks=[
                {
                    "index": i,
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"],
                }
                for i, tc in enumerate(msg.tool_calls)
            ] if msg.tool_calls else [],
        )
        gen_chunk = ChatGenerationChunk(message=ai_chunk)

        if run_manager:
            await run_manager.on_llm_new_token(msg.content, chunk=gen_chunk)

        yield gen_chunk

    # ---- bind_tools ------------------------------------------------------ #

    def bind_tools(self, tools: list, **kwargs: Any) -> "ActusResponsesModel":
        """Return a new ActusResponsesModel with tool schemas bound for LLM calls.

        Uses LangChain's convert_to_openai_tool to normalize tool definitions,
        then converts from Chat Completions format to Responses API format.

        **Codex audit HIGH #3 fix**: preserve ``provider_name`` and
        ``_telemetry`` on the clone so bound invocations still record
        correct provider + telemetry events (LangGraph binds once, so
        every subsequent LLM call goes through this clone, not ``self``).
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        # First convert to standard OpenAI Chat Completions format
        chat_format = [convert_to_openai_tool(t) for t in tools]
        # Then convert to Responses API format
        responses_format = self._convert_tools(chat_format)

        # Create a new instance with the same config but tools bound
        new_model = ActusResponsesModel(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,  # D5.1: must propagate
            supports_vision=self.supports_vision,
            supports_pdf_input=self.supports_pdf_input,
            provider_name=self.provider_name,
        )
        new_model._bound_tools = responses_format
        # Preserve tool_choice from kwargs (critical for with_structured_output)
        if "tool_choice" in kwargs:
            new_model._bound_tool_choice = kwargs["tool_choice"]
        # Preserve telemetry port attachment so bound model invocations
        # still emit record_llm_invocation events. ``_telemetry_lang``
        # carries the attach-time language (post-audit LOW #4).
        object.__setattr__(
            new_model, "_telemetry", getattr(self, "_telemetry", None)
        )
        object.__setattr__(
            new_model, "_telemetry_lang", getattr(self, "_telemetry_lang", "zh")
        )
        return new_model
