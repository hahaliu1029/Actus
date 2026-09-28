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
- _astream: consumes Responses SSE events; tool calls commit only on completion
- bind_tools: returns new instance with tools bound, tools converted via _convert_tools
"""

from __future__ import annotations

import json
import inspect
import logging
from copy import deepcopy
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
    UsageMetadata,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
import httpx
import openai
from openai import AsyncOpenAI

from pydantic import Field

from app.application.errors.exceptions import ServerRequestsError
from app.infrastructure.external.llm._timeout_helpers import (
    translate_transient,
    with_llm_timeout,
)

logger = logging.getLogger(__name__)


def _default_generic_profile() -> Any:
    """Default ProviderProfile factory — generic_openai.

    Used by Pydantic Field(default_factory=...) when adapter is constructed
    without explicit profile (e.g. existing test fixtures). _build_llm always
    passes an explicit profile, so this default only fires in direct-
    instantiation paths.
    """
    from app.domain.services.provider_profiles import get_profile
    return get_profile("generic_openai")


def _resp_get(obj: Any, name: str) -> Any:
    """Read ``name`` from either a pydantic-style SDK object or a dict.

    ``_agenerate`` accepts both: the OpenAI SDK returns a pydantic model
    (attribute access), while proxies and some test doubles hand back a
    ``dict`` (mapping access). Without this shim the dict path silently
    produced ``cost_status=unknown`` because ``getattr(dict, "usage")`` is
    always None.
    """
    from collections.abc import Mapping

    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_responses_usage_metadata(usage: Any) -> Optional[UsageMetadata]:
    """Translate an OpenAI Responses API ``ResponseUsage`` into LangChain ``UsageMetadata``.

    Responses API surface (differs from Chat Completions):
        input_tokens                                  -> input_tokens
        output_tokens                                 -> output_tokens
        total_tokens (or input+output if absent)      -> total_tokens
        input_tokens_details.cached_tokens            -> input_token_details.cache_read
        output_tokens_details.reasoning_tokens        -> output_token_details.reasoning

    Returns ``None`` when the provider omitted the usage block so B4
    ``CostCallbackHandler`` can flag the row as ``unknown``. Accepts both
    pydantic SDK objects and dicts (proxies / test doubles routinely hand
    back a dict; see ``_agenerate``'s ``isinstance(response, dict)``
    branch).
    """
    if usage is None:
        return None

    # Require at least one authoritative token counter field to be
    # present. An empty ``usage`` payload (``{}`` / ``SimpleNamespace()`` /
    # every counter None) must return ``None`` so
    # ``CostCallbackHandler`` stamps ``cost_status=unknown`` — otherwise
    # a CostRecord would land as "actual $0" and under-report real cost.
    raw_input = _resp_get(usage, "input_tokens")
    raw_output = _resp_get(usage, "output_tokens")
    raw_total = _resp_get(usage, "total_tokens")
    if raw_input is None and raw_output is None and raw_total is None:
        return None

    input_tokens = int(raw_input or 0)
    output_tokens = int(raw_output or 0)
    total_tokens = int(raw_total or (input_tokens + output_tokens))

    result: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    input_details = _resp_get(usage, "input_tokens_details")
    if input_details is not None:
        cached = _resp_get(input_details, "cached_tokens")
        if cached is not None:
            result["input_token_details"] = {"cache_read": int(cached)}

    output_details = _resp_get(usage, "output_tokens_details")
    if output_details is not None:
        reasoning = _resp_get(output_details, "reasoning_tokens")
        if reasoning is not None:
            result["output_token_details"] = {"reasoning": int(reasoning)}

    return result


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
            RetryPolicy. Also propagates to httpx as the read/write/pool
            phase ceiling; 0 means unlimited for those phases.
        connect_timeout_seconds: httpx connect-phase timeout (TCP+TLS
            handshake). Default 60s — an order of magnitude above the OpenAI
            SDK default of 5s, which is too tight for slow cross-border TLS
            and was firing before timeout_seconds could take effect.
    """

    # ---- Pydantic config fields ------------------------------------------ #

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-5.4-pro"
    temperature: float = 0.7
    max_tokens: int = 8192
    supports_vision: bool = True
    supports_pdf_input: bool = False
    # Parity with ActusChatModel: a user-level gate that forces ``response_format``
    # (and its Responses-API rename ``text.format``) to be stripped even if the
    # profile would otherwise support it. Used when the caller knows the
    # concrete deployment does not honor ``response_format`` / ``text.format``.
    supports_response_format: bool = True
    # B5 C0a: provider identification for prompt rendering (system-reminder format etc.)
    # Currently all Actus LLM adapters target OpenAI-compatible endpoints; B5.1 may
    # introduce real Anthropic routing via LLMConfig.provider field.
    provider_name: Literal["openai", "anthropic"] = "openai"
    # D5.1: per-call hard timeout (seconds). 0 disables the wait_for wrap.
    # See docs/superpowers/specs/2026-04-13-per-operation-llm-timeout-design.md
    timeout_seconds: float = 120.0
    # D5.2: httpx connect-phase timeout (TCP+TLS). Separate from
    # timeout_seconds because the SDK default 5s fires before the outer
    # asyncio.wait_for can rescue slow-handshake cases — see CHANGELOG.
    connect_timeout_seconds: float = 60.0
    # A7 P0.1: profile injection with default_factory=generic_openai. In P0.1
    # the Responses adapter's _agenerate / _astream bodies are unchanged
    # (spec §8 rollout — "Responses 走 generic_openai 等价行为"); this field
    # is set for P0.4 future use. Typed Any to override LangChain
    # BaseChatModel.profile (ModelProfile | None).
    profile: Any = Field(default_factory=_default_generic_profile)

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

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Expose ``model`` + ``provider_id`` to LangChain callbacks.

        ``BaseChatModel._get_invocation_params`` merges this dict into the
        payload given to ``on_chat_model_start(**kwargs).invocation_params``.
        B4's ``CostCallbackHandler`` reads ``model`` / ``provider_id`` from
        there; without this override it would see only ``{'_type': 'actus-responses'}``.
        """
        return {
            "model": self.model_name,
            "provider_id": getattr(self.profile, "provider_id", None) or "unknown",
        }

    # ---- B5 C11: telemetry hook ----------------------------------------- #

    def attach_telemetry(self, telemetry: Any, lang: str = "zh") -> None:
        """Attach a ``PromptTelemetryPort`` for LLM-invocation logging.

        Non-blocking — see ``ActusChatModel.attach_telemetry`` docstring.
        """
        from app.infrastructure.external.llm._telemetry_mixin import (
            attach_telemetry,
        )

        attach_telemetry(self, telemetry, lang=lang)

    # ---- A7 P0.1: WARN emit (adapter-scoped dedup) ---------------------- #

    def _emit_warnings(self, warnings: list) -> None:
        """A7 adapter-scoped WARN dedup.

        Dedup by w.code across adapter instance lifetime.
        level='warning' → logger.warning; 'debug' → logger.debug.
        In P0.1 the Responses adapter does not call this; present for P0.4.
        """
        if not warnings:
            return
        seen = self.__dict__.setdefault("_emitted_warning_codes", set())
        for w in warnings:
            if w.code in seen:
                continue
            seen.add(w.code)
            if w.level == "debug":
                logger.debug("[A7] %s", w.message or w.code)
            else:
                logger.warning("[A7] %s", w.message or w.code)

    # ---- Client factory -------------------------------------------------- #

    def _get_client(self) -> AsyncOpenAI:
        """Create AsyncOpenAI client. Extracted as method for testability.

        D5.1: ``max_retries=0`` disables SDK-level retry so the LangGraph
        ``RetryPolicy(max_attempts=3)`` at ``react_graph.llm_node`` and
        ``main_graph.planner_node`` is the single retry authority.

        D5.2: Pass an explicit ``httpx.Timeout`` so the connect phase has a
        dedicated budget. Without this, the SDK default connect=5s fires on
        slow TLS handshakes before the outer ``asyncio.wait_for`` window is
        reached, and bumping ``timeout_seconds`` has no effect on that class
        of failure. ``timeout_seconds==0`` (escape hatch) sets read/write/pool
        to unlimited but still bounds connect so a dead endpoint fails fast.
        """
        default_timeout: float | None = (
            self.timeout_seconds if self.timeout_seconds > 0 else None
        )
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0,
            timeout=httpx.Timeout(
                default_timeout,
                connect=self.connect_timeout_seconds,
            ),
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
                output_items = msg.additional_kwargs.get("responses_output_items")
                if output_items:
                    # Keep native reasoning items (including encrypted_content) and
                    # their ordering. Chat adapters ignore this private carrier.
                    entry["responses_output_items"] = deepcopy(output_items)
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                content = msg.content or ""
                # R2 CS2.14: inject error prefix from artifact when
                # status == "error". Mirrors ActusChatModel Task 33 exactly
                # — see that module for the full rationale (generic
                # "[TOOL_ERROR]" fallback on missing artifact, conditional
                # separator to avoid trailing space, ``isinstance(str)``
                # guard for list content).
                if getattr(msg, "status", None) == "error" and isinstance(
                    content, str
                ):
                    from app.infrastructure.external.llm._error_prefix import (
                        _format_error_prefix,
                    )

                    prefix = (
                        _format_error_prefix(getattr(msg, "artifact", None))
                        or "[TOOL_ERROR]"
                    )
                    content = f"{prefix} {content}" if content else prefix
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": content,
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

            native_items = message.get("responses_output_items")
            if role == "assistant" and native_items:
                normalized = ActusResponsesModel._normalize_response({"output": native_items})
                if (
                    (normalized.get("content") or "") == (message.get("content") or "")
                    and ActusResponsesModel._parse_tool_calls(normalized.get("tool_calls"))
                    == ActusResponsesModel._parse_tool_calls(message.get("tool_calls"))
                ):
                    # Replay exactly once. Do not append reconstructed tool calls
                    # as well, which would duplicate call_ids in the next request.
                    converted.extend(deepcopy(native_items))
                    continue
                # Context management may edit/truncate an AIMessage. Respect its
                # current text/tools instead of restoring stale native output.
                converted.extend(
                    deepcopy(item) for item in native_items
                    if item.get("type") == "reasoning"
                )

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

            converted.append({
                key: value for key, value in message.items()
                if key != "responses_output_items"
            })

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
        if not isinstance(output_items, list):
            raise ServerRequestsError("Responses API returned invalid output items")

        content_text = ""
        refusals: list[str] = []
        tool_calls: List[dict[str, Any]] = []

        for item in output_items:
            if not isinstance(item, dict):
                raise ServerRequestsError("Responses API returned invalid output item")
            item_type = item.get("type")

            if item_type == "message":
                parts = item.get("content")
                if not isinstance(parts, list):
                    raise ServerRequestsError("Responses API returned invalid message content")
                for part in parts:
                    if not isinstance(part, dict):
                        raise ServerRequestsError("Responses API returned invalid message part")
                    if part.get("type") == "output_text":
                        text = part.get("text")
                        if not isinstance(text, str):
                            raise ServerRequestsError("Responses API returned invalid output text")
                        content_text += text
                    elif part.get("type") == "refusal":
                        refusal = part.get("refusal")
                        if not isinstance(refusal, str):
                            raise ServerRequestsError("Responses API returned invalid refusal")
                        content_text += refusal
                        refusals.append(refusal)

            elif item_type == "function_call":
                tool_calls.append({
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments"),
                    },
                })

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content_text or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if refusals:
            message["refusal"] = "".join(refusals)

        return message

    # ------------------------------------------------------------------
    # Parse tool calls from normalized response into LangChain format
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[dict]:
        """Parse tool_calls from normalized response into LangChain format."""
        if not raw_tool_calls:
            return []
        if not isinstance(raw_tool_calls, list):
            raise ServerRequestsError("Responses API returned invalid tool calls")

        tool_calls = []
        seen_ids: set[str] = set()
        for tc in raw_tool_calls:
            if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
                raise ServerRequestsError("Responses API returned an invalid tool call")
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments")

            if isinstance(fn_args, str):
                try:
                    def reject_constant(value: str) -> None:
                        raise ValueError(f"non-JSON constant: {value}")

                    fn_args = json.loads(fn_args, parse_constant=reject_constant)
                except (ValueError, TypeError) as exc:
                    raise ServerRequestsError(
                        "Responses API returned invalid JSON tool arguments"
                    ) from exc
            if not isinstance(fn_args, dict):
                raise ServerRequestsError(
                    "Responses API tool arguments must be a JSON object"
                )
            try:
                json.dumps(fn_args, allow_nan=False)
            except (ValueError, TypeError) as exc:
                raise ServerRequestsError(
                    "Responses API returned invalid JSON tool arguments"
                ) from exc
            call_id = tc.get("id")
            if (
                not isinstance(fn_name, str) or not fn_name.strip()
                or not isinstance(call_id, str) or not call_id.strip()
            ):
                raise ServerRequestsError("Responses API returned an invalid tool call")
            if call_id in seen_ids:
                raise ServerRequestsError("Responses API returned duplicate tool call IDs")
            seen_ids.add(call_id)

            tool_calls.append({
                "id": call_id,
                "name": fn_name,
                "args": fn_args,
            })
        return tool_calls

    def _response_to_message(self, response: Any) -> AIMessage:
        """Validate the provider's terminal state before admitting any tool calls."""
        dumped = response.model_dump() if hasattr(response, "model_dump") else response
        if not isinstance(dumped, dict):
            raise ServerRequestsError(
                f"LLM ({self.model_name}) returned unexpected response "
                f"(type={type(response).__name__})"
            )
        status = dumped.get("status")
        if (
            status != "completed"
            or dumped.get("error") is not None
            or dumped.get("incomplete_details") is not None
        ):
            raise ServerRequestsError(
                f"LLM ({self.model_name}) returned non-completed response "
                f"(status={status or 'unknown'})"
            )
        output = dumped.get("output")
        if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
            raise ServerRequestsError(f"LLM ({self.model_name}) returned invalid output items")
        if any(item.get("status") not in (None, "completed") for item in output):
            raise ServerRequestsError(f"LLM ({self.model_name}) returned incomplete output item")
        normalized = self._normalize_response(dumped)
        content = normalized.get("content") or ""
        tool_calls = self._parse_tool_calls(normalized.get("tool_calls"))
        if not content and not tool_calls:
            raise ServerRequestsError(
                f"LLM ({self.model_name}) returned empty response (no content, no tool_calls)"
            )
        additional_kwargs: dict[str, Any] = {}
        # Only native conversation items belong in a subsequent input. Other
        # built-in tool events are not ordinary assistant messages.
        native_items = [
            deepcopy(item) for item in output
            if item.get("type") in ("reasoning", "message", "function_call")
        ]
        if native_items:
            additional_kwargs["responses_output_items"] = native_items
        if "refusal" in normalized:
            additional_kwargs["refusal"] = normalized["refusal"]
        metadata = {
            key: dumped[key] for key in ("id", "status", "model")
            if dumped.get(key) is not None
        }
        return AIMessage(
            content=content,
            tool_calls=tool_calls,
            additional_kwargs=additional_kwargs,
            response_metadata=metadata,
            usage_metadata=_extract_responses_usage_metadata(_resp_get(response, "usage")),
        )

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

    def _build_request_params(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build one Responses request for both synchronous and SSE responses.

        A7 P0.4: routes through the 6-step pipeline (tool_choice resolve,
        outbound rewrites, response_format resolve, WARN emit, wire serialize,
        build_sdk_params) plus Responses-specific field remap
        (``messages → input``, ``max_tokens → max_output_tokens``,
        ``response_format → text.format``). Q3 boundary: the rewrites pass
        ``is_chat_completions_api=False`` so cross-turn reasoning strip +
        wire reasoning injection are skipped — the Responses API has its
        own reasoning shape (output items) handled by the SDK.
        """
        from app.application.errors.exceptions import InternalError
        from app.domain.services.provider_profiles._base import RewriteWarning
        from app.domain.services.provider_profiles._rewrites import (
            apply_outbound_rewrites,
            build_sdk_params,
            detect_per_call_thinking,
            resolve_response_format,
            resolve_tool_choice,
        )

        profile = self.profile

        # Step 1: tool_choice 归口 (per_call + bound)
        per_call_tc = kwargs.pop("tool_choice", None)
        # A7 P1 fix: detect per-call thinking for opt-in profiles (Anthropic
        # extra_body.thinking / DashScope enable_thinking / Gemini reasoning_effort)
        # — profile.thinking_always_on is False for these, so the forbidden-
        # when-thinking contract was dead without this OR.
        thinking_enabled = profile.thinking_always_on or detect_per_call_thinking(kwargs, profile)
        resolved_tc, tc_warnings = resolve_tool_choice(
            per_call_value=per_call_tc,
            bound_value=self._bound_tool_choice,
            profile=profile,
            thinking_enabled=thinking_enabled,
        )

        # Step 2: deep-copy messages + sampling param strip + image URL assertion
        # Q3 boundary: is_chat_completions_api=False — Responses API has its
        # own reasoning shape (output items), so cross-turn reasoning_content
        # strip is a no-op for this path.
        try:
            rewritten_messages, rewritten_kwargs, rewrite_warnings = (
                apply_outbound_rewrites(
                    messages, kwargs, profile, is_chat_completions_api=False,
                )
            )
        except InternalError as e:
            logger.error("[A7] rewrite invariant violated: %s", e)
            raise

        # Step 3: response_format shape normalization (will be remapped to
        # text.format below, after build_sdk_params).
        request_rf = rewritten_kwargs.pop("response_format", None)
        resolved_rf, rf_warning = resolve_response_format(request_rf, profile)
        # User-level ``supports_response_format=False`` gate: even if the
        # profile would accept the shape, the caller has opted out (e.g. the
        # concrete deployment does not honor text.format). Strip after
        # resolve_response_format so any profile-level warning is still
        # surfaced for diagnostic parity with ActusChatModel.
        if not self.supports_response_format and (
            resolved_rf is not None or request_rf is not None
        ):
            resolved_rf = None
            if rf_warning is None:
                rf_warning = RewriteWarning(
                    code="response_format_disabled_by_config",
                    level="warning",
                    message=(
                        "response_format stripped because "
                        "LLMConfig.supports_response_format=False "
                        "(user-level gate, not profile-level)"
                    ),
                )

        # Step 4: WARN emit 唯一出口 (adapter-scoped dedup by code)
        self._emit_warnings(
            [*tc_warnings, *rewrite_warnings,
             *([rf_warning] if rf_warning else [])]
        )

        # Step 5: wire serialization — convert BaseMessage to OpenAI Chat dicts.
        # Responses-specific ``_convert_input_messages_from_dicts`` runs below
        # on the built ``messages`` param (function_call / function_call_output
        # items, multimodal blocks).
        openai_messages = self._to_openai_messages(rewritten_messages)

        # Step 6: build_sdk_params — produces Chat-shape dict (messages,
        # max_tokens, response_format). Responses-specific field remap runs
        # after this step.
        params = build_sdk_params(
            rewritten_kwargs,
            profile,
            adapter_defaults={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            base_params={"model": self.model_name, "messages": openai_messages},
            resolved_response_format=resolved_rf,
            resolved_tool_choice=resolved_tc,
        )

        # Tools: merge bound + per-call (kwargs-sourced). Per-call tools come
        # in Chat Completions format, so convert via ``_convert_tools`` before
        # merging with already-converted ``_bound_tools``.
        all_tools = list(self._bound_tools or [])
        extra_tools = params.pop("tools", None)
        if extra_tools:
            all_tools.extend(self._convert_tools(extra_tools))
        if all_tools:
            params["tools"] = all_tools

        if stop:
            raise ValueError("Responses API does not support stop sequences")

        # --- Responses-specific field remap (runs AFTER build_sdk_params) --- #
        # messages → input (Responses API uses input items, not chat messages)
        chat_messages = params.pop("messages")
        params["input"] = self._convert_input_messages_from_dicts(chat_messages)
        # max_tokens → max_output_tokens rename.
        # Precedence: an explicit caller-supplied ``max_output_tokens`` wins
        # over the ``max_tokens`` that ``build_sdk_params`` injects from
        # ``adapter_defaults``. Without this the caller's explicit value is
        # silently overwritten by ``self.max_tokens``.
        explicit_max_output = params.pop("max_output_tokens", None)
        fallback_max_tokens = params.pop("max_tokens", None)
        effective_max_output = (
            explicit_max_output
            if explicit_max_output is not None
            else fallback_max_tokens
        )
        if effective_max_output is not None:
            params["max_output_tokens"] = effective_max_output
        # Chat JSON Schema nests name/schema/strict under json_schema;
        # Responses text.format puts them directly beside type.
        if "response_format" in params:
            response_format = params.pop("response_format")
            if response_format.get("type") == "json_schema":
                schema_config = response_format.get("json_schema")
                if isinstance(schema_config, dict):
                    response_format = {**schema_config, "type": "json_schema"}
            params["text"] = {**(params.get("text") or {}), "format": response_format}
        choice = params.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function" and "function" in choice:
            params["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
        elif isinstance(choice, str) and choice not in ("auto", "none", "required"):
            params["tool_choice"] = {"type": "function", "name": choice}
        # Stateless reasoning requires the encrypted item on the next input;
        # preserve explicit include choices while asking for that payload.
        if params.get("store") is False:
            include = list(params.get("include") or [])
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            params["include"] = include

        # B5 C11: emit telemetry (non-blocking — any failure is swallowed).
        from app.infrastructure.external.llm._telemetry_mixin import (
            emit_invocation_telemetry,
        )

        emit_invocation_telemetry(self, messages, all_tools)

        logger.info(
            "ActusResponsesModel request: model=%s, tools=%d, tool_choice=%s, "
            "provider=%s",
            self.model_name, len(all_tools), resolved_tc, profile.provider_id,
        )
        return params

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from app.domain.services.provider_profiles._classify import classify_error

        params = self._build_request_params(messages, stop=stop, **kwargs)
        client = self._get_client()

        try:
            response = await with_llm_timeout(
                self, client.responses.create(**params)
            )
        except Exception as exc:
            err_class = classify_error(exc, self.profile)
            logger.debug(
                "[A7] Responses adapter exception classified: provider=%s class=%s exc=%s",
                self.profile.provider_id, err_class, type(exc).__name__,
            )
            raise

        ai_message = self._response_to_message(response)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ---- LangChain interface: _astream (async streaming) ----------------- #

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream text immediately; admit tools only after response.completed.

        Function argument deltas are deliberately buffered by the provider's
        final response. Passing partial JSON to AIMessageChunk would let its
        permissive parser repair a truncated call into an executable one.
        """
        params = self._build_request_params(messages, stop=stop, **kwargs)
        params["stream"] = True
        client = self._get_client()

        async def obtain_stream():
            pending = client.responses.create(**params)
            return await pending if inspect.isawaitable(pending) else pending

        stream = await with_llm_timeout(self, obtain_stream())
        if not hasattr(stream, "__aiter__"):
            raise ServerRequestsError(f"LLM ({self.model_name}) did not return a Responses stream")
        emitted_text = ""
        completed = False
        try:
            async for event in stream:
                event_type = _resp_get(event, "type")
                if event_type in ("response.failed", "response.incomplete", "response.cancelled"):
                    raise ServerRequestsError(
                        f"LLM ({self.model_name}) stream ended with {event_type}"
                    )
                if event_type == "error":
                    raise ServerRequestsError(
                        f"LLM ({self.model_name}) Responses stream error "
                        f"(code={_resp_get(event, 'code') or 'unknown'})"
                    )
                if event_type in ("response.output_text.delta", "response.refusal.delta"):
                    delta = _resp_get(event, "delta")
                    if not isinstance(delta, str):
                        raise ServerRequestsError("Responses stream returned invalid text delta")
                    emitted_text += delta
                    chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                elif event_type == "response.completed":
                    response = _resp_get(event, "response")
                    msg = self._response_to_message(response)
                    if not msg.content.startswith(emitted_text):
                        raise ServerRequestsError("Responses stream final text does not match its deltas")
                    chunk = ChatGenerationChunk(message=AIMessageChunk(
                        content=msg.content[len(emitted_text):],
                        tool_call_chunks=[
                            {"index": i, "id": tc["id"], "name": tc["name"], "args": json.dumps(tc["args"])}
                            for i, tc in enumerate(msg.tool_calls)
                        ],
                        additional_kwargs=msg.additional_kwargs,
                        response_metadata=msg.response_metadata,
                        usage_metadata=msg.usage_metadata,
                        chunk_position="last",
                    ))
                    completed = True
                else:
                    # Lifecycle / reasoning / tool argument events are retained
                    # losslessly in the completed response's output items.
                    continue
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk
                if completed:
                    break
        except openai.APIError as exc:
            raise translate_transient(self, exc) from exc
        finally:
            close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
            if callable(close):
                try:
                    closing = close()
                    if inspect.isawaitable(closing):
                        await closing
                except Exception:
                    logger.warning("Could not close Responses stream", exc_info=True)
        if not completed:
            raise ServerRequestsError(
                f"LLM ({self.model_name}) stream ended without response.completed"
            )

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
            connect_timeout_seconds=self.connect_timeout_seconds,  # D5.2: must propagate
            supports_vision=self.supports_vision,
            supports_pdf_input=self.supports_pdf_input,
            # User-level response_format gate: must propagate or a bound clone
            # silently re-enables ``text.format`` even though the caller
            # explicitly opted out on the base instance.
            supports_response_format=self.supports_response_format,
            provider_name=self.provider_name,
            profile=self.profile,  # A7 P0.1: propagate profile to clone
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
