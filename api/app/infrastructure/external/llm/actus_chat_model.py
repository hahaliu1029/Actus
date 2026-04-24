"""ActusChatModel — BaseChatModel wrapping OpenAI Chat Completions API.

Direct LangChain BaseChatModel implementation that calls the AsyncOpenAI
Chat Completions endpoint. Replaces the old LLM Protocol + LLMAdapter
indirection with a single unified class.

Implements:
- _generate: raises NotImplementedError (project is async-only)
- _agenerate: calls AsyncOpenAI chat.completions.create, returns ChatResult
- _astream: calls AsyncOpenAI with stream=True, yields ChatGenerationChunk
- bind_tools: returns new instance with tools bound via convert_to_openai_tool
"""

from __future__ import annotations

import json
import logging
import re
import uuid
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
import httpx
import openai
from openai import AsyncOpenAI

from pydantic import Field

from app.application.errors.exceptions import ServerRequestsError
from app.infrastructure.external.llm._timeout_helpers import (
    TRANSIENT_OPENAI_EXCEPTIONS,
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


class ActusChatModel(BaseChatModel):
    """BaseChatModel that wraps the OpenAI Chat Completions API directly.

    Configuration fields (Pydantic):
        base_url: OpenAI-compatible API base URL.
        api_key: API key for authentication.
        model_name: Model identifier (e.g. "gpt-4o", "deepseek-chat").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        supports_response_format: Whether the model supports response_format param.
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
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 8192
    supports_response_format: bool = True
    supports_vision: bool = True
    supports_pdf_input: bool = False
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
    # A7 P0.1: profile injection with default_factory=generic_openai so
    # existing test fixtures that construct ActusChatModel(...) without
    # profile still work. _build_llm always passes an explicit profile.
    # Typed Any to override LangChain BaseChatModel.profile (ModelProfile | None).
    profile: Any = Field(default_factory=_default_generic_profile)

    # Tools bound via bind_tools() — None means no tools bound
    _bound_tools: Optional[list[dict[str, Any]]] = None
    # Cached tool names from _bound_tools (computed once in bind_tools)
    _bound_tool_names: frozenset[str] = frozenset()
    # tool_choice bound via bind_tools() — critical for with_structured_output
    _bound_tool_choice: Optional[Any] = None
    # B5 C11: telemetry port attached via attach_telemetry(). None means
    # the _agenerate hook no-ops. Never serialized.
    _telemetry: Optional[Any] = None

    # ---- Properties ------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        return "actus-chat"

    # ---- B5 C11: telemetry hook ----------------------------------------- #

    def attach_telemetry(self, telemetry: Any, lang: str = "zh") -> None:
        """Attach a ``PromptTelemetryPort`` for LLM-invocation logging.

        Passing ``None`` detaches. Non-blocking — telemetry failures
        never propagate to the main call path. Call once at session
        setup time (typically from ``AgentTaskRunner.__init__``).

        ``lang`` is recorded on every subsequent invocation telemetry
        event. Defaults to ``"zh"`` matching the pre-audit hardcode.
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

    # ---- Message conversion (private) ------------------------------------ #

    def _to_openai_messages(self, messages: List[BaseMessage]) -> list[dict]:
        """Convert LangChain BaseMessage list to OpenAI Chat dict format."""
        result: list[dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                # content may be str (text) or list[dict] (multimodal with image blocks).
                # OpenAI Chat Completions API accepts both formats natively.
                if isinstance(msg.content, list):
                    from app.infrastructure.external.llm.message_sanitizer import (
                        sanitize_multimodal_blocks,
                    )
                    content = sanitize_multimodal_blocks(
                        msg.content,
                        supports_vision=self.supports_vision,
                        supports_pdf_input=self.supports_pdf_input,
                    )
                    block_types = [b.get("type", "?") for b in content if isinstance(b, dict)]
                    image_count = sum(1 for t in block_types if t == "image_url")
                    logger.info(
                        "[MULTIMODAL] HumanMessage has %d content blocks (%d images), block_types=%s",
                        len(content), image_count, block_types,
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
                # A7 P0.1: inject provider-specific reasoning key into wire entry.
                # No-op when profile.supports_thinking=False (generic_openai default).
                # Guard for ``__new__``-constructed test instances that skip
                # Pydantic init (see test_actus_chat_model_error_prefix fixture).
                _profile = getattr(self, "profile", None)
                if _profile is not None:
                    from app.domain.services.provider_profiles._wire import (
                        inject_reasoning_into_wire_entry,
                    )
                    inject_reasoning_into_wire_entry(
                        entry,
                        msg.additional_kwargs or {},
                        _profile,
                        is_chat_completions_api=True,
                    )
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                content = msg.content or ""
                # R2 CS2.14: inject error prefix from artifact when
                # status == "error". The OpenAI tool-message schema has no
                # status field, so the R2 success/error signal would be
                # invisible to the LLM without this encoding step. See
                # ``_error_prefix.py`` for the full rationale. Import is
                # local (function body) to keep this module's import
                # graph flat and avoid any domain/infrastructure circular
                # risk if _error_prefix grows.
                #
                # Fallback: if status=="error" but artifact is missing /
                # malformed, fall back to the legacy generic "[TOOL_ERROR]"
                # marker so the LLM still sees _some_ error signal (matches
                # R1 behavior). Silently dropping the prefix would let a
                # producer bug in Layer 3 leak error outcomes to the model
                # as plain success content.
                #
                # List ``content`` (multimodal blocks) is only produced by
                # Passthrough (status=="success"), so the error branch
                # won't hit it in practice. Keep the ``isinstance(str)``
                # guard defensive — stringifying a block list would both
                # break the multimodal wire format and drop the prefix.
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
                    # Conditional separator avoids a trailing space when
                    # ``content`` is empty: prefer "[TOOL_FAILED: x]" over
                    # "[TOOL_FAILED: x] ".
                    content = f"{prefix} {content}" if content else prefix
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": content,
                })
            else:
                # Fallback for unknown message types
                result.append({"role": "user", "content": str(msg.content)})
        return result

    # ---- Response conversion --------------------------------------------- #

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[dict]:
        """Parse OpenAI tool_calls from response into LangChain format."""
        if not raw_tool_calls:
            return []

        tool_calls = []
        for tc in raw_tool_calls:
            # Handle both object and dict formats
            if hasattr(tc, "function"):
                fn = tc.function
                # Guard: some providers return function as a string
                if isinstance(fn, str):
                    fn_name = fn
                    fn_args = "{}"
                else:
                    fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
                    fn_args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
                tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
            elif isinstance(tc, dict):
                fn = tc.get("function", {})
                if isinstance(fn, str):
                    fn_name = fn
                    fn_args = "{}"
                else:
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", "{}")
                tc_id = tc.get("id", "")
            else:
                continue

            # Deserialize JSON arguments
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}

            tool_calls.append({
                "id": tc_id,
                "name": fn_name,
                "args": fn_args,
            })
        return tool_calls

    # ---- Fallback: extract tool calls from content text -------------------- #

    # Skip fallback scanning on excessively long content to avoid
    # performance issues from regex/brace-scanning on large outputs.
    _MAX_CONTENT_TO_SCAN = 32_000

    def _extract_tool_calls_from_content(
        self, content: str,
    ) -> tuple[list[dict], str]:
        """Fallback: extract tool calls embedded in content text.

        Some LLM providers (e.g. MiniMax) return tool calls as XML or JSON
        inside the ``content`` field instead of the structured ``tool_calls``
        response field. This method attempts to detect and parse them.

        Only tool names that match bound tools are accepted (avoids false
        positives).

        Limitations:
        - Nested XML inside parameter values is not supported (e.g.
          ``<command>echo <b>hi</b></command>`` will not parse correctly).
        - Content longer than ``_MAX_CONTENT_TO_SCAN`` chars is skipped.

        Returns ``(tool_calls, cleaned_content)`` where *tool_calls* is in
        LangChain format and *cleaned_content* has the matched text removed.
        """
        if not content or not self._bound_tool_names:
            return [], content
        if len(content) > self._MAX_CONTENT_TO_SCAN:
            return [], content

        valid_names = self._bound_tool_names

        # --- Strategy 1: XML <invoke name="...">...</invoke> --- #
        tool_calls, cleaned = self._try_extract_xml_invoke(content, valid_names)
        if tool_calls:
            return tool_calls, cleaned

        # --- Strategy 2: JSON object with "name" + "arguments" --- #
        tool_calls, cleaned = self._try_extract_json_tool_call(content, valid_names)
        if tool_calls:
            return tool_calls, cleaned

        return [], content

    # -- XML extraction ---------------------------------------------------- #

    # Bounded quantifiers prevent catastrophic backtracking on malformed input.
    _RE_INVOKE = re.compile(
        r'<invoke\s+name="([^"]{1,256})"[^>]{0,256}>([\s\S]{0,16384}?)</invoke>',
        re.DOTALL,
    )
    _RE_XML_PARAM = re.compile(r'<(\w{1,64})>(.{0,4096}?)</\1>', re.DOTALL)
    # Tags inside <invoke> that are control metadata, not tool arguments
    _XML_CONTROL_TAGS = frozenset({"end_turn"})
    # Marker that immediately precedes XML tool calls (e.g. "minimax:tool_call")
    # Anchored to line start via MULTILINE to avoid corrupting normal text.
    _RE_TOOL_CALL_MARKER = re.compile(r'^\w+:tool_call\s*$', re.MULTILINE)

    def _try_extract_xml_invoke(
        self, content: str, valid_names: frozenset[str],
    ) -> tuple[list[dict], str]:
        """Try to extract ``<invoke name="tool">`` blocks from *content*.

        Only removes matched (accepted) spans — unrecognised tool names are
        left intact in the returned content.
        """
        tool_calls: list[dict] = []
        accepted_spans: list[tuple[int, int]] = []

        for match in self._RE_INVOKE.finditer(content):
            tool_name = match.group(1)
            inner = match.group(2)

            if tool_name not in valid_names:
                continue

            # Parse child XML elements as arguments
            args: dict[str, Any] = {}
            for pm in self._RE_XML_PARAM.finditer(inner):
                key = pm.group(1)
                if key in self._XML_CONTROL_TAGS:
                    continue
                value = pm.group(2).strip()
                # Try to interpret as JSON value (number, bool, null, etc.)
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    args[key] = value

            tool_calls.append({
                "id": f"fallback_{uuid.uuid4().hex[:8]}",
                "name": tool_name,
                "args": args,
            })
            accepted_spans.append((match.start(), match.end()))

        if not tool_calls:
            return [], content

        # Remove only accepted spans (reverse order to preserve indices)
        cleaned = content
        for start, end in reversed(accepted_spans):
            cleaned = cleaned[:start] + cleaned[end:]
        # Remove line-anchored tool_call markers (e.g. "minimax:tool_call")
        cleaned = self._RE_TOOL_CALL_MARKER.sub("", cleaned)
        cleaned = cleaned.strip()

        logger.info(
            "[FALLBACK_TOOL_CALL] Extracted %d XML tool call(s) from content "
            "(model=%s): %s",
            len(tool_calls), self.model_name,
            [tc["name"] for tc in tool_calls],
        )
        return tool_calls, cleaned

    # -- JSON extraction --------------------------------------------------- #

    @staticmethod
    def _find_json_objects(text: str) -> list[tuple[int, int, dict]]:
        """Find top-level JSON objects in *text* via brace-depth scanning.

        Returns list of ``(start, end, parsed_dict)`` tuples.
        Uses ``str.find`` to skip non-brace characters efficiently.
        """
        results: list[tuple[int, int, dict]] = []
        i = 0
        length = len(text)
        while i < length:
            next_brace = text.find('{', i)
            if next_brace == -1:
                break
            i = next_brace
            depth = 0
            end = i
            for j in range(i, length):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if depth == 0 and end > i:
                try:
                    obj = json.loads(text[i:end])
                    if isinstance(obj, dict):
                        results.append((i, end, obj))
                except (json.JSONDecodeError, ValueError):
                    pass
                i = end
            else:
                i += 1
        return results

    _RE_CODE_FENCE = re.compile(r'```(?:json)?\s*\n?\s*```', re.MULTILINE)

    def _try_extract_json_tool_call(
        self, content: str, valid_names: frozenset[str],
    ) -> tuple[list[dict], str]:
        """Try to extract JSON tool call objects from *content*.

        Recognises objects like ``{"name": "tool", "arguments": {...}}``.
        """
        tool_calls: list[dict] = []
        spans_to_remove: list[tuple[int, int]] = []

        for start, end, obj in self._find_json_objects(content):
            name = obj.get("name", "")
            arguments = obj.get("arguments")

            if not name or name not in valid_names:
                continue
            if not isinstance(arguments, dict):
                continue

            tool_calls.append({
                "id": f"fallback_{uuid.uuid4().hex[:8]}",
                "name": name,
                "args": arguments,
            })
            spans_to_remove.append((start, end))

        if not tool_calls:
            return [], content

        # Remove matched JSON spans (reverse order to preserve indices)
        cleaned = content
        for start, end in reversed(spans_to_remove):
            cleaned = cleaned[:start] + cleaned[end:]
        # Strip leftover empty code fences
        cleaned = self._RE_CODE_FENCE.sub('', cleaned)
        cleaned = cleaned.strip()

        logger.info(
            "[FALLBACK_TOOL_CALL] Extracted %d JSON tool call(s) from content "
            "(model=%s): %s",
            len(tool_calls), self.model_name,
            [tc["name"] for tc in tool_calls],
        )
        return tool_calls, cleaned

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
        """Call AsyncOpenAI Chat Completions API and return ChatResult.

        A7 P0.1: routes through the 6-step pipeline (tool_choice resolve,
        outbound rewrites, response_format resolve, WARN emit, wire serialize,
        build_sdk_params). With the default ``generic_openai`` profile,
        behavior is equivalent to the pre-A7 adapter.
        """
        from app.application.errors.exceptions import InternalError
        from app.domain.services.provider_profiles._parse import (
            parse_chat_completion_message,
        )
        from app.domain.services.provider_profiles._rewrites import (
            apply_outbound_rewrites,
            build_sdk_params,
            detect_per_call_thinking,
            resolve_response_format,
            resolve_tool_choice,
        )

        client = self._get_client()
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

        # Step 2: 深拷贝 messages + 采样参数 strip + image URL assertion
        try:
            rewritten_messages, rewritten_kwargs, rewrite_warnings = (
                apply_outbound_rewrites(
                    messages, kwargs, profile, is_chat_completions_api=True,
                )
            )
        except InternalError as e:
            logger.error("[A7] rewrite invariant violated: %s", e)
            raise

        # Step 3: response_format shape 归一化
        # Combine profile-level gate with adapter-level supports_response_format
        # (legacy toggle — pre-A7 many tests rely on it). A7 resolve_response_format
        # handles profile support/strip; we additionally respect the adapter
        # toggle so supports_response_format=False still drops the key.
        request_rf = rewritten_kwargs.pop("response_format", None)
        if request_rf is not None and not self.supports_response_format:
            resolved_rf, rf_warning = None, None
        else:
            resolved_rf, rf_warning = resolve_response_format(request_rf, profile)

        # Step 4: WARN emit 唯一出口 (adapter-scoped dedup by code)
        self._emit_warnings(
            [*tc_warnings, *rewrite_warnings,
             *([rf_warning] if rf_warning else [])]
        )

        # Step 5: wire 序列化 (inject_reasoning_into_wire_entry 已在 _to_openai_messages 内)
        openai_messages = self._to_openai_messages(rewritten_messages)

        # Step 6: build_sdk_params
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

        # tools: merge bound + per-call (kwargs-sourced) — params may already
        # contain a ``tools`` key from rewritten_kwargs; merge instead of clobber.
        all_tools = list(self._bound_tools or [])
        extra_tools = params.pop("tools", None)
        if extra_tools:
            all_tools.extend(extra_tools)
        if all_tools:
            params["tools"] = all_tools

        # stop sequences (not handled by the 6-step pipeline)
        if stop:
            params["stop"] = stop

        # B5 C11: emit telemetry (non-blocking — any failure is swallowed)
        from app.infrastructure.external.llm._telemetry_mixin import (
            emit_invocation_telemetry,
        )

        emit_invocation_telemetry(self, messages, all_tools)

        # 统计多模态内容块数量用于调试
        multimodal_count = sum(
            1 for m in openai_messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        )
        logger.info(
            "ActusChatModel._agenerate: model=%s, tools=%d, tool_choice=%s, "
            "multimodal_messages=%d, provider=%s",
            self.model_name, len(all_tools), resolved_tc, multimodal_count,
            profile.provider_id,
        )

        response = await with_llm_timeout(
            self, client.chat.completions.create(**params)
        )

        # Validate response — some OpenAI-compatible proxies may return raw
        # strings (e.g. error text with 200 status).  Raise ServerRequestsError
        # so LangGraph's RetryPolicy can retry the call automatically.
        if not hasattr(response, "choices") or not response.choices:
            raw = str(response)[:200]
            raise ServerRequestsError(
                f"LLM ({self.model_name}) returned unexpected response "
                f"(type={type(response).__name__}): {raw}"
            )

        # Extract message from response
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls = self._parse_tool_calls(message.tool_calls)

        # A7 P0.1: parse reasoning_content out of the provider-specific key.
        # For generic_openai (supports_thinking=False) this returns {}.
        ak = parse_chat_completion_message(message, profile)

        # Fallback: if no structured tool_calls, try extracting from content
        if not tool_calls and content:
            tool_calls, content = self._extract_tool_calls_from_content(content)

        # Validate: entirely empty response (no content, no tool_calls) is
        # almost always a provider-side error (e.g. 404 wrapped in 200).
        # Raise ServerRequestsError so RetryPolicy / fallback can act on it.
        if not content and not tool_calls:
            raise ServerRequestsError(
                f"LLM ({self.model_name}) returned empty response "
                f"(no content, no tool_calls)"
            )

        ai_message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            additional_kwargs=ak,
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    # ---- LangChain interface: _astream (async streaming) ----------------- #

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Call AsyncOpenAI Chat Completions with stream=True, yield ChatGenerationChunk.

        A7 P0.1: routes through the 6-step pipeline (tool_choice resolve,
        outbound rewrites, response_format resolve, WARN emit, wire serialize,
        build_sdk_params) mirroring ``_agenerate``. Reasoning chunks are
        parsed via ``parse_chat_completion_stream_chunk`` and aggregated on
        the emitted AIMessageChunk's additional_kwargs.
        """
        from app.application.errors.exceptions import InternalError
        from app.domain.services.provider_profiles._parse import (
            parse_chat_completion_stream_chunk,
        )
        from app.domain.services.provider_profiles._rewrites import (
            apply_outbound_rewrites,
            build_sdk_params,
            detect_per_call_thinking,
            resolve_response_format,
            resolve_tool_choice,
        )

        client = self._get_client()
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
        try:
            rewritten_messages, rewritten_kwargs, rewrite_warnings = (
                apply_outbound_rewrites(
                    messages, kwargs, profile, is_chat_completions_api=True,
                )
            )
        except InternalError as e:
            logger.error("[A7] rewrite invariant violated: %s", e)
            raise

        # Step 3: response_format shape normalization
        request_rf = rewritten_kwargs.pop("response_format", None)
        if request_rf is not None and not self.supports_response_format:
            resolved_rf, rf_warning = None, None
        else:
            resolved_rf, rf_warning = resolve_response_format(request_rf, profile)

        # Step 4: WARN emit (adapter-scoped dedup by code)
        self._emit_warnings(
            [*tc_warnings, *rewrite_warnings,
             *([rf_warning] if rf_warning else [])]
        )

        # Step 5: wire serialization
        openai_messages = self._to_openai_messages(rewritten_messages)

        # Step 6: build_sdk_params
        params = build_sdk_params(
            rewritten_kwargs,
            profile,
            adapter_defaults={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            base_params={"model": self.model_name, "messages": openai_messages,
                         "stream": True},
            resolved_response_format=resolved_rf,
            resolved_tool_choice=resolved_tc,
        )

        # tools: merge bound + per-call — rewritten_kwargs may already carry
        # a ``tools`` key which build_sdk_params copied onto params; merge
        # instead of clobber so bind_tools + per-call combine correctly.
        all_tools = list(self._bound_tools or [])
        extra_tools = params.pop("tools", None)
        if extra_tools:
            all_tools.extend(extra_tools)
        if all_tools:
            params["tools"] = all_tools

        if stop:
            params["stop"] = stop

        # B5 C11: emit telemetry (non-blocking — any failure is swallowed)
        from app.infrastructure.external.llm._telemetry_mixin import (
            emit_invocation_telemetry,
        )

        emit_invocation_telemetry(self, messages, all_tools)

        multimodal_count = sum(
            1 for m in openai_messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        )
        logger.info(
            "ActusChatModel._astream: model=%s, multimodal_messages=%d, provider=%s",
            self.model_name, multimodal_count, profile.provider_id,
        )

        # D5.1: Bound the "obtain stream object" step with a hard timeout.
        # Because AsyncOpenAI's stream=True call returns an awaitable that
        # resolves to an async iterator, but some mocks return the iterator
        # directly, we encapsulate both branches in an inner async helper.
        # wait_for bounds the helper's coroutine as a whole.
        #
        # Asymmetry to note:
        # - ``await response`` branch (real AsyncOpenAI): wait_for bounds the
        #   HTTP connection setup network I/O — this is the actual protection.
        # - ``__aiter__`` branch (direct async generator, typically mocks):
        #   _obtain_stream() returns instantly, so wait_for wraps a near-no-op
        #   await. This branch is not usefully bounded by the timeout — it
        #   exists purely to keep mocks working and to keep the code symmetric.
        #
        # Chunk iteration below is intentionally unwrapped because legitimate
        # long streams run for minutes; mid-stream stalls are handled by D5
        # ExecutionWatchdog idle_timeout_seconds at the graph level.
        #
        # NOTE: if a second streaming adapter is added in the future (e.g. a
        # real Responses API streaming path), extract this helper to
        # _timeout_helpers.py as a free function taking the ``client`` and
        # ``params`` captured by the closure.
        async def _obtain_stream():
            response = client.chat.completions.create(**params)
            if hasattr(response, "__aiter__"):
                return response
            return await response

        stream = await with_llm_timeout(self, _obtain_stream())

        # The ``wait_for`` wrap in ``with_llm_timeout`` only bounds the
        # "obtain stream" step; the chunk loop below is deliberately
        # unbounded (long streams can run for minutes). But SDK-level
        # transport failures mid-stream (read timeout, TCP reset, upstream
        # 5xx) still surface here as ``openai.APITimeoutError`` /
        # ``APIConnectionError`` / ``InternalServerError`` / ``RateLimitError``.
        # In addition, the SDK raises the bare ``openai.APIError`` when it
        # receives an SSE ``error`` event from the provider mid-stream
        # (see ``openai/_streaming.py`` lines 75/92/178/195 in SDK 2.14.0)
        # — this class is NOT a subclass of the transient tuple.
        # Translate both so ``RetryPolicy(retry_on=ServerRequestsError)`` at
        # ``react_graph.llm_node`` can actually retry — otherwise they'd
        # leak out past both ``ActusFallbackChatModel`` (not a protocol
        # signal) and the graph-level retry and become terminal.
        #
        # Note: once streaming has started, ``ActusFallbackChatModel``'s
        # cross-protocol escalation is no longer applicable (we've already
        # committed to chat.completions and yielded partial output), so
        # funneling every mid-stream ``APIError`` to the same-endpoint
        # retry path is the correct routing regardless of subclass.
        has_content = False
        try:
            async for chunk in stream:
                # Guard against proxies yielding raw strings or malformed chunks
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Extract content
                content = delta.content or ""

                # Extract tool_call_chunks for streaming aggregation
                tool_call_chunks = []
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        fn = tc.function if hasattr(tc, "function") else None
                        tool_call_chunks.append({
                            "index": tc.index if hasattr(tc, "index") else 0,
                            "id": tc.id if hasattr(tc, "id") and tc.id else None,
                            "name": fn.name if fn and hasattr(fn, "name") and fn.name else None,
                            "args": fn.arguments if fn and hasattr(fn, "arguments") else "",
                        })

                # A7 P0.1: parse reasoning_content out of the provider-specific
                # key on the delta. For generic_openai (supports_thinking=False)
                # this returns {} and the chunk passes through unchanged.
                chunk_ak = parse_chat_completion_stream_chunk(delta, profile)

                if content or tool_call_chunks or chunk_ak:
                    has_content = True

                ai_chunk = AIMessageChunk(
                    content=content,
                    tool_call_chunks=tool_call_chunks if tool_call_chunks else [],
                    additional_kwargs=chunk_ak,
                )
                gen_chunk = ChatGenerationChunk(message=ai_chunk)

                if run_manager:
                    await run_manager.on_llm_new_token(content, chunk=gen_chunk)

                yield gen_chunk
        except TRANSIENT_OPENAI_EXCEPTIONS as exc:
            raise translate_transient(self, exc) from exc
        except openai.APIError as exc:
            # See comment above: bare ``APIError`` (SSE error events) and
            # any other APIError subclass not in TRANSIENT_OPENAI_EXCEPTIONS
            # (APIResponseValidationError, and the protocol/permanent
            # subclasses which almost never fire mid-stream). Funnel them
            # all to ServerRequestsError so the llm_node retry fires.
            raise translate_transient(self, exc) from exc

        # Validate: stream produced zero useful chunks (same 404-in-200 scenario)
        if not has_content:
            raise ServerRequestsError(
                f"LLM ({self.model_name}) stream returned empty response "
                f"(no content, no tool_calls in any chunk)"
            )

    # ---- bind_tools ------------------------------------------------------ #

    def bind_tools(self, tools: list, **kwargs: Any) -> "ActusChatModel":
        """Return a new ActusChatModel with tool schemas bound for LLM calls.

        Uses LangChain's convert_to_openai_tool to normalize tool definitions.

        **Codex audit HIGH #3 fix**: the clone must preserve
        ``provider_name`` (otherwise it defaults back to ``"openai"``
        even if the source was Anthropic) AND ``_telemetry`` (otherwise
        bound models never emit invocation events because LangGraph
        binds tools once per react_graph and all LLM calls go through
        the clone).
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        converted = [convert_to_openai_tool(t) for t in tools]

        # Create a new instance with the same config but tools bound.
        # ``provider_name`` flows through the Pydantic config field so the
        # cloned model correctly identifies itself to telemetry consumers.
        new_model = ActusChatModel(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,  # D5.1: must propagate
            connect_timeout_seconds=self.connect_timeout_seconds,  # D5.2: must propagate
            supports_response_format=self.supports_response_format,
            supports_vision=self.supports_vision,
            supports_pdf_input=self.supports_pdf_input,
            provider_name=self.provider_name,
            profile=self.profile,  # A7 P0.1: propagate profile to clone
        )
        new_model._bound_tools = converted
        new_model._bound_tool_names = frozenset(
            t["function"]["name"]
            for t in converted
            if t.get("function", {}).get("name")
        )
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
