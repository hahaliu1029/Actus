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

import inspect
import json
import logging
from xml.etree import ElementTree
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
    UsageMetadata,
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


def _extract_usage_metadata(usage: Any) -> Optional[UsageMetadata]:
    """Translate an OpenAI SDK ``CompletionUsage`` into LangChain ``UsageMetadata``.

    Returns ``None`` when the provider omitted the usage block, so callers
    (e.g. B4 ``CostCallbackHandler``) can correctly distinguish actual usage
    from a missing-data case and flag the row as ``estimated`` instead of
    silently persisting zeros.

    Mapping (OpenAI Chat Completions surface):
        prompt_tokens                                    -> input_tokens
        completion_tokens                                -> output_tokens
        total_tokens (or input+output if absent)         -> total_tokens
        prompt_tokens_details.cached_tokens              -> input_token_details.cache_read
        completion_tokens_details.reasoning_tokens       -> output_token_details.reasoning
    """
    if usage is None:
        return None

    # Require at least one authoritative token counter field to be
    # present. An empty ``usage`` object (``SimpleNamespace()`` /
    # ``{}`` / ``usage=None-on-every-field``) must flow through as
    # ``None`` so ``CostCallbackHandler`` stamps ``cost_status=unknown``
    # instead of silently recording an "actual $0" row with zero tokens.
    raw_prompt = getattr(usage, "prompt_tokens", None)
    raw_completion = getattr(usage, "completion_tokens", None)
    raw_total = getattr(usage, "total_tokens", None)
    if raw_prompt is None and raw_completion is None and raw_total is None:
        return None

    input_tokens = int(raw_prompt or 0)
    output_tokens = int(raw_completion or 0)
    total_tokens = int(raw_total or (input_tokens + output_tokens))

    result: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached = getattr(prompt_details, "cached_tokens", None)
        if cached is not None:
            result["input_token_details"] = {"cache_read": int(cached)}

    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details is not None:
        reasoning = getattr(completion_details, "reasoning_tokens", None)
        if reasoning is not None:
            result["output_token_details"] = {"reasoning": int(reasoning)}

    return result


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

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Expose ``model`` + ``provider_id`` to LangChain callbacks.

        ``BaseChatModel._get_invocation_params`` merges this dict into the
        payload handed to ``on_chat_model_start(**kwargs).invocation_params``.
        B4's ``CostCallbackHandler`` reads ``model`` and ``provider_id``
        from there to stamp the CostRecord — without this override it would
        see only ``{'_type': 'actus-chat'}`` and fall back to
        ``model='unknown'`` / heuristic provider inference.
        """
        return {
            "model": self.model_name,
            "provider_id": getattr(self.profile, "provider_id", None) or "unknown",
        }

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
        """Accept only complete, identifiable function calls with JSON objects.

        Never repair partial JSON or substitute empty arguments: the result is
        executable input, so a malformed call invalidates the whole response.
        """
        def field(value: Any, name: str) -> Any:
            return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

        calls: list[dict] = []
        seen: set[str] = set()
        for tc in raw_tool_calls or []:
            fn = field(tc, "function")
            name, raw, call_id = field(fn, "name"), field(fn, "arguments"), field(tc, "id")
            if not isinstance(name, str) or not name.strip() or not isinstance(call_id, str) or not call_id.strip():
                raise ServerRequestsError("LLM returned a tool call without a valid name or id")
            if call_id in seen:
                raise ServerRequestsError("LLM returned duplicate tool call ids")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (ValueError, TypeError) as exc:
                raise ServerRequestsError("LLM returned invalid or incomplete tool arguments") from exc
            if not isinstance(args, dict):
                raise ServerRequestsError("LLM tool arguments must be a JSON object")
            try:
                json.dumps(args, allow_nan=False)
            except (ValueError, TypeError) as exc:
                raise ServerRequestsError("LLM tool arguments contain invalid JSON values") from exc
            seen.add(call_id)
            calls.append({"id": call_id, "name": name, "args": args})
        return calls

    @staticmethod
    def _validate_finish_reason(reason: Any) -> str:
        if reason not in ("stop", "tool_calls"):
            label = reason if isinstance(reason, str) else "missing"
            raise ServerRequestsError(f"LLM response did not finish successfully (finish_reason={label})")
        return reason

    # ---- Explicit provider content-tool compatibility ------------------- #

    _MAX_CONTENT_TO_SCAN = 32_000

    def _extract_tool_calls_from_content(self, content: str) -> tuple[list[dict], str]:
        """Decode a complete MiniMax tool envelope only for opted-in profiles.

        Prose, bare JSON/XML and code examples remain content. A bound tool name
        alone is not evidence that the model intended an executable call.
        """
        if (not self.profile.emits_tool_calls_in_content or not content
                or not self._bound_tool_names or len(content) > self._MAX_CONTENT_TO_SCAN):
            return [], content
        body = content.strip()
        if body.startswith("<minimax:tool_call>") and body.endswith("</minimax:tool_call>"):
            body = body[len("<minimax:tool_call>"):-len("</minimax:tool_call>")]
        elif body.startswith("minimax:tool_call\n"):
            body = body[len("minimax:tool_call\n"):]
        else:
            return [], content
        # Forbid declarations/entities and code fences before XML parsing.
        if "<!" in body or "<?" in body or "```" in body:
            return [], content
        try:
            root = ElementTree.fromstring(f"<calls>{body}</calls>")
        except ElementTree.ParseError:
            return [], content
        if root.text and root.text.strip():
            return [], content
        calls: list[dict] = []
        for invoke in root:
            name = invoke.get("name")
            if (invoke.tag != "invoke" or name not in self._bound_tool_names
                    or set(invoke.attrib) != {"name"}
                    or (invoke.text and invoke.text.strip())
                    or (invoke.tail and invoke.tail.strip())):
                return [], content
            args: dict[str, Any] = {}
            for param in invoke:
                key = param.get("name") if param.tag == "parameter" else param.tag
                if (not key or len(param) or (param.tail and param.tail.strip())
                        or (param.attrib and not (param.tag == "parameter" and set(param.attrib) == {"name"}))
                        or key in args):
                    return [], content
                if key == "end_turn":
                    continue
                value = (param.text or "").strip()
                try:
                    args[key] = json.loads(value)
                except ValueError:
                    args[key] = value
            calls.append({"id": f"fallback_{uuid.uuid4().hex}", "name": name, "args": args})
        try:
            json.dumps(calls, allow_nan=False)
        except (ValueError, TypeError):
            return [], content
        return (calls, "") if calls else ([], content)

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
        finish_reason = self._validate_finish_reason(getattr(choice, "finish_reason", None))
        message = choice.message
        content = message.content or ""
        refusal = getattr(message, "refusal", None)
        refusal = refusal if isinstance(refusal, str) else None
        tool_calls = self._parse_tool_calls(message.tool_calls)

        # A7 P0.1: parse reasoning_content out of the provider-specific key.
        # For generic_openai (supports_thinking=False) this returns {}.
        ak = parse_chat_completion_message(message, profile)
        if refusal:
            ak["refusal"] = refusal
            content = content or refusal

        # Fallback: if no structured tool_calls, try extracting from content
        if not tool_calls and content and not refusal:
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
            response_metadata={"finish_reason": finish_reason},
            usage_metadata=_extract_usage_metadata(getattr(response, "usage", None)),
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

        # B4 M0: opt into upstream usage reporting on streaming responses so
        # CostCallbackHandler can read usage_metadata off the aggregated
        # AIMessageChunk. Setdefault preserves any caller override.
        stream_options = dict(params.get("stream_options") or {})
        stream_options.setdefault("include_usage", True)
        params["stream_options"] = stream_options

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
        has_refusal = False
        finish_reason: str | None = None
        pending_calls: list[dict] = []
        current_calls: dict[int, dict] = {}
        buffered_content: list[str] = []
        # Content-tool compatibility needs the complete envelope before it can
        # distinguish visible text from a call. Other profiles stream normally.
        buffer_content = bool(profile.emits_tool_calls_in_content and self._bound_tool_names)
        try:
            async for chunk in stream:
                # B4 M0: OpenAI with stream_options.include_usage=true sends a
                # trailing chunk where choices=[] and usage is populated. Emit
                # it as a terminal AIMessageChunk carrying usage_metadata so
                # CostCallbackHandler (reading the aggregated message) sees
                # provider-reported counters. Still continue — there is no
                # content/tool delta on that chunk.
                if not hasattr(chunk, "choices") or not chunk.choices:
                    usage_meta = _extract_usage_metadata(
                        getattr(chunk, "usage", None)
                    )
                    if usage_meta is not None:
                        usage_msg = AIMessageChunk(
                            content="", usage_metadata=usage_meta
                        )
                        usage_gen = ChatGenerationChunk(message=usage_msg)
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                "", chunk=usage_gen
                            )
                        yield usage_gen
                    continue

                choice = chunk.choices[0]
                reason = getattr(choice, "finish_reason", None)
                delta = choice.delta

                content = getattr(delta, "content", None) or ""
                refusal = getattr(delta, "refusal", None)
                refusal = refusal if isinstance(refusal, str) else None
                if finish_reason is not None and (content or refusal or getattr(delta, "tool_calls", None)):
                    raise ServerRequestsError("LLM stream returned output after its finish reason")
                if reason is not None:
                    finish_reason = self._validate_finish_reason(reason)
                if refusal:
                    has_refusal = True
                    content = content or refusal
                if buffer_content:
                    buffered_content.append(content)

                # Do not expose partially repaired tool_calls from LangChain's
                # parse_partial_json. Release the whole batch only after the
                # stream has terminated successfully and every call is valid.
                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(tc, "index", None)
                    if not isinstance(idx, int) or idx < 0:
                        raise ServerRequestsError("LLM tool stream omitted a valid call index")
                    call_id = getattr(tc, "id", None)
                    entry = current_calls.get(idx)
                    if entry is None or (call_id and entry["id"] and call_id != entry["id"]):
                        entry = {"index": idx, "id": "", "function": {"name": "", "arguments": ""}}
                        current_calls[idx] = entry
                        pending_calls.append(entry)
                    if call_id:
                        entry["id"] = call_id
                    fn = getattr(tc, "function", None)
                    for key in ("name", "arguments"):
                        part = getattr(fn, key, None)
                        if part is not None:
                            if not isinstance(part, str):
                                raise ServerRequestsError("LLM tool stream contains a non-string function delta")
                            entry["function"][key] += part

                # A7 P0.1: parse reasoning_content out of the provider-specific
                # key on the delta. For generic_openai (supports_thinking=False)
                # this returns {} and the chunk passes through unchanged.
                chunk_ak = parse_chat_completion_stream_chunk(delta, profile)

                if refusal:
                    chunk_ak["refusal"] = refusal
                if content or pending_calls:
                    has_content = True

                # Some providers attach usage to the final delta chunk (rather
                # than a separate no-choices chunk). Pick it up here too.
                chunk_usage_meta = _extract_usage_metadata(
                    getattr(chunk, "usage", None)
                )

                ai_chunk = AIMessageChunk(
                    content="" if buffer_content else content,
                    additional_kwargs=chunk_ak,
                    usage_metadata=chunk_usage_meta,
                )
                gen_chunk = ChatGenerationChunk(message=ai_chunk)

                if run_manager:
                    await run_manager.on_llm_new_token(ai_chunk.content, chunk=gen_chunk)

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
        finally:
            close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

        self._validate_finish_reason(finish_reason)
        ordered_calls = sorted(pending_calls, key=lambda call: call["index"])
        tool_calls = self._parse_tool_calls(ordered_calls)
        final_content = "".join(buffered_content)
        if buffer_content and not tool_calls and not has_refusal:
            tool_calls, final_content = self._extract_tool_calls_from_content(final_content)

        # Validate: stream produced zero useful chunks (same 404-in-200 scenario)
        if not has_content:
            raise ServerRequestsError(
                f"LLM ({self.model_name}) stream returned empty response "
                f"(no content, no tool_calls in any chunk)"
            )

        terminal = ChatGenerationChunk(message=AIMessageChunk(
            content=final_content,
            tool_call_chunks=[{
                "index": i, "id": call["id"], "name": call["name"],
                "args": json.dumps(call["args"], ensure_ascii=False),
            } for i, call in enumerate(tool_calls)],
            response_metadata={"finish_reason": finish_reason},
        ))
        if run_manager:
            await run_manager.on_llm_new_token(final_content, chunk=terminal)
        yield terminal

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
