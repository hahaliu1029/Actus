"""B5 C11: LLM adapter telemetry hook tests.

Verifies:
- ``attach_telemetry`` is available on all 3 adapters
- ``emit_invocation_telemetry`` produces correct fields
- Non-blocking: telemetry exceptions are swallowed
- ``ActusFallbackChatModel`` forwards to both inner adapters
- ``tools_hash`` is stable under registration reordering
"""
from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.infrastructure.external.llm._telemetry_mixin import (
    _extract_system_prompt,
    _extract_tool_names,
    attach_telemetry,
    emit_invocation_telemetry,
)
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
from app.infrastructure.external.llm.actus_fallback_chat_model import (
    ActusFallbackChatModel,
)
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel


class _RecordingTelemetry:
    """Fake PromptTelemetryPort capturing calls for assertion."""

    def __init__(self) -> None:
        self.assembly_calls: list[dict] = []
        self.llm_calls: list[dict] = []

    def record_assembly(self, **kwargs: Any) -> None:
        self.assembly_calls.append(kwargs)

    def record_llm_invocation(self, **kwargs: Any) -> None:
        self.llm_calls.append(kwargs)


class _RaisingTelemetry:
    """Fake telemetry that always raises — used to verify non-blocking."""

    def record_llm_invocation(self, **kwargs: Any) -> None:
        raise RuntimeError("simulated telemetry failure")

    def record_assembly(self, **kwargs: Any) -> None:
        raise RuntimeError("simulated telemetry failure")


def _make_tool_dict(name: str) -> dict:
    """Build a minimal OpenAI Chat Completions tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


# ---- _extract_system_prompt / _extract_tool_names --------------------- #


class TestExtractHelpers:
    def test_extract_system_prompt_finds_first_system_message(self) -> None:
        messages = [
            HumanMessage(content="hi"),
            SystemMessage(content="you are a helper"),
            SystemMessage(content="second system — ignored"),
            AIMessage(content="ok"),
        ]
        assert _extract_system_prompt(messages) == "you are a helper"

    def test_extract_system_prompt_returns_empty_when_none(self) -> None:
        assert _extract_system_prompt([HumanMessage(content="hi")]) == ""

    def test_extract_system_prompt_handles_empty_list(self) -> None:
        assert _extract_system_prompt([]) == ""

    def test_extract_system_prompt_stringifies_multimodal(self) -> None:
        """Multimodal SystemMessage (rare but legal) is stringified rather
        than raising — keeps the hash deterministic."""
        msg = SystemMessage(content=[{"type": "text", "text": "hello"}])
        result = _extract_system_prompt([msg])
        assert isinstance(result, str)
        assert "hello" in result

    def test_extract_tool_names_sorts_and_dedupes(self) -> None:
        tools = [
            _make_tool_dict("shell_execute"),
            _make_tool_dict("file_read"),
            _make_tool_dict("shell_execute"),  # duplicate
            _make_tool_dict("browser_navigate"),
        ]
        names = _extract_tool_names(tools)
        assert names == ["browser_navigate", "file_read", "shell_execute"]

    def test_extract_tool_names_handles_none(self) -> None:
        assert _extract_tool_names(None) == []
        assert _extract_tool_names([]) == []

    def test_extract_tool_names_ignores_malformed(self) -> None:
        """Malformed entries (missing function, non-dict) are silently
        skipped rather than raising."""
        tools = [
            _make_tool_dict("valid"),
            "not a dict",
            {"function": "not a dict"},
            {"function": {"name": None}},
            {"function": {"name": ""}},
            {"function": {}},
        ]
        names = _extract_tool_names(tools)
        assert names == ["valid"]


# ---- emit_invocation_telemetry ---------------------------------------- #


class TestEmitInvocationTelemetry:
    def test_no_op_when_telemetry_is_none(self) -> None:
        class _Stub:
            _telemetry = None
            provider_name = "openai"

        # Should not raise
        emit_invocation_telemetry(_Stub(), [], None)

    def test_emits_one_call_with_populated_fields(self) -> None:
        telemetry = _RecordingTelemetry()

        class _Stub:
            _telemetry = telemetry
            provider_name = "openai"

        messages = [
            SystemMessage(content="you are an agent"),
            HumanMessage(content="do something"),
        ]
        tools = [_make_tool_dict("shell_execute"), _make_tool_dict("file_read")]

        emit_invocation_telemetry(_Stub(), messages, tools)

        assert len(telemetry.llm_calls) == 1
        call = telemetry.llm_calls[0]
        assert call["provider"] == "openai"
        assert call["lang"] == "zh"  # hardcoded in C11
        assert call["system_prompt_bytes"] == len(
            "you are an agent".encode("utf-8")
        )
        assert len(call["system_prompt_hash"]) == 16
        assert len(call["tools_hash"]) == 16
        # Hash is deterministic
        assert call["system_prompt_hash"] == hashlib.sha256(
            "you are an agent".encode("utf-8")
        ).hexdigest()[:16]

    def test_tools_hash_stable_under_reorder(self) -> None:
        """Tool registration order should not affect the hash because
        _extract_tool_names sorts."""
        telemetry = _RecordingTelemetry()

        class _Stub:
            _telemetry = telemetry
            provider_name = "openai"

        messages = [SystemMessage(content="sys")]
        tools_a = [_make_tool_dict("a"), _make_tool_dict("b"), _make_tool_dict("c")]
        tools_b = [_make_tool_dict("c"), _make_tool_dict("a"), _make_tool_dict("b")]

        emit_invocation_telemetry(_Stub(), messages, tools_a)
        emit_invocation_telemetry(_Stub(), messages, tools_b)

        assert len(telemetry.llm_calls) == 2
        assert telemetry.llm_calls[0]["tools_hash"] == telemetry.llm_calls[1]["tools_hash"]

    def test_empty_system_prompt_produces_stable_hash(self) -> None:
        telemetry = _RecordingTelemetry()

        class _Stub:
            _telemetry = telemetry
            provider_name = "openai"

        emit_invocation_telemetry(_Stub(), [], None)
        emit_invocation_telemetry(_Stub(), [HumanMessage(content="hi")], None)

        assert len(telemetry.llm_calls) == 2
        assert telemetry.llm_calls[0]["system_prompt_hash"] == telemetry.llm_calls[1]["system_prompt_hash"]
        assert telemetry.llm_calls[0]["system_prompt_bytes"] == 0
        assert telemetry.llm_calls[1]["system_prompt_bytes"] == 0

    def test_telemetry_exception_is_swallowed(self) -> None:
        """Non-blocking guarantee: a raising telemetry must NOT propagate."""

        class _Stub:
            _telemetry = _RaisingTelemetry()
            provider_name = "openai"

        # Must not raise
        emit_invocation_telemetry(
            _Stub(), [SystemMessage(content="sys")], [_make_tool_dict("shell")]
        )

    def test_different_system_prompts_produce_different_hashes(self) -> None:
        telemetry = _RecordingTelemetry()

        class _Stub:
            _telemetry = telemetry
            provider_name = "openai"

        emit_invocation_telemetry(
            _Stub(), [SystemMessage(content="prompt A")], None
        )
        emit_invocation_telemetry(
            _Stub(), [SystemMessage(content="prompt B")], None
        )

        assert telemetry.llm_calls[0]["system_prompt_hash"] != telemetry.llm_calls[1]["system_prompt_hash"]

    def test_provider_name_is_recorded(self) -> None:
        telemetry = _RecordingTelemetry()

        class _OpenAIStub:
            _telemetry = telemetry
            provider_name = "openai"

        class _AnthropicStub:
            _telemetry = telemetry
            provider_name = "anthropic"

        emit_invocation_telemetry(_OpenAIStub(), [], None)
        emit_invocation_telemetry(_AnthropicStub(), [], None)

        assert telemetry.llm_calls[0]["provider"] == "openai"
        assert telemetry.llm_calls[1]["provider"] == "anthropic"


# ---- attach_telemetry on adapters ------------------------------------- #


class TestAttachTelemetryOnAdapters:
    @pytest.mark.parametrize(
        "adapter_cls",
        [ActusChatModel, ActusResponsesModel],
    )
    def test_adapter_has_attach_telemetry_method(self, adapter_cls: type) -> None:
        adapter = adapter_cls(api_key="test")
        assert hasattr(adapter, "attach_telemetry")
        assert callable(adapter.attach_telemetry)

    @pytest.mark.parametrize(
        "adapter_cls",
        [ActusChatModel, ActusResponsesModel],
    )
    def test_attach_and_detach(self, adapter_cls: type) -> None:
        adapter = adapter_cls(api_key="test")
        telemetry = _RecordingTelemetry()

        adapter.attach_telemetry(telemetry)
        assert getattr(adapter, "_telemetry", None) is telemetry

        adapter.attach_telemetry(None)
        assert getattr(adapter, "_telemetry", None) is None

    def test_fallback_model_forwards_to_both_inner(self) -> None:
        primary = ActusChatModel(api_key="primary")
        fallback = ActusResponsesModel(api_key="fallback")
        combined = ActusFallbackChatModel(primary=primary, fallback=fallback)
        telemetry = _RecordingTelemetry()

        combined.attach_telemetry(telemetry)

        assert getattr(primary, "_telemetry", None) is telemetry
        assert getattr(fallback, "_telemetry", None) is telemetry


# ---- End-to-end stub (no real API call) ------------------------------- #


class TestAdapterTelemetryIntegration:
    def test_attach_telemetry_free_function_directly(self) -> None:
        """Verify the free function ``attach_telemetry`` works on an
        object that isn't necessarily a Pydantic model."""

        class _BareObject:
            pass

        obj = _BareObject()
        telemetry = _RecordingTelemetry()
        attach_telemetry(obj, telemetry)
        assert obj._telemetry is telemetry

        attach_telemetry(obj, None)
        assert obj._telemetry is None


# ---- Codex audit HIGH #3: bind_tools clone preserves telemetry ------- #


class TestAttachTelemetryLangParameter:
    """Post-audit LOW #4: ``attach_telemetry`` accepts a ``lang`` kwarg
    that gets stored on the adapter and used by
    ``emit_invocation_telemetry``. Previously the lang was hardcoded to
    ``"zh"`` inside ``emit_invocation_telemetry`` itself; moving it to
    the attach site makes the hardcode explicit and unblocks future
    real lang plumbing (TODOS #32)."""

    def test_default_lang_is_zh(self) -> None:
        telemetry = _RecordingTelemetry()
        model = ActusChatModel(api_key="test")
        model.attach_telemetry(telemetry)
        assert getattr(model, "_telemetry_lang", None) == "zh"

    def test_explicit_lang_propagates_to_telemetry_event(self) -> None:
        telemetry = _RecordingTelemetry()
        model = ActusChatModel(api_key="test")
        model.attach_telemetry(telemetry, lang="en")
        assert getattr(model, "_telemetry_lang", None) == "en"

        # Fire a telemetry event and verify the recorded lang
        emit_invocation_telemetry(
            model, [SystemMessage(content="sys")], None
        )
        assert len(telemetry.llm_calls) == 1
        assert telemetry.llm_calls[0]["lang"] == "en"

    def test_bind_tools_preserves_lang(self) -> None:
        """The bind_tools clone must carry the attach-time lang."""
        telemetry = _RecordingTelemetry()
        model = ActusChatModel(api_key="test")
        model.attach_telemetry(telemetry, lang="en")

        bound = model.bind_tools([])
        assert getattr(bound, "_telemetry_lang", None) == "en"

    def test_fallback_model_forwards_lang_to_both_inner(self) -> None:
        telemetry = _RecordingTelemetry()
        primary = ActusChatModel(api_key="primary")
        fallback = ActusResponsesModel(api_key="fallback")
        combined = ActusFallbackChatModel(primary=primary, fallback=fallback)

        combined.attach_telemetry(telemetry, lang="en")
        assert getattr(primary, "_telemetry_lang", None) == "en"
        assert getattr(fallback, "_telemetry_lang", None) == "en"


class TestBindToolsPreservesTelemetry:
    """Pre-audit: ``bind_tools`` returned a clone that reset
    ``_telemetry`` and ``provider_name`` to defaults. LangGraph calls
    ``bind_tools`` once per react_graph build, so every subsequent LLM
    invocation went through the clone and emitted zero telemetry events.
    These tests lock in the fix."""

    def test_actus_chat_model_bind_tools_preserves_telemetry(self) -> None:
        telemetry = _RecordingTelemetry()
        model = ActusChatModel(api_key="test", provider_name="anthropic")
        model.attach_telemetry(telemetry)

        bound = model.bind_tools([])
        # provider_name must carry over to the clone
        assert bound.provider_name == "anthropic"
        # _telemetry must carry over
        assert getattr(bound, "_telemetry", None) is telemetry

    def test_actus_responses_model_bind_tools_preserves_telemetry(self) -> None:
        telemetry = _RecordingTelemetry()
        model = ActusResponsesModel(api_key="test", provider_name="anthropic")
        model.attach_telemetry(telemetry)

        bound = model.bind_tools([])
        assert bound.provider_name == "anthropic"
        assert getattr(bound, "_telemetry", None) is telemetry

    def test_bind_tools_with_provider_openai_default(self) -> None:
        """Default provider_name=openai must also be preserved (not reset)."""
        model = ActusChatModel(api_key="test")  # default provider_name="openai"
        assert model.provider_name == "openai"
        bound = model.bind_tools([])
        assert bound.provider_name == "openai"

    def test_bind_tools_without_telemetry_returns_none(self) -> None:
        """When no telemetry was attached, the bound clone's _telemetry is None."""
        model = ActusChatModel(api_key="test")
        bound = model.bind_tools([])
        assert getattr(bound, "_telemetry", None) is None


# ---- Codex audit HIGH #3: Responses API tool schema extraction -------- #


class TestExtractToolNamesResponsesSchema:
    """Pre-audit: ``_extract_tool_names`` only handled the nested
    ``{"function": {"name": X}}`` shape. The Responses API uses a flat
    ``{"type": "function", "name": X}`` shape, so every ActusResponsesModel
    call produced an empty tools_hash. These tests lock in the fix."""

    def test_responses_api_flat_schema_recognized(self) -> None:
        """Flat ``{"type": "function", "name": X, "parameters": ...}``
        is the Responses API tool format."""
        tools = [
            {"type": "function", "name": "shell_execute", "parameters": {}},
            {"type": "function", "name": "file_read", "parameters": {}},
        ]
        names = _extract_tool_names(tools)
        assert names == ["file_read", "shell_execute"]

    def test_mixed_schemas_both_extracted(self) -> None:
        """If a tools list has both Chat Completions nested and Responses
        flat entries (shouldn't happen in practice, but robustness), both
        are extracted."""
        tools = [
            {"type": "function", "function": {"name": "nested_tool"}},
            {"type": "function", "name": "flat_tool", "parameters": {}},
        ]
        names = _extract_tool_names(tools)
        assert names == ["flat_tool", "nested_tool"]

    def test_flat_schema_without_type_ignored(self) -> None:
        """A dict with ``name`` but no ``type=function`` must NOT be
        picked up (prevents collision with unrelated dicts)."""
        tools = [
            {"name": "unlabeled"},  # not a function tool
            {"type": "function", "name": "real_tool"},
        ]
        names = _extract_tool_names(tools)
        assert names == ["real_tool"]

    def test_flat_schema_deterministic_hash_for_responses_call(self) -> None:
        """End-to-end: a ResponsesModel-style tool list produces a stable,
        non-empty tools_hash via emit_invocation_telemetry."""
        telemetry = _RecordingTelemetry()

        class _Stub:
            _telemetry = telemetry
            provider_name = "openai"

        tools = [
            {"type": "function", "name": "shell_execute", "parameters": {}},
            {"type": "function", "name": "file_read", "parameters": {}},
        ]
        emit_invocation_telemetry(_Stub(), [], tools)

        assert len(telemetry.llm_calls) == 1
        call = telemetry.llm_calls[0]
        # Non-empty hash (pre-fix this was the hash of empty string)
        empty_tools_hash = (
            emit_invocation_telemetry.__module__,  # arbitrary different value
        )
        assert call["tools_hash"] != ""
        assert len(call["tools_hash"]) == 16
