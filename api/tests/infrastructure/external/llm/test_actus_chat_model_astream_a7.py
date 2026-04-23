"""Tests for ActusChatModel._astream — A7 6-step pipeline parity (Task 1.8 P1 fix).

These tests lock in the Task 1.8 P1 fix: ActusChatModel._astream must mirror
_agenerate's 6-step pipeline (resolve_tool_choice, apply_outbound_rewrites,
resolve_response_format, emit warnings, wire serialize, build_sdk_params),
including reasoning chunk parsing via parse_chat_completion_stream_chunk.

The pre-A7 _astream code path hardcoded "any" -> "required" and did not
consult the profile for tool_choice alias, forbidden sampling params, or
reasoning_content field name, which meant streaming calls silently dropped
A7 behavior that _agenerate had picked up.

Spec anchors:
- A7 spec section 4.3 apply_outbound_rewrites
- A7 spec section 4.6 resolve_tool_choice + resolve_response_format
- A7 spec section 4.3a parse_chat_completion_stream_chunk
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.domain.services.provider_profiles._base import ProviderProfile
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _kimi_profile(
    *,
    forbidden_sampling_params: frozenset[str] = frozenset(),
    reasoning_content_field_name: str = "reasoning_content",
) -> ProviderProfile:
    """Kimi-style profile: tool_choice_any_alias=auto, forbidden required under thinking."""
    return ProviderProfile(
        provider_id="kimi_k2",
        human_name="Kimi K2",
        default_api_mode="chat_completions",
        api_mode_fallback_enabled=False,
        tool_choice_any_alias="auto",
        tool_choice_forbidden_when_thinking=frozenset({"required"}),
        supports_thinking=True,
        thinking_always_on=True,
        reasoning_content_field_name=reasoning_content_field_name,
        forbidden_sampling_params=forbidden_sampling_params,
    )


def _text_delta_chunks(
    content: str = "",
    *,
    reasoning: str | None = None,
    reasoning_key: str = "reasoning_content",
) -> list[SimpleNamespace]:
    """Build streaming chunks. When *reasoning* is given, the first chunk delta
    exposes a provider-specific reasoning field under *reasoning_key*.
    """
    chunks: list[SimpleNamespace] = []
    if reasoning is not None:
        delta_fields: dict[str, Any] = {
            "content": None,
            "role": None,
            "tool_calls": None,
            reasoning_key: reasoning,
        }
        delta = SimpleNamespace(**delta_fields)
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
        chunks.append(SimpleNamespace(choices=[choice]))

    if content:
        delta = SimpleNamespace(content=content, role=None, tool_calls=None)
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
        chunks.append(SimpleNamespace(choices=[choice]))

    final_delta = SimpleNamespace(content=None, role=None, tool_calls=None)
    final_choice = SimpleNamespace(index=0, delta=final_delta, finish_reason="stop")
    chunks.append(SimpleNamespace(choices=[final_choice]))
    return chunks


def _make_model_with_profile(profile: ProviderProfile) -> ActusChatModel:
    return ActusChatModel(
        base_url="https://api.test.com/v1",
        api_key="test-key",
        model_name="test-model",
        temperature=0.5,
        max_tokens=1024,
        profile=profile,
    )


async def _collect_stream(model: ActusChatModel, **kwargs: Any) -> list:
    collected = []
    async for chunk in model._astream([HumanMessage(content="hi")], **kwargs):
        collected.append(chunk)
    return collected


class TestAStreamToolChoiceRewrite:
    """_astream must apply profile-driven tool_choice resolution."""

    async def test_any_rewrites_via_profile_alias_not_hardcoded_required(
        self,
    ) -> None:
        """Kimi profile maps tool_choice=any -> auto (NOT pre-A7 hardcoded 'required')."""
        model = _make_model_with_profile(_kimi_profile())
        captured: dict[str, Any] = {}

        async def fake_create(**kwargs: Any):
            captured.update(kwargs)
            async def _gen():
                for chunk in _text_delta_chunks("hi"):
                    yield chunk
            return _gen()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = fake_create

        with patch.object(model, "_get_client", return_value=mock_client):
            await _collect_stream(model, tool_choice="any")

        # Kimi under thinking_always_on: any -> auto via profile alias; "required"
        # would additionally be forbidden under thinking and rewritten to auto.
        assert captured.get("tool_choice") == "auto"
        assert captured.get("tool_choice") != "required"

    async def test_required_forbidden_under_thinking_rewrites_to_alias(
        self,
    ) -> None:
        """Kimi profile forbids 'required' under thinking -> rewrite to 'auto'."""
        model = _make_model_with_profile(_kimi_profile())
        captured: dict[str, Any] = {}

        async def fake_create(**kwargs: Any):
            captured.update(kwargs)
            async def _gen():
                for chunk in _text_delta_chunks("x"):
                    yield chunk
            return _gen()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = fake_create

        with patch.object(model, "_get_client", return_value=mock_client):
            await _collect_stream(model, tool_choice="required")

        assert captured.get("tool_choice") == "auto"


class TestAStreamForbiddenSamplingParams:
    """_astream must strip forbidden sampling params via apply_outbound_rewrites."""

    async def test_forbidden_param_stripped(self) -> None:
        """profile.forbidden_sampling_params={logprobs} -> logprobs not in SDK kwargs."""
        model = _make_model_with_profile(
            _kimi_profile(forbidden_sampling_params=frozenset({"logprobs"}))
        )
        captured: dict[str, Any] = {}

        async def fake_create(**kwargs: Any):
            captured.update(kwargs)
            async def _gen():
                for chunk in _text_delta_chunks("ok"):
                    yield chunk
            return _gen()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = fake_create

        with patch.object(model, "_get_client", return_value=mock_client):
            await _collect_stream(model, logprobs=True)

        assert "logprobs" not in captured


class TestAStreamReasoningAggregation:
    """_astream must parse reasoning chunks via parse_chat_completion_stream_chunk."""

    async def test_reasoning_content_field_aggregated(self) -> None:
        """Kimi profile reasoning_content_field_name='reasoning_content' -> aggregated."""
        model = _make_model_with_profile(_kimi_profile())

        async def fake_create(**kwargs: Any):
            async def _gen():
                for chunk in _text_delta_chunks(
                    content="answer",
                    reasoning="think step",
                    reasoning_key="reasoning_content",
                ):
                    yield chunk
            return _gen()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = fake_create

        with patch.object(model, "_get_client", return_value=mock_client):
            chunks = await _collect_stream(model)

        # Aggregate additional_kwargs across all yielded chunks. The streaming
        # contract allows the reasoning blob to be split across chunks, or to
        # appear on a dedicated chunk.
        seen_reasoning = ""
        for c in chunks:
            ak = getattr(c.message, "additional_kwargs", {}) or {}
            r = ak.get("reasoning_content")
            if r:
                seen_reasoning += r
        assert "think step" in seen_reasoning

    async def test_k26_style_reasoning_wire_key(self) -> None:
        """K2.6-style profile: reasoning_content_field_name='reasoning' normalizes to internal key."""
        model = _make_model_with_profile(
            _kimi_profile(reasoning_content_field_name="reasoning")
        )

        async def fake_create(**kwargs: Any):
            async def _gen():
                for chunk in _text_delta_chunks(
                    content="answer",
                    reasoning="think",
                    reasoning_key="reasoning",  # wire key used by K2.6
                ):
                    yield chunk
            return _gen()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = fake_create

        with patch.object(model, "_get_client", return_value=mock_client):
            chunks = await _collect_stream(model)

        seen_reasoning = ""
        for c in chunks:
            ak = getattr(c.message, "additional_kwargs", {}) or {}
            r = ak.get("reasoning_content")
            if r:
                seen_reasoning += r
        assert "think" in seen_reasoning
