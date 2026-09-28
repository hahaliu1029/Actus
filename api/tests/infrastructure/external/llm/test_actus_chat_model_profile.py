"""A7 P0.2 — ActusChatModel real Kimi profile integration tests (T25/T26/T27/T28b/T29).

Covers:
- T27: adapter WARN dedup by code
- T28b: per-call + bound tool_choice rewrite to auto under Kimi
- T29: streaming reasoning_content accumulation (K2 + K2.6)
- T25/T26: wire serializer injects reasoning with provider-specific key
"""
from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.domain.services.provider_profiles import get_profile
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _mock_openai_response(content: str = "answer"):
    """Build a fake OpenAI ChatCompletion response object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.model_dump = lambda exclude_none=False: {"content": content}
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _capture_create_params():
    """Return (fake_create_async, captured_list)."""
    captured: list[dict] = []

    async def fake_create(**params):
        captured.append(params)
        return _mock_openai_response()

    return fake_create, captured


# ---------- T27 dedup ----------

async def test_adapter_emit_warnings_dedup_by_code(caplog: pytest.LogCaptureFixture) -> None:
    """T27: same warning code emits once per adapter instance lifetime."""
    p = get_profile("kimi_k2")
    p = replace(p, forbidden_sampling_params=frozenset({"logprobs"}))

    model = ActusChatModel(
        base_url="https://api.moonshot.ai/v1",
        api_key="k", model_name="kimi-k2",
        profile=p,
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=_mock_openai_response())

    with patch.object(model, "_get_client", return_value=fake_client):
        with caplog.at_level(
            logging.WARNING,
            logger="app.infrastructure.external.llm.actus_chat_model",
        ):
            await model._agenerate(
                [HumanMessage("q")],
                tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
                logprobs=True,
            )
            await model._agenerate(
                [HumanMessage("q2")],
                tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
                logprobs=True,
            )

    # _emit_warnings logs "[A7] <warning.message>" where warning.message is
    # "{provider_id} forbids sampling param '{p}'; stripped". Dedup-by-code
    # means the same code fires once per adapter lifetime — verify via the
    # unique "'logprobs'" substring.
    warn_records = [
        r for r in caplog.records
        if "'logprobs'" in r.message and r.levelno == logging.WARNING
    ]
    assert len(warn_records) == 1


# ---------- T28b tool_choice rewrite ----------

async def test_chat_adapter_per_call_required_rewrites_to_auto_for_kimi() -> None:
    """T28b (a): per-call tool_choice='required' + Kimi → SDK receives 'auto'"""
    p = get_profile("kimi_k2")
    model = ActusChatModel(
        base_url="https://api.moonshot.ai/v1", api_key="k",
        model_name="kimi-k2", profile=p,
    )
    fake_create, captured = _capture_create_params()
    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        await model._agenerate(
            [HumanMessage("q")],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice="required",
        )
    assert captured[0]["tool_choice"] == "auto"


async def test_chat_adapter_bound_required_also_rewrites_to_auto_for_kimi() -> None:
    """T28b (b): bind_tools(tool_choice='required') + no per-call → SDK still 'auto'"""
    p = get_profile("kimi_k2")
    base = ActusChatModel(
        base_url="https://api.moonshot.ai/v1", api_key="k",
        model_name="kimi-k2", profile=p,
    )
    bound = base.bind_tools(
        [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice="required",
    )
    fake_create, captured = _capture_create_params()
    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_create

    with patch.object(bound, "_get_client", return_value=fake_client):
        await bound._agenerate([HumanMessage("q")])
    assert captured[0]["tool_choice"] == "auto"


async def test_chat_adapter_per_call_wins_over_bound() -> None:
    """T28b (c): per-call 'auto' + bound 'required' → SDK 'auto' no warning"""
    p = get_profile("kimi_k2")
    base = ActusChatModel(
        base_url="https://api.moonshot.ai/v1", api_key="k",
        model_name="kimi-k2", profile=p,
    )
    bound = base.bind_tools(
        [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice="required",
    )
    fake_create, captured = _capture_create_params()
    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_create

    with patch.object(bound, "_get_client", return_value=fake_client):
        await bound._agenerate([HumanMessage("q")], tool_choice="auto")
    assert captured[0]["tool_choice"] == "auto"


# ---------- T29 streaming reasoning ----------

async def test_chat_adapter_streaming_accumulates_reasoning_content_for_kimi() -> None:
    """T29: Kimi streaming — reasoning_content accumulated into AIMessageChunk"""
    p = get_profile("kimi_k2")
    model = ActusChatModel(
        base_url="https://api.moonshot.ai/v1", api_key="k",
        model_name="kimi-k2", profile=p,
    )

    deltas = [
        SimpleNamespace(content=None, reasoning_content="think", tool_calls=None),
        SimpleNamespace(content=None, reasoning_content="ing...", tool_calls=None),
        SimpleNamespace(content="answer", reasoning_content=None, tool_calls=None),
    ]

    async def fake_stream(**params):
        for i, d in enumerate(deltas):
            chunk = MagicMock()
            choice = MagicMock()
            choice.delta = d
            choice.finish_reason = "stop" if i == len(deltas) - 1 else None
            chunk.choices = [choice]
            yield chunk

    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_stream

    with patch.object(model, "_get_client", return_value=fake_client):
        chunks = []
        async for c in model._astream([HumanMessage("q")]):
            chunks.append(c)

    accumulated = "".join(
        c.message.additional_kwargs.get("reasoning_content", "") for c in chunks
    )
    assert accumulated == "thinking..."
    final_content = "".join(c.message.content or "" for c in chunks)
    assert "answer" in final_content


async def test_chat_adapter_streaming_kimi_k2_6_uses_reasoning_field() -> None:
    """T29 (d): K2.6 delta uses 'reasoning' key → normalized to internal 'reasoning_content'"""
    p = get_profile("kimi_k2_6")
    model = ActusChatModel(
        base_url="https://api.moonshot.ai/v1", api_key="k",
        model_name="kimi-k2.6", profile=p,
    )
    deltas = [SimpleNamespace(content="a", reasoning="think", tool_calls=None)]

    async def fake_stream(**params):
        for i, d in enumerate(deltas):
            chunk = MagicMock()
            choice = MagicMock()
            choice.delta = d
            choice.finish_reason = "stop" if i == len(deltas) - 1 else None
            chunk.choices = [choice]
            yield chunk

    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_stream

    with patch.object(model, "_get_client", return_value=fake_client):
        chunks = []
        async for c in model._astream([HumanMessage("q")]):
            chunks.append(c)

    acc = "".join(c.message.additional_kwargs.get("reasoning_content", "") for c in chunks)
    assert acc == "think"


# ---------- T25/T26 wire serializer ----------

def test_chat_model_serializer_injects_reasoning_for_kimi_k2() -> None:
    """T25: Kimi K2 → entry['reasoning_content']"""
    p = get_profile("kimi_k2")
    model = ActusChatModel(
        base_url="https://api.moonshot.ai/v1", api_key="k",
        model_name="kimi-k2", profile=p,
    )
    msgs = [
        HumanMessage("q"),
        AIMessage(content="a", additional_kwargs={"reasoning_content": "think"}),
    ]
    out = model._to_openai_messages(msgs)
    assert out[-1].get("reasoning_content") == "think"


def test_chat_model_serializer_maps_reasoning_for_kimi_k2_6() -> None:
    """T26: K2.6 → entry['reasoning'], no 'reasoning_content'"""
    p = get_profile("kimi_k2_6")
    model = ActusChatModel(
        base_url="https://api.moonshot.ai/v1", api_key="k",
        model_name="kimi-k2.6", profile=p,
    )
    msgs = [
        HumanMessage("q"),
        AIMessage(content="a", additional_kwargs={"reasoning_content": "think"}),
    ]
    out = model._to_openai_messages(msgs)
    assert out[-1].get("reasoning") == "think"
    assert "reasoning_content" not in out[-1]


# ---------- Task 3.7 DeepSeek SDK rewrites + cross-turn strip ----------


async def test_chat_adapter_outbound_params_reflect_rewrites_for_deepseek() -> None:
    """T28: DeepSeek → SDK params 全维度 strip + 字段规则抵达"""
    p = get_profile("deepseek_reasoner")
    model = ActusChatModel(
        base_url="https://api.deepseek.com/", api_key="k",
        model_name="deepseek-reasoner", profile=p,
    )
    fake_create, captured = _capture_create_params()
    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        await model._agenerate(
            [HumanMessage("q")],
            tool_choice="any",
            logprobs=True,
            top_logprobs=5,
            temperature=0.7,
            response_format={"type": "json_schema", "json_schema": {}},
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        )
    params = captured[0]
    assert params["tool_choice"] == "required"
    assert "logprobs" not in params
    assert "top_logprobs" not in params
    assert "temperature" not in params
    assert "response_format" not in params


async def test_chat_adapter_serializes_rewritten_messages_stripping_deepseek_cross_turn_reasoning() -> None:
    """T30: DeepSeek 跨轮剥离真的抵达 SDK；对比 Kimi 保留。"""
    history = [
        HumanMessage("q1"),
        AIMessage(content="a1", additional_kwargs={"reasoning_content": "think1"}),
        HumanMessage("q2"),
    ]

    p_ds = get_profile("deepseek_reasoner")
    model_ds = ActusChatModel(
        base_url="https://api.deepseek.com/", api_key="k",
        model_name="deepseek-reasoner", profile=p_ds,
    )
    fake_create_ds, captured_ds = _capture_create_params()
    fake_client_ds = MagicMock()
    fake_client_ds.chat.completions.create = fake_create_ds
    with patch.object(model_ds, "_get_client", return_value=fake_client_ds):
        await model_ds._agenerate(history)
    ai_entry_ds = next(m for m in captured_ds[0]["messages"]
                       if m["role"] == "assistant")
    assert "reasoning_content" not in ai_entry_ds

    p_kimi = get_profile("kimi_k2")
    model_kimi = ActusChatModel(
        base_url="https://api.moonshot.ai/v1", api_key="k",
        model_name="kimi-k2", profile=p_kimi,
    )
    fake_create_k, captured_k = _capture_create_params()
    fake_client_k = MagicMock()
    fake_client_k.chat.completions.create = fake_create_k
    with patch.object(model_kimi, "_get_client", return_value=fake_client_k):
        await model_kimi._agenerate(history)
    ai_entry_k = next(m for m in captured_k[0]["messages"]
                      if m["role"] == "assistant")
    assert ai_entry_k.get("reasoning_content") == "think1"
