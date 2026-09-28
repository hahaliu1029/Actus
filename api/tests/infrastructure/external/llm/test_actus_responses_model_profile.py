"""A7 P0.4 — ActusResponsesModel real-profile integration (T31/T31b/T32).

Covers:
- T31: Responses adapter → SDK + field remap (messages→input, max_tokens→max_output_tokens,
  response_format→text.format) + tool_choice rewrite + sampling strip for DeepSeek Reasoner.
- T31(e): Kimi ``json_object`` response_format maps into ``text={"format": ...}``.
- T31b: per-call + bound ``tool_choice="required"`` is rewritten to ``"auto"`` under Kimi K2.
- T32: ``_astream`` uses the same request pipeline as ``_agenerate``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.domain.services.provider_profiles import get_profile
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _MockResponseObj:
    """Mimics Pydantic v2 Responses API response: plain class exposing
    ``.model_dump()`` returning a real dict.

    A bare ``MagicMock`` would cause ``_normalize_response`` to receive a
    ``MagicMock`` from ``model_dump()``, which is not a dict, and the adapter
    would raise ``ServerRequestsError('empty response')`` before any
    assertion runs.
    """

    def __init__(self, dump: dict):
        self._dump = dump

    def model_dump(self) -> dict:
        return self._dump


def _fake_responses_output(content: str = "answer") -> _MockResponseObj:
    return _MockResponseObj({
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": content}],
            },
        ],
    })


def _capture_responses_params():
    captured: list[dict] = []

    async def fake_create(**params):
        captured.append(params)
        if params.get("stream"):
            async def events():
                yield {"type": "response.completed", "response": _fake_responses_output()}
            return events()
        return _fake_responses_output()

    return fake_create, captured


# ---------- T31 (part a) DeepSeek Reasoner: rewrites + field remap (text-only) ----------

async def test_responses_adapter_rewrites_and_field_remap_for_deepseek_text_only() -> None:
    """T31: Responses adapter → SDK + field remap on a *text-only* history.

    DeepSeek Reasoner declares ``supports_vision=False``; in production
    ``_build_llm`` would apply the vision ceiling so no image block ever
    reaches this adapter. Using a text-only history here validates the
    DeepSeek-specific rewrites (tool_choice / logprobs / temperature /
    json_schema / max_tokens rename + function_call items) under the real
    production-reachable state. The multimodal input-items shape is locked
    by ``test_responses_adapter_input_items_shape_for_vision_profile`` below,
    which uses a profile that legitimately supports vision.
    """
    p = get_profile("deepseek_reasoner")
    model = ActusResponsesModel(
        base_url="https://api.deepseek.com/",
        api_key="k",
        model_name="deepseek-reasoner",
        profile=p,
    )
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create

    history = [
        HumanMessage("please use the tool"),
        AIMessage(
            content="use tool",
            tool_calls=[{"id": "call_1", "name": "foo", "args": {}}],
        ),
        ToolMessage(tool_call_id="call_1", content="ok"),
    ]

    with patch.object(model, "_get_client", return_value=fake_client):
        await model._agenerate(
            history,
            tool_choice="any",
            logprobs=True,
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_schema", "json_schema": {}},
        )
    params = captured[0]

    # tool_choice normalised to "required" (DeepSeek supports it natively).
    assert params["tool_choice"] == "required"
    # Sampling params stripped by apply_outbound_rewrites for reasoner.
    assert "logprobs" not in params
    assert "temperature" not in params
    # json_schema stripped → no text.format for schema (may be absent entirely).
    assert (
        "text" not in params
        or params.get("text") is None
        or "format" not in params.get("text", {})
    )
    # Responses-specific field remap.
    assert params.get("max_output_tokens") == 4096
    assert "max_tokens" not in params
    assert "input" in params
    assert "messages" not in params

    input_items = params["input"]
    # assistant tool_calls surfaced as function_call items.
    assert any(
        it.get("type") == "function_call" and it.get("call_id") == "call_1"
        for it in input_items
    )
    # ToolMessage → function_call_output with matching call_id + output.
    assert any(
        it.get("type") == "function_call_output"
        and it.get("call_id") == "call_1"
        and it.get("output") == "ok"
        for it in input_items
    )


# ---------- T31 (part b) Vision-capable profile: input_text / input_image ----------

async def test_responses_adapter_input_items_shape_for_vision_profile() -> None:
    """T31: multimodal user message → ``input_text`` / ``input_image`` items.

    Uses ``openai_official`` (``supports_vision=True``, ``accepts_image_url=True``)
    so the assertion reflects the real production-reachable path for Responses
    API + vision. DeepSeek cannot reach this path at runtime because the
    vision ceiling in ``_build_llm`` forces ``supports_vision=False`` for
    DeepSeek profiles regardless of user config.
    """
    p = get_profile("openai_official")
    assert p.supports_vision is True
    model = ActusResponsesModel(
        base_url="https://api.openai.com/v1",
        api_key="k",
        model_name="gpt-4o",
        profile=p,
    )
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create

    history = [
        HumanMessage(content=[
            {"type": "text", "text": "hi"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/pic.png"},
            },
        ]),
    ]

    with patch.object(model, "_get_client", return_value=fake_client):
        await model._agenerate(history)

    params = captured[0]
    assert "input" in params

    input_items = params["input"]
    user_item = next(
        it for it in input_items
        if it.get("role") == "user" and isinstance(it.get("content"), list)
    )
    types = [b.get("type") for b in user_item["content"]]
    assert "input_text" in types
    assert "input_image" in types


# ---------- P1 regression guard: max_output_tokens precedence ----------

async def test_explicit_max_output_tokens_wins_over_default_max_tokens() -> None:
    """Caller-supplied ``max_output_tokens`` must survive the Responses field
    remap — the adapter default ``max_tokens`` must NOT silently overwrite it.
    """
    p = get_profile("openai_official")
    # Adapter default ``max_tokens`` is whatever the class default supplies;
    # the explicit caller value (123) must land on the wire intact.
    model = ActusResponsesModel(
        base_url="https://api.openai.com/v1",
        api_key="k",
        model_name="gpt-4o",
        profile=p,
    )
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        await model._agenerate([HumanMessage("q")], max_output_tokens=123)

    params = captured[0]
    assert params["max_output_tokens"] == 123
    assert "max_tokens" not in params


async def test_default_max_tokens_renamed_when_no_explicit_max_output_tokens() -> None:
    """Regression sibling: without an explicit ``max_output_tokens``, the
    adapter default ``max_tokens`` still renames to ``max_output_tokens`` so
    the Responses SDK receives a valid budget.
    """
    p = get_profile("openai_official")
    model = ActusResponsesModel(
        base_url="https://api.openai.com/v1",
        api_key="k",
        model_name="gpt-4o",
        profile=p,
        max_tokens=4321,
    )
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        await model._agenerate([HumanMessage("q")])

    params = captured[0]
    assert params["max_output_tokens"] == 4321
    assert "max_tokens" not in params


# ---------- P1 supports_response_format gate on Responses path ----------

async def test_supports_response_format_false_strips_text_format_on_responses() -> None:
    """LLMConfig.supports_response_format=False must strip the Responses-side
    ``text.format`` payload, parity with ActusChatModel's response_format gate.
    """
    p = get_profile("openai_official")
    model = ActusResponsesModel(
        base_url="https://api.openai.com/v1",
        api_key="k",
        model_name="gpt-4o",
        profile=p,
        supports_response_format=False,
    )
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        await model._agenerate(
            [HumanMessage("q")],
            response_format={"type": "json_object"},
        )

    params = captured[0]
    # Neither the Chat-shape key nor the Responses-rename key may leak through.
    assert "response_format" not in params
    assert "text" not in params


async def test_supports_response_format_true_keeps_text_format_on_responses() -> None:
    """Regression sibling: when the gate is True (default), text.format IS
    emitted — proves the strip path doesn't over-fire."""
    p = get_profile("openai_official")
    model = ActusResponsesModel(
        base_url="https://api.openai.com/v1",
        api_key="k",
        model_name="gpt-4o",
        profile=p,
        supports_response_format=True,
    )
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        await model._agenerate(
            [HumanMessage("q")],
            response_format={"type": "json_object"},
        )

    params = captured[0]
    assert params.get("text") == {"format": {"type": "json_object"}}


async def test_supports_response_format_gate_propagates_to_bind_tools_clone() -> None:
    """bind_tools() returns a new adapter instance. The
    ``supports_response_format=False`` gate must flow to that clone too —
    otherwise ``with_structured_output`` / any bound path silently
    re-enables ``text.format`` even though the caller opted out on the base.
    """
    p = get_profile("openai_official")
    base = ActusResponsesModel(
        base_url="https://api.openai.com/v1",
        api_key="k",
        model_name="gpt-4o",
        profile=p,
        supports_response_format=False,  # opt-out on base
    )
    bound = base.bind_tools(
        [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    )

    # Clone-level state must reflect the opt-out.
    assert bound.supports_response_format is False, (
        "bind_tools clone lost supports_response_format=False — the user's "
        "explicit opt-out was silently reverted to the default True"
    )

    # Behavioral proof: invoking the clone with a response_format must still
    # strip both ``response_format`` and the Responses-API ``text.format``.
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create

    with patch.object(bound, "_get_client", return_value=fake_client):
        await bound._agenerate(
            [HumanMessage("q")],
            response_format={"type": "json_object"},
        )

    params = captured[0]
    assert "response_format" not in params
    assert "text" not in params


# ---------- T31(e) Kimi json_object → text.format ----------

async def test_responses_adapter_kimi_json_object_maps_to_text_format() -> None:
    """T31(e): Kimi json_object → params['text'] = {'format': {...}}."""
    p = get_profile("kimi_k2")
    model = ActusResponsesModel(
        base_url="https://api.moonshot.ai/v1",
        api_key="k",
        model_name="kimi-k2",
        profile=p,
    )
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create
    with patch.object(model, "_get_client", return_value=fake_client):
        await model._agenerate(
            [HumanMessage("q")],
            response_format={"type": "json_object"},
        )
    params = captured[0]
    assert params["text"] == {"format": {"type": "json_object"}}
    assert "response_format" not in params


# ---------- T31b per-call + bound tool_choice rewrite ----------

async def test_responses_adapter_per_call_and_bound_tool_choice_kimi() -> None:
    """T31b: per-call + bound paths mirror T28b.

    Kimi K2 rewrites ``tool_choice="required"`` to ``"auto"`` at the SDK layer.
    """
    p = get_profile("kimi_k2")
    base = ActusResponsesModel(
        base_url="https://api.moonshot.ai/v1",
        api_key="k",
        model_name="kimi-k2",
        profile=p,
    )

    # Per-call path.
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create
    with patch.object(base, "_get_client", return_value=fake_client):
        await base._agenerate([HumanMessage("q")], tool_choice="required")
    assert captured[0]["tool_choice"] == "auto"

    # Bound path (via bind_tools(tool_choice=...)).
    bound = base.bind_tools(
        [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice="required",
    )
    fake_create_b, captured_b = _capture_responses_params()
    fake_client_b = MagicMock()
    fake_client_b.responses.create = fake_create_b
    with patch.object(bound, "_get_client", return_value=fake_client_b):
        await bound._agenerate([HumanMessage("q2")])
    assert captured_b[0]["tool_choice"] == "auto"


# ---------- T32 _astream wrapper inherits the pipeline ----------

async def test_responses_adapter_astream_inherits_shared_fields() -> None:
    """SSE requests share tool-choice, sampling and field-remap rewrites."""
    p = get_profile("deepseek_reasoner")
    model = ActusResponsesModel(
        base_url="https://api.deepseek.com/",
        api_key="k",
        model_name="deepseek-reasoner",
        profile=p,
    )
    fake_create, captured = _capture_responses_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create
    with patch.object(model, "_get_client", return_value=fake_client):
        chunks = []
        async for c in model._astream(
            [HumanMessage("q")],
            tool_choice="any",
            logprobs=True,
            max_tokens=2048,
        ):
            chunks.append(c)
    params = captured[0]
    assert params["tool_choice"] == "required"
    assert "logprobs" not in params
    assert params.get("max_output_tokens") == 2048
    assert "max_tokens" not in params
    assert "messages" not in params
    assert "input" in params
    assert params["stream"] is True
