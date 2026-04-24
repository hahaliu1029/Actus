"""T-P1-BLOCKER-1 integration: opt-in thinking kwargs must trigger
``tool_choice_forbidden_when_thinking`` rewrite at the adapter SDK boundary.

Regression surface: before the fix, adapters passed only
``thinking_enabled=profile.thinking_always_on`` to ``resolve_tool_choice``.
For Anthropic (``thinking_always_on=False`` + opt-in via
``extra_body.thinking``), the ``tool_choice_forbidden_when_thinking={"required"}``
contract was dead on arrival: thinking got turned on at the call site,
but the adapter kept ``tool_choice="required"``, so Anthropic 400 on the
very combination the profile field was meant to protect.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.domain.services.provider_profiles import get_profile
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------- Chat Completions: capture helpers (mirror test_actus_chat_model_profile.py) ----------


def _mock_chat_completion_response(content: str = "answer"):
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


def _capture_chat_create_params():
    captured: list[dict] = []

    async def fake_create(**params):
        captured.append(params)
        return _mock_chat_completion_response()

    return fake_create, captured


# ---------- Responses API: capture helpers (mirror test_actus_responses_model_profile.py) ----------


class _MockResponsesResponse:
    """Plain class exposing a real-dict ``model_dump`` so
    ``_normalize_response`` doesn't choke on a MagicMock.
    """

    def __init__(self, dump: dict):
        self._dump = dump

    def model_dump(self) -> dict:
        return self._dump


def _fake_responses_output(content: str = "answer") -> _MockResponsesResponse:
    return _MockResponsesResponse({
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": content}],
            },
        ],
    })


def _capture_responses_create_params():
    captured: list[dict] = []

    async def fake_create(**params):
        captured.append(params)
        return _fake_responses_output()

    return fake_create, captured


# ========== ActusChatModel × Anthropic ==========


async def test_chat_model_anthropic_thinking_downgrades_tool_choice_required(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P1 fix: per-call ``extra_body.thinking`` triggers the
    ``forbidden_when_thinking`` rewrite — SDK sees ``tool_choice="auto"``
    (Anthropic profile's ``tool_choice_any_alias``), NOT ``"required"``.
    """
    profile = get_profile("anthropic_compat")
    assert profile.thinking_always_on is False
    assert profile.thinking_toggle_style == "extra_body_thinking"
    assert "required" in profile.tool_choice_forbidden_when_thinking

    model = ActusChatModel(
        base_url="https://api.anthropic.com/v1/",
        api_key="k",
        model_name="claude-sonnet-4-6",
        profile=profile,
    )

    fake_create, captured = _capture_chat_create_params()
    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        with caplog.at_level(
            logging.WARNING,
            logger="app.infrastructure.external.llm.actus_chat_model",
        ):
            await model._agenerate(
                [HumanMessage("hi")],
                tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
                tool_choice="required",
                extra_body={"thinking": {"type": "enabled", "budget_tokens": 10_000}},
            )

    assert captured, "SDK create was not invoked"
    assert captured[0]["tool_choice"] == "auto"

    # Verify the forbidden-when-thinking WARN fired by its stable code.
    warn_records = [
        r for r in caplog.records
        if "forbids tool_choice='required' under thinking" in r.message
        and r.levelno == logging.WARNING
    ]
    assert warn_records, (
        "Expected tool_choice_forbidden/required warning to be emitted "
        f"when thinking is opt-in for Anthropic. Captured records: "
        f"{[r.message for r in caplog.records]}"
    )


async def test_chat_model_anthropic_no_thinking_preserves_tool_choice_required(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Control case: without ``extra_body.thinking``, Anthropic profile has
    ``thinking_always_on=False``, so thinking is OFF and the
    ``forbidden_when_thinking`` rule should NOT fire — SDK gets
    ``tool_choice="required"`` untouched.
    """
    profile = get_profile("anthropic_compat")
    model = ActusChatModel(
        base_url="https://api.anthropic.com/v1/",
        api_key="k",
        model_name="claude-sonnet-4-6",
        profile=profile,
    )

    fake_create, captured = _capture_chat_create_params()
    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        with caplog.at_level(
            logging.WARNING,
            logger="app.infrastructure.external.llm.actus_chat_model",
        ):
            await model._agenerate(
                [HumanMessage("hi")],
                tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
                tool_choice="required",
                # NO extra_body.thinking — thinking stays off.
            )

    assert captured[0]["tool_choice"] == "required"

    # No forbidden-when-thinking WARN without thinking active.
    warn_records = [
        r for r in caplog.records
        if "forbids tool_choice='required' under thinking" in r.message
        and r.levelno == logging.WARNING
    ]
    assert warn_records == [], (
        "Did not expect tool_choice_forbidden/required warning without "
        "thinking active. Captured: "
        f"{[r.message for r in caplog.records]}"
    )


# ========== ActusResponsesModel × Anthropic ==========


async def test_responses_model_anthropic_thinking_downgrades_tool_choice_required(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mirror of the Chat-side fix on the Responses API adapter path.

    Same contract: opt-in thinking via ``extra_body.thinking`` on the
    Responses adapter must also trigger the forbidden-when-thinking rewrite.
    """
    profile = get_profile("anthropic_compat")
    model = ActusResponsesModel(
        base_url="https://api.anthropic.com/v1/",
        api_key="k",
        model_name="claude-sonnet-4-6",
        profile=profile,
    )

    fake_create, captured = _capture_responses_create_params()
    fake_client = MagicMock()
    fake_client.responses.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        with caplog.at_level(
            logging.WARNING,
            logger="app.infrastructure.external.llm.actus_responses_model",
        ):
            await model._agenerate(
                [HumanMessage("hi")],
                tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
                tool_choice="required",
                extra_body={"thinking": {"type": "enabled", "budget_tokens": 10_000}},
            )

    assert captured, "Responses SDK create was not invoked"
    assert captured[0]["tool_choice"] == "auto"

    warn_records = [
        r for r in caplog.records
        if "forbids tool_choice='required' under thinking" in r.message
        and r.levelno == logging.WARNING
    ]
    assert warn_records, (
        "Expected tool_choice_forbidden/required warning on the Responses "
        "adapter path when thinking is opt-in for Anthropic. "
        f"Captured records: {[r.message for r in caplog.records]}"
    )


# ========== ActusChatModel._astream × Anthropic ==========


def _stream_minimal_chunks() -> list["SimpleNamespace"]:
    """Minimal streaming chunks: one content delta + terminal stop.

    A single finish_reason='stop' chunk with no content/tool_calls trips the
    _astream empty-response guard, so tests need at least one content chunk
    ahead of the terminator to let the stream complete cleanly.
    """
    content_delta = SimpleNamespace(content="x", role=None, tool_calls=None)
    content_choice = SimpleNamespace(index=0, delta=content_delta, finish_reason=None)
    stop_delta = SimpleNamespace(content=None, role=None, tool_calls=None)
    stop_choice = SimpleNamespace(index=0, delta=stop_delta, finish_reason="stop")
    return [
        SimpleNamespace(choices=[content_choice]),
        SimpleNamespace(choices=[stop_choice]),
    ]


async def test_astream_anthropic_thinking_downgrades_tool_choice_required(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P2 gap close: _astream must also consult per-call thinking.

    Mirror of test_chat_model_anthropic_thinking_downgrades_tool_choice_required
    on the streaming path. Before the P1 fix, _astream hardcoded
    ``thinking_enabled=profile.thinking_always_on`` and SDK still received
    ``tool_choice="required"`` under Anthropic opt-in thinking. After the
    fix, the per-call detector flips thinking_enabled=True and the SDK
    receives ``"auto"``.
    """
    profile = get_profile("anthropic_compat")
    assert profile.thinking_always_on is False
    assert profile.thinking_toggle_style == "extra_body_thinking"
    assert "required" in profile.tool_choice_forbidden_when_thinking

    model = ActusChatModel(
        base_url="https://api.anthropic.com/v1/",
        api_key="k",
        model_name="claude-sonnet-4-6",
        profile=profile,
    )

    captured: list[dict] = []

    async def fake_create(**kwargs):
        captured.append(kwargs)

        async def _gen():
            for chunk in _stream_minimal_chunks():
                yield chunk

        return _gen()

    fake_client = AsyncMock()
    fake_client.chat.completions.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        with caplog.at_level(
            logging.WARNING,
            logger="app.infrastructure.external.llm.actus_chat_model",
        ):
            async for _ in model._astream(
                [HumanMessage("hi")],
                tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
                tool_choice="required",
                extra_body={"thinking": {"type": "enabled", "budget_tokens": 10_000}},
            ):
                pass

    assert captured, "SDK create was not invoked"
    assert captured[0]["tool_choice"] == "auto", (
        f"_astream failed to rewrite tool_choice under Anthropic opt-in thinking. "
        f"Got: {captured[0].get('tool_choice')!r}"
    )

    warn_records = [
        r for r in caplog.records
        if "forbids tool_choice='required' under thinking" in r.message
        and r.levelno == logging.WARNING
    ]
    assert warn_records, (
        "Expected tool_choice_forbidden/required warning on _astream path "
        "when thinking is opt-in for Anthropic. "
        f"Captured records: {[r.message for r in caplog.records]}"
    )


async def test_astream_anthropic_no_thinking_preserves_tool_choice_required(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Streaming control case: without per-call thinking, _astream must
    preserve ``tool_choice="required"`` (thinking_always_on=False for
    Anthropic means no always-on rewrite fires either).
    """
    profile = get_profile("anthropic_compat")
    model = ActusChatModel(
        base_url="https://api.anthropic.com/v1/",
        api_key="k",
        model_name="claude-sonnet-4-6",
        profile=profile,
    )

    captured: list[dict] = []

    async def fake_create(**kwargs):
        captured.append(kwargs)

        async def _gen():
            for chunk in _stream_minimal_chunks():
                yield chunk

        return _gen()

    fake_client = AsyncMock()
    fake_client.chat.completions.create = fake_create

    with patch.object(model, "_get_client", return_value=fake_client):
        with caplog.at_level(
            logging.WARNING,
            logger="app.infrastructure.external.llm.actus_chat_model",
        ):
            async for _ in model._astream(
                [HumanMessage("hi")],
                tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
                tool_choice="required",
                # NO extra_body.thinking — thinking stays off
            ):
                pass

    assert captured[0]["tool_choice"] == "required"

    warn_records = [
        r for r in caplog.records
        if "forbids tool_choice='required' under thinking" in r.message
        and r.levelno == logging.WARNING
    ]
    assert warn_records == [], (
        "Did not expect forbidden-when-thinking warning without thinking on _astream. "
        f"Captured: {[r.message for r in caplog.records]}"
    )
