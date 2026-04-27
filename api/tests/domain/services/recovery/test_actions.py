from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.domain.services.provider_profiles._base import ProviderProfile
from app.domain.services.recovery._base import RecoveryContext


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _profile(thinking_toggle_style: str = "extra_body_enable_thinking") -> ProviderProfile:
    return ProviderProfile(
        provider_id="p",
        human_name="Test",
        default_api_mode="chat_completions",
        api_mode_fallback_enabled=True,
        thinking_toggle_style=thinking_toggle_style,
    )


def _ctx(profile=None, callback=None) -> RecoveryContext:
    return RecoveryContext(
        profile=profile or _profile(),
        api_mode="chat_completions",
        attempt_index=0,
        attempted_action_codes=frozenset(),
        on_context_overflow=callback,
    )


async def test_strip_response_format_success():
    from app.domain.services.recovery._actions import StripResponseFormat
    out = await StripResponseFormat().apply(
        [HumanMessage(content="x")],
        {"response_format": {"type": "json_schema"}},
        _ctx(),
    )
    assert out is not None
    _, new_kwargs = out
    assert new_kwargs["response_format"] is None


async def test_strip_response_format_already_none_skips():
    from app.domain.services.recovery._actions import StripResponseFormat
    out = await StripResponseFormat().apply(
        [HumanMessage(content="x")], {"response_format": None}, _ctx(),
    )
    assert out is None


async def test_disable_thinking_extra_body_enable_thinking():
    from app.domain.services.recovery._actions import DisableThinking
    out = await DisableThinking().apply(
        [HumanMessage(content="x")],
        {"extra_body": {"enable_thinking": True}},
        _ctx(profile=_profile("extra_body_enable_thinking")),
    )
    assert out is not None
    _, new_kwargs = out
    assert new_kwargs["extra_body"]["enable_thinking"] is False


async def test_disable_thinking_already_off_skips():
    from app.domain.services.recovery._actions import DisableThinking
    out = await DisableThinking().apply(
        [HumanMessage(content="x")],
        {"extra_body": {"enable_thinking": False}},
        _ctx(profile=_profile("extra_body_enable_thinking")),
    )
    assert out is None


async def test_downgrade_tool_choice_to_auto():
    from app.domain.services.recovery._actions import DowngradeToolChoiceToAuto
    out = await DowngradeToolChoiceToAuto().apply(
        [HumanMessage(content="x")], {"tool_choice": "required"}, _ctx(),
    )
    assert out is not None
    _, new_kwargs = out
    assert new_kwargs["tool_choice"] == "auto"


async def test_downgrade_tool_choice_already_auto_skips():
    from app.domain.services.recovery._actions import DowngradeToolChoiceToAuto
    out = await DowngradeToolChoiceToAuto().apply(
        [HumanMessage(content="x")], {"tool_choice": "auto"}, _ctx(),
    )
    assert out is None


async def test_downgrade_tool_choice_writes_auto_when_per_call_is_none():
    """Audit Round 16 P1 #2: per-call kwarg may be None when tool_choice was
    set via bind_tools. Recovery must still write per-call "auto" so it
    overrides the bound value at resolve_tool_choice's per_call > bound
    precedence. See actus_chat_model.py:670-683 / :837-850.
    """
    from app.domain.services.recovery._actions import DowngradeToolChoiceToAuto
    out = await DowngradeToolChoiceToAuto().apply(
        [HumanMessage(content="x")], {}, _ctx(),  # no tool_choice in kwargs
    )
    assert out is not None, (
        "DowngradeToolChoiceToAuto must NOT short-circuit when per-call "
        "tool_choice is None — bound value via bind_tools may still be "
        "'required', and per-call 'auto' is the only override path."
    )
    _, new_kwargs = out
    assert new_kwargs["tool_choice"] == "auto"


async def test_downgrade_tool_choice_writes_auto_when_per_call_is_required():
    """Regression: pre-Round-16 behavior on the explicit per-call path."""
    from app.domain.services.recovery._actions import DowngradeToolChoiceToAuto
    out = await DowngradeToolChoiceToAuto().apply(
        [HumanMessage(content="x")], {"tool_choice": "required"}, _ctx(),
    )
    assert out is not None
    _, new_kwargs = out
    assert new_kwargs["tool_choice"] == "auto"


async def test_trigger_recompact_calls_callback_and_returns_messages():
    from app.domain.services.recovery._actions import TriggerRecompact
    compressed = [HumanMessage(content="compressed")]

    async def cb(m, k): return compressed

    out = await TriggerRecompact().apply(
        [HumanMessage(content="big"), HumanMessage(content="more")],
        {}, _ctx(callback=cb),
    )
    assert out is not None
    new_msgs, _ = out
    assert new_msgs == compressed


async def test_trigger_recompact_callback_returns_none_skips():
    from app.domain.services.recovery._actions import TriggerRecompact
    async def cb(m, k): return None
    assert await TriggerRecompact().apply(
        [HumanMessage(content="x")], {}, _ctx(callback=cb),
    ) is None


async def test_trigger_recompact_no_callback_skips():
    from app.domain.services.recovery._actions import TriggerRecompact
    assert await TriggerRecompact().apply(
        [HumanMessage(content="x")], {}, _ctx(callback=None),
    ) is None


async def test_trigger_recompact_callback_exception_propagates():
    from app.domain.services.recovery._actions import TriggerRecompact

    async def cb(m, k): raise RuntimeError("compact failed")

    with pytest.raises(RuntimeError, match="compact failed"):
        await TriggerRecompact().apply([HumanMessage(content="x")], {}, _ctx(callback=cb))


async def test_I5_callback_receives_shallow_copy_of_kwargs():
    """Callback mutations of its kwargs must not leak back to caller's dict."""
    from app.domain.services.recovery._actions import TriggerRecompact

    original = {"k": "v"}

    async def cb(m, kwargs):
        kwargs["k"] = "mutated"
        return [HumanMessage(content="x")]

    await TriggerRecompact().apply([HumanMessage(content="x")], original, _ctx(callback=cb))
    assert original["k"] == "v"
