from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from app.domain.services.provider_profiles._base import ProviderProfile
from app.domain.services.provider_profiles._parse import (
    _to_dict,
    parse_chat_completion_message,
    parse_chat_completion_stream_chunk,
)


def _profile(**overrides) -> ProviderProfile:
    defaults = dict(
        provider_id="x", human_name="X",
        default_api_mode="chat_completions", api_mode_fallback_enabled=False,
    )
    defaults.update(overrides)
    return ProviderProfile(**defaults)


class _FakeChatCompletionMessage(BaseModel):
    """Pydantic v2 model mimicking openai.ChatCompletionMessage with extras allowed."""
    model_config = ConfigDict(extra="allow")
    content: str
    tool_calls: list = []


# ---- _to_dict ----

def test_to_dict_dict_passthrough() -> None:
    d = {"a": 1, "b": 2}
    assert _to_dict(d) == d


def test_to_dict_pydantic_model_dump() -> None:
    m = _FakeChatCompletionMessage(content="hi", reasoning_content="think")
    d = _to_dict(m)
    assert d["content"] == "hi"
    assert d["reasoning_content"] == "think"


def test_to_dict_simple_object_fallback() -> None:
    """Mapping-like obj (no model_dump) falls through dict()."""
    class M:
        def keys(self): return ["a"]
        def __getitem__(self, k): return 1
    assert _to_dict(M()) == {"a": 1}


def test_to_dict_unhashable_non_mapping_raises_typeerror() -> None:
    """Non-mapping non-Pydantic → TypeError; exposes adapter boundary bug."""
    with pytest.raises(TypeError):
        _to_dict("not-a-dict")


# ---- parse_chat_completion_message ----

def test_parse_kimi_k2_dict_input() -> None:
    p = _profile(supports_thinking=True, reasoning_content_field_name="reasoning_content")
    raw = {"content": "answer", "tool_calls": [], "reasoning_content": "think"}
    kw = parse_chat_completion_message(raw, p)
    assert kw == {"reasoning_content": "think"}


def test_parse_kimi_k2_sdk_object_input() -> None:
    p = _profile(supports_thinking=True)
    obj = _FakeChatCompletionMessage(content="answer", reasoning_content="think")
    kw = parse_chat_completion_message(obj, p)
    assert kw == {"reasoning_content": "think"}


def test_parse_kimi_k2_6_wire_field_maps_to_internal_key() -> None:
    p = _profile(supports_thinking=True, reasoning_content_field_name="reasoning")
    raw = {"content": "answer", "reasoning": "think"}
    kw = parse_chat_completion_message(raw, p)
    assert kw == {"reasoning_content": "think"}


def test_parse_both_field_names_profile_wins() -> None:
    p = _profile(supports_thinking=True, reasoning_content_field_name="reasoning_content")
    raw = {"content": "a", "reasoning_content": "k2", "reasoning": "k26"}
    kw = parse_chat_completion_message(raw, p)
    assert kw == {"reasoning_content": "k2"}


def test_parse_skipped_when_profile_not_thinking() -> None:
    p = _profile(supports_thinking=False)
    raw = {"content": "a", "reasoning_content": "think"}
    kw = parse_chat_completion_message(raw, p)
    assert kw == {}


def test_parse_empty_reasoning_not_set() -> None:
    p = _profile(supports_thinking=True)
    raw = {"content": "a", "reasoning_content": ""}
    kw = parse_chat_completion_message(raw, p)
    assert kw == {}


# ---- parse_chat_completion_stream_chunk ----

def test_parse_stream_chunk_dict() -> None:
    p = _profile(supports_thinking=True)
    delta = {"content": "a", "reasoning_content": "think"}
    kw = parse_chat_completion_stream_chunk(delta, p)
    assert kw == {"reasoning_content": "think"}


def test_parse_stream_chunk_sdk_object() -> None:
    p = _profile(supports_thinking=True)
    delta = SimpleNamespace(content="a", reasoning_content="think")
    kw = parse_chat_completion_stream_chunk(delta, p)
    assert kw == {"reasoning_content": "think"}


def test_parse_stream_chunk_k2_6_field_name() -> None:
    p = _profile(supports_thinking=True, reasoning_content_field_name="reasoning")
    delta = {"content": "a", "reasoning": "think"}
    kw = parse_chat_completion_stream_chunk(delta, p)
    assert kw == {"reasoning_content": "think"}
