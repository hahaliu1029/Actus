from app.domain.services.provider_profiles._base import ProviderProfile
from app.domain.services.provider_profiles._rewrites import resolve_response_format


def _profile(**overrides) -> ProviderProfile:
    defaults = dict(
        provider_id="x", human_name="X",
        default_api_mode="chat_completions", api_mode_fallback_enabled=False,
    )
    defaults.update(overrides)
    return ProviderProfile(**defaults)


def test_none_input_returns_none_none() -> None:
    v, w = resolve_response_format(None, _profile())
    assert v is None
    assert w is None


def test_json_object_passthrough_when_supported() -> None:
    p = _profile(supports_response_format_json_object=True,
                 supports_response_format_json_schema=True)
    rf = {"type": "json_object"}
    v, w = resolve_response_format(rf, p)
    assert v == rf
    assert w is None


def test_json_schema_stripped_when_unsupported() -> None:
    """T23: DeepSeek reasoner — json_schema 不支持, strip + warning"""
    p = _profile(provider_id="deepseek_reasoner",
                 supports_response_format_json_object=True,
                 supports_response_format_json_schema=False)
    rf = {"type": "json_schema", "json_schema": {"schema": {}}}
    v, w = resolve_response_format(rf, p)
    assert v is None
    assert w is not None
    assert "json_schema" in w.code
    assert "deepseek_reasoner" in w.code


def test_silently_ignored_strips_all() -> None:
    """T24: Anthropic-compat — silently_ignored=True → 全 strip"""
    p = _profile(provider_id="anthropic_compat",
                 response_format_silently_ignored=True)
    rf1 = {"type": "json_object"}
    v, w = resolve_response_format(rf1, p)
    assert v is None
    assert "ignored" in w.code

    rf2 = {"type": "json_schema", "json_schema": {}}
    v, w = resolve_response_format(rf2, p)
    assert v is None
    assert "ignored" in w.code


def test_unknown_type_stripped() -> None:
    p = _profile()
    v, w = resolve_response_format({"type": "weird_shape"}, p)
    assert v is None
    assert w is not None
    assert "unknown" in w.code


# ---------- Task 3.6 real DeepSeek profile ----------

from app.domain.services.provider_profiles import get_profile  # noqa: E402


def test_deepseek_reasoner_json_schema_stripped_real_profile() -> None:
    p = get_profile("deepseek_reasoner")
    rf = {"type": "json_schema", "json_schema": {"schema": {"a": 1}}}
    v, w = resolve_response_format(rf, p)
    assert v is None
    assert "json_schema" in w.code
    assert "deepseek_reasoner" in w.code


def test_deepseek_reasoner_json_object_passthrough_real_profile() -> None:
    p = get_profile("deepseek_reasoner")
    rf = {"type": "json_object"}
    v, w = resolve_response_format(rf, p)
    assert v == rf
    assert w is None


# ---------- Task 4.2 Responses-path shared helpers ----------


def test_responses_api_rewrites_skip_reasoning_echo() -> None:
    """T19: is_chat_completions_api=False → apply_outbound_rewrites preserves
    cross-turn reasoning_content; only strips sampling params."""
    from app.domain.services.provider_profiles._rewrites import apply_outbound_rewrites
    from langchain_core.messages import AIMessage, HumanMessage

    p = get_profile("deepseek_reasoner")
    msgs = [
        HumanMessage("q1"),
        AIMessage(content="a", additional_kwargs={"reasoning_content": "think"}),
        HumanMessage("q2"),
    ]
    rewritten, kwargs, _ = apply_outbound_rewrites(
        msgs, {"logprobs": True, "temperature": 0.5}, p,
        is_chat_completions_api=False,
    )
    # Q3 boundary: cross-turn reasoning preserved (not handled by rewrites at all).
    assert rewritten[1].additional_kwargs.get("reasoning_content") == "think"
    # Sampling params still strip.
    assert "logprobs" not in kwargs
    assert "temperature" not in kwargs


def test_responses_api_classify_error_shared_fingerprints() -> None:
    """T20: Responses-path exceptions share the profile's error fingerprints."""
    from app.domain.services.provider_profiles._classify import classify_error
    import httpx
    import openai
    from app.domain.services.provider_profiles._base import ErrorClass

    profile = get_profile("deepseek_reasoner")
    req = httpx.Request("POST", "https://api.deepseek.com/responses")
    resp = httpx.Response(400, request=req, text="Missing reasoning_content x")
    exc = openai.BadRequestError(
        "Missing reasoning_content x", response=resp,
        body={"message": "Missing reasoning_content x"},
    )
    assert classify_error(exc, profile) == ErrorClass.COMPAT_QUIRK
