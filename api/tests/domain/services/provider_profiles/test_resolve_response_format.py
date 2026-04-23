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
