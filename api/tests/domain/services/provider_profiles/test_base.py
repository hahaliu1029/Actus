from dataclasses import FrozenInstanceError
import pytest
from app.domain.services.provider_profiles._base import (
    ProviderProfile, ErrorClass, ErrorFingerprint, RewriteWarning,
)
from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE
from app.domain.services.provider_profiles.openai_official import OPENAI_OFFICIAL_PROFILE


def test_provider_profile_frozen() -> None:
    p = ProviderProfile(
        provider_id="x", human_name="X",
        default_api_mode="chat_completions", api_mode_fallback_enabled=False,
    )
    with pytest.raises(FrozenInstanceError):
        p.provider_id = "y"  # type: ignore[misc]


def test_provider_profile_defaults() -> None:
    p = ProviderProfile(
        provider_id="x", human_name="X",
        default_api_mode="chat_completions", api_mode_fallback_enabled=False,
    )
    assert p.tool_choice_any_alias == "required"
    assert p.tool_choice_forbidden_when_thinking == frozenset()
    assert p.emits_tool_calls_in_content is False
    assert p.supports_thinking is False
    assert p.thinking_always_on is False
    assert p.reasoning_content_field_name == "reasoning_content"
    assert p.reasoning_echo_in_tool_loop is False
    assert p.reasoning_echo_across_user_turns is False
    assert p.supports_thinking_with_tools is True
    assert p.accepts_image_url is True
    assert p.accepts_image_base64 is True
    assert p.image_max_bytes == 5 * 1024 * 1024
    assert p.supports_vision is True
    assert p.supports_pdf_input is False
    assert p.silently_ignored_sampling_params == frozenset()
    assert p.forbidden_sampling_params == frozenset()
    assert p.supports_response_format_json_object is True
    assert p.supports_response_format_json_schema is True
    assert p.response_format_silently_ignored is False
    assert p.error_fingerprints == ()
    assert p.default_context_window == 128_000
    assert p.default_max_output_tokens == 8_192
    assert p.downgrade_targets == ()
    assert isinstance(p.tool_choice_forbidden_when_thinking, frozenset)
    assert isinstance(p.silently_ignored_sampling_params, frozenset)
    assert isinstance(p.forbidden_sampling_params, frozenset)
    assert isinstance(p.error_fingerprints, tuple)
    assert isinstance(p.downgrade_targets, tuple)


def test_error_class_values() -> None:
    assert ErrorClass.TRANSIENT_RATE_LIMIT == "transient_rate_limit"
    assert ErrorClass.TRANSIENT_CONNECTION == "transient_connection"
    assert ErrorClass.TRANSIENT_AUTH == "transient_auth"
    assert ErrorClass.COMPAT_QUIRK == "compat_quirk"
    assert ErrorClass.CONTEXT_OVERFLOW == "context_overflow"
    assert ErrorClass.PERMANENT_4XX == "permanent_4xx"
    assert ErrorClass.UNKNOWN == "unknown"


def test_error_fingerprint_frozen() -> None:
    f = ErrorFingerprint(status_code=400, body_substring="Missing",
                         error_class=ErrorClass.COMPAT_QUIRK)
    with pytest.raises(FrozenInstanceError):
        f.status_code = 500  # type: ignore[misc]


def test_rewrite_warning_frozen_and_defaults() -> None:
    w = RewriteWarning(code="sampling_forbidden/logprobs")
    assert w.level == "warning"
    assert w.message == ""
    assert w.context is None
    w2 = RewriteWarning(code="x", level="debug", message="m", context={"k": 1})
    assert w2.level == "debug"
    assert w2.message == "m"
    assert w2.context == {"k": 1}
    with pytest.raises(FrozenInstanceError):
        w.code = "y"  # type: ignore[misc]


def test_generic_openai_profile_is_conservative() -> None:
    """Generic 兜底 profile：所有字段默认值；允许 all；不声明 strip。"""
    p = GENERIC_OPENAI_PROFILE
    assert p.provider_id == "generic_openai"
    assert p.default_api_mode == "chat_completions"
    assert p.api_mode_fallback_enabled is True   # 未知 provider 允许 chat→responses 升级
    assert p.tool_choice_any_alias == "required"
    assert p.silently_ignored_sampling_params == frozenset()
    assert p.forbidden_sampling_params == frozenset()
    assert p.supports_thinking is False
    assert p.supports_vision is True


def test_openai_official_profile_identity() -> None:
    p = OPENAI_OFFICIAL_PROFILE
    assert p.provider_id == "openai_official"
    assert p.api_mode_fallback_enabled is True
    assert p.supports_thinking is False
    # OpenAI 官方无 forbidden/silently_ignored 采样参数
    assert p.silently_ignored_sampling_params == frozenset()
    assert p.forbidden_sampling_params == frozenset()
