"""T-P1-BLOCKER-1: per-call thinking detection for opt-in profiles (A7 P1 blocker fix)."""
from __future__ import annotations

import pytest

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._rewrites import detect_per_call_thinking
# Ensure all profiles registered:
import app.domain.services.provider_profiles  # noqa: F401


# -------- extra_body_thinking (Anthropic) --------


def test_detects_anthropic_extra_body_thinking_enabled_dict() -> None:
    p = get_profile("anthropic_compat")
    kw = {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 10_000}}}
    assert detect_per_call_thinking(kw, p) is True


def test_does_not_detect_anthropic_extra_body_thinking_disabled_dict() -> None:
    p = get_profile("anthropic_compat")
    kw = {"extra_body": {"thinking": {"type": "disabled"}}}
    assert detect_per_call_thinking(kw, p) is False


def test_detects_anthropic_extra_body_thinking_truthy_bool() -> None:
    """Tolerate callers passing thinking as a raw boolean (non-canonical but defensive)."""
    p = get_profile("anthropic_compat")
    assert detect_per_call_thinking({"extra_body": {"thinking": True}}, p) is True
    assert detect_per_call_thinking({"extra_body": {"thinking": False}}, p) is False


def test_no_thinking_for_anthropic_when_extra_body_absent() -> None:
    p = get_profile("anthropic_compat")
    assert detect_per_call_thinking({}, p) is False
    assert detect_per_call_thinking({"extra_body": {}}, p) is False
    assert detect_per_call_thinking({"extra_body": None}, p) is False


# -------- extra_body_enable_thinking (DashScope Qwen) --------


def test_detects_dashscope_enable_thinking_true() -> None:
    p = get_profile("dashscope_qwen")
    assert detect_per_call_thinking({"extra_body": {"enable_thinking": True}}, p) is True


def test_does_not_detect_dashscope_enable_thinking_false() -> None:
    p = get_profile("dashscope_qwen")
    assert detect_per_call_thinking({"extra_body": {"enable_thinking": False}}, p) is False


def test_no_thinking_for_dashscope_when_field_absent() -> None:
    p = get_profile("dashscope_qwen")
    assert detect_per_call_thinking({}, p) is False


# -------- openai_reasoning_effort (Gemini 2.5 Flash) --------


@pytest.mark.parametrize("effort", ["low", "medium", "high", "LOW", "Medium"])
def test_detects_gemini_reasoning_effort_active(effort: str) -> None:
    p = get_profile("gemini_compat")
    assert detect_per_call_thinking({"reasoning_effort": effort}, p) is True


def test_does_not_detect_gemini_reasoning_effort_none() -> None:
    p = get_profile("gemini_compat")
    assert detect_per_call_thinking({"reasoning_effort": "none"}, p) is False
    assert detect_per_call_thinking({"reasoning_effort": "None"}, p) is False


def test_no_thinking_for_gemini_when_field_absent() -> None:
    p = get_profile("gemini_compat")
    assert detect_per_call_thinking({}, p) is False
    assert detect_per_call_thinking({"reasoning_effort": None}, p) is False


# -------- thinking_toggle_style="none" (GLM, generic_openai) --------


def test_glm_never_detects_per_call_thinking() -> None:
    """GLM thinking_toggle_style='none' — no per-call toggle regardless of kwargs."""
    p = get_profile("glm")
    assert detect_per_call_thinking({}, p) is False
    assert detect_per_call_thinking({"extra_body": {"thinking": True}}, p) is False
    assert detect_per_call_thinking({"reasoning_effort": "high"}, p) is False


# -------- always-on profiles (Kimi): per-call detector returns False, always_on handles it --------


def test_kimi_always_on_profile_per_call_detector_returns_false() -> None:
    """Kimi K2 thinking_toggle_style='none' + thinking_always_on=True.
    Per-call detector is a no-op for always-on profiles — adapter ORs with thinking_always_on.
    """
    p = get_profile("kimi_k2")
    assert p.thinking_always_on is True
    assert p.thinking_toggle_style == "none"
    assert detect_per_call_thinking({}, p) is False  # per-call path inert
