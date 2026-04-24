"""T-P1-SMOKE-2: Anthropic tool_choice thinking-downgrade (Spec §8.4).

锁定 spec §5.3 的 v2 修复：alias="auto" + forbidden_when_thinking={"required"}
组合，thinking 开启时 "required" 被正确降为 "auto" (不是 rewrite loop)。
防止未来回退到 alias="required" (Round 1 P1-2)。
"""
from __future__ import annotations

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._rewrites import resolve_tool_choice
import app.domain.services.provider_profiles.anthropic_compat  # noqa: F401


def test_anthropic_tool_choice_required_under_thinking_downgrades_to_auto() -> None:
    """thinking=True + per_call='required' → 'auto' + warning."""
    p = get_profile("anthropic_compat")
    resolved, warnings = resolve_tool_choice(
        per_call_value="required", bound_value=None, profile=p, thinking_enabled=True,
    )
    assert resolved == "auto"
    codes = [w.code for w in warnings]
    assert "tool_choice_forbidden/required" in codes
    # 关键断言：warning.context 显示 rewritten_to=auto (NOT required — 防 rewrite loop)
    ctx = next(w.context for w in warnings if w.code == "tool_choice_forbidden/required")
    assert ctx == {"original": "required", "rewritten_to": "auto"}


def test_anthropic_tool_choice_required_without_thinking_preserved() -> None:
    """thinking=False + per_call='required' → 'required' (preserved, no warning)."""
    p = get_profile("anthropic_compat")
    resolved, warnings = resolve_tool_choice(
        per_call_value="required", bound_value=None, profile=p, thinking_enabled=False,
    )
    assert resolved == "required"
    assert warnings == []


def test_anthropic_tool_choice_any_aliased_to_auto() -> None:
    """'any' → tool_choice_any_alias='auto' unconditionally (accepted trade-off, D-Q1).

    Alias step runs before forbidden-thinking step. Outside thinking, there is
    no forbidden-thinking rewrite, and the alias step does not emit warnings.
    """
    p = get_profile("anthropic_compat")
    resolved, warnings = resolve_tool_choice(
        per_call_value="any", bound_value=None, profile=p, thinking_enabled=False,
    )
    assert resolved == "auto"
    assert warnings == []
