"""Recovery rule registration (PR-2). Called by __init__.py at import time."""
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.recovery._actions import (
    DisableThinking,
    DowngradeToolChoiceToAuto,
    StripResponseFormat,
    TriggerRecompact,
)
from app.domain.services.recovery._registry import register_rule


def register_all() -> None:
    # R1: DashScope Qwen — json mode + thinking collision
    register_rule(
        ("dashscope_qwen", "chat_completions", ErrorClass.COMPAT_QUIRK, "json_mode_with_thinking"),
        (StripResponseFormat(), DisableThinking()),
    )
    # R2: DashScope Qwen — tool_choice string forbidden
    register_rule(
        ("dashscope_qwen", "chat_completions", ErrorClass.COMPAT_QUIRK, "tool_choice_string_forbidden"),
        (DowngradeToolChoiceToAuto(),),
    )
    # R3: Anthropic-compat Sonnet/Haiku — thinking + tool_choice=required
    register_rule(
        ("anthropic_compat", "chat_completions", ErrorClass.COMPAT_QUIRK, "thinking_forbidden_with_tool_choice"),
        (DowngradeToolChoiceToAuto(),),
    )
    # R4: global CONTEXT_OVERFLOW wildcard (api_mode=* is the only cross-protocol exception)
    register_rule(
        ("*", "*", ErrorClass.CONTEXT_OVERFLOW, "*"),
        (TriggerRecompact(),),
    )
