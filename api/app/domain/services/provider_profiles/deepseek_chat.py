"""DeepSeek Chat profile — derived from reasoner but non-thinking."""
from __future__ import annotations

from dataclasses import replace

from app.domain.services.provider_profiles._registry import register_profile
from app.domain.services.provider_profiles.deepseek_reasoner import DEEPSEEK_REASONER_PROFILE


DEEPSEEK_CHAT_PROFILE = replace(
    DEEPSEEK_REASONER_PROFILE,
    provider_id="deepseek_chat",
    human_name="DeepSeek Chat",
    supports_thinking=False,
    thinking_always_on=False,
    reasoning_echo_in_tool_loop=False,
    reasoning_echo_across_user_turns=False,
    silently_ignored_sampling_params=frozenset(),
    forbidden_sampling_params=frozenset({"logprobs", "top_logprobs"}),
    downgrade_targets=(),
)

register_profile(DEEPSEEK_CHAT_PROFILE)
