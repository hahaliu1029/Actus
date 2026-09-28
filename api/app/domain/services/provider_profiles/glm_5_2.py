"""GLM-5.2 reasoning round-trip, without changing legacy GLM models.

https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode documents
interleaved reasoning and Coding Plan's default preserved thinking.
Standard API keeps reasoning within a tool turn; Coding Plan keeps all turns.
"""
from dataclasses import replace

from app.domain.services.provider_profiles._registry import register_profile
from app.domain.services.provider_profiles.glm import GLM_PROFILE


GLM_5_2_PROFILE = replace(
    GLM_PROFILE,
    provider_id="glm_5_2",
    human_name="GLM-5.2",
    # The model's official input modality is text; do not inherit legacy V models.
    supports_vision=False,
    supports_thinking=True,
    thinking_toggle_style="extra_body_thinking",
    supports_thinking_with_tools=True,
    reasoning_echo_in_tool_loop=True,
    reasoning_echo_across_user_turns=False,
)
GLM_5_2_CODING_PROFILE = replace(
    GLM_5_2_PROFILE,
    provider_id="glm_5_2_coding",
    human_name="GLM-5.2 Coding Plan",
    reasoning_echo_across_user_turns=True,
)

register_profile(GLM_5_2_PROFILE)
register_profile(GLM_5_2_CODING_PROFILE)
