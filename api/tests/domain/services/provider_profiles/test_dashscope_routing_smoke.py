"""T-P1-SMOKE-1: DashScope routing + profile consistency end-to-end (Spec §8.4).

锁定 v5 allowlist：HIT → 命中 P1 profile；MISS → 回退 generic_openai。
防止未来 allowlist prefix 被静默回退 (§5.1 / §5.2 / §7.2 / §8.3 一致性)。
"""
from __future__ import annotations

import pytest

from app.domain.services.provider_profiles import get_profile, infer_provider_from_base_url
# Ensure all P1 profiles registered (package-level side effects):
import app.domain.services.provider_profiles  # noqa: F401


_DASHSCOPE_BU = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@pytest.mark.parametrize("model_name,expected_id,expected_vision", [
    # Allowlist HIT
    ("qwen3-vl-plus",  "dashscope_qwen_vl", True),
    ("qwen-plus",      "dashscope_qwen",    False),
    ("qwen-flash",     "dashscope_qwen",    False),
    # Allowlist MISS → generic_openai (Round 4 Fact #1-3, Round 3 Fact #1)
    # generic_openai default supports_vision=True (see generic_openai.py / test_base)
    ("qwen-vl-max",    "generic_openai",    True),
    ("qwen-max",       "generic_openai",    True),
    ("qwen3.5-plus",   "generic_openai",    True),
])
def test_dashscope_routing_resolves_to_expected_profile(
    model_name: str, expected_id: str, expected_vision: bool,
) -> None:
    inferred_id = infer_provider_from_base_url(_DASHSCOPE_BU, model_name=model_name)
    assert inferred_id == expected_id
    profile = get_profile(inferred_id)
    assert profile.provider_id == expected_id
    assert profile.supports_vision is expected_vision


def test_dashscope_qwen_text_does_not_carry_vision_gate() -> None:
    """§5.1 / §5.2 拆分契约：text flagship 不带 vision；VL 才带。"""
    text_p = get_profile("dashscope_qwen")
    vl_p = get_profile("dashscope_qwen_vl")
    assert text_p.supports_vision is False
    assert text_p.accepts_image_base64 is False
    assert vl_p.supports_vision is True
    assert vl_p.accepts_image_base64 is True
