"""DashScope Qwen-VL (vision) profile (Spec §5.2).

Target models: qwen3-vl-plus / qwen3-vl-flash (Qwen3-VL family:
  hybrid thinking + tool-capable, thinking togglable + default-off).

不覆盖（heuristic 路由到 generic_openai）：
- qwen-vl-max / qwen-vl-plus — Qwen2.5-VL: Deep thinking / Tool calling "Not supported"
  (Round 4 Fact #3)
- qwen3.5-plus / qwen3.5-flash — hybrid default-ON (不是默认关)
- qwen-omni-turbo / qwen-omni-turbo-latest — 不支持 enable_thinking 参数
- qwen3-omni-* / qwen3.5-omni-* — P2 子 profile 待 rigorous research

通过 replace() 从 dashscope_qwen 派生，只改图像相关字段：其他字段（含 7 条 fingerprint）全部继承。
"""
from __future__ import annotations

from dataclasses import replace

from app.domain.services.provider_profiles._registry import register_profile
from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE


DASHSCOPE_QWEN_VL_PROFILE = replace(
    DASHSCOPE_QWEN_PROFILE,
    provider_id="dashscope_qwen_vl",
    human_name="DashScope Qwen-VL (vision)",
    accepts_image_url=True,
    accepts_image_base64=True,
    # DashScope docs: 10 MB post-base64 for local file, 20 MB for some public-URL models.
    # Actus agent_task_runner:812 比较 raw bytes; sanitizer:9 硬编码 5 MiB ceiling —
    # 这个字段声明 vendor 上限但非 end-to-end wired (sanitizer 仍截 5 MiB)。
    # Raw / base64 / URL 三路径语义拆分是 Finding 1 partial-consumption follow-up。
    image_max_bytes=10 * 1024 * 1024,
    supports_vision=True,
    # error_fingerprints inherited (same error body patterns across family)
)

register_profile(DASHSCOPE_QWEN_VL_PROFILE)
