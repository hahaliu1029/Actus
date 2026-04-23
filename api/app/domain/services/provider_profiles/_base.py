"""ProviderProfile + error taxonomy + rewrite warning data classes.

Reasoning roundtrip 兼容承诺范围（C3，spec §3.6）：
Kimi/DeepSeek reasoning_content 的 echo / strip 规则仅覆盖未经
Memory.compact() 的 in-flight replay。compacted 历史（memory.py:75-80
无条件删除 reasoning_content）不在 A7 承诺范围内。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class ErrorClass(StrEnum):
    """Taxonomy for provider-returned exceptions. Consumed by B2 recovery chain."""

    TRANSIENT_RATE_LIMIT = "transient_rate_limit"
    TRANSIENT_CONNECTION = "transient_connection"
    TRANSIENT_AUTH = "transient_auth"
    COMPAT_QUIRK = "compat_quirk"
    CONTEXT_OVERFLOW = "context_overflow"
    PERMANENT_4XX = "permanent_4xx"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorFingerprint:
    """Match (status_code, body_substring) → error_class. status_code=0 = any status."""

    status_code: int
    body_substring: str | None
    error_class: ErrorClass


@dataclass(frozen=True)
class RewriteWarning:
    """Warnings-as-data (spec §4.3 round-3 P1).

    Pure functions (apply_outbound_rewrites / resolve_tool_choice /
    resolve_response_format) return warnings; adapter instances dedup via
    ``_emitted_warning_codes: set[str]`` and emit to logger.

    code: stable string dedup key, e.g. "tool_choice_forbidden/required"
          or "sampling_forbidden/logprobs".
    level: "warning" | "debug" — adapter dispatches to logger.warning/debug.
    message: human-readable text for the log line.
    context: optional structured telemetry fields (not used for dedup).
    """

    code: str
    level: Literal["warning", "debug"] = "warning"
    message: str = ""
    context: dict[str, object] | None = None


@dataclass(frozen=True)
class ProviderProfile:
    """Declarative provider capability record. spec §4.2."""

    # --- Identity ---
    provider_id: str
    human_name: str
    default_api_mode: Literal["chat_completions", "responses", "auto"]
    api_mode_fallback_enabled: bool

    # --- Tool-call / tool_choice ---
    tool_choice_any_alias: Literal["required", "auto", "none"] = "required"
    tool_choice_forbidden_when_thinking: frozenset[str] = frozenset()
    emits_tool_calls_in_content: bool = False

    # --- Thinking / reasoning (Chat Completions 语义) ---
    supports_thinking: bool = False
    thinking_always_on: bool = False
    thinking_toggle_style: Literal[
        "none",
        "openai_reasoning_effort",
        "extra_body_enable_thinking",
        "extra_body_thinking",
    ] = "none"
    reasoning_content_field_name: str = "reasoning_content"
    reasoning_echo_in_tool_loop: bool = False
    reasoning_echo_across_user_turns: bool = False
    supports_thinking_with_tools: bool = True

    # --- Multimodal ---
    accepts_image_url: bool = True
    accepts_image_base64: bool = True
    image_max_bytes: int = 5 * 1024 * 1024
    supports_vision: bool = True
    supports_pdf_input: bool = False

    # --- Sampling params ---
    silently_ignored_sampling_params: frozenset[str] = frozenset()
    forbidden_sampling_params: frozenset[str] = frozenset()

    # --- response_format ---
    supports_response_format_json_object: bool = True
    supports_response_format_json_schema: bool = True
    response_format_silently_ignored: bool = False

    # --- Errors ---
    error_fingerprints: tuple[ErrorFingerprint, ...] = field(default_factory=tuple)

    # --- Context window ---
    default_context_window: int = 128_000
    default_max_output_tokens: int = 8_192

    # --- Fallback chain hooks (consumed by B2) ---
    downgrade_targets: tuple[str, ...] = ()
