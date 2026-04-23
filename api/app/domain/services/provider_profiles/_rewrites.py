"""Pure functions for profile-driven outbound rewrites.

All functions here are pure: no state, no logging, no I/O. They return
(messages, kwargs, warnings) / (value, warning) tuples. Adapters consume
warnings via `_emit_warnings(...)` and do their own dedup.

spec §4.3 apply_outbound_rewrites, §4.6 resolve_tool_choice +
resolve_response_format, §4.6a build_sdk_params.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.application.errors.exceptions import InternalError
from app.domain.services.provider_profiles._base import ProviderProfile, RewriteWarning


# ========== resolve_tool_choice ==========

def resolve_tool_choice(
    per_call_value: Any,
    bound_value: Any,
    profile: ProviderProfile,
    *,
    thinking_enabled: bool,
) -> tuple[Any, list[RewriteWarning]]:
    """Unify per-call + bind_tools tool_choice into single effective value."""
    chosen = per_call_value if per_call_value is not None else bound_value
    if chosen is None:
        return None, []

    original_chosen = chosen
    warnings: list[RewriteWarning] = []

    # round-10 P1 type guard: function-specific dict passes through
    if isinstance(chosen, str):
        # Step A: "any" → alias
        if chosen == "any":
            chosen = profile.tool_choice_any_alias

        # Step B: forbidden-when-thinking strong constraint
        if thinking_enabled and chosen in profile.tool_choice_forbidden_when_thinking:
            warnings.append(RewriteWarning(
                code=f"tool_choice_forbidden/{original_chosen}",
                level="warning",
                message=(
                    f"{profile.provider_id} forbids tool_choice={original_chosen!r} "
                    f"under thinking; rewriting to {profile.tool_choice_any_alias!r}"
                ),
                context={
                    "original": original_chosen,
                    "rewritten_to": profile.tool_choice_any_alias,
                },
            ))
            chosen = profile.tool_choice_any_alias

    return chosen, warnings


# ========== resolve_response_format ==========

def resolve_response_format(
    request_rf: dict | None,
    profile: ProviderProfile,
) -> tuple[dict | None, RewriteWarning | None]:
    """Branch on response_format.type; strip when not supported."""
    if request_rf is None:
        return None, None

    if profile.response_format_silently_ignored:
        return None, RewriteWarning(
            code=f"response_format_ignored/{profile.provider_id}",
            level="warning",
            message=f"{profile.provider_id} silently ignores response_format; stripped",
        )

    rf_type = request_rf.get("type")

    if rf_type == "json_object":
        if profile.supports_response_format_json_object:
            return request_rf, None
        return None, RewriteWarning(
            code=f"response_format_stripped/json_object/{profile.provider_id}",
            level="warning",
            message=f"{profile.provider_id} does not support json_object; stripped",
        )

    if rf_type == "json_schema":
        if profile.supports_response_format_json_schema:
            return request_rf, None
        return None, RewriteWarning(
            code=f"response_format_stripped/json_schema/{profile.provider_id}",
            level="warning",
            message=(
                f"{profile.provider_id} does not support json_schema; stripped "
                "(Phase 1 does not auto-downgrade to json_object)"
            ),
        )

    return None, RewriteWarning(
        code=f"response_format_unknown/{rf_type}",
        level="warning",
        message=f"unknown response_format.type={rf_type!r}; stripped",
    )


# ========== apply_outbound_rewrites ==========

def _last_human_index(messages: list[BaseMessage]) -> int:
    """Return index of last HumanMessage. -1 if none."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return -1


def _assert_no_https_image_url(messages: list[BaseMessage], profile: ProviderProfile) -> None:
    """Fail-fast: upstream must strip HTTPS URLs for this profile."""
    if profile.accepts_image_url:
        return
    for msg in messages:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "image_url":
                continue
            url = block.get("image_url", {})
            if isinstance(url, dict):
                url = url.get("url", "")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                raise InternalError(
                    f"[A7] HTTPS image_url reached rewrite layer but profile "
                    f"{profile.provider_id}.accepts_image_url=False — agent_task_runner "
                    f"must have converted to base64 or skipped. url={url[:64]}..."
                )


def _strip_sampling_params(
    kwargs: dict, profile: ProviderProfile,
) -> tuple[dict, list[RewriteWarning]]:
    out = dict(kwargs)
    warnings: list[RewriteWarning] = []
    for p in list(out.keys()):
        if p in profile.forbidden_sampling_params:
            out.pop(p)
            warnings.append(RewriteWarning(
                code=f"sampling_forbidden/{p}",
                level="warning",
                message=f"{profile.provider_id} forbids sampling param {p!r}; stripped",
            ))
        elif p in profile.silently_ignored_sampling_params:
            out.pop(p)
            warnings.append(RewriteWarning(
                code=f"sampling_ignored/{p}",
                level="debug",
                message=f"{profile.provider_id} silently ignores {p!r}; stripped",
            ))
    return out, warnings


def _strip_cross_turn_reasoning(
    messages: list[BaseMessage], profile: ProviderProfile,
) -> list[BaseMessage]:
    """Strip reasoning_content from AIMessages before the last HumanMessage."""
    if profile.reasoning_echo_across_user_turns:
        return messages
    last_h = _last_human_index(messages)
    for i, msg in enumerate(messages):
        if i >= last_h:
            continue
        if not isinstance(msg, AIMessage):
            continue
        msg.additional_kwargs.pop("reasoning_content", None)
    return messages


def apply_outbound_rewrites(
    messages: list[BaseMessage],
    kwargs: dict,
    profile: ProviderProfile,
    *,
    is_chat_completions_api: bool,
) -> tuple[list[BaseMessage], dict, list[RewriteWarning]]:
    """Pure function. Returns deep-copied messages + kwargs + warnings list."""
    deep_messages = deepcopy(messages)
    all_warnings: list[RewriteWarning] = []

    _assert_no_https_image_url(deep_messages, profile)

    rewritten_kwargs, w1 = _strip_sampling_params(kwargs, profile)
    all_warnings.extend(w1)

    if is_chat_completions_api:
        deep_messages = _strip_cross_turn_reasoning(deep_messages, profile)

    return deep_messages, rewritten_kwargs, all_warnings


# ========== build_sdk_params ==========

def build_sdk_params(
    rewritten_kwargs: dict,
    profile: ProviderProfile,
    *,
    adapter_defaults: dict,
    base_params: dict,
    resolved_response_format: dict | None,
    resolved_tool_choice: Any = None,
) -> dict:
    """Build Chat-shape SDK params dict."""
    stripped_keys = (
        profile.silently_ignored_sampling_params
        | profile.forbidden_sampling_params
    )
    effective_defaults = {
        k: v for k, v in adapter_defaults.items() if k not in stripped_keys
    }

    params: dict[str, Any] = {**base_params, **effective_defaults, **rewritten_kwargs}

    if resolved_response_format is not None:
        params["response_format"] = resolved_response_format

    if resolved_tool_choice is not None:
        params["tool_choice"] = resolved_tool_choice

    return params
