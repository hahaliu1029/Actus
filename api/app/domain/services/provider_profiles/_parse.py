"""Inbound response parsing — normalize provider reasoning fields to internal key.

spec §4.3a. Boundary contract: accepts both dict and SDK Pydantic objects (via
_to_dict normalization); adapter does not need to call model_dump() first.
"""
from __future__ import annotations

from typing import Any

from app.domain.services.provider_profiles._base import ProviderProfile

_INTERNAL_KEY = "reasoning_content"


def _to_dict(raw: Any) -> dict:
    """Normalize SDK Pydantic v2 / dict / mapping-like to dict.

    Non-mapping, non-Pydantic → raises TypeError (exposes adapter boundary bug).

    Precedence:
      1. Pydantic v2 model (``model_dump``)
      2. dict pass-through
      3. Mapping protocol (``keys`` + ``__getitem__``) via ``dict(raw)``
      4. Object with ``__dict__`` (SimpleNamespace / plain classes) via ``vars()``
      5. Otherwise TypeError
    """
    if hasattr(raw, "model_dump"):
        return raw.model_dump(exclude_none=False)
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "keys") and hasattr(raw, "__getitem__"):
        # Catch both TypeError and ValueError: `dict(raw)` raises ValueError on
        # non-mapping iterables of strings (e.g. `dict("abc")`), not TypeError.
        # Do not narrow this without re-verifying test_to_dict_unhashable_non_mapping_raises_typeerror.
        try:
            return dict(raw)
        except (TypeError, ValueError):
            raise TypeError(
                f"_to_dict: cannot convert {type(raw).__name__} to dict; "
                f"adapter boundary error — pass dict or SDK Pydantic object"
            )
    if hasattr(raw, "__dict__"):
        return dict(vars(raw))
    raise TypeError(
        f"_to_dict: cannot convert {type(raw).__name__} to dict; "
        f"adapter boundary error — pass dict or SDK Pydantic object"
    )


def parse_chat_completion_message(
    message: dict | Any,
    profile: ProviderProfile,
) -> dict:
    """Normalize provider reasoning field to additional_kwargs['reasoning_content']."""
    if not profile.supports_thinking:
        return {}

    msg = _to_dict(message)
    wire_key = profile.reasoning_content_field_name
    value = msg.get(wire_key)
    if not value and wire_key != _INTERNAL_KEY:
        value = msg.get(_INTERNAL_KEY)
    if not value:
        return {}
    return {_INTERNAL_KEY: value}


def parse_chat_completion_stream_chunk(
    chunk: dict | Any,
    profile: ProviderProfile,
) -> dict:
    """Same as parse_chat_completion_message but for streaming delta chunks."""
    return parse_chat_completion_message(chunk, profile)
