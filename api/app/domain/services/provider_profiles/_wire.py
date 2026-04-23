"""Outbound wire serialization — inject reasoning field with provider-specific key.

spec §4.3c. Single point where internal 'reasoning_content' key maps to wire key
(K2: 'reasoning_content', K2.6: 'reasoning'). Entry is the already-built OpenAI
dict; we mutate it in place (wire entry is local to serialization, not state).
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import ProviderProfile

_INTERNAL_KEY = "reasoning_content"


def inject_reasoning_into_wire_entry(
    entry: dict,
    additional_kwargs: dict,
    profile: ProviderProfile,
    *,
    is_chat_completions_api: bool,
) -> dict:
    """Mutate *entry* by injecting reasoning under profile.reasoning_content_field_name.

    No-op when:
    - profile.supports_thinking=False
    - additional_kwargs lacks 'reasoning_content' or value is empty
    - is_chat_completions_api=False (Q3 boundary: Responses API has its own shape)
    """
    if not is_chat_completions_api:
        return entry
    if not profile.supports_thinking:
        return entry
    value = (additional_kwargs or {}).get(_INTERNAL_KEY)
    if not value:
        return entry
    entry[profile.reasoning_content_field_name] = value
    return entry
