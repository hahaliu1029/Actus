"""Moonshot Kimi K2.6 profile — replace() on K2 with field-name change.

K2.6 renamed the reasoning field from 'reasoning_content' to 'reasoning' on wire.
Everything else matches K2.5 semantics (Q1 decision: separate profile, explicit
LLMConfig.provider required; heuristic falls back to K2 not K2.6).
"""
from __future__ import annotations

from dataclasses import replace

from app.domain.services.provider_profiles._registry import register_profile
from app.domain.services.provider_profiles.kimi_k2 import KIMI_K2_PROFILE


KIMI_K2_6_PROFILE = replace(
    KIMI_K2_PROFILE,
    provider_id="kimi_k2_6",
    human_name="Moonshot Kimi K2.6",
    reasoning_content_field_name="reasoning",
)

register_profile(KIMI_K2_6_PROFILE)
