"""B4 M0 post-audit: every registered provider is either priced or explicitly allow-listed.

The audit flagged that ``kimi_k2`` was a registered provider profile but
had no pricing entry, silently collapsing real Kimi calls to
``cost_status=unknown`` / ``total_usd=0``. This coverage gate prevents the
class of regression where a new ``register_profile(...)`` call ships
without a matching pricing decision: every registered provider must be
either keyed in ``PRICING_TABLE`` or listed in
``UNPRICED_PROVIDER_ALLOWLIST`` (with a review trail).
"""

from __future__ import annotations

from app.domain.services.pricing.static_pricing import (
    PRICING_TABLE,
    UNPRICED_PROVIDER_ALLOWLIST,
    get_price,
)
from app.domain.services.provider_profiles._registry import _REGISTRY


def test_every_registered_provider_is_priced_or_allowlisted() -> None:
    registered = set(_REGISTRY.keys())
    priced = set(PRICING_TABLE.keys())
    unpriced_ok = set(UNPRICED_PROVIDER_ALLOWLIST)

    unclassified = registered - priced - unpriced_ok
    assert not unclassified, (
        f"Registered provider(s) {sorted(unclassified)!r} are neither "
        f"priced (PRICING_TABLE) nor allowlisted (UNPRICED_PROVIDER_ALLOWLIST). "
        "Add real prices or opt-in to the allowlist with a review comment."
    )


def test_allowlist_does_not_overlap_priced() -> None:
    overlap = set(PRICING_TABLE.keys()) & set(UNPRICED_PROVIDER_ALLOWLIST)
    assert not overlap, (
        f"Providers {sorted(overlap)!r} appear in both PRICING_TABLE and "
        "UNPRICED_PROVIDER_ALLOWLIST. Priced providers should not be "
        "allow-listed as unpriced."
    )


def test_kimi_models_are_priced() -> None:
    """Regression: Kimi provider was registered with no pricing; fix this first."""
    assert get_price("kimi-k2", "kimi_k2") is not None
    assert get_price("kimi-k2-6", "kimi_k2_6") is not None


def test_registry_routable_prefixes_resolve_to_a_price() -> None:
    """Every prefix the registry routes to a priced provider must resolve.

    Previous regression: anthropic_compat profile targets ``claude-sonnet-4-6``
    / ``claude-haiku-4-5`` (per the registry's
    ``_ANTHROPIC_COMPAT_MODEL_PREFIXES``) but the pricing table was keyed on
    wrong names (``claude-sonnet-4`` / ``claude-opus-4``) — real production
    Anthropic calls silently resolved to ``cost_status=unknown``. This test
    fails loudly on that class of bug.
    """
    from app.domain.services.provider_profiles._registry import (
        _ANTHROPIC_COMPAT_MODEL_PREFIXES,
        _DASHSCOPE_QWEN_TEXT_PREFIXES,
        _DASHSCOPE_QWEN_VL_PREFIXES,
        _GEMINI_COMPAT_MODEL_PREFIXES,
    )

    # Representative routable model names per priced provider, sourced
    # where possible from the registry's classifier constants. Providers
    # that infer purely by base_url (kimi / deepseek / openai_official)
    # still need at least one representative model tested — those are
    # listed by hand. UNPRICED_PROVIDER_ALLOWLIST entries are skipped.
    representatives: dict[str, tuple[str, ...]] = {
        "openai_official": ("gpt-4o", "gpt-4o-mini", "o1", "o1-mini"),
        "deepseek_chat": ("deepseek-chat",),
        "deepseek_reasoner": ("deepseek-reasoner",),
        "anthropic_compat": tuple(_ANTHROPIC_COMPAT_MODEL_PREFIXES),
        "kimi_k2": ("kimi-k2",),
        "kimi_k2_6": ("kimi-k2-6",),
    }

    # Sanity: priced providers are exactly those in ``representatives``
    # (minus UNPRICED_PROVIDER_ALLOWLIST). If not, this test is stale.
    priced = set(PRICING_TABLE.keys())
    expected = set(representatives.keys())
    missing_in_test = priced - expected
    assert not missing_in_test, (
        f"Priced provider(s) {sorted(missing_in_test)!r} have no "
        "representative model entry in this test's ``representatives`` "
        "dict — add at least one known routable model name so the "
        "coverage gate can check its price."
    )

    # Also touch the dashscope/gemini prefix tuples so a rename in the
    # registry trips this test (and the author decides whether to add
    # pricing or extend UNPRICED_PROVIDER_ALLOWLIST explicitly).
    _ = _DASHSCOPE_QWEN_TEXT_PREFIXES
    _ = _DASHSCOPE_QWEN_VL_PREFIXES
    _ = _GEMINI_COMPAT_MODEL_PREFIXES

    unresolved: list[tuple[str, str]] = []
    for provider_id, models in representatives.items():
        for model in models:
            if get_price(model, provider_id) is None:
                unresolved.append((provider_id, model))
    assert not unresolved, (
        "Registry-routable (provider, model) pairs with no price: "
        f"{unresolved!r}. Either add prices or update the classifier."
    )


def test_prefix_fallback_handles_versioned_model_names() -> None:
    """Real configs often carry suffixes (``-latest``, dated IDs)."""
    # exact match still wins
    assert get_price("claude-sonnet-4-6", "anthropic_compat") is not None
    # versioned/dated suffix falls back to the prefix key
    assert get_price("claude-sonnet-4-6-latest", "anthropic_compat") is not None
    assert get_price("claude-sonnet-4-6-20260101", "anthropic_compat") is not None
    assert get_price("claude-haiku-4-5-latest", "anthropic_compat") is not None
    # genuinely unknown model still returns None
    assert get_price("claude-opus-6", "anthropic_compat") is None
