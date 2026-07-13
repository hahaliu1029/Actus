"""B4 M0 Phase D: static pricing table + deterministic pricing_version (Issue 2B).

Prices are USD per 1M tokens — the industry-standard public unit. The
in-module dict is the source of truth; any change shifts ``PRICING_VERSION``
and stamps the new value onto every ``CostRecord`` written thereafter, so
the audit trail can distinguish rows priced under different regimes.

Determinism (Issue 2B): ``compute_pricing_version`` hashes the canonical
JSON (sorted keys, Decimal→str) so the version is stable across Python
restarts, independent of dict insertion order, and changes iff a number
changes.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Mapping, Optional


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def compute_pricing_version(
    pricing: Mapping[str, Mapping[str, Decimal]]
) -> str:
    """Compute a deterministic 12-hex-char version tag from a pricing dict."""
    payload = _canonical_json(pricing)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:12]


def _price(
    input_: str,
    output: str,
    cache_read: str = "0",
    cache_write: str = "0",
    reasoning: str = "0",
) -> dict[str, Decimal]:
    return {
        "input": Decimal(input_),
        "output": Decimal(output),
        "cache_read": Decimal(cache_read),
        "cache_write": Decimal(cache_write),
        "reasoning": Decimal(reasoning),
    }


PRICING_TABLE: dict[str, dict[str, dict[str, Decimal]]] = {
    # Provider id matches ``ProviderProfile.provider_id`` (the canonical key
    # carried on ``_identifying_params.provider_id``). A model name that
    # exists across providers (e.g. a self-hosted gpt-oss alias) lands
    # under a different provider_id here, so ``generic_openai`` cannot
    # accidentally price same-named models at ``openai_official`` rates.
    "openai_official": {
        "gpt-4o":      _price("2.5", "10.0", cache_read="1.25"),
        "gpt-4o-mini": _price("0.15", "0.6", cache_read="0.075"),
        "o1":          _price("15.0", "60.0", cache_read="7.5", reasoning="60.0"),
        "o1-mini":     _price("3.0", "12.0", cache_read="1.5", reasoning="12.0"),
        # GPT-5.4 family is intentionally NOT priced — see
        # ``UNPRICED_MODEL_ALLOWLIST`` below. Once OpenAI publishes a
        # dated rate card for these models, move them back here with a
        # source link.
    },
    "deepseek_chat": {
        "deepseek-chat": _price("0.27", "1.1", cache_read="0.07"),
    },
    "deepseek_reasoner": {
        "deepseek-reasoner": _price(
            "0.55", "2.19", cache_read="0.14", reasoning="2.19"
        ),
    },
    "anthropic_compat": {
        # Registry routes these two prefixes to ``anthropic_compat``
        # (see provider_profiles/_registry.py::_ANTHROPIC_COMPAT_MODEL_PREFIXES).
        # Keys here MUST match the prefix so ``get_price``'s prefix-fallback
        # catches full model ids like ``claude-sonnet-4-6-latest`` or dated
        # variants ``claude-sonnet-4-6-20260101``.
        "claude-sonnet-4-6": _price(
            "3.0", "15.0", cache_read="0.3", cache_write="3.75"
        ),
        "claude-haiku-4-5": _price(
            "0.8", "4.0", cache_read="0.08", cache_write="1.0"
        ),
    },
    # Moonshot / Kimi. Public Moonshot pricing as of 2026-04: tune as the
    # vendor's rate card evolves. The two registered profile ids
    # (``kimi_k2`` / ``kimi_k2_6``) map to the same model family today;
    # keep them as separate entries so future model-specific rates don't
    # have to refactor the lookup shape.
    "kimi_k2": {
        "kimi-k2": _price("0.60", "2.50", cache_read="0.15"),
    },
    "kimi_k2_6": {
        "kimi-k2-6": _price("0.60", "2.50", cache_read="0.15"),
    },
    "glm": {
        # Z.AI official USD pricing as of 2026-05:
        # https://docs.z.ai/guides/overview/pricing
        "glm-5v-turbo": _price(
            "1.2", "4.0", cache_read="0.24", reasoning="4.0"
        ),
        # GLM-5.x text family — same source, re-fetched 2026-07-13. The page
        # lists no separate thinking-token rate; mirror the glm-5v-turbo
        # convention (reasoning billed at the output rate) so adapters that
        # report reasoning tokens separately don't undercount.
        # [child-budget fix] Unpriced glm-5.2 put every coordinator child on
        # the wallclock-only budget rung and stamped cost_status=unknown on
        # every glm CostRecord.
        "glm-5.2": _price("1.4", "4.4", cache_read="0.26", reasoning="4.4"),
        "glm-5.1": _price("1.4", "4.4", cache_read="0.26", reasoning="4.4"),
        "glm-5": _price("1.0", "3.2", cache_read="0.2", reasoning="3.2"),
    },
}


# Providers registered in ``app.domain.services.provider_profiles`` that are
# intentionally NOT priced here — either because they forward to provider-
# specific routes upstream (``generic_openai``) or because the M0 scope
# doesn't cover them yet. The registry-coverage test pins this list so a
# new provider doesn't silently become ``cost_status=unknown`` without
# someone deliberately opting it in.
UNPRICED_PROVIDER_ALLOWLIST: frozenset[str] = frozenset(
    {
        "generic_openai",    # fallback catch-all; forwards to real provider
        "gemini_compat",     # TODO: add Gemini prices in a later milestone
        "dashscope_qwen",    # TODO: add DashScope prices
        "dashscope_qwen_vl",
        "minimax",           # TODO: add MiniMax prices
    }
)


# Per-(provider, model) opt-out from pricing. Distinct from
# ``UNPRICED_PROVIDER_ALLOWLIST``: those are entire providers whose pricing
# isn't M0 scope. This is for models inside an *otherwise priced* provider
# whose published rate card we don't have yet — silently using a placeholder
# price would let ``cost_status=actual`` certify a number we made up.
# Keeping them here means ``get_price`` returns None → handler stamps
# ``cost_status=unknown`` (honest) and the example-config coverage gate
# still passes.
UNPRICED_MODEL_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        # GPT-5.4 family — recommended in ``api/config.yaml.example``
        # (lines 31 / 42) but no public OpenAI rate card available yet
        # in this repo's data set. Replace with real prices the moment
        # OpenAI publishes a dated rate card.
        ("openai_official", "gpt-5.4"),
        ("openai_official", "gpt-5.4-pro"),
    }
)


PRICING_VERSION: str = compute_pricing_version(PRICING_TABLE)


def get_price(
    model: str,
    provider_id: Optional[str] = None,
) -> Optional[dict[str, Decimal]]:
    """Return the per-dim price dict for ``(provider_id, model)``, or None.

    ``provider_id`` is required — without it we can't distinguish
    ``openai_official/gpt-4o`` (real OpenAI pricing) from a third-party
    compat endpoint also serving a model called "gpt-4o" at a different
    rate. ``None`` / unknown provider returns None so the CostRecord is
    flagged ``cost_status=unknown`` instead of silently mispricing.

    Lookup is exact-first, then prefix-fallback. Prefix-fallback exists
    because user configs often include suffixes like ``-latest`` or dated
    variants (e.g. ``claude-sonnet-4-6-20260101``) that aren't enumerable
    at pricing-table authoring time. The prefix match mirrors how the
    registry itself classifies models (see
    ``provider_profiles/_registry.py::_ANTHROPIC_COMPAT_MODEL_PREFIXES``).
    """
    if not provider_id or not model:
        return None
    provider_table = PRICING_TABLE.get(provider_id)
    if provider_table is None:
        return None
    exact = provider_table.get(model)
    if exact is not None:
        return exact
    # Prefix fallback — longest-prefix wins so e.g. ``claude-sonnet-4-6-X``
    # matches the ``claude-sonnet-4-6`` key even if a ``claude-sonnet-4``
    # key existed.
    best_key: Optional[str] = None
    for key in provider_table:
        if model.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    if best_key is not None:
        return provider_table[best_key]
    return None


def compute_cost(
    usage_metadata: Optional[Mapping[str, object]],
    price: Mapping[str, Decimal],
) -> Optional[Decimal]:
    """Compute total USD cost from a LangChain UsageMetadata + per-dim prices.

    Returns ``None`` when ``usage_metadata`` is ``None`` so the caller can
    mark the resulting CostRecord as ``estimated``.

    Fresh input = input_tokens - cache_read - cache_write; output_visible =
    output_tokens - reasoning. All arithmetic stays in ``Decimal`` so
    thousands of small rows don't accumulate float rounding error.
    """
    if usage_metadata is None:
        return None

    input_total = int(usage_metadata.get("input_tokens", 0) or 0)
    output_total = int(usage_metadata.get("output_tokens", 0) or 0)

    input_details = dict(usage_metadata.get("input_token_details") or {})
    output_details = dict(usage_metadata.get("output_token_details") or {})

    cache_read = int(input_details.get("cache_read", 0) or 0)
    cache_write = int(
        input_details.get("cache_creation", 0)
        or input_details.get("cache_write", 0)
        or 0
    )
    reasoning = int(output_details.get("reasoning", 0) or 0)

    fresh_input = max(0, input_total - cache_read - cache_write)
    visible_output = max(0, output_total - reasoning)

    per_million = Decimal(1_000_000)

    cost = (
        (Decimal(fresh_input) * price["input"]) / per_million
        + (Decimal(visible_output) * price["output"]) / per_million
        + (Decimal(cache_read) * price["cache_read"]) / per_million
        + (Decimal(cache_write) * price["cache_write"]) / per_million
        + (Decimal(reasoning) * price["reasoning"]) / per_million
    )
    return cost
