"""B4 M0 Phase D: static_pricing module contract.

Locks Issue 2B — ``compute_pricing_version`` is deterministic across Python
restarts — and the 5-dim cost calculation. If this drifts, every
``CostRecord.pricing_version`` becomes meaningless (audit trail corruption).
"""

from __future__ import annotations

from decimal import Decimal

from app.domain.services.pricing.static_pricing import (
    PRICING_TABLE,
    compute_cost,
    compute_pricing_version,
    get_price,
)


class TestComputePricingVersion:
    def test_same_dict_yields_same_hash_across_calls(self) -> None:
        """Issue 2B: must be stable so audit trail is meaningful."""
        dict_a = {"gpt-4o": {"input": Decimal("2.5"), "output": Decimal("10")}}
        dict_b = {"gpt-4o": {"input": Decimal("2.5"), "output": Decimal("10")}}
        assert compute_pricing_version(dict_a) == compute_pricing_version(dict_b)

    def test_key_order_irrelevant(self) -> None:
        """Stability must not depend on Python dict insertion order."""
        a = {"m1": {"input": Decimal("1"), "output": Decimal("2")}}
        b = {"m1": {"output": Decimal("2"), "input": Decimal("1")}}
        assert compute_pricing_version(a) == compute_pricing_version(b)

    def test_changed_price_changes_version(self) -> None:
        base = {"gpt-4o": {"input": Decimal("2.5"), "output": Decimal("10")}}
        bumped = {"gpt-4o": {"input": Decimal("3.0"), "output": Decimal("10")}}
        assert compute_pricing_version(base) != compute_pricing_version(bumped)

    def test_version_is_short_hex(self) -> None:
        v = compute_pricing_version({"x": {"input": Decimal("1")}})
        assert isinstance(v, str)
        assert len(v) == 12, f"pricing_version must be 12 hex chars, got {len(v)}"
        assert all(c in "0123456789abcdef" for c in v)

    def test_live_pricing_table_version_is_populated(self) -> None:
        v = compute_pricing_version(PRICING_TABLE)
        assert v


class TestGetPrice:
    def test_known_model_returns_dict_with_all_five_dims(self) -> None:
        for provider_id, provider_table in PRICING_TABLE.items():
            for model_key, dims in provider_table.items():
                for dim in (
                    "input",
                    "output",
                    "cache_read",
                    "cache_write",
                    "reasoning",
                ):
                    assert dim in dims, (
                        f"PRICING_TABLE[{provider_id!r}][{model_key!r}] "
                        f"missing dim {dim!r}"
                    )
                    assert isinstance(dims[dim], Decimal), (
                        f"PRICING_TABLE[{provider_id!r}][{model_key!r}]"
                        f"[{dim!r}] must be Decimal"
                    )

    def test_unknown_model_returns_none(self) -> None:
        assert get_price("totally-made-up-model", "openai_official") is None

    def test_no_provider_returns_none(self) -> None:
        """Without provider_id, pricing lookup must fail shut — don't
        silently pick an arbitrary provider's price."""
        assert get_price("gpt-4o") is None
        assert get_price("gpt-4o", None) is None

    def test_known_pair_returns_dict(self) -> None:
        price = get_price("gpt-4o", "openai_official")
        assert price is not None
        assert price["input"] == Decimal("2.5")

    def test_cross_provider_miss_returns_none(self) -> None:
        """generic_openai/gpt-4o must NOT inherit openai_official's price."""
        assert get_price("gpt-4o", "generic_openai") is None


class TestComputeCost:
    def test_simple_input_output_calculation(self) -> None:
        """Price per 1M tokens: 1000 input at $2.50/M = $0.0025; etc."""
        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
        }
        price = {
            "input": Decimal("2.5"),
            "output": Decimal("10.0"),
            "cache_read": Decimal("0"),
            "cache_write": Decimal("0"),
            "reasoning": Decimal("0"),
        }
        cost = compute_cost(usage, price)
        assert cost == Decimal("0.0075")

    def test_cache_read_discount_applied(self) -> None:
        """cache_read dim comes out of input_token_details.cache_read."""
        usage = {
            "input_tokens": 1000,
            "output_tokens": 0,
            "total_tokens": 1000,
            "input_token_details": {"cache_read": 800},
        }
        price = {
            "input": Decimal("2.5"),
            "output": Decimal("10.0"),
            "cache_read": Decimal("0.25"),
            "cache_write": Decimal("0"),
            "reasoning": Decimal("0"),
        }
        assert compute_cost(usage, price) == Decimal("0.0007")

    def test_reasoning_tokens_priced_separately(self) -> None:
        usage = {
            "input_tokens": 0,
            "output_tokens": 50,
            "total_tokens": 50,
            "output_token_details": {"reasoning": 30},
        }
        price = {
            "input": Decimal("0"),
            "output": Decimal("10.0"),
            "cache_read": Decimal("0"),
            "cache_write": Decimal("0"),
            "reasoning": Decimal("60.0"),
        }
        assert compute_cost(usage, price) == Decimal("0.002")

    def test_none_usage_returns_none_cost(self) -> None:
        price = {
            dim: Decimal("1")
            for dim in ("input", "output", "cache_read", "cache_write", "reasoning")
        }
        assert compute_cost(None, price) is None


class TestGlm5FamilyPriced:
    """[child-budget fix] glm-5.2 is the live deployed model (root + coordinator
    children). Unpriced, every coordinator child lands on the wallclock-only
    budget rung ("token budget enforcement DISABLED" warning) and every glm
    CostRecord stamps cost_status=unknown ($0.0000 in the UI). Rates from the
    SAME source the existing ``glm`` entries cite: Z.AI official USD pricing
    (https://docs.z.ai/guides/overview/pricing, fetched 2026-07-13).
    """

    def test_glm_5_2_priced(self) -> None:
        p = get_price("glm-5.2", "glm")
        assert p is not None
        assert p["input"] == Decimal("1.4")
        assert p["output"] == Decimal("4.4")
        assert p["cache_read"] == Decimal("0.26")

    def test_glm_5_1_priced(self) -> None:
        p = get_price("glm-5.1", "glm")
        assert p is not None
        assert p["input"] == Decimal("1.4")
        assert p["output"] == Decimal("4.4")

    def test_glm_5_priced(self) -> None:
        p = get_price("glm-5", "glm")
        assert p is not None
        assert p["input"] == Decimal("1.0")
        assert p["output"] == Decimal("3.2")

    def test_glm_5_2_suffixed_variant_prefix_falls_back(self) -> None:
        """Dated/suffixed variants must longest-prefix-match glm-5.2, not
        the shorter glm-5 key."""
        assert get_price("glm-5.2-20260701", "glm") == get_price("glm-5.2", "glm")

    def test_glm_5v_turbo_still_exact_match(self) -> None:
        """Adding the glm-5 key must NOT shadow the existing exact entry."""
        p = get_price("glm-5v-turbo", "glm")
        assert p is not None
        assert p["input"] == Decimal("1.2")
