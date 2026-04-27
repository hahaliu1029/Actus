"""B4 M0 post-audit: every model recommended in ``api/config.yaml.example`` is priced.

The audit reproducer: ``api/config.yaml.example`` recommends ``gpt-5.4``
and ``gpt-5.4-pro`` as the OpenAI Official examples (lines 31 / 42),
but the M0 ``PRICING_TABLE`` only covered the ``gpt-4o`` family. Result:
every user who copied the recommended config saw
``cost_status=unknown, total_usd=0`` even though usage was reported
correctly.

This gate ties the example file to the pricing table so they can't
silently drift apart again. Every ``llm_config`` block in
``api/config.yaml.example`` (commented or active) must resolve to a
price via ``get_price(model, provider_id)`` — using either the explicit
``provider:`` line or the registry's ``infer_provider_from_base_url``
heuristic, mirroring what ``_build_llm`` does at runtime.

Providers in ``UNPRICED_PROVIDER_ALLOWLIST`` are exempt (the example
config doesn't currently exercise them; if it ever does, this test
fails so the author has to make a deliberate choice).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, NamedTuple

from app.domain.services.pricing.static_pricing import (
    UNPRICED_MODEL_ALLOWLIST,
    UNPRICED_PROVIDER_ALLOWLIST,
    get_price,
)
from app.domain.services.provider_profiles._registry import (
    infer_provider_from_base_url,
)


class _LLMConfigSample(NamedTuple):
    """One llm_config block extracted from the example file."""

    block_label: str
    base_url: str
    model_name: str
    explicit_provider: str | None


def _example_config_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = next(
        p
        for p in here.parents
        if (p / "config.yaml.example").is_file()
        or (p / "api" / "config.yaml.example").is_file()
    )
    candidate = repo_root / "config.yaml.example"
    return candidate if candidate.is_file() else repo_root / "api" / "config.yaml.example"


_FIELD = re.compile(
    r"""^\s*\#?\s*  # optional comment marker
        (base_url|model_name|provider)\s*:\s*
        (.+?)\s*(?:\#.*)?$  # value up to optional trailing comment
    """,
    re.VERBOSE,
)


def _parse_llm_config_blocks(text: str) -> List[_LLMConfigSample]:
    """Extract every ``llm_config`` block (active OR commented) from the file."""
    lines = text.splitlines()
    samples: List[_LLMConfigSample] = []

    in_block = False
    block_label = ""
    base_url: str | None = None
    model_name: str | None = None
    explicit_provider: str | None = None

    def _flush() -> None:
        nonlocal base_url, model_name, explicit_provider, block_label
        if base_url and model_name:
            samples.append(
                _LLMConfigSample(
                    block_label=block_label,
                    base_url=base_url,
                    model_name=model_name,
                    explicit_provider=explicit_provider,
                )
            )
        base_url = model_name = explicit_provider = None
        block_label = ""

    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        # Detect a block header: either "llm_config:" or "# llm_config:"
        if re.match(r"^#?\s*llm_config\s*:\s*$", stripped):
            if in_block:
                _flush()
            in_block = True
            block_label = f"L{idx}"
            continue

        if in_block:
            # Lightweight sentinel: a non-comment top-level key (no leading
            # space, ends with ':') marks the end of the llm_config block
            # (e.g. "agent_config:").
            if (
                raw
                and not raw.startswith("#")
                and not raw.startswith(" ")
                and stripped.endswith(":")
                and not stripped.startswith("llm_config")
            ):
                _flush()
                in_block = False
                continue
            m = _FIELD.match(raw)
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip().strip("'\"")
            if key == "base_url":
                base_url = value
            elif key == "model_name":
                model_name = value
            elif key == "provider":
                explicit_provider = value

    if in_block:
        _flush()

    return samples


def _resolve_provider(sample: _LLMConfigSample) -> str:
    if sample.explicit_provider:
        return sample.explicit_provider
    return infer_provider_from_base_url(
        sample.base_url, model_name=sample.model_name
    )


def test_example_file_parses_at_least_three_blocks() -> None:
    """Sanity: the example file's structure hasn't changed underneath us."""
    text = _example_config_path().read_text(encoding="utf-8")
    samples = _parse_llm_config_blocks(text)
    # As of 2026-04: deepseek default + gpt-5.4 + gpt-5.4-pro.
    assert len(samples) >= 3, (
        f"Expected ≥ 3 llm_config blocks in the example file; got {len(samples)}. "
        "Either the file changed shape or the parser is broken."
    )


def test_every_example_config_model_resolves_to_a_price() -> None:
    text = _example_config_path().read_text(encoding="utf-8")
    samples = _parse_llm_config_blocks(text)

    failures: list[str] = []
    for sample in samples:
        provider = _resolve_provider(sample)
        if provider in UNPRICED_PROVIDER_ALLOWLIST:
            continue
        if (provider, sample.model_name) in UNPRICED_MODEL_ALLOWLIST:
            # Model is intentionally unpriced — handler will stamp
            # ``cost_status=unknown`` (honest) instead of forging an
            # actual cost from a placeholder rate.
            continue
        price = get_price(sample.model_name, provider)
        if price is None:
            failures.append(
                f"  block {sample.block_label}: "
                f"model={sample.model_name!r} provider={provider!r} → no price"
            )

    assert not failures, (
        "Recommended example configurations have no price entry. Either:\n"
        "  - add the model to PRICING_TABLE under the right provider with a "
        "    sourced rate, OR\n"
        "  - add (provider, model) to UNPRICED_MODEL_ALLOWLIST so the "
        "    handler honestly stamps cost_status=unknown, OR\n"
        "  - add the provider to UNPRICED_PROVIDER_ALLOWLIST.\n"
        "Failures:\n"
        + "\n".join(failures)
    )


def test_gpt_5_4_models_are_intentionally_unpriced() -> None:
    """Audit: don't fake ACTUAL with placeholder GPT-5.4 prices.

    Until OpenAI publishes a dated rate card for gpt-5.4 / gpt-5.4-pro,
    these stay in UNPRICED_MODEL_ALLOWLIST so the handler reports
    ``cost_status=unknown`` truthfully.
    """
    assert ("openai_official", "gpt-5.4") in UNPRICED_MODEL_ALLOWLIST
    assert ("openai_official", "gpt-5.4-pro") in UNPRICED_MODEL_ALLOWLIST
    assert get_price("gpt-5.4", "openai_official") is None
    assert get_price("gpt-5.4-pro", "openai_official") is None
