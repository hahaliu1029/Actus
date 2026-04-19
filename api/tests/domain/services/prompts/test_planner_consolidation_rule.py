"""Fix B: planner prompts must include a hard constraint that consolidation
/ summarization steps reuse prior outputs instead of re-running searches.

Context: same run trace as Fix A. Even after Fix A exposes
``prior_step_outputs`` to the executor, a "整理汇总" step still chose to
issue 29 fresh ``search_web`` calls because the planner's step description
did not forbid it. This test pins the rule's presence in both CREATE and
UPDATE planner prompts, in both ZH and EN bundles.
"""
from __future__ import annotations

import pytest

from app.domain.services.prompts import get_prompt_bundle


ZH_MARKER_SNIPPETS = (
    "汇总",
    "整理",
    "前序",
)

EN_MARKER_SNIPPETS = (
    "prior",
    "reuse",
)


class TestPlannerReuseConstraint:
    @pytest.mark.parametrize("prompt_attr", ["CREATE_PLAN_PROMPT", "UPDATE_PLAN_PROMPT"])
    def test_zh_has_reuse_constraint(self, prompt_attr: str) -> None:
        bundle = get_prompt_bundle("zh")
        prompt = getattr(bundle, prompt_attr)
        for snippet in ZH_MARKER_SNIPPETS:
            assert snippet in prompt, (
                f"ZH {prompt_attr} must mention '{snippet}' to describe the "
                f"consolidation / reuse constraint"
            )
        # Must also explicitly forbid redundant search for consolidation-type steps.
        assert ("禁止" in prompt) or ("不得" in prompt) or ("避免" in prompt), (
            f"ZH {prompt_attr} must include a hard constraint keyword (禁止/不得/避免) "
            f"so executors don't re-issue searches"
        )

    @pytest.mark.parametrize("prompt_attr", ["CREATE_PLAN_PROMPT", "UPDATE_PLAN_PROMPT"])
    def test_en_has_reuse_constraint(self, prompt_attr: str) -> None:
        bundle = get_prompt_bundle("en")
        prompt = getattr(bundle, prompt_attr)
        lowered = prompt.lower()
        for snippet in EN_MARKER_SNIPPETS:
            assert snippet in lowered, (
                f"EN {prompt_attr} must mention '{snippet}' (case-insensitive) "
                f"to describe the consolidation / reuse constraint"
            )
        # Must forbid redundant searches via an explicit negative (must
        # not / forbidden / avoid).
        assert (
            "must not" in lowered
            or "forbidden" in lowered
            or "avoid" in lowered
        ), (
            f"EN {prompt_attr} must include a hard constraint keyword "
            f"(must not / forbidden / avoid) so executors don't re-issue searches"
        )
