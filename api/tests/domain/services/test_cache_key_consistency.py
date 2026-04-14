"""Cache key invariant: _last_skill_ids must equal _last_initialized_skill_ids
after any _apply_preselected_skills call.

This invariant is critical because _build_lc_tools_for_step uses
_last_skill_ids as part of its cache key, and the cached tools are
closures over self._skill_tool. If the two fields drift out of sync,
the cache can return tool objects that reference a SkillTool internal
state different from what the cache key implies.

See spec §1.3 (inconsistency mechanism) and §6.1 #3 (cache-key consistency
fix for the 4 bypass sites).
"""
from __future__ import annotations

from typing import Any

import pytest

# Reuse the runner fixture from the apply_preselected_skills test file
from tests.domain.services.test_agent_task_runner_apply_preselected_skills import (
    _build_skill,
    _make_runner,
)


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.parametrize(
    "site_label,skills",
    [
        ("site_1_bootstrap_empty", []),
        ("site_1_bootstrap_one", [_build_skill("init_one")]),
        ("site_2_new_message", [_build_skill("msg_a"), _build_skill("msg_b")]),
        ("site_3_step_lock", [_build_skill("step_x")]),
        ("site_4_unknown_tool_reselect", [_build_skill("reselect_y")]),
        ("transition_to_empty", []),
    ],
)
async def test_cache_key_consistent_after_apply(
    site_label: str,
    skills: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For every bypass site, after _apply_preselected_skills,
    _last_skill_ids must equal _last_initialized_skill_ids.

    The site_label is just for test ID readability — all 4 sites converge
    on the same helper, so testing the helper directly with various inputs
    covers all paths.
    """
    runner = _make_runner(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skill_list, scores=None: f"ctx-{len(skill_list)}",
    )

    await runner._apply_preselected_skills(skills)

    assert runner._last_skill_ids == runner._last_initialized_skill_ids, (
        f"Cache key drift detected for {site_label}: "
        f"_last_skill_ids={runner._last_skill_ids}, "
        f"_last_initialized_skill_ids={runner._last_initialized_skill_ids}"
    )


async def test_cache_key_consistent_across_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Across multiple consecutive applies, the two fields stay in sync."""
    runner = _make_runner(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skill_list, scores=None: f"ctx-{len(skill_list)}",
    )

    sequence = [
        [_build_skill("a"), _build_skill("b")],
        [_build_skill("c")],
        [],
        [_build_skill("d"), _build_skill("e"), _build_skill("f")],
        [_build_skill("a"), _build_skill("b")],
    ]

    for skills in sequence:
        await runner._apply_preselected_skills(skills)
        assert runner._last_skill_ids == runner._last_initialized_skill_ids, (
            f"Drift after applying {[s.id for s in skills]}: "
            f"_last_skill_ids={runner._last_skill_ids}, "
            f"_last_initialized_skill_ids={runner._last_initialized_skill_ids}"
        )
