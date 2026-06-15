"""[C2 PR-1 Task 1.10] Planner prompt teaching for parallel_work_units.

Verifies that the bilingual teaching constants are defined and contain
the spec keywords the planner needs to learn (the actual prompt
injection into the assembled planner prompt is deferred — see the
teaching constants' docstrings).
"""
from __future__ import annotations


def test_en_bundle_has_teaching():
    from app.domain.services.prompts.sections.parallel_work_units_teaching import (
        PARALLEL_WORK_UNITS_TEACHING_EN,
    )

    assert "parallel_work_units" in PARALLEL_WORK_UNITS_TEACHING_EN
    assert "exploration" in PARALLEL_WORK_UNITS_TEACHING_EN


def test_zh_bundle_has_teaching():
    from app.domain.services.prompts.sections.parallel_work_units_teaching import (
        PARALLEL_WORK_UNITS_TEACHING_ZH,
    )

    assert "parallel_work_units" in PARALLEL_WORK_UNITS_TEACHING_ZH
