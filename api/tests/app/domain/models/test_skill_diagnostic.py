"""SkillDiagnostic domain 模型（B9 PR-0，spec §8 P-1 钉子）。"""
from datetime import datetime

import pytest

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.skill_diagnostic import SkillDiagnostic


def _make_skill() -> Skill:
    return Skill(
        id="s1", slug="s1", name="S1", description="", version="0.1.0",
        source_type=SkillSourceType.LOCAL, source_ref="ref",
        runtime_type=SkillRuntimeType.NATIVE, manifest={},
        enabled=True, installed_by=None,
        created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1),
    )


def test_ok_diagnostic_carries_skill():
    diag = SkillDiagnostic(skill_key="s1", ok=True, skill=_make_skill())
    assert diag.ok is True
    assert diag.error_code is None
    assert diag.relative_file is None
    assert diag.skill is not None


def test_error_diagnostic_shape():
    diag = SkillDiagnostic(
        skill_key="broken-dir", ok=False,
        error_code="parse_error", relative_file="meta.json",
    )
    assert diag.skill is None
    assert diag.error_code == "parse_error"
    assert diag.relative_file == "meta.json"


def test_frozen():
    diag = SkillDiagnostic(skill_key="x", ok=False, error_code="missing_meta")
    with pytest.raises(Exception):
        diag.ok = True  # type: ignore[misc]


def test_abc_declares_list_with_diagnostics():
    from app.domain.repositories.skill_repository import SkillRepository
    assert "list_with_diagnostics" in SkillRepository.__abstractmethods__
