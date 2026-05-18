"""PE-1 §3.1 Round 3 P1#5 — SkillTool.resolve_skill_dir public port contract."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.services.tools.skill import SkillTool


def _make_skill(
    *,
    skill_id: str = "sk1",
    scan_report: dict | None = None,
    trust_origin: str = "user_installed",
) -> Skill:
    """Build a minimally-valid Skill satisfying Pydantic schema.

    Skill requires ``slug`` / ``source_type`` / ``source_ref`` in addition to
    the fields the T5 contract exercises. Pass-through values are chosen to
    be irrelevant to resolve_skill_dir's behavior — only ``id`` matters.
    """
    return Skill(
        id=skill_id,
        slug="t5-fixture",
        name="s",
        source_type=SkillSourceType.LOCAL,
        source_ref="local:t5",
        manifest={},
        runtime_type=SkillRuntimeType.NATIVE,
        scan_report=scan_report,
        trust_origin=trust_origin,
        enabled=True,
    )


def _make_skilltool() -> SkillTool:
    sandbox = AsyncMock()
    mcp = MagicMock()
    a2a = MagicMock()
    return SkillTool(sandbox=sandbox, mcp_tool=mcp, a2a_tool=a2a)


class TestResolveSkillDirContract:
    def test_method_exists(self):
        assert hasattr(SkillTool, "resolve_skill_dir")
        assert callable(SkillTool.resolve_skill_dir)

    def test_signature_takes_only_tool_name(self):
        sig = inspect.signature(SkillTool.resolve_skill_dir)
        params = list(sig.parameters)
        assert params == ["self", "tool_name"]
        # skill.py uses ``from __future__ import annotations`` (PEP 563) so
        # signature annotations are stored as strings — resolve via
        # get_type_hints to compare the actual type.
        hints = get_type_hints(SkillTool.resolve_skill_dir)
        assert hints["tool_name"] is str

    def test_return_annotation_is_path_or_none(self):
        hints = get_type_hints(SkillTool.resolve_skill_dir)
        # Expect Optional[Path] (str → Path | None)
        ret = hints["return"]
        assert ret == (Path | None) or repr(ret) in {
            "Path | None",
            "Optional[Path]",
            "typing.Optional[pathlib.Path]",
        }


class TestResolveSkillDirBehavior:
    def test_returns_none_when_binding_missing(self):
        t = _make_skilltool()
        assert t.resolve_skill_dir("nope") is None

    def test_returns_none_when_no_bundle_sync_manager(self):
        t = _make_skilltool()
        # Inject a binding but no bundle manager
        skill = _make_skill(
            skill_id="sk1",
            scan_report={"verdict": "safe", "content_hash": "h"},
            trust_origin="user_installed",
        )
        t._tool_bindings = {
            "t1": {
                "skill": skill,
                "runtime_type": SkillRuntimeType.NATIVE,
                "manifest_tool": {},
                "final_risk": "low",
                "trust_origin": "user_installed",
                "scan_verdict": "safe",
            }
        }
        assert t._bundle_sync_manager is None
        assert t.resolve_skill_dir("t1") is None

    def test_returns_none_when_skills_root_missing(self):
        t = _make_skilltool()
        mgr = MagicMock()
        mgr._skills_root_dir = None
        t._bundle_sync_manager = mgr
        skill = _make_skill(
            skill_id="sk1",
            scan_report=None,
            trust_origin="user_installed",
        )
        t._tool_bindings = {
            "t1": {
                "skill": skill,
                "runtime_type": SkillRuntimeType.NATIVE,
                "manifest_tool": {},
                "final_risk": "high",
                "trust_origin": "user_installed",
                "scan_verdict": "dangerous",
            }
        }
        assert t.resolve_skill_dir("t1") is None

    def test_returns_path_skills_root_div_skill_id(self, tmp_path):
        t = _make_skilltool()
        mgr = MagicMock()
        mgr._skills_root_dir = tmp_path
        t._bundle_sync_manager = mgr
        skill = _make_skill(
            skill_id="sk_xyz",
            scan_report={"verdict": "safe", "content_hash": "h"},
            trust_origin="builtin",
        )
        t._tool_bindings = {
            "t1": {
                "skill": skill,
                "runtime_type": SkillRuntimeType.NATIVE,
                "manifest_tool": {},
                "final_risk": "low",
                "trust_origin": "builtin",
                "scan_verdict": "safe",
            }
        }
        out = t.resolve_skill_dir("t1")
        assert out == tmp_path / "sk_xyz"

    def test_does_not_require_skill_dir_existence(self, tmp_path):
        """Refresher decides what to do when the path doesn't exist —
        port returns the resolved path regardless."""
        t = _make_skilltool()
        mgr = MagicMock()
        mgr._skills_root_dir = tmp_path / "does_not_exist"
        t._bundle_sync_manager = mgr
        skill = _make_skill(
            skill_id="sk_xyz",
            scan_report=None,
            trust_origin="user_installed",
        )
        t._tool_bindings = {
            "t1": {
                "skill": skill,
                "runtime_type": SkillRuntimeType.NATIVE,
                "manifest_tool": {},
                "final_risk": "high",
                "trust_origin": "user_installed",
                "scan_verdict": "dangerous",
            }
        }
        # skills_root truthy → returns Path even if dir absent
        out = t.resolve_skill_dir("t1")
        assert out == tmp_path / "does_not_exist" / "sk_xyz"
