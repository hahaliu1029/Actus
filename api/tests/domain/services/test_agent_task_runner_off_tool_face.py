"""SPM Task 24 — runner-side off-mode tool-face contraction.

Drives the two thin helpers directly on a ``__new__``-bypass runner (repo's
established runner unit-test convention):
- ``_apply_off_native_skill_filter`` — off removes NATIVE skills, keeps mcp/a2a.
- ``_get_native_tool_names_by_category`` — off summary has zero sandbox
  categories (file / shell / browser / file_view).
- ``_sandbox_provision_off`` — defensive False when settings unavailable.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.services.agent_task_runner import AgentTaskRunner


def _skill(slug: str, runtime: SkillRuntimeType) -> Skill:
    return Skill(
        slug=slug,
        name=slug.upper(),
        source_type=SkillSourceType.LOCAL,
        source_ref=f"ref-{slug}",
        runtime_type=runtime,
    )


def _bare_runner() -> AgentTaskRunner:
    return object.__new__(AgentTaskRunner)


class TestNativeSkillPoolFilter:
    def test_off_pool_excludes_native_keeps_mcp_a2a(self) -> None:
        runner = _bare_runner()
        runner._sandbox_provision_off = lambda: True  # type: ignore[method-assign]
        pool = [
            _skill("nat", SkillRuntimeType.NATIVE),
            _skill("mcp", SkillRuntimeType.MCP),
            _skill("a2a", SkillRuntimeType.A2A),
        ]
        out = runner._apply_off_native_skill_filter(pool)
        assert [s.slug for s in out] == ["mcp", "a2a"]

    def test_non_off_pool_is_identity(self) -> None:
        runner = _bare_runner()
        runner._sandbox_provision_off = lambda: False  # type: ignore[method-assign]
        pool = [
            _skill("nat", SkillRuntimeType.NATIVE),
            _skill("mcp", SkillRuntimeType.MCP),
        ]
        # byte-zero: same list object returned when not off.
        assert runner._apply_off_native_skill_filter(pool) is pool


class TestNativeToolCategorySummary:
    def _summary_runner(self, *, off: bool) -> AgentTaskRunner:
        runner = _bare_runner()
        runner._sandbox_accessor = MagicMock()
        runner._browser_accessor = MagicMock()
        runner._search_engine = MagicMock()
        runner._file_processor_lookup = None
        runner._supports_vision = True
        runner._supports_pdf_input = False
        runner._build_memory_mount_scope = lambda: None  # type: ignore[method-assign]
        runner._sandbox_provision_off = lambda: off  # type: ignore[method-assign]
        return runner

    def test_off_summary_has_zero_sandbox_categories(self) -> None:
        runner = self._summary_runner(off=True)
        groups = runner._get_native_tool_names_by_category()
        assert "shell" not in groups
        assert "file" not in groups
        assert "browser" not in groups
        # message + search survive.
        assert "message" in groups
        assert "search" in groups
        # no empty-title residue: every present category is non-empty.
        assert all(names for names in groups.values())

    def test_non_off_summary_includes_sandbox_categories(self) -> None:
        runner = self._summary_runner(off=False)
        groups = runner._get_native_tool_names_by_category()
        assert "shell" in groups
        assert "file" in groups
        assert "browser" in groups
        assert "message" in groups
        assert "search" in groups


class TestSandboxProvisionOffHelper:
    def test_defensive_false_when_settings_unavailable(self, monkeypatch) -> None:
        import core.config as cfg

        def _boom() -> None:
            raise RuntimeError("no settings in test env")

        monkeypatch.setattr(cfg, "get_settings", _boom)
        runner = _bare_runner()
        assert runner._sandbox_provision_off() is False
