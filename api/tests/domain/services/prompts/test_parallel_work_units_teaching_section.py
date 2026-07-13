"""[C2b rollout WS0] Unit tests for the flag-gated parallel_work_units teaching section."""
from __future__ import annotations

import pytest

from app.domain.services.prompts.section import MINIMAL_MODE_ALLOWLIST, RenderContext
from app.domain.services.prompts.sections.parallel_work_units_teaching import (
    parallel_work_units_teaching_section,
)

_FLAG = "ACTUS_C2_COORDINATOR_ENABLED"


def test_flag_off_renders_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    out = parallel_work_units_teaching_section.render(RenderContext(lang="zh"))
    assert out.text is None


def test_flag_on_renders_zh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "true")
    out = parallel_work_units_teaching_section.render(RenderContext(lang="zh"))
    assert out.text is not None
    assert "parallel_work_units" in out.text
    assert "并行工作单元" in out.text  # ZH marker
    assert out.metadata.get("coordinator_teaching") is True


def test_flag_on_renders_en(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "true")
    out = parallel_work_units_teaching_section.render(RenderContext(lang="en"))
    assert out.text is not None
    assert "parallel_work_units" in out.text
    assert "Parallel Work Units" in out.text  # EN marker


def test_flag_on_teaches_directory_qualified_paths_en(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[single-path contract] The planner teaching must instruct directory-
    qualified workspace-relative proposed_paths (no bare filename) so a real
    provider doesn't propose ``part_a.md`` that fails the lease contract."""
    monkeypatch.setenv(_FLAG, "true")
    out = parallel_work_units_teaching_section.render(RenderContext(lang="en"))
    assert out.text is not None
    assert "directory" in out.text.lower()
    assert "workspace/" in out.text  # the new-file convention example


def test_flag_on_teaches_directory_qualified_paths_zh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_FLAG, "true")
    out = parallel_work_units_teaching_section.render(RenderContext(lang="zh"))
    assert out.text is not None
    assert "目录" in out.text  # 带目录路径
    assert "workspace/" in out.text


def test_section_metadata() -> None:
    s = parallel_work_units_teaching_section
    assert s.id == "parallel_work_units_teaching"
    assert s.priority == 8
    assert s.cacheable is False
    assert s.dynamic is True


def test_not_in_minimal_allowlist() -> None:
    # Children must NOT learn parallel dispatch (they are leaf workers).
    assert "parallel_work_units_teaching" not in MINIMAL_MODE_ALLOWLIST


import os
import subprocess
import sys


def test_flag_on_bundle_import_does_not_raise_section_validation() -> None:
    """[R2 P1#2] Under flag-ON, SectionRegistry.__post_init__ eager-renders the
    teaching against _FIXTURE_CTX at bundle import. The teaching text must
    validate (no dangling skill-tool refs) — pin it via a clean subprocess
    import with the flag set BEFORE any prompt module loads."""
    code = (
        "import app.domain.services.prompts.bundles.en as en;"
        "import app.domain.services.prompts.bundles.zh as zh;"
        "assert any(s.id == 'parallel_work_units_teaching' "
        "for s in en.EN_PLANNER_REGISTRY.sections);"
        "print('OK')"
    )
    env = dict(os.environ)
    env["ACTUS_C2_COORDINATOR_ENABLED"] = "true"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"import raised under flag-on:\n{result.stderr}"
    assert "OK" in result.stdout


def _assemble_planner_prompt(ctx: RenderContext | None = None):
    from app.domain.services.graphs.token_estimator import TokenEstimator
    from app.domain.services.prompts.assembler import PromptAssembler
    from app.domain.services.prompts.budget import SystemPromptBudget
    from app.domain.services.prompts.bundles.en import EN_PLANNER_REGISTRY
    from app.domain.services.prompts.section import PromptMode

    assembler = PromptAssembler(
        budget=SystemPromptBudget(max_tokens=10000),
        token_estimator=TokenEstimator(),
    )
    return assembler.assemble(
        EN_PLANNER_REGISTRY, ctx or RenderContext(lang="en"), PromptMode.FULL
    )


def test_flag_off_assembled_planner_prompt_omits_teaching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    result = _assemble_planner_prompt()
    assert "parallel_work_units" not in result.text
    assert "parallel_work_units_teaching" not in result.sections_included


def test_flag_on_assembled_planner_prompt_includes_teaching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_FLAG, "true")
    result = _assemble_planner_prompt()
    assert "parallel_work_units" in result.text
    assert "parallel_work_units_teaching" in result.sections_included


def test_flag_on_child_ctx_renders_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """[child-pwu fix] A coordinator child cannot dispatch parallel work
    units (no subgraph wired, depth cap = 1) — teaching it the schema only
    makes its planner emit steps that get stripped at the parse boundary.
    ctx.parallel_dispatch_allowed=False must suppress the section even
    flag-ON."""
    monkeypatch.setenv(_FLAG, "true")
    out = parallel_work_units_teaching_section.render(
        RenderContext(lang="zh", parallel_dispatch_allowed=False)
    )
    assert out.text is None


def test_flag_on_child_assembled_planner_prompt_omits_teaching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[child-pwu fix] End-to-end: a child-shaped RenderContext keeps the
    teaching out of the assembled planner prompt."""
    monkeypatch.setenv(_FLAG, "true")
    result = _assemble_planner_prompt(
        RenderContext(lang="en", parallel_dispatch_allowed=False)
    )
    assert "parallel_work_units_teaching" not in result.sections_included


def test_teaching_covers_user_named_absolute_root_paths_en(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[dispatch-fallback fix] Live 2026-07-13: user asked for
    /home/ubuntu/part_*.md, planner copied the absolute paths verbatim and
    dispatch rejected pre-spawn. Teach the produce-then-move pattern
    explicitly so the planner keeps parallelism AND satisfies the user's
    literal target location."""
    monkeypatch.setenv(_FLAG, "true")
    out = parallel_work_units_teaching_section.render(RenderContext(lang="en"))
    assert out.text is not None
    assert "absolute" in out.text.lower()
    assert "move" in out.text.lower()


def test_teaching_covers_user_named_absolute_root_paths_zh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_FLAG, "true")
    out = parallel_work_units_teaching_section.render(RenderContext(lang="zh"))
    assert out.text is not None
    assert "绝对路径" in out.text
    assert "移动" in out.text
