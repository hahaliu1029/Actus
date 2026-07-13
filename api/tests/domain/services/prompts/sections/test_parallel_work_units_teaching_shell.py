import pytest

from app.domain.services.prompts.sections import (
    parallel_work_units_teaching as teach_mod,
)
from app.domain.services.prompts.sections.parallel_work_units_teaching import (
    parallel_work_units_teaching_section,
)

_SHELL_FIVE = (
    "shell_execute", "shell_wait_process", "shell_kill_process",
    "shell_write_input", "shell_read_output",
)


class _Ctx:
    def __init__(self, lang="en", parallel_dispatch_allowed=True):
        self.lang = lang
        # [child-pwu fix] mirror RenderContext's default so the stub keeps
        # satisfying the section contract (render gates on this attr).
        self.parallel_dispatch_allowed = parallel_dispatch_allowed


def _render(ctx):
    return parallel_work_units_teaching_section.render(ctx)


def test_coordinator_off_emits_nothing(monkeypatch):
    monkeypatch.setattr(teach_mod, "is_coordinator_enabled", lambda: False)
    monkeypatch.setattr(teach_mod, "is_coordinator_shell_mode_enabled",
                        lambda: True)
    out = _render(_Ctx("en"))
    assert out.text is None


def test_coordinator_on_shell_off_typed_only(monkeypatch):
    monkeypatch.setattr(teach_mod, "is_coordinator_enabled", lambda: True)
    monkeypatch.setattr(teach_mod, "is_coordinator_shell_mode_enabled",
                        lambda: False)
    out = _render(_Ctx("en"))
    assert out.text is not None
    assert "proposed_paths" in out.text
    assert "shell_mode" not in out.text
    assert "proposed_trees" not in out.text


def test_both_flags_on_teaches_shell_tree_en(monkeypatch):
    monkeypatch.setattr(teach_mod, "is_coordinator_enabled", lambda: True)
    monkeypatch.setattr(teach_mod, "is_coordinator_shell_mode_enabled",
                        lambda: True)
    out = _render(_Ctx("en"))
    assert out.text is not None
    assert "shell_mode" in out.text
    assert "proposed_trees" in out.text
    assert "shell_execute" in out.text


def test_both_flags_on_teaches_zh(monkeypatch):
    monkeypatch.setattr(teach_mod, "is_coordinator_enabled", lambda: True)
    monkeypatch.setattr(teach_mod, "is_coordinator_shell_mode_enabled",
                        lambda: True)
    out = _render(_Ctx("zh"))
    assert out.text is not None
    assert "shell_mode" in out.text
    assert "proposed_trees" in out.text


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_shell_teaching_lists_all_five_shell_tools(monkeypatch, lang):
    # Gate step-1 requires every CALLED tool to be in allowed_tools; if the
    # teaching's example omits any of the 5 shell tools, a flag-on provider
    # learns an allowlist that bounces shell_wait_process / shell_read_output /
    # shell_write_input / shell_kill_process with OUT_OF_TOOL_ALLOWLIST at call.
    # Lock that all 5 names appear in the shell teaching (so the example +
    # the REQUIRED instruction both name them).
    monkeypatch.setattr(teach_mod, "is_coordinator_enabled", lambda: True)
    monkeypatch.setattr(teach_mod, "is_coordinator_shell_mode_enabled",
                        lambda: True)
    out = _render(_Ctx(lang))
    assert out.text is not None
    for tool in _SHELL_FIVE:
        assert tool in out.text, f"{tool} missing from {lang} shell teaching"
