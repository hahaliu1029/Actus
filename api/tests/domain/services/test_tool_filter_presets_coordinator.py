"""C2 PR-1 Task 1.6/1.7 — coordinator_step preset + Session Literal extension."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.session import Session
from app.domain.services.tool_filter_presets import (
    COORDINATOR_STEP_BASE_ALLOWED_TOOLS,
    TOOL_FILTER_PRESETS,
    resolve_preset,
)


class TestCoordinatorStepPreset:
    def test_includes_reads(self):
        for t in ("search_web", "file_read", "file_list", "file_view"):
            assert t in COORDINATOR_STEP_BASE_ALLOWED_TOOLS

    def test_includes_typed_writes(self):
        for t in ("file_write", "file_str_replace"):
            assert t in COORDINATOR_STEP_BASE_ALLOWED_TOOLS

    def test_excludes_shell(self):
        for t in ("shell_execute", "shell_wait", "shell_kill", "shell_input"):
            assert t not in COORDINATOR_STEP_BASE_ALLOWED_TOOLS

    def test_excludes_browser(self):
        for t in ("browser_view", "browser_navigate"):
            assert t not in COORDINATOR_STEP_BASE_ALLOWED_TOOLS

    def test_excludes_user_interaction(self):
        for t in ("message_ask_user", "message_notify_user"):
            assert t not in COORDINATOR_STEP_BASE_ALLOWED_TOOLS

    def test_excludes_memory_save(self):
        assert "memory_save" not in COORDINATOR_STEP_BASE_ALLOWED_TOOLS
        assert "memory_search" in COORDINATOR_STEP_BASE_ALLOWED_TOOLS
        assert "memory_get" in COORDINATOR_STEP_BASE_ALLOWED_TOOLS

    def test_registered(self):
        assert "coordinator_step" in TOOL_FILTER_PRESETS
        assert TOOL_FILTER_PRESETS["coordinator_step"] is COORDINATOR_STEP_BASE_ALLOWED_TOOLS

    def test_resolve(self):
        assert resolve_preset("coordinator_step") == COORDINATOR_STEP_BASE_ALLOWED_TOOLS

    def test_subagent_research_intact(self):
        assert "subagent_research" in TOOL_FILTER_PRESETS


class TestSessionPresetExtended:
    def test_coordinator_step_accepted(self):
        s = Session(
            id="s1",
            user_id="u1",
            worker_type="subagent",
            parent_session_id="p1",
            tool_filter_preset="coordinator_step",
        )
        assert s.tool_filter_preset == "coordinator_step"

    def test_subagent_research_still_accepted(self):
        s = Session(
            id="s1",
            user_id="u1",
            worker_type="subagent",
            parent_session_id="p1",
            tool_filter_preset="subagent_research",
        )
        assert s.tool_filter_preset == "subagent_research"

    def test_unknown_rejected(self):
        with pytest.raises(ValidationError):
            Session(
                id="s1",
                user_id="u1",
                worker_type="subagent",
                parent_session_id="p1",
                tool_filter_preset="bogus",  # type: ignore[arg-type]
            )
