"""Smoke tests for R2 ReactGraphState field additions (Task 8).

These tests verify the presence of 5 new fields added to ReactGraphState
for the CS2 tool_node split (exactly-once prefix closure + interrupt bridge).
They are field-presence smoke checks only — behavioral coverage comes from
the Day-4 hard-gate tests (Tasks 16-19) in test_react_graph_node_split.py.
"""
from __future__ import annotations

from app.domain.services.graphs.state import ReactGraphState


def test_react_state_has_completed_tool_call_prefix():
    assert "completed_tool_call_prefix" in ReactGraphState.__annotations__


def test_react_state_has_approved_tool_call_ids():
    assert "approved_tool_call_ids" in ReactGraphState.__annotations__


def test_react_state_has_pending_ask_outcome():
    assert "pending_ask_outcome" in ReactGraphState.__annotations__


def test_react_state_has_pending_ask_tool_call_id():
    assert "pending_ask_tool_call_id" in ReactGraphState.__annotations__


def test_react_state_has_pending_ask_artifact():
    assert "pending_ask_artifact" in ReactGraphState.__annotations__


def test_react_state_has_pending_ask_tool_args():
    """Original tool_call.args preserved for the deny audit path."""
    assert "pending_ask_tool_args" in ReactGraphState.__annotations__
