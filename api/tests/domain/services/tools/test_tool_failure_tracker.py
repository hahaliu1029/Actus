"""Tests for ToolFailureTracker."""

import pytest

from app.domain.services.tools.tool_failure_tracker import ToolFailureTracker


class TestSignature:
    def test_same_args_same_hash(self):
        sig1 = ToolFailureTracker._signature("tool_a", {"key": "val"})
        sig2 = ToolFailureTracker._signature("tool_a", {"key": "val"})
        assert sig1 == sig2

    def test_different_order_same_hash(self):
        sig1 = ToolFailureTracker._signature("tool_a", {"a": 1, "b": 2})
        sig2 = ToolFailureTracker._signature("tool_a", {"b": 2, "a": 1})
        assert sig1 == sig2

    def test_different_args_different_hash(self):
        sig1 = ToolFailureTracker._signature("tool_a", {"key": "val1"})
        sig2 = ToolFailureTracker._signature("tool_a", {"key": "val2"})
        assert sig1 != sig2

    def test_different_tools_different_hash(self):
        sig1 = ToolFailureTracker._signature("tool_a", {"key": "val"})
        sig2 = ToolFailureTracker._signature("tool_b", {"key": "val"})
        assert sig1 != sig2

    def test_empty_args(self):
        sig = ToolFailureTracker._signature("tool_a", {})
        assert sig.startswith("tool_a:")
        assert len(sig) > len("tool_a:")

    def test_non_serializable_fallback(self):
        """Non-serializable args should fall back to tool_name only."""
        class Circular:
            pass
        obj = Circular()
        obj.self_ref = obj
        # default=str should handle this, but if not, try/except catches
        sig = ToolFailureTracker._signature("tool_a", {"obj": obj})
        assert "tool_a" in sig


class TestRecordFailure:
    def test_first_failure_not_blocked(self):
        t = ToolFailureTracker(max_same_failures=3)
        assert t.record_failure("tool_a", {"x": 1}) is False
        assert not t.is_blocked("tool_a", {"x": 1})

    def test_nth_failure_blocks(self):
        t = ToolFailureTracker(max_same_failures=3)
        t.record_failure("tool_a", {"x": 1})
        t.record_failure("tool_a", {"x": 1})
        assert t.record_failure("tool_a", {"x": 1}) is True
        assert t.is_blocked("tool_a", {"x": 1})

    def test_different_signature_independent(self):
        t = ToolFailureTracker(max_same_failures=2)
        t.record_failure("tool_a", {"x": 1})
        t.record_failure("tool_a", {"x": 1})
        assert t.is_blocked("tool_a", {"x": 1})
        assert not t.is_blocked("tool_a", {"x": 2})


class TestRecordSuccess:
    def test_resets_failure_count(self):
        t = ToolFailureTracker(max_same_failures=3)
        t.record_failure("tool_a", {"x": 1})
        t.record_failure("tool_a", {"x": 1})
        t.record_success("tool_a", {"x": 1})
        # After success, 2 more failures needed to block
        assert not t.is_blocked("tool_a", {"x": 1})
        t.record_failure("tool_a", {"x": 1})
        assert not t.is_blocked("tool_a", {"x": 1})

    def test_removes_from_blocked(self):
        t = ToolFailureTracker(max_same_failures=1)
        t.record_failure("tool_a", {"x": 1})
        assert t.is_blocked("tool_a", {"x": 1})
        t.record_success("tool_a", {"x": 1})
        assert not t.is_blocked("tool_a", {"x": 1})


class TestGetBlockedSummary:
    def test_empty_when_none_blocked(self):
        t = ToolFailureTracker()
        assert t.get_blocked_summary() == ""

    def test_has_content_when_blocked(self):
        t = ToolFailureTracker(max_same_failures=1)
        t.record_failure("tool_a", {"x": 1})
        summary = t.get_blocked_summary()
        assert "tool_a" in summary
        assert len(summary) > 0


class TestResetBlocked:
    def test_clears_blocked_preserves_failure_map(self):
        t = ToolFailureTracker(max_same_failures=2)
        t.record_failure("tool_a", {"x": 1})
        t.record_failure("tool_a", {"x": 1})
        assert t.is_blocked("tool_a", {"x": 1})
        t.reset_blocked()
        assert not t.is_blocked("tool_a", {"x": 1})
        # failure_map preserved → one more failure re-blocks
        # (failure_map has count=2 still, next failure → count=3 ≥ 2 → blocked again)
        t.record_failure("tool_a", {"x": 1})
        assert t.is_blocked("tool_a", {"x": 1})
