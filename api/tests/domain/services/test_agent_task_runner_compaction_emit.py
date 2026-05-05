"""Test agent_task_runner threads compaction_id from CompactionResult into the yielded CompactionEvent."""
import pytest
from unittest.mock import MagicMock

from app.domain.models.event import CompactionEvent, ContextStatusEvent
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.graphs.compaction import CompactionResult


def test_build_compaction_events_yields_compaction_event_with_id():
    """[P1.1 fix] _build_compaction_events_if_any returns CompactionEvent with the compaction_id."""
    runner = AgentTaskRunner.__new__(AgentTaskRunner)  # bypass __init__

    # Stub the flow with the minimal fields the helper reads
    flow_mock = MagicMock()
    flow_mock._last_compaction_result = CompactionResult(
        messages=(), level_applied=2, tokens_before=1000, tokens_after=500,
        summary_injected=True, messages_removed=10, usage_ratio_after=0.5,
        summary_text="s", operations=[{"kind": "llm_summary"}],
        compaction_id="aabbccddeeff0011",
    )
    flow_mock._overflow_config = None  # use defaults
    flow_mock._execution_metrics = None  # skip D5 path
    runner._flow = flow_mock

    events = runner._build_compaction_events_if_any()

    # ContextStatusEvent always emitted; CompactionEvent only when level_applied > 0
    assert any(isinstance(e, ContextStatusEvent) for e in events)
    compaction_evs = [e for e in events if isinstance(e, CompactionEvent)]
    assert len(compaction_evs) == 1
    assert compaction_evs[0].compaction_id == "aabbccddeeff0011"
    assert compaction_evs[0].level == 2

    # Helper is idempotent — subsequent call returns []
    assert runner._build_compaction_events_if_any() == []


def test_build_compaction_events_no_op_when_no_compaction():
    """Helper returns [] when _last_compaction_result is None (idempotent)."""
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    flow_mock = MagicMock()
    flow_mock._last_compaction_result = None
    runner._flow = flow_mock
    assert runner._build_compaction_events_if_any() == []


def test_build_compaction_events_skips_compaction_event_when_level_zero():
    """level_applied=0 => ContextStatusEvent only, no CompactionEvent."""
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    flow_mock = MagicMock()
    flow_mock._last_compaction_result = CompactionResult(
        messages=(), level_applied=0, tokens_before=100, tokens_after=100,
        summary_injected=False, messages_removed=0, usage_ratio_after=0.1,
    )
    flow_mock._overflow_config = None
    flow_mock._execution_metrics = None
    runner._flow = flow_mock
    events = runner._build_compaction_events_if_any()
    assert any(isinstance(e, ContextStatusEvent) for e in events)
    assert not any(isinstance(e, CompactionEvent) for e in events)
