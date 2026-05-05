"""ContextAssembler emits record_decision('context_assembler_trim', outcome='phase_N')
after each phase that produced a change. NO attrs per [R3-P2-6].

P1.2 fix: ContextAssembler now accepts an injected ``decision_recorder`` callable
rather than importing ``record_decision`` directly from infrastructure. Tests pass
the recorder as a mock so there is no OTel dependency in the domain test path.
"""
import pytest
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from app.domain.services.graphs.context_assembler import ContextAssembler
from app.domain.services.graphs.token_estimator import TokenEstimator


@pytest.fixture
def messages_above_threshold():
    """Build messages large enough that ContextAssembler triggers Phase 1+2+3."""
    return [
        SystemMessage(content="sys"),
        # Add enough content to trigger trimming
        HumanMessage(content="x" * 5000),
        AIMessage(
            content="x" * 5000,
            tool_calls=[{"id": "c1", "name": "f", "args": {}}],
        ),
        ToolMessage(content="x" * 5000, tool_call_id="c1", name="f"),
        HumanMessage(content="x" * 5000),
        AIMessage(content="x" * 5000),
    ]


def test_assembler_emits_decision_for_each_active_phase(messages_above_threshold):
    """ContextAssembler should emit record_decision per active phase. NO extra attrs.

    P1.2: recorder is injected at construction time; the test passes a MagicMock
    directly — no monkeypatching of the infrastructure module required.
    """
    mock = MagicMock()
    estimator = TokenEstimator()
    assembler = ContextAssembler(
        estimator=estimator,
        effective_window=2000,  # Tight budget to force all phases
        decision_recorder=mock,
    )

    result = assembler.assemble(messages_above_threshold)

    # Verify record_decision was called
    assert mock.called, "record_decision was never called"

    # Extract calls for 'context_assembler_trim'
    trim_calls = [
        c for c in mock.call_args_list if c.args and c.args[0] == "context_assembler_trim"
    ]
    assert len(trim_calls) > 0, "No 'context_assembler_trim' decisions recorded"

    # Lock the exact set of phases the fixture is expected to trigger so that a
    # future regression that silently drops phase_2 or phase_3 emit is caught.
    # The fixture (messages_above_threshold) is engineered to hit ALL THREE
    # phases (compress tool outputs + drop tool_call groups + drop oldest turns).
    outcomes = [c.kwargs.get("outcome") for c in trim_calls]
    assert all(o and o.startswith("phase_") for o in outcomes), (
        f"Not all outcomes start with 'phase_': {outcomes}"
    )
    assert set(outcomes) == {"phase_1", "phase_2", "phase_3"}, (
        f"Expected all three phases to fire, got {sorted(set(outcomes))}"
    )

    # Ensure NO attrs were passed beyond outcome [R3-P2-6]
    for c in trim_calls:
        kwargs_keys = set(c.kwargs.keys())
        assert kwargs_keys <= {"outcome"}, (
            f"Unexpected kwargs keys: {kwargs_keys - {'outcome'}}"
        )

    # The result should have messages (possibly trimmed)
    assert result.messages is not None
    assert result.original_tokens > 0


def test_assembler_noop_when_no_recorder(messages_above_threshold):
    """ContextAssembler works correctly with no recorder injected (None = no-op)."""
    estimator = TokenEstimator()
    assembler = ContextAssembler(
        estimator=estimator,
        effective_window=2000,
        decision_recorder=None,
    )
    # Should not raise even though decision_recorder is None
    result = assembler.assemble(messages_above_threshold)
    assert result.messages is not None
    assert result.original_tokens > 0
