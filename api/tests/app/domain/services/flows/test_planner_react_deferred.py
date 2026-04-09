"""Test that PlannerReActFlow.invoke() defers persist for normal completion."""
import pytest
import inspect


def test_invoke_finally_checks_should_interrupt():
    """The finally block in invoke() must branch on should_interrupt."""
    from app.domain.services.flows.planner_react import PlannerReActFlow
    source = inspect.getsource(PlannerReActFlow.invoke)
    assert "should_interrupt" in source, \
        "invoke() finally must check bridge.final_state.get('should_interrupt')"


def test_deferred_attributes_exist():
    """PlannerReActFlow must have _deferred_final_state and _deferred_summaries attributes."""
    from app.domain.services.flows.planner_react import PlannerReActFlow
    source = inspect.getsource(PlannerReActFlow.__init__)
    assert "_deferred_final_state" in source, \
        "__init__ must initialize _deferred_final_state"
    assert "_deferred_summaries" in source, \
        "__init__ must initialize _deferred_summaries"


def test_summary_llm_property_exposed():
    """PlannerReActFlow must expose summary_llm as a property."""
    from app.domain.services.flows.planner_react import PlannerReActFlow
    assert isinstance(
        PlannerReActFlow.summary_llm, property
    ), "summary_llm must be a property"
