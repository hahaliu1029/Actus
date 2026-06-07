"""SSM ABC must be uninstantiable; subclasses must implement the required methods (9 after A4-1)."""

import pytest

from app.domain.services.session.session_state_machine import SessionStateMachine


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        SessionStateMachine()  # type: ignore[abstract]


def test_subclass_missing_method_cannot_instantiate():
    class Incomplete(SessionStateMachine):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_required_method_names():
    required = {
        "get_mode",
        "get_mode_with_revision",
        "request_takeover",
        "release_takeover",
        "enter_finishing",
        "complete",
        "transition",
        "set_mode",  # A4-1: caller-owned non-terminal status write
        "terminate",  # A4-1: caller-owned terminal status write
    }
    actual = {
        name for name, attr in SessionStateMachine.__dict__.items()
        if getattr(attr, "__isabstractmethod__", False)
    }
    assert required.issubset(actual)
