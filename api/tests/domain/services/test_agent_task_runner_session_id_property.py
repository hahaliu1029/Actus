"""B5.5 T1 — AgentTaskRunner exposes a public ``session_id`` read accessor so
the task layer (RedisStreamTask) can bind the observability session context
without reaching into the runner's private ``_session_id`` field.
"""
from app.domain.services.agent_task_runner import AgentTaskRunner


def test_session_id_is_a_public_read_property():
    # The accessor is part of the public contract (a read-only property),
    # not just an attribute the task layer happens to find.
    assert isinstance(AgentTaskRunner.session_id, property)
    assert AgentTaskRunner.session_id.fset is None  # read-only


def test_session_id_property_returns_private_field():
    # object.__new__ avoids the heavy __init__ dependency graph — we only
    # exercise the property accessor, mirroring the existing
    # half-constructed-runner pattern used elsewhere in the suite.
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "sess-xyz"
    assert runner.session_id == "sess-xyz"
