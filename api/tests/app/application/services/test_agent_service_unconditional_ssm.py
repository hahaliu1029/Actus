from __future__ import annotations

from app.application.services.agent_service import AgentService


def test_agent_service_builds_unconditional_ssm(monkeypatch) -> None:
    """`self._ssm` is built independent of the PE/confirmation flag.

    A4-1 §6: SSM must be present on every production write path. The build is
    extracted into _build_unconditional_ssm() (called by __init__) so it is
    never None regardless of the tool-confirmation master switch.
    """
    captured: dict[str, object] = {}
    sentinel = object()

    def _fake_build(*, uow_factory, redis=None):
        captured["uow_factory"] = uow_factory
        captured["redis"] = redis
        return sentinel

    import app.application.composition.graph_assembly as ga
    monkeypatch.setattr(ga, "build_session_state_machine", _fake_build)

    svc = object.__new__(AgentService)
    svc._uow_factory = lambda: None
    svc._redis_client = None
    svc._build_unconditional_ssm()

    assert svc._ssm is sentinel
    assert captured["uow_factory"] is svc._uow_factory
    assert captured["redis"] is None
