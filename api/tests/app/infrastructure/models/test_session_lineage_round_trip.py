"""S3 PR-1: depth/root_session_id round-trip through the ORM mapping."""
from __future__ import annotations

from datetime import datetime

from app.domain.models.session import Session
from app.infrastructure.models.session import SessionModel


def _hydrate_timestamps(model: SessionModel) -> SessionModel:
    """Pin created_at/updated_at on a never-flushed ORM instance.

    ``SessionModel.created_at`` / ``updated_at`` carry only a SQL
    ``server_default``; without a real DB roundtrip they are ``None`` and
    ``Session.model_validate(orm, from_attributes=True)`` (inside
    ``to_domain``) fails the non-optional datetime check. Mirrors the
    established ``_hydrate_orm`` helper in
    ``tests/app/application/services/test_agent_service_t12_resume.py``.
    """
    now = datetime.now()
    model.created_at = now
    model.updated_at = now
    return model


def test_domain_session_lineage_defaults():
    s = Session(title="t", user_id="u1")
    assert s.depth == 0
    assert s.root_session_id is None


def test_child_lineage_round_trips_through_orm():
    s = Session(
        id="c",
        parent_session_id="root",
        worker_type="subagent",
        depth=1,
        root_session_id="root",
        user_id="u1",
        tool_filter_preset="subagent_research",
    )
    model = _hydrate_timestamps(SessionModel.from_domain(s))
    assert model.depth == 1
    assert model.root_session_id == "root"

    back = model.to_domain()
    assert back.depth == 1
    assert back.root_session_id == "root"


def test_root_lineage_round_trips_through_orm():
    s = Session(id="r", worker_type="root", depth=0, root_session_id=None, user_id="u1")
    model = _hydrate_timestamps(SessionModel.from_domain(s))
    assert model.depth == 0
    assert model.root_session_id is None
    back = model.to_domain()
    assert back.depth == 0
    assert back.root_session_id is None
