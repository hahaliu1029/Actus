from __future__ import annotations

from app.domain.models.session import Session, SessionStatus
from app.infrastructure.models.session import SessionModel


def _domain_session(session_id: str, status: SessionStatus) -> Session:
    return Session(id=session_id, user_id="user-1", status=status)


def test_update_from_domain_does_not_overwrite_status() -> None:
    """save()-UPDATE path must NOT transition an existing row's status.

    A4-1 §4: `status` is excluded from update_from_domain so the only legal
    status writes are the three repo mutators (SSM-owned). `mode_revision` is
    ORM-only (absent from the domain Session) so save() never bumped the race
    fence either — neutralization removes a latent lost-update.
    """
    base = _domain_session("s1", SessionStatus.RUNNING)
    model = SessionModel.from_domain(base)
    # INSERT path legitimately sets the genesis status.
    assert model.status == SessionStatus.RUNNING.value

    # A domain Session carrying a DIFFERENT status must NOT mutate the row.
    changed = _domain_session("s1", SessionStatus.COMPLETED)
    model.update_from_domain(changed)
    assert model.status == SessionStatus.RUNNING.value
