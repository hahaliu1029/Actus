"""C3 PR-1 — DestroyReason 4 new mailbox-driven values (spec §7.2) +
subagent_control_plane Literal narrowing (codex round 11 P2)."""

import pytest
from pydantic import ValidationError

from app.domain.models.session import DestroyReason, Session


def test_destroy_reason_includes_c3_values():
    members = {r.value for r in DestroyReason}
    assert {
        "subagent_terminal_result",
        "cancel_ack_observed",
        "orphan_timeout",
        "force_terminate",
    }.issubset(members)


def test_destroy_reason_legacy_values_preserved():
    members = {r.value for r in DestroyReason}
    assert {"session_delete", "watchdog_timeout", "reconcile_orphan"}.issubset(members)


def test_destroy_reason_values_fit_string64():
    for r in DestroyReason:
        assert len(r.value) <= 64


def test_subagent_control_plane_accepts_legacy():
    s = Session(subagent_control_plane="legacy")
    assert s.subagent_control_plane == "legacy"


def test_subagent_control_plane_accepts_mailbox():
    s = Session(subagent_control_plane="mailbox")
    assert s.subagent_control_plane == "mailbox"


def test_subagent_control_plane_accepts_none():
    s = Session(subagent_control_plane=None)
    assert s.subagent_control_plane is None


def test_subagent_control_plane_default_is_none():
    """NULL ≡ legacy per R1 P2.3 contract; pre-C3 rows must stay representable."""
    s = Session()
    assert s.subagent_control_plane is None


def test_subagent_control_plane_rejects_typo():
    """Literal narrowing catches typo'd values at the domain boundary so they
    never reach the DB CHECK constraint."""
    with pytest.raises(ValidationError):
        Session(subagent_control_plane="mailbbox")


def test_subagent_control_plane_rejects_arbitrary_string():
    with pytest.raises(ValidationError):
        Session(subagent_control_plane="external_writer_value")


# [C2 PR-1 Task 1.8 Override 5] coordinator lineage fields on domain Session
# (codex round-4 P2 — without these the infrastructure → domain rehydrate path
# silently drops coordinator_run_id / work_unit_id / coordinator_attempts).

def test_coordinator_fields_default_to_empty():
    s = Session()
    assert s.coordinator_run_id is None
    assert s.work_unit_id is None
    assert s.coordinator_attempts == {}


def test_coordinator_fields_roundtrip_via_from_attributes():
    """The ORM → domain rehydrate path uses `model_validate(orm, from_attributes=True)`.
    Mimic that with a stub object so the regression test does not need a DB.
    """

    class _OrmLike:
        id = "sess-c2-roundtrip"
        coordinator_run_id = "sess-c2-roundtrip:abcd1234abcd1234:a1"
        work_unit_id = "abcd1234abcd1234.a1.0"
        coordinator_attempts = {"step_login_01": 2}

    s = Session.model_validate(_OrmLike(), from_attributes=True)
    assert s.coordinator_run_id == "sess-c2-roundtrip:abcd1234abcd1234:a1"
    assert s.work_unit_id == "abcd1234abcd1234.a1.0"
    assert s.coordinator_attempts == {"step_login_01": 2}
