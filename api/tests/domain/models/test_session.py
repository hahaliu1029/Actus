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
