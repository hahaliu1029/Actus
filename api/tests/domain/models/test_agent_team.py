import pytest
from pydantic import ValidationError

from app.domain.models.agent_team import TeamBundle, TeamMember


def _member(**kw):
    base = dict(role="explorer", description="map the code", system_prompt="You explore.")
    base.update(kw)
    return TeamMember(**base)


def test_member_minimal_defaults():
    m = _member()
    assert m.skills == ()
    assert m.default_phase is None
    assert m.shell_mode is False


def test_member_is_frozen_and_extra_forbidden():
    m = _member()
    with pytest.raises(ValidationError):
        m.role = "other"  # frozen
    with pytest.raises(ValidationError):
        TeamMember(role="r", description="d", system_prompt="s", bogus=1)  # extra


def test_member_exploration_must_not_be_shell_capable():
    with pytest.raises(ValidationError):
        _member(default_phase="exploration", shell_mode=True)
    # exploration + non-shell is fine
    assert _member(default_phase="exploration", shell_mode=False).shell_mode is False


def test_member_system_prompt_max_length_rejected():
    # R10-4: the child prompt's minimality is a security invariant.
    with pytest.raises(ValidationError):
        _member(system_prompt="x" * 9000)


def test_bundle_requires_non_empty_members():
    with pytest.raises(ValidationError):
        TeamBundle(slug="t", name="T", members=())


def test_bundle_rejects_duplicate_roles():
    with pytest.raises(ValidationError):
        TeamBundle(
            slug="t", name="T",
            members=(_member(role="a"), _member(role="a")),
        )


def test_bundle_happy_path():
    b = TeamBundle(
        slug="migration-squad", name="Migration Squad",
        members=(_member(role="explorer"), _member(role="implementer", default_phase="write")),
    )
    assert b.version == "0.1.0"
    assert [m.role for m in b.members] == ["explorer", "implementer"]
