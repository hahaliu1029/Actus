from app.domain.services.prompts.section import RenderContext


def _min_ctx(**kw):
    # [codex-R3-F6] RenderContext's ONLY required field is lang (section.py:68);
    # everything else defaults. This is the full executable fixture.
    return RenderContext(lang="zh", **kw)


def test_team_members_field_defaults_none():
    assert _min_ctx().team_members is None


def test_team_members_set_is_immutable_tuple():
    ctx = _min_ctx(team_members=(("explorer", "map the code"),))
    assert ctx.team_members == (("explorer", "map the code"),)
    assert isinstance(ctx.team_members, tuple)
