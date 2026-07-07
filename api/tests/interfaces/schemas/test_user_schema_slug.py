"""B11 §9: ToolWithPreference gains an additive optional `slug` field."""
from __future__ import annotations

from app.interfaces.schemas.user import ToolWithPreference


def test_slug_defaults_none():
    t = ToolWithPreference(
        tool_id="s1",
        tool_name="Repo Map",
        enabled_global=True,
        enabled_user=True,
    )
    assert t.slug is None


def test_slug_accepts_value():
    t = ToolWithPreference(
        tool_id="s1",
        tool_name="Repo Map",
        enabled_global=True,
        enabled_user=True,
        slug="repo-map",
    )
    assert t.slug == "repo-map"
