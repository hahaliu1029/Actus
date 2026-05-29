"""PR-9b-B — GroupLineageFields value object for CoordinatorApplyEvent lineage.

Apply event is GROUP-LEVEL (post-reduce), so only 3 of the 5
CoordinatorLineageMixin fields are meaningful: root_session_id,
parent_session_id, coordinator_run_id. Per-child fields stay None.
"""
from __future__ import annotations

import pytest

from app.application.services.group_lineage import GroupLineageFields


def test_construct_with_two_required_fields():
    g = GroupLineageFields(root_session_id="root-1", parent_session_id="parent-1")
    assert g.root_session_id == "root-1"
    assert g.parent_session_id == "parent-1"


def test_optional_default_none():
    g = GroupLineageFields()
    assert g.root_session_id is None
    assert g.parent_session_id is None


def test_frozen():
    g = GroupLineageFields(root_session_id="r", parent_session_id="p")
    with pytest.raises(Exception):
        g.root_session_id = "x"  # type: ignore[misc]
