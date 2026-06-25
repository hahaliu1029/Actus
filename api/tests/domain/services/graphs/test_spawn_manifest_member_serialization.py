import json

from app.domain.models.work_unit import WorkUnit
from app.domain.services.graphs.parallel_execution_subgraph import (
    _serialize_spawn_manifest,
)


def _wu(**kw):
    base = dict(work_unit_id="w.a0.0", objective="o", phase="exploration")
    base.update(kw)
    return WorkUnit(**base)


def test_empty_member_fields_omitted_byte_identical():
    # INV-0: a unit with no member fields serializes EXACTLY as a pre-S4 unit.
    # Non-vacuous: independently reconstruct the pre-S4 payload (the 8 existing
    # keys + their no-member values) and assert TRUE byte identity. This fixes
    # all 8 keys, their values, AND the sort_keys=True ordering, so any drift in
    # an existing key / default / order fails the test (not just the absence of
    # the two new keys).
    expected = {
        "work_unit_id": "w.a0.0",
        "objective": "o",
        "phase": "exploration",
        "allowed_tools": [],
        "write_lease": [],
        "write_tree_lease": [],
        "shell_mode": False,
        "expected_result_schema": None,
    }
    expected_bytes = json.dumps(expected, sort_keys=True).encode("utf-8")
    assert _serialize_spawn_manifest(_wu()) == expected_bytes
    # Keep the explicit "two new keys absent" assertions (cheap + explicit).
    data = json.loads(_serialize_spawn_manifest(_wu()))
    assert "member_skill_tools" not in data
    assert "member_skill_slugs" not in data


def test_non_empty_member_fields_sorted_deterministic():
    wu = _wu(
        member_skill_tools=frozenset({"skill_b_y", "skill_a_x"}),
        member_skill_slugs=("repo-map", "grep-pro"),
    )
    data = json.loads(_serialize_spawn_manifest(wu))
    assert data["member_skill_tools"] == ["skill_a_x", "skill_b_y"]   # sorted
    assert data["member_skill_slugs"] == ["grep-pro", "repo-map"]     # sorted
    assert _serialize_spawn_manifest(wu) == _serialize_spawn_manifest(wu)  # determinism
