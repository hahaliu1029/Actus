from app.domain.models.work_unit import WorkUnit
from app.domain.services.permission.child_permission_context import (
    ChildBudget,
    ChildPermissionContext,
    SpawnManifest,
)


def test_work_unit_member_fields_default_empty():
    wu = WorkUnit(work_unit_id="w.a0.0", objective="o", phase="exploration")
    assert wu.member_skill_tools == frozenset()
    assert wu.member_skill_slugs == ()


def test_spawn_manifest_member_fields_tail_defaulted():
    # all pre-S4 (5-field) constructions stay valid + inert
    m = SpawnManifest(allowed_tools=frozenset({"file_read"}), path_leases=(), runtime_caps=frozenset())
    assert m.member_skill_tools == frozenset()
    assert m.member_skill_slugs == ()
    m2 = SpawnManifest(
        allowed_tools=frozenset(), path_leases=(), runtime_caps=frozenset(),
        member_skill_tools=frozenset({"skill_repo_map_search"}),
        member_skill_slugs=("repo-map",),
    )
    assert m2.member_skill_tools == frozenset({"skill_repo_map_search"})


def test_child_permission_context_member_fields_tail_defaulted():
    cpc = ChildPermissionContext(
        parent_session_id="p", child_session_id="c", coordinator_run_id="r",
        work_unit_id="w", spawn_manifest=SpawnManifest(
            allowed_tools=frozenset(), path_leases=(), runtime_caps=frozenset()),
        session_mode_revision=0,
        budget=ChildBudget(max_tool_calls=1, max_token_cost_usd=1.0, max_wallclock_seconds=1),
    )
    assert cpc.member_skill_tools == frozenset()
    assert cpc.member_skill_slugs == ()
