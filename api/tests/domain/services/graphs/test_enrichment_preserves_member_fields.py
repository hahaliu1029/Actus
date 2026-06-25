from app.domain.models.work_unit import PathLease, TreeLease, WorkUnit
from app.domain.services.graphs.parallel_execution_subgraph import (
    _rebuild_enriched_unit,
)


def test_enrichment_preserves_all_member_fields():
    # A WRITE unit with an ORIGINAL lease + a DISTINCT new_leases. This proves
    # the whole point of _rebuild_enriched_unit: the enriched unit takes the NEW
    # leases (seed enrichment), NOT the source unit's lease — and that all 12
    # WorkUnit fields survive the rebuild.
    #
    # [codex-R2-nit] write_tree_lease + shell_mode are set to NON-DEFAULT values
    # (a non-empty TreeLease ⇒ shell_mode=True per the validator) so the two
    # assertions below are TRUE drop-detectors: if _rebuild_enriched_unit dropped
    # either field the rebuilt unit would fall back to the [] / False default and
    # the assertion would fail. With both at their defaults the test could not
    # distinguish "carried" from "dropped".
    original_lease = PathLease(path="src/original.py", op="add")
    src = WorkUnit(
        work_unit_id="w.a0.0",
        objective="o",
        phase="write",
        allowed_tools=["file_write"],
        write_lease=[original_lease],
        write_tree_lease=[TreeLease(prefix="workspace", ops=frozenset({"add"}))],
        shell_mode=True,  # non-empty tree lease ⇒ shell_mode=True (validator)
        expected_result_schema="schema",
        role="implementer",
        system_prompt="persona",
        member_skill_tools=frozenset({"skill_x_y"}),
        member_skill_slugs=("x",),
    )
    # DISTINCT new lease (different path) so write_lease != src.write_lease.
    new_leases = [PathLease(path="src/enriched.py", op="add")]

    out = _rebuild_enriched_unit(src, new_leases=new_leases)

    # The mapping is to the NEW leases, not the source's (the whole point).
    assert out.write_lease == new_leases
    assert out.write_lease != src.write_lease

    # EVERY OTHER live WorkUnit field is carried from src (11 non-write_lease).
    assert out.work_unit_id == src.work_unit_id
    assert out.objective == src.objective
    assert out.phase == src.phase
    assert out.allowed_tools == src.allowed_tools
    assert out.write_tree_lease == src.write_tree_lease
    assert out.shell_mode == src.shell_mode
    assert out.expected_result_schema == src.expected_result_schema
    assert out.role == src.role
    assert out.system_prompt == src.system_prompt
    assert out.member_skill_tools == src.member_skill_tools
    assert out.member_skill_slugs == src.member_skill_slugs
