# api/tests/domain/models/test_tree_lease.py
import pytest
from pydantic import ValidationError

from app.domain.models.work_unit import (
    PathLease,
    ProposedPath,
    ProposedTree,
    TreeLease,
    WorkUnit,
    WorkUnitRequest,
)


class TestTreeLeaseModel:
    def test_minimal_add_only(self):
        t = TreeLease(prefix="workspace", ops=frozenset({"add"}))
        assert t.prefix == "workspace"
        assert t.ops == frozenset({"add"})

    def test_prefix_canonicalized(self):
        t = TreeLease(prefix="./workspace/", ops=frozenset({"add"}))
        assert t.prefix == "workspace"

    def test_frozen(self):
        t = TreeLease(prefix="workspace", ops=frozenset({"add"}))
        with pytest.raises(ValidationError):
            t.prefix = "other"

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            TreeLease(prefix="workspace", ops=frozenset({"add"}), bogus=1)

    def test_ops_rejects_non_add(self):
        # v1 is ADD-ONLY: Literal["add"] makes "modify"/"delete" illegal.
        with pytest.raises(ValidationError):
            TreeLease(prefix="workspace", ops=frozenset({"modify"}))

    def test_bare_root_prefix_rejected(self):
        with pytest.raises(ValidationError):
            TreeLease(prefix="/home/ubuntu", ops=frozenset({"add"}))


class TestProposedTreeModel:
    def test_minimal(self):
        p = ProposedTree(prefix="workspace", ops=frozenset({"add"}))
        assert p.prefix == "workspace"

    def test_frozen_extra_forbid(self):
        with pytest.raises(ValidationError):
            ProposedTree(prefix="workspace", ops=frozenset({"add"}), x=1)


class TestPathLeaseExtraForbid:
    def test_path_lease_rejects_extra(self):
        # §3.3: lease models gain extra="forbid".
        with pytest.raises(ValidationError):
            PathLease(path="a/b.py", op="modify", base_digest="b" * 64, bogus=1)


class TestWorkUnitTreeLease:
    def test_write_phase_satisfied_by_tree_lease_only(self):
        # §3.3: phase=write is satisfied by write_lease OR write_tree_lease.
        # tree lease implies shell_mode=True.
        wu = WorkUnit(
            work_unit_id="wu-1", objective="o", phase="write",
            allowed_tools=["file_write"],
            write_lease=[],
            write_tree_lease=[TreeLease(prefix="workspace", ops=frozenset({"add"}))],
            shell_mode=True,
        )
        assert wu.write_tree_lease[0].prefix == "workspace"
        assert wu.shell_mode is True

    def test_write_phase_with_path_lease_only_still_ok(self):
        wu = WorkUnit(
            work_unit_id="wu-1", objective="o", phase="write",
            allowed_tools=["file_write"],
            write_lease=[PathLease(path="a/b.py", op="add")],
        )
        assert wu.shell_mode is False  # default inert

    def test_write_phase_empty_both_rejected(self):
        with pytest.raises(ValidationError):
            WorkUnit(
                work_unit_id="wu-1", objective="o", phase="write",
                allowed_tools=["file_write"], write_lease=[], write_tree_lease=[],
            )

    def test_exploration_with_tree_lease_rejected(self):
        with pytest.raises(ValidationError):
            WorkUnit(
                work_unit_id="wu-1", objective="o", phase="exploration",
                allowed_tools=["file_read"],
                write_tree_lease=[TreeLease(prefix="workspace", ops=frozenset({"add"}))],
                shell_mode=True,
            )

    def test_exploration_with_shell_mode_rejected(self):
        # exploration is read-only — shell_mode True is a contradiction.
        with pytest.raises(ValidationError):
            WorkUnit(
                work_unit_id="wu-1", objective="o", phase="exploration",
                allowed_tools=["file_read"], shell_mode=True,
            )

    def test_tree_lease_without_shell_mode_rejected(self):
        # §3.3: a non-empty write_tree_lease IMPLIES shell_mode=True.
        with pytest.raises(ValidationError):
            WorkUnit(
                work_unit_id="wu-1", objective="o", phase="write",
                allowed_tools=["file_write"], write_lease=[],
                write_tree_lease=[TreeLease(prefix="workspace", ops=frozenset({"add"}))],
                shell_mode=False,
            )

    def test_shell_mode_defaults_false(self):
        wu = WorkUnit(
            work_unit_id="wu-1", objective="o", phase="write",
            allowed_tools=["file_write"],
            write_lease=[PathLease(path="a/b.py", op="add")],
        )
        assert wu.shell_mode is False


class TestWorkUnitRequestProposedTrees:
    def test_write_satisfied_by_proposed_trees_only(self):
        req = WorkUnitRequest(
            objective="o", phase="write", allowed_tools=["file_write"],
            proposed_paths=[],
            proposed_trees=[ProposedTree(prefix="workspace", ops=frozenset({"add"}))],
        )
        assert req.proposed_trees[0].prefix == "workspace"

    def test_write_empty_both_rejected(self):
        with pytest.raises(ValidationError):
            WorkUnitRequest(
                objective="o", phase="write", allowed_tools=["file_write"],
                proposed_paths=[], proposed_trees=[],
            )

    def test_request_extra_forbid(self):
        with pytest.raises(ValidationError):
            WorkUnitRequest(
                objective="o", phase="exploration", allowed_tools=["file_read"],
                bogus=1,
            )

    def test_proposed_trees_defaults_empty(self):
        req = WorkUnitRequest(
            objective="o", phase="exploration", allowed_tools=["file_read"],
        )
        assert req.proposed_trees == []


class TestWorkUnitRequestShellMode:
    def test_shell_mode_defaults_false(self):
        # §3.5: shell_mode is a positive signal defaulting to False (inert).
        req = WorkUnitRequest(
            objective="o", phase="exploration", allowed_tools=["file_read"],
        )
        assert req.shell_mode is False

    def test_extra_forbid_no_longer_rejects_shell_mode(self):
        # P0 REGRESSION GUARD: before the fix, extra="forbid" + a missing
        # shell_mode FIELD made `WorkUnitRequest(..., shell_mode=True, ...)` a
        # ValidationError — exactly the construction PR-5/PR-6 planner payloads
        # use. It must now be ACCEPTED.
        req = WorkUnitRequest(
            objective="o", phase="write", allowed_tools=["file_write"],
            proposed_paths=[ProposedPath(path="a/b.py", op="add")],
            shell_mode=True,
        )
        assert req.shell_mode is True

    def test_shell_mode_with_paths_only_no_trees(self):
        # §3.5: shell_mode requestable with ONLY exact file leases (no trees).
        # Tree leases are SUFFICIENT but NOT NECESSARY for shell mode.
        req = WorkUnitRequest(
            objective="o", phase="write", allowed_tools=["file_write"],
            proposed_paths=[ProposedPath(path="a/b.py", op="add")],
            proposed_trees=[],
            shell_mode=True,
        )
        assert req.shell_mode is True
        assert req.proposed_trees == []

    def test_proposed_trees_coerce_shell_mode_true(self):
        # §3.3: a non-empty proposed_trees coerces shell_mode True at the
        # request level even when the planner left it default-False.
        req = WorkUnitRequest(
            objective="o", phase="write", allowed_tools=["file_write"],
            proposed_paths=[],
            proposed_trees=[ProposedTree(prefix="workspace", ops=frozenset({"add"}))],
        )
        assert req.shell_mode is True

    def test_exploration_shell_mode_rejected(self):
        # §3.5: read-only phase contradicts raw-shell write.
        with pytest.raises(ValidationError):
            WorkUnitRequest(
                objective="o", phase="exploration", allowed_tools=["file_read"],
                shell_mode=True,
            )

    def test_exploration_proposed_trees_rejected_at_request_boundary(self):
        # codex-review FIX (request-boundary validator ordering gap): an
        # exploration request carrying a non-empty proposed_trees with the
        # DEFAULT shell_mode=False is an EFFECTIVE shell-intent contradiction
        # (a tree lease implies shell mode, §3.3). It MUST be rejected by
        # _request_shell_mode_consistency at the REQUEST boundary — NOT silently
        # coerced to shell_mode=True and left for the downstream WorkUnit
        # validator to catch. This fails before the fix (the flag-only check
        # passes because shell_mode is still False when it runs, then the
        # coercion produces a contradictory exploration-with-tree object).
        with pytest.raises(ValidationError):
            WorkUnitRequest(
                objective="o", phase="exploration", allowed_tools=["file_read"],
                proposed_trees=[ProposedTree(prefix="workspace", ops=frozenset({"add"}))],
            )


class TestWorkUnitRequestExtraForbidByteForBytePrecision:
    """[codex-review FIX P2 — §5 invariant 6 precision] LOCK the one intentional
    deviation from a literal "flag OFF ⇒ byte-for-byte current behavior".

    Today ``WorkUnitRequest`` has NO ``model_config``, so an unknown ROOT field
    on a payload is silently IGNORED. PR-3 adds ``extra="forbid"`` (§3.3 [B6] —
    prevents a mistyped ``proposed_trees``/``shell_mode`` from being silently
    dropped). We KEEP it: the precise guarantee is "byte-for-byte for VALID
    current planner payloads; the only new rejection is of payloads carrying
    UNKNOWN fields, which no current producer (the planner) emits."

    This pair of assertions pins BOTH halves of that contract so a future edit
    can't quietly (a) regress the new field defaults / loosen the validators for
    a CURRENT valid payload, or (b) drop ``extra="forbid"`` and re-open the
    silent-drop hole.
    """

    def test_current_valid_payload_unchanged_and_unknown_field_rejected(self):
        # (a) A representative CURRENT valid planner payload — ONLY the fields the
        #     planner emits today (objective / phase / allowed_tools /
        #     proposed_paths / expected_result_schema), NO extra/unknown fields,
        #     NO new S2 fields. It is STILL ACCEPTED, and the new S2 fields
        #     DEFAULT to the inert/off state so behavior is unchanged for it.
        req = WorkUnitRequest(
            objective="implement X",
            phase="write",
            allowed_tools=["file_read", "file_write"],
            proposed_paths=[ProposedPath(path="api/x.py", op="add")],
            expected_result_schema="summary",
        )
        # Validators accept it (no spurious new requirement).
        assert req.objective == "implement X"
        assert req.phase == "write"
        assert [p.path for p in req.proposed_paths] == ["api/x.py"]
        assert req.expected_result_schema == "summary"
        # New S2 fields default OFF ⇒ shell mode dormant, no tree leases.
        assert req.shell_mode is False
        assert req.proposed_trees == []

        # And it builds the SAME WorkUnit shape it always did — phase=write with
        # a single ADD file lease, no tree lease, shell_mode off (the build maps
        # proposed_paths -> write_lease 1:1; the new fields stay at their inert
        # defaults). This is the "byte-for-byte for a VALID payload" half.
        wu = WorkUnit(
            work_unit_id="wu-current-1",
            objective=req.objective,
            phase=req.phase,
            allowed_tools=list(req.allowed_tools),
            write_lease=[
                PathLease(path=p.path, op=p.op) for p in req.proposed_paths
            ],
            expected_result_schema=req.expected_result_schema,
        )
        assert wu.shell_mode is False
        assert wu.write_tree_lease == []
        assert [l.path for l in wu.write_lease] == ["api/x.py"]

        # (b) The DELIBERATE TRADEOFF, locked: the SAME payload shape PLUS one
        #     UNKNOWN root field — silently ignored BEFORE this PR — is now
        #     REJECTED. This is the only new rejection vs. current behavior, and
        #     no current producer (the planner) emits such a field.
        with pytest.raises(ValidationError):
            WorkUnitRequest(
                objective="implement X",
                phase="write",
                allowed_tools=["file_read", "file_write"],
                proposed_paths=[ProposedPath(path="api/x.py", op="add")],
                expected_result_schema="summary",
                unknown_root_field="silently-dropped-before-PR-3",
            )


def test_tree_lease_empty_ops_rejected():
    # [codex PR-3 R1 P1] v1 is ADD-only: ops must be non-empty (exactly
    # {"add"} given the Literal). An empty frozenset authorizes no operation
    # yet would still make a unit shell_mode=True (bool(write_tree_lease)) and
    # carry write-tree intent with no add grant — a broken model contract.
    # Reject it at the model boundary.
    with pytest.raises(ValidationError):
        TreeLease(prefix="workspace", ops=frozenset())


def test_proposed_tree_empty_ops_rejected():
    with pytest.raises(ValidationError):
        ProposedTree(prefix="workspace", ops=frozenset())
