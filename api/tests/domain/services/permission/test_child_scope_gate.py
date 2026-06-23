"""C2 PR-2 §5.4 — ChildScopeGate 4-way intersection tests."""
import dataclasses
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.domain.models.session import SessionStatus
from app.domain.models.work_unit import PathLease, TreeLease
from app.domain.services.permission import child_scope_gate as gate_mod
from app.domain.services.permission.child_permission_context import (
    ChildBudget,
    ChildPermissionContext,
    SpawnManifest,
)
from app.domain.services.permission.child_scope_gate import (
    HARD_BLOCKED_FOR_CHILDREN,
    TYPED_WRITE_TOOL_NAMES,
    ChildScopeGate,
    ScopeDecision,
)
from app.domain.services.permission.context import EvaluationContext

pytestmark = pytest.mark.anyio

_SHELL_FIVE = frozenset({
    "shell_execute", "shell_wait_process", "shell_kill_process",
    "shell_write_input", "shell_read_output",
})
_NON_SHELL_HARD_BLOCKED = HARD_BLOCKED_FOR_CHILDREN - _SHELL_FIVE


def _call(tool_name, **args):
    m = MagicMock()
    m.tool_name = tool_name
    m.tool_args = args
    return m


def _cctx(
    *,
    allowed=frozenset({"file_read"}),
    leases=(),
    tree_leases=(),
    caps=frozenset(),
    shell_mode=False,
    max_tool_calls=100,
    session_mode_revision=1,
) -> ChildPermissionContext:
    return ChildPermissionContext(
        parent_session_id="p1",
        child_session_id="c1",
        coordinator_run_id="r1",
        work_unit_id="wu1",
        spawn_manifest=SpawnManifest(
            allowed_tools=allowed,
            path_leases=leases,
            runtime_caps=caps,
            tree_leases=tree_leases,
            shell_mode=shell_mode,
        ),
        session_mode_revision=session_mode_revision,
        budget=ChildBudget(
            max_tool_calls=max_tool_calls,
            max_token_cost_usd=1.0,
            max_wallclock_seconds=600,
        ),
        shell_mode=shell_mode,
    )


def _ctx(c: ChildPermissionContext, *, session_mode_revision: int = 1) -> EvaluationContext:
    return EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=session_mode_revision,
        child_permission_context=c,
    )


@pytest.fixture
def gate() -> ChildScopeGate:
    return ChildScopeGate()


class TestAllowlist:
    async def test_in_allowlist(self, gate):
        c = _cctx()
        assert await gate.check_in_scope(_call("file_read"), _ctx(c), c) == ScopeDecision.IN_SCOPE

    async def test_out_of_allowlist(self, gate):
        c = _cctx(allowed=frozenset({"file_read"}))
        assert await gate.check_in_scope(_call("shell_execute"), _ctx(c), c) == ScopeDecision.OUT_OF_TOOL_ALLOWLIST


class TestHardBlocked:
    """Shell entries are conditional; non-shell entries unconditional."""

    @pytest.mark.parametrize("tool", sorted(_NON_SHELL_HARD_BLOCKED))
    async def test_non_shell_overrides_allowlist_unconditionally(
        self, gate, tool, monkeypatch,
    ):
        # Even flag-on + shell_mode=True must NOT un-block non-shell hard-blocks.
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(allowed=frozenset({tool}), shell_mode=True)
        assert await gate.check_in_scope(
            _call(tool), _ctx(c), c
        ) == ScopeDecision.HARD_BLOCKED

    @pytest.mark.parametrize("tool", sorted(_SHELL_FIVE))
    async def test_shell_blocked_when_flag_off(self, gate, tool, monkeypatch):
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: False)
        c = _cctx(allowed=frozenset({tool}), shell_mode=True)
        assert await gate.check_in_scope(
            _call(tool), _ctx(c), c
        ) == ScopeDecision.HARD_BLOCKED

    @pytest.mark.parametrize("tool", sorted(_SHELL_FIVE))
    async def test_shell_blocked_when_shell_mode_off(self, gate, tool, monkeypatch):
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(allowed=frozenset({tool}), shell_mode=False)
        assert await gate.check_in_scope(
            _call(tool), _ctx(c), c
        ) == ScopeDecision.HARD_BLOCKED

    def test_full_live_blocked_set(self):
        """Lock the full HARD_BLOCKED_FOR_CHILDREN canonical membership.

        If a maintainer accidentally removes any entry, this test fails.
        """
        expected = frozenset({
            "shell_execute", "shell_wait_process", "shell_kill_process",
            "shell_write_input", "shell_read_output",
            "message_ask_user", "message_notify_user",
            "memory_save",
            "spawn_subagent",
            "install_skill",
            "set_tool_approval",
            "publish_mailbox_envelope",
        })
        assert HARD_BLOCKED_FOR_CHILDREN == expected, (
            f"HARD_BLOCKED_FOR_CHILDREN drifted: missing {expected - HARD_BLOCKED_FOR_CHILDREN}, "
            f"extra {HARD_BLOCKED_FOR_CHILDREN - expected}"
        )


class TestShellModeConditional:
    @pytest.mark.parametrize("tool", sorted(_SHELL_FIVE))
    async def test_shell_unblocked_when_flag_on_and_shell_mode(
        self, gate, tool, monkeypatch,
    ):
        # flag_on AND shell_mode → the 5 shell entries fall through steps 3-6
        # (no path-lease check, no path arg) → IN_SCOPE.
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(allowed=frozenset({tool}), shell_mode=True)
        assert await gate.check_in_scope(
            _call(tool), _ctx(c), c
        ) == ScopeDecision.IN_SCOPE

    async def test_shell_not_in_allowlist_still_out_of_allowlist(
        self, gate, monkeypatch,
    ):
        # Un-block does NOT bypass step-1 allowlist: a shell tool absent from
        # allowed_tools is still OUT_OF_TOOL_ALLOWLIST even flag-on+shell_mode.
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(allowed=frozenset({"file_read"}), shell_mode=True)
        assert await gate.check_in_scope(
            _call("shell_execute"), _ctx(c), c
        ) == ScopeDecision.OUT_OF_TOOL_ALLOWLIST

    async def test_shell_unblock_respects_budget_exhaustion(
        self, gate, monkeypatch,
    ):
        # Falling through steps 3-6 means an exhausted budget still trips.
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(allowed=frozenset({"shell_execute"}), shell_mode=True,
                  max_tool_calls=0)
        assert await gate.check_in_scope(
            _call("shell_execute"), _ctx(c), c
        ) == ScopeDecision.BUDGET_EXHAUSTED


class TestPathLease:
    async def test_write_in_lease(self, gate):
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        assert await gate.check_in_scope(_call("file_write", path="/x"), _ctx(c), c) == ScopeDecision.IN_SCOPE

    async def test_write_out_of_lease(self, gate):
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        assert await gate.check_in_scope(_call("file_write", path="/y"), _ctx(c), c) == ScopeDecision.OUT_OF_PATH_LEASE

    async def test_write_missing_path_arg(self, gate):
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        assert await gate.check_in_scope(_call("file_write"), _ctx(c), c) == ScopeDecision.OUT_OF_PATH_LEASE

    async def test_read_skips_lease(self, gate):
        c = _cctx(allowed=frozenset({"file_read"}), leases=())
        assert await gate.check_in_scope(_call("file_read", path="/any"), _ctx(c), c) == ScopeDecision.IN_SCOPE

    async def test_write_add_lease_op_compatible(self, gate):
        """op='add' should accept TYPED_WRITE_TOOL_NAMES per _op_compatible."""
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/new", op="add"),),
        )
        assert await gate.check_in_scope(_call("file_write", path="/new"), _ctx(c), c) == ScopeDecision.IN_SCOPE

    async def test_typed_write_constant_membership(self):
        assert "file_write" in TYPED_WRITE_TOOL_NAMES
        assert "file_str_replace" in TYPED_WRITE_TOOL_NAMES

    async def test_write_in_lease_via_filepath_key(self, gate):
        """Live tools use 'filepath' (not 'path') — gate must match."""
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        assert await gate.check_in_scope(_call("file_write", filepath="/x"), _ctx(c), c) == ScopeDecision.IN_SCOPE

    async def test_write_out_of_lease_via_filepath_key(self, gate):
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        assert await gate.check_in_scope(_call("file_write", filepath="/y"), _ctx(c), c) == ScopeDecision.OUT_OF_PATH_LEASE

    @pytest.mark.parametrize(
        "target_path",
        [
            "/home/ubuntu/workspace/a.py",
            "./workspace/a.py",
            "workspace//a.py",
        ],
    )
    async def test_write_matches_lease_after_coordinator_path_canonicalization(
        self, gate, target_path
    ):
        """A child tool may send an absolute/non-canonical spelling of the same
        workspace path the coordinator stored in the lease."""
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(
                PathLease(path="workspace/a.py", op="modify", base_digest="abc"),
            ),
        )

        assert (
            await gate.check_in_scope(
                _call("file_write", filepath=target_path), _ctx(c), c
            )
            == ScopeDecision.IN_SCOPE
        )

    @pytest.mark.parametrize("target_path", ["/etc/passwd", "a.py"])
    async def test_invalid_coordinator_path_spelling_still_denied(
        self, gate, target_path
    ):
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(
                PathLease(path="workspace/a.py", op="modify", base_digest="abc"),
            ),
        )

        assert (
            await gate.check_in_scope(
                _call("file_write", filepath=target_path), _ctx(c), c
            )
            == ScopeDecision.OUT_OF_PATH_LEASE
        )

    async def test_empty_filepath_does_not_fall_through_to_path(self, gate):
        """Explicit empty filepath must not silently fall back to 'path' kwarg."""
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        # filepath="" is present-but-falsy; should NOT fall back to 'path' kwarg.
        m = MagicMock()
        m.tool_name = "file_write"
        m.tool_args = {"filepath": "", "path": "/x"}
        assert await gate.check_in_scope(m, _ctx(c), c) == ScopeDecision.OUT_OF_PATH_LEASE

    async def test_filepath_wins_over_path_when_both_present(self, gate):
        """When tool_args has both 'filepath' and 'path', extract_target_path MUST use filepath.

        Mutation guard: if a future change reorders the `or` fallback in extract_target_path,
        this test catches it before merging.
        """
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        # filepath="/x" (matches lease), path="/y" (different) — gate must lease-check filepath
        m = MagicMock()
        m.tool_name = "file_write"
        m.tool_args = {"filepath": "/x", "path": "/y"}
        assert await gate.check_in_scope(m, _ctx(c), c) == ScopeDecision.IN_SCOPE

    async def test_filepath_wins_over_path_negative(self, gate):
        """When filepath has non-leased path and path has leased path, filepath still wins -> deny."""
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        m = MagicMock()
        m.tool_name = "file_write"
        m.tool_args = {"filepath": "/forbidden", "path": "/x"}
        assert await gate.check_in_scope(m, _ctx(c), c) == ScopeDecision.OUT_OF_PATH_LEASE

    async def test_non_string_filepath_rejected(self, gate):
        """If filepath is a list/int (model malformed), extract_target_path returns None
        and gate denies OUT_OF_PATH_LEASE. Mutation guard for isinstance(path, str) check."""
        c = _cctx(
            allowed=frozenset({"file_write"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        for bad_filepath in ([1, 2, 3], 42, {"nested": "/x"}, None):
            m = MagicMock()
            m.tool_name = "file_write"
            m.tool_args = {"filepath": bad_filepath}
            decision = await gate.check_in_scope(m, _ctx(c), c)
            assert decision == ScopeDecision.OUT_OF_PATH_LEASE, (
                f"non-string filepath {bad_filepath!r} must yield OUT_OF_PATH_LEASE"
            )


class TestBudget:
    async def test_under_cap(self, gate):
        c = _cctx(max_tool_calls=10)
        assert await gate.check_in_scope(_call("file_read"), _ctx(c), c) == ScopeDecision.IN_SCOPE

    async def test_at_cap(self, gate):
        c = _cctx(max_tool_calls=0)
        assert await gate.check_in_scope(_call("file_read"), _ctx(c), c) == ScopeDecision.BUDGET_EXHAUSTED


class TestRevisionDrift:
    async def test_match(self, gate):
        c = _cctx(session_mode_revision=5)
        assert await gate.check_in_scope(
            _call("file_read"), _ctx(c, session_mode_revision=5), c
        ) == ScopeDecision.IN_SCOPE

    async def test_drift(self, gate):
        c = _cctx(session_mode_revision=5)
        assert await gate.check_in_scope(
            _call("file_read"), _ctx(c, session_mode_revision=6), c
        ) == ScopeDecision.REVISION_DRIFT


class TestLeaseExpiry:
    """[r3 P1-2] lease_expiry check."""

    async def test_lease_not_expired(self, gate):
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        c = dataclasses.replace(_cctx(), lease_expiry=future)
        assert await gate.check_in_scope(_call("file_read"), _ctx(c), c) == ScopeDecision.IN_SCOPE

    async def test_lease_expired(self, gate):
        past = datetime.now(timezone.utc) - timedelta(minutes=10)
        c = dataclasses.replace(_cctx(), lease_expiry=past)
        assert await gate.check_in_scope(_call("file_read"), _ctx(c), c) == ScopeDecision.LEASE_EXPIRED

    async def test_lease_naive_expiry_treated_as_utc(self, gate):
        """Naive datetime in lease_expiry must NOT crash; treated as UTC."""
        from datetime import datetime, timedelta
        # naive datetime (no tzinfo)
        past_naive = datetime.utcnow() - timedelta(minutes=10)
        c = dataclasses.replace(_cctx(), lease_expiry=past_naive)
        assert await gate.check_in_scope(_call("file_read"), _ctx(c), c) == ScopeDecision.LEASE_EXPIRED

    async def test_lease_naive_future_not_expired(self, gate):
        from datetime import datetime, timedelta
        future_naive = datetime.utcnow() + timedelta(minutes=10)
        c = dataclasses.replace(_cctx(), lease_expiry=future_naive)
        assert await gate.check_in_scope(_call("file_read"), _ctx(c), c) == ScopeDecision.IN_SCOPE


class TestDeleteLease:
    """file_delete must be subject to path lease enforcement via PATH_LEASED_TOOL_NAMES."""

    async def test_delete_in_lease(self, gate):
        c = _cctx(
            allowed=frozenset({"file_delete"}),
            leases=(PathLease(path="/old", op="delete"),),
        )
        assert await gate.check_in_scope(_call("file_delete", path="/old"), _ctx(c), c) == ScopeDecision.IN_SCOPE

    async def test_delete_out_of_lease(self, gate):
        c = _cctx(
            allowed=frozenset({"file_delete"}),
            leases=(PathLease(path="/old", op="delete"),),
        )
        assert await gate.check_in_scope(_call("file_delete", path="/other"), _ctx(c), c) == ScopeDecision.OUT_OF_PATH_LEASE

    async def test_delete_with_modify_lease_op_mismatch(self, gate):
        """A delete call against a modify lease should return OP_MISMATCH."""
        c = _cctx(
            allowed=frozenset({"file_delete"}),
            leases=(PathLease(path="/x", op="modify", base_digest="abc"),),
        )
        assert await gate.check_in_scope(_call("file_delete", path="/x"), _ctx(c), c) == ScopeDecision.OP_MISMATCH

    async def test_delete_in_lease_via_filepath_key(self, gate):
        c = _cctx(
            allowed=frozenset({"file_delete"}),
            leases=(PathLease(path="/old", op="delete"),),
        )
        assert await gate.check_in_scope(_call("file_delete", filepath="/old"), _ctx(c), c) == ScopeDecision.IN_SCOPE


class TestPathLeasedConstant:
    def test_path_leased_includes_delete(self):
        from app.domain.services.permission.child_scope_gate import PATH_LEASED_TOOL_NAMES, TYPED_WRITE_TOOL_NAMES
        assert PATH_LEASED_TOOL_NAMES == TYPED_WRITE_TOOL_NAMES | {"file_delete"}


class TestNoneChildRaises:
    async def test_none_raises(self, gate):
        ctx = EvaluationContext(
            session_mode=SessionStatus.RUNNING,
            session_mode_revision=1,
        )
        with pytest.raises(ValueError, match="child_ctx=None"):
            await gate.check_in_scope(_call("file_read"), ctx, None)  # type: ignore[arg-type]


class TestSpawnManifestShellModeCarrier:
    def test_spawn_manifest_defaults_inert(self):
        from app.domain.services.permission.child_permission_context import (
            SpawnManifest,
        )
        m = SpawnManifest(
            allowed_tools=frozenset({"file_read"}),
            path_leases=(),
            runtime_caps=frozenset(),
        )
        # §3.5/§3.3: both new fields default to the inert state.
        assert m.shell_mode is False
        assert m.tree_leases == ()

    def test_spawn_manifest_carries_tree_lease_and_shell_mode(self):
        from app.domain.services.permission.child_permission_context import (
            SpawnManifest,
        )
        tl = TreeLease(prefix="workspace", ops=frozenset({"add"}))
        m = SpawnManifest(
            allowed_tools=frozenset({"file_write"}),
            path_leases=(),
            runtime_caps=frozenset(),
            tree_leases=(tl,),
            shell_mode=True,
        )
        assert m.shell_mode is True
        assert m.tree_leases == (tl,)

    def test_child_permission_context_shell_mode_default(self):
        from app.domain.services.permission.child_permission_context import (
            ChildBudget,
            ChildPermissionContext,
            SpawnManifest,
        )
        ctx = ChildPermissionContext(
            parent_session_id="p1", child_session_id="c1",
            coordinator_run_id="r1", work_unit_id="wu1",
            spawn_manifest=SpawnManifest(
                allowed_tools=frozenset({"file_read"}),
                path_leases=(), runtime_caps=frozenset(),
            ),
            session_mode_revision=1,
            budget=ChildBudget(
                max_tool_calls=10, max_token_cost_usd=1.0, max_wallclock_seconds=600,
            ),
        )
        assert ctx.shell_mode is False


class TestTreeLeaseAdd:
    async def test_tree_add_typed_write_honored(self, gate, monkeypatch):
        # flag-on + shell_mode: file_write to a path under a tree-lease prefix,
        # no exact file lease → IN_SCOPE (op=add inferred from absence of exact
        # lease; tree is ADD-only).
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(
            allowed=frozenset({"file_write"}),
            tree_leases=(TreeLease(prefix="workspace", ops=frozenset({"add"})),),
            shell_mode=True,
        )
        assert await gate.check_in_scope(
            _call("file_write", filepath="workspace/new.py"), _ctx(c), c
        ) == ScopeDecision.IN_SCOPE

    async def test_tree_add_out_of_lease_when_flag_off(self, gate, monkeypatch):
        # FAIL-SAFE: flag OFF + a covering tree lease (shell_mode True on a
        # stale/hand-crafted manifest) → the tree branch is gated on the master
        # flag, so it NEVER widens the gate → OUT_OF_PATH_LEASE. This locks the
        # default-OFF invariant for the tree-add path (mirror of the step-2
        # shell un-block flag gate).
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: False)
        c = _cctx(
            allowed=frozenset({"file_write"}),
            tree_leases=(TreeLease(prefix="workspace", ops=frozenset({"add"})),),
            shell_mode=True,
        )
        assert await gate.check_in_scope(
            _call("file_write", filepath="workspace/new.py"), _ctx(c), c
        ) == ScopeDecision.OUT_OF_PATH_LEASE

    async def test_tree_add_no_coverage_out_of_lease(self, gate, monkeypatch):
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(
            allowed=frozenset({"file_write"}),
            tree_leases=(TreeLease(prefix="workspace", ops=frozenset({"add"})),),
            shell_mode=True,
        )
        assert await gate.check_in_scope(
            _call("file_write", filepath="api/other.py"), _ctx(c), c
        ) == ScopeDecision.OUT_OF_PATH_LEASE

    async def test_exact_file_lease_wins_over_tree_op_mismatch(
        self, gate, monkeypatch,
    ):
        # F21: exact file lease (op=add) + a covering tree lease on same prefix;
        # a file_delete (op=delete shape) → exact wins → OP_MISMATCH, NO tree
        # fallback widening the op.
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(
            allowed=frozenset({"file_delete"}),
            leases=(PathLease(path="workspace/x.py", op="add"),),
            tree_leases=(TreeLease(prefix="workspace", ops=frozenset({"add"})),),
            shell_mode=True,
        )
        assert await gate.check_in_scope(
            _call("file_delete", filepath="workspace/x.py"), _ctx(c), c
        ) == ScopeDecision.OP_MISMATCH

    async def test_tree_lease_does_not_cover_delete(self, gate, monkeypatch):
        # A file_delete under a tree prefix with NO exact lease → tree is
        # ADD-only, so a delete shape is never covered → OUT_OF_PATH_LEASE.
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(
            allowed=frozenset({"file_delete"}),
            tree_leases=(TreeLease(prefix="workspace", ops=frozenset({"add"})),),
            shell_mode=True,
        )
        assert await gate.check_in_scope(
            _call("file_delete", filepath="workspace/gone.py"), _ctx(c), c
        ) == ScopeDecision.OUT_OF_PATH_LEASE

    async def test_tree_add_out_of_lease_when_shell_mode_off(
        self, gate, monkeypatch,
    ):
        # [codex PR-5 R2 P2] FAIL-SAFE truth-table cell: master flag ON but the
        # child's shell_mode is False + a covering tree lease (stale/hand-crafted
        # manifest) → the tree branch requires BOTH the flag AND
        # child_ctx.shell_mode, so it NEVER widens the gate → OUT_OF_PATH_LEASE.
        # Completes the {flag, shell_mode} truth table for the tree-add path.
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(
            allowed=frozenset({"file_write"}),
            tree_leases=(TreeLease(prefix="workspace", ops=frozenset({"add"})),),
            shell_mode=False,
        )
        assert await gate.check_in_scope(
            _call("file_write", filepath="workspace/new.py"), _ctx(c), c
        ) == ScopeDecision.OUT_OF_PATH_LEASE

    async def test_tree_add_canonicalizes_absolute_target(
        self, gate, monkeypatch,
    ):
        # [codex PR-5 R2 P2] The tree branch canonicalizes the target before
        # tree_contains (mirror of _lookup_lease). An ABSOLUTE workspace target
        # (/home/ubuntu/workspace/new.py) canonicalizes to "workspace/new.py" and
        # is covered → IN_SCOPE. Before the fix the raw absolute path was handed
        # to tree_contains (which needs canonical workspace-relative) → wrongly
        # rejected. NON-VACUOUS: this returns OUT_OF_PATH_LEASE without the fix.
        monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                            lambda: True)
        c = _cctx(
            allowed=frozenset({"file_write"}),
            tree_leases=(TreeLease(prefix="workspace", ops=frozenset({"add"})),),
            shell_mode=True,
        )
        assert await gate.check_in_scope(
            _call("file_write", filepath="/home/ubuntu/workspace/new.py"),
            _ctx(c), c,
        ) == ScopeDecision.IN_SCOPE
