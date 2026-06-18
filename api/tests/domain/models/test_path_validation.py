"""C2 PR-4 r3 P1#6 — path validator tests.

Pins the rejection rules so a future relaxation requires explicit test
removal (high friction = intentional change). Tests directly target the
helper + verify the wired models (FilePatchEntry, ProposedPath, PathLease)
all share the same rejection contract.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.path_validation import (
    CoordinatorPathContractError,
    to_workspace_relative,
    validate_coordinator_path,
    validate_directory_qualified_relative_path,
    validate_relative_path,
    validate_relative_path_strict,
)


class TestValidatorRejections:
    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_relative_path("")

    def test_non_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be str"):
            validate_relative_path(123)  # type: ignore[arg-type]

    def test_nul_byte_rejected(self) -> None:
        with pytest.raises(ValueError, match="NUL byte"):
            validate_relative_path("a\x00b")

    def test_newline_rejected(self) -> None:
        with pytest.raises(ValueError, match="newline"):
            validate_relative_path("a\nb")

    def test_carriage_return_rejected(self) -> None:
        with pytest.raises(ValueError, match="newline"):
            validate_relative_path("a\rb")

    def test_dotdot_segment_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\.\.'? segment"):
            validate_relative_path("a/../b")

    def test_dotdot_at_start_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\.\.'? segment"):
            validate_relative_path("../etc/passwd")

    def test_dotdot_at_end_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\.\.'? segment"):
            validate_relative_path("foo/..")


class TestValidatorAccepts:
    def test_simple_relative(self) -> None:
        assert validate_relative_path("foo/bar.py") == "foo/bar.py"

    def test_dotted_substring_allowed(self) -> None:
        """``foo..bar`` is NOT traversal (no path separator)."""
        assert validate_relative_path("foo..bar") == "foo..bar"

    def test_single_dot_segment_allowed(self) -> None:
        """``./foo`` is a no-op identity, not traversal."""
        assert validate_relative_path("./foo") == "./foo"

    def test_absolute_path_allowed_by_loose(self) -> None:
        """[r5 P1#2] LOOSE validator permits absolute paths for compatibility
        with PR-3 dispatch fixtures + ChildScopeGate lease conventions.
        Used by PathLease / ProposedPath only."""
        assert validate_relative_path("/absolute") == "/absolute"


class TestStrictValidatorRejections:
    """[r5 P1#2] STRICT validator additionally rejects absolute paths.
    Used by FilePatchEntry where the path is a sandbox-relative write target."""

    def test_strict_rejects_absolute(self) -> None:
        with pytest.raises(ValueError, match="must be relative"):
            validate_relative_path_strict("/etc/passwd")

    def test_strict_rejects_tmp(self) -> None:
        with pytest.raises(ValueError, match="must be relative"):
            validate_relative_path_strict("/tmp/escape")

    def test_strict_still_rejects_traversal(self) -> None:
        with pytest.raises(ValueError, match=r"\.\.'? segment"):
            validate_relative_path_strict("a/../b")

    def test_strict_accepts_relative(self) -> None:
        assert validate_relative_path_strict("a/b.py") == "a/b.py"

    def test_strict_rejects_non_canonical_dot_segment(self) -> None:
        """[codex R3 P2#6] Strict now rejects ``./foo`` because the
        reducer's cross-worker conflict detection compares paths as
        raw strings — ``foo`` and ``./foo`` would point to the same
        sandbox file but slip past CONFLICT. Loose validator (used by
        PathLease) continues to accept them."""
        with pytest.raises(ValueError, match="canonical"):
            validate_relative_path_strict("./foo")

    def test_strict_rejects_redundant_dot_segment(self) -> None:
        """``a/./b`` normalizes to ``a/b`` — non-canonical."""
        with pytest.raises(ValueError, match="canonical"):
            validate_relative_path_strict("a/./b")

    def test_strict_rejects_double_slash(self) -> None:
        """``a//b`` normalizes to ``a/b`` — non-canonical."""
        with pytest.raises(ValueError, match="canonical"):
            validate_relative_path_strict("a//b")

    def test_strict_loose_divergence(self) -> None:
        """Pin the strict-vs-loose contract: loose validator (used by
        PathLease) keeps accepting ``./foo`` for clarity in leases;
        strict (used by FilePatchEntry write target) does NOT."""
        from app.domain.models.path_validation import validate_relative_path
        assert validate_relative_path("./foo") == "./foo"
        with pytest.raises(ValueError):
            validate_relative_path_strict("./foo")


class TestToWorkspaceRelative:
    """Single-path contract: canonicalize a child write/lease path to its
    workspace-relative form (shared domain helper; the application-layer
    ``coordinator_child_runner._to_workspace_relative`` delegates to it)."""

    def test_relative_passthrough(self) -> None:
        assert to_workspace_relative("sub/x.py") == "sub/x.py"

    def test_bare_relative_passthrough(self) -> None:
        """The helper only strips the workspace prefix — the directory check
        is a separate concern (validate_coordinator_path / manifest validator)."""
        assert to_workspace_relative("x.py") == "x.py"

    def test_absolute_under_root_stripped(self) -> None:
        assert to_workspace_relative("/home/ubuntu/sub/b.py") == "sub/b.py"

    def test_absolute_root_level_stripped_to_bare(self) -> None:
        assert to_workspace_relative("/home/ubuntu/part_a.md") == "part_a.md"

    def test_absolute_outside_workspace_rejected(self) -> None:
        with pytest.raises(CoordinatorPathContractError):
            to_workspace_relative("/etc/passwd")

    def test_workspace_root_itself_rejected(self) -> None:
        with pytest.raises(CoordinatorPathContractError):
            to_workspace_relative("/home/ubuntu")


class TestValidateDirectoryQualifiedRelativePath:
    """Manifest boundary (FilePatchEntry.path). Strict-relative PLUS a required
    directory component — bare filenames are rejected at the wire schema, not
    just late at the parent ``atomic_write_file`` guard."""

    def test_accepts_directory_qualified(self) -> None:
        assert validate_directory_qualified_relative_path("workspace/part_a.md") == "workspace/part_a.md"
        assert validate_directory_qualified_relative_path("api/foo.py") == "api/foo.py"

    def test_rejects_bare_filename(self) -> None:
        with pytest.raises(ValueError, match="directory"):
            validate_directory_qualified_relative_path("part_a.md")

    def test_rejects_absolute(self) -> None:
        """Inherits the strict validator's absolute rejection."""
        with pytest.raises(ValueError):
            validate_directory_qualified_relative_path("/home/ubuntu/sub/x.py")

    def test_rejects_traversal(self) -> None:
        with pytest.raises(ValueError):
            validate_directory_qualified_relative_path("a/../b")

    def test_rejects_non_canonical(self) -> None:
        with pytest.raises(ValueError):
            validate_directory_qualified_relative_path("./a/b.py")


class TestValidateCoordinatorPath:
    """Planner/lease boundary (``_build_work_units_from_requests``). Validates a
    planner-proposed / lease path: hygiene + canonicalize-to-workspace-relative +
    require a directory component. Keeps the loose validator's absolute-allowed
    policy (N5) — an absolute path is fine as long as its workspace-relative form
    carries a directory."""

    def test_accepts_relative_directory_qualified(self) -> None:
        assert validate_coordinator_path("api/foo.py") == "api/foo.py"

    def test_accepts_absolute_directory_qualified(self) -> None:
        """Absolute lease under the workspace root is allowed (N5) as long as
        its workspace-relative tail has a directory component — and is
        CANONICALIZED to that single relative form (see
        TestValidateCoordinatorPathCanonicalizes for the full rationale)."""
        assert validate_coordinator_path("/home/ubuntu/sub/b.py") == "sub/b.py"

    def test_rejects_bare_relative(self) -> None:
        with pytest.raises(CoordinatorPathContractError, match="directory"):
            validate_coordinator_path("part_a.md")

    def test_rejects_workspace_root_absolute_bare(self) -> None:
        """``/home/ubuntu/part_a.md`` canonicalizes to bare ``part_a.md`` — the
        exact §14 live-repro path that produced ``apply_status=write_io_error``."""
        with pytest.raises(CoordinatorPathContractError, match="directory"):
            validate_coordinator_path("/home/ubuntu/part_a.md")

    def test_rejects_outside_workspace(self) -> None:
        with pytest.raises(CoordinatorPathContractError):
            validate_coordinator_path("/etc/passwd")

    def test_rejects_traversal(self) -> None:
        with pytest.raises(CoordinatorPathContractError):
            validate_coordinator_path("sub/../../etc/x.py")

    def test_error_is_value_error_subclass(self) -> None:
        """CoordinatorPathContractError must be a ValueError so it composes with
        pydantic AfterValidator semantics and generic ValueError handlers."""
        assert issubclass(CoordinatorPathContractError, ValueError)


class TestValidateCoordinatorPathCanonicalizes:
    """[single-path contract — codex CONTRACT findings] ``validate_coordinator_path``
    returns the CANONICAL directory-qualified workspace-relative form — the SAME
    form the manifest validator requires — so a lease can NEVER pass the lease
    boundary in a form (absolute / ``./`` / ``//``) that the strict manifest
    validator (or ChildScopeGate) would later reject mid-flight, stranding the
    run. This is the §14 'pin ONE canonical relative form' resolution."""

    def test_absolute_under_root_canonicalized_to_relative(self) -> None:
        assert validate_coordinator_path("/home/ubuntu/sub/b.py") == "sub/b.py"

    def test_dot_prefix_canonicalized(self) -> None:
        assert validate_coordinator_path("./api/foo.py") == "api/foo.py"

    def test_double_slash_canonicalized(self) -> None:
        assert validate_coordinator_path("api//foo.py") == "api/foo.py"

    def test_double_slash_absolute_canonicalized(self) -> None:
        assert validate_coordinator_path("/home/ubuntu//api/foo.py") == "api/foo.py"

    def test_redundant_dot_segment_canonicalized(self) -> None:
        assert validate_coordinator_path("api/./foo.py") == "api/foo.py"

    def test_canonical_output_always_passes_manifest_validator(self) -> None:
        """The CRITICAL cross-boundary invariant: whatever ``validate_coordinator_path``
        accepts at the lease boundary, its output must ALWAYS be accepted by the
        manifest validator — no lease-passes-but-manifest-fails strand."""
        for raw in [
            "./api/foo.py", "api//foo.py", "api/./foo.py",
            "/home/ubuntu/sub/b.py", "/home/ubuntu//api/foo.py",
            "workspace/x.py", "a/b/c.py",
        ]:
            canon = validate_coordinator_path(raw)
            # Must round-trip through the strict manifest validator unchanged.
            assert validate_directory_qualified_relative_path(canon) == canon

    def test_canonical_relative_unchanged(self) -> None:
        assert validate_coordinator_path("api/foo.py") == "api/foo.py"


class TestFilePatchEntryPathLengthCap:
    """[codex R11 P1] FilePatchEntry.path has Field(max_length=2048)
    aligned with coordinator_apply_audit.failed_at_path DB column +
    snapshot store NAME_MAX bounds. Test pins the schema-level cap
    so the validator stays the single source of truth.
    """

    def test_path_at_cap_accepted(self) -> None:
        """A 2048-char path is the cap boundary — accepted. Directory-qualified
        (single-path contract) while still hitting the length boundary."""
        from app.domain.models.patch_manifest import FilePatchEntry
        path = "d/" + "a" * 2046
        entry = FilePatchEntry(
            path=path, op="add",
            new_digest="a" * 64,
            content_ref="ref",
            content_size=1,
        )
        assert entry.path == path
        assert len(entry.path) == 2048

    def test_path_over_cap_rejected(self) -> None:
        """Anything > 2048 chars must be rejected by the schema so
        the DB ``failed_at_path String(2048)`` write can never
        overflow."""
        from pydantic import ValidationError
        from app.domain.models.patch_manifest import FilePatchEntry
        with pytest.raises(ValidationError):
            FilePatchEntry(
                path="a" * 2049, op="add",
                new_digest="a" * 64,
                content_ref="ref",
                content_size=1,
            )


class TestModelsWireValidator:
    def test_file_patch_entry_rejects_absolute(self) -> None:
        """[r5 P1#2 enforcement] FilePatchEntry uses the strict validator —
        absolute paths now rejected at the wire boundary (the trust point
        consumed by PR-5 PatchApplier)."""
        from app.domain.models.patch_manifest import FilePatchEntry
        with pytest.raises(ValidationError):
            FilePatchEntry(path="/etc/passwd", op="delete", base_digest="x")

    def test_path_lease_allows_absolute(self) -> None:
        """[r5 P1#2 split] PathLease keeps loose validator — absolute paths
        permitted for ChildScopeGate lease comparison compatibility."""
        from app.domain.models.work_unit import PathLease
        # Should NOT raise — absolute path legal on PathLease.
        PathLease(path="/leased", op="add")

    def test_proposed_path_allows_absolute(self) -> None:
        from app.domain.models.work_unit import ProposedPath
        ProposedPath(path="/proposed", op="add")

    def test_file_patch_entry_rejects_traversal(self) -> None:
        from app.domain.models.patch_manifest import FilePatchEntry
        with pytest.raises(ValidationError):
            FilePatchEntry(path="a/../b", op="delete", base_digest="x")

    def test_proposed_path_rejects_traversal(self) -> None:
        from app.domain.models.work_unit import ProposedPath
        with pytest.raises(ValidationError):
            ProposedPath(path="x/../y", op="add")

    def test_path_lease_rejects_traversal(self) -> None:
        from app.domain.models.work_unit import PathLease
        with pytest.raises(ValidationError):
            PathLease(path="x/../y", op="add")

    def test_file_patch_entry_rejects_nul(self) -> None:
        from app.domain.models.patch_manifest import FilePatchEntry
        with pytest.raises(ValidationError):
            FilePatchEntry(path="a\x00b", op="delete", base_digest="x")

    def test_path_lease_rejects_newline(self) -> None:
        from app.domain.models.work_unit import PathLease
        with pytest.raises(ValidationError):
            PathLease(path="a\nb", op="add")
