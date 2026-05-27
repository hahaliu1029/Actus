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


class TestFilePatchEntryPathLengthCap:
    """[codex R11 P1] FilePatchEntry.path has Field(max_length=2048)
    aligned with coordinator_apply_audit.failed_at_path DB column +
    snapshot store NAME_MAX bounds. Test pins the schema-level cap
    so the validator stays the single source of truth.
    """

    def test_path_at_cap_accepted(self) -> None:
        """A 2048-char path is the cap boundary — accepted."""
        from app.domain.models.patch_manifest import FilePatchEntry
        path = "a" * 2048
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
