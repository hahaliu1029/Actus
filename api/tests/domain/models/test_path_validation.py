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

    def test_strict_accepts_dot_segment(self) -> None:
        """``./foo`` is current-dir not traversal — strict allows it like loose."""
        assert validate_relative_path_strict("./foo") == "./foo"


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
