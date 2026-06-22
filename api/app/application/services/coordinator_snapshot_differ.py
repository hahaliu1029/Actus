"""C2-full S2 §3.1/§3.2 — pure snapshot differ (tree-INDEPENDENT slice).

PR-1 scope: identity-tuple change detection between PRE and POST
``WorkspaceScan``s + the structural zero-apply signals that need no lease/tree
input (special/kind transitions, truncation, non-dir-qualified path). PR-4
extends this with lease revalidation, TreeLease enforcement, parent-precheck,
and the byte-capture→MinIO step.

Diff identity tuple = ``(kind, sha256, size, mode, link_target)``. Two entries
with equal tuples are a no-op (F15). An entry present in POST but not PRE (by
equal tuple) is an add; a tuple change at an existing path is a modify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from app.domain.external.parent_sandbox import (
    FileKind,
    WorkspaceScan,
    WorkspaceScanEntry,
)


def _identity(e: WorkspaceScanEntry) -> tuple:
    return (e.kind, e.sha256, e.size, e.mode, e.link_target)


@dataclass(frozen=True)
class SnapshotChange:
    """One detected change. ``post_entry`` is the POST-scan inode; PR-4 turns
    it into a FilePatchEntry via byte capture."""

    rel_path: str
    kind: "FileKind"
    change: Literal["add", "modify"]
    post_entry: WorkspaceScanEntry


@dataclass(frozen=True)
class SnapshotDiffResult:
    """Result of a PRE/POST diff. ``zero_apply`` (with reason) is the
    group-level abort signal — PR-4 maps it to GroupOutcome / NEEDS_AUTHORIZATION
    reason codes."""

    changes: list[SnapshotChange] = field(default_factory=list)
    zero_apply: bool = False
    zero_apply_reason: Optional[str] = None


def diff_snapshots(
    pre: WorkspaceScan, post: WorkspaceScan
) -> SnapshotDiffResult:
    """Tree-independent identity-tuple diff (PR-1 slice).

    PR-4 prepends the lease/tree/parent-precheck gates and the truncation /
    special-transition / non-dir-qualified zero-apply rules; this function is
    the inner change-detection it builds on.
    """
    # F17 — a truncated scan (either side) is an incomplete view; never apply
    # a partial diff. Group zero-apply.
    if pre.truncated or post.truncated:
        return SnapshotDiffResult(zero_apply=True, zero_apply_reason="scan_truncated")

    changes: list[SnapshotChange] = []
    for rel_path, post_entry in post.entries.items():
        # F18 — a changed path must be directory-qualified (carry a '/').
        # A bare top-level file forces group zero-apply with the canonical
        # §3.4 reason ``out_of_path_lease`` (the same reason PR-4's capture
        # path emits for this case).
        pre_entry = pre.entries.get(rel_path)
        if pre_entry is not None and _identity(pre_entry) == _identity(post_entry):
            continue  # F15 no-op — unchanged paths never trigger F18
        if "/" not in rel_path:
            return SnapshotDiffResult(
                zero_apply=True, zero_apply_reason="out_of_path_lease",
            )
        # F8/F12 — any non-regular POST inode (symlink most commonly) is not
        # appliable; group zero-apply. This also covers a regular->symlink kind
        # change at an existing path (the tuple already differs).
        if post_entry.kind != "regular":
            return SnapshotDiffResult(
                zero_apply=True, zero_apply_reason="symlink",
            )
        # F12 (other direction) / kind change INTO regular from a non-regular
        # PRE — also not a clean content modify; zero-apply.
        if pre_entry is not None and pre_entry.kind != "regular":
            return SnapshotDiffResult(
                zero_apply=True, zero_apply_reason="symlink",
            )
        # F11 — same content (sha256 + size) but mode changed only; reducer
        # cannot represent a mode-only patch -> group zero-apply.
        if (
            pre_entry is not None
            and pre_entry.sha256 == post_entry.sha256
            and pre_entry.size == post_entry.size
            and pre_entry.mode != post_entry.mode
        ):
            return SnapshotDiffResult(
                zero_apply=True, zero_apply_reason="mode_only_change",
            )
        if pre_entry is None:
            changes.append(
                SnapshotChange(
                    rel_path=rel_path,
                    kind=post_entry.kind,
                    change="add",
                    post_entry=post_entry,
                )
            )
            continue
        changes.append(
            SnapshotChange(
                rel_path=rel_path,
                kind=post_entry.kind,
                change="modify",
                post_entry=post_entry,
            )
        )
    return SnapshotDiffResult(changes=changes)
