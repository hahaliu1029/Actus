"""C2 PR-4 r3 P1#6 — shared canonical-relative-path validator.

Used by FilePatchEntry / PathLease / ProposedPath to reject path strings that
would otherwise become injection vectors when consumed by PatchApplier or
ChildScopeGate (PR-5+):
- absolute paths bypass parent sandbox boundary
- ``..`` segments traverse outside the leased subtree
- NUL / newline / control characters confuse downstream parsers + audit logs
- empty strings (already covered by min_length, kept here for completeness)

Defense in depth: PatchApplier will re-validate against the actual lease at
apply time. The wire-schema validator just catches the obvious injection
attempts at the earliest possible point (envelope construction).
"""
from __future__ import annotations

import posixpath


# The sandbox workspace root (``~`` for the sandbox user); consistent across the
# codebase (langchain_mcp ``_SANDBOX_PATH_PREFIX``, skill bundle root, the
# Sandbox Workspace Isolation epic's ``workspace_root``). The single-path
# contract canonicalizes a coordinator path against this root before checking
# for a directory component.
WORKSPACE_ROOT = "/home/ubuntu"


class CoordinatorPathContractError(ValueError):
    """Single-path contract violation at the planner/lease boundary.

    Raised by ``validate_coordinator_path`` / ``to_workspace_relative`` when a
    planner-proposed or lease path is NOT a directory-qualified
    workspace-relative path — i.e. a bare filename (``part_a.md``), a
    workspace-root absolute that canonicalizes to bare
    (``/home/ubuntu/part_a.md``), or an absolute path outside the workspace
    root. Rejecting at this boundary stops a bad path BEFORE children spawn,
    instead of letting it surface late as the parent ``atomic_write_file``
    ``write_io_error`` (the §14 live-repro). A ``ValueError`` subclass so it
    composes with pydantic AfterValidator semantics and generic handlers, while
    still being catchable specifically (``_run_parallel_backend`` converts it to
    a graceful failed ``ParallelBackendOutcome``)."""


def to_workspace_relative(path: str) -> str:
    """Canonicalize a coordinator write/lease path to its workspace-relative form.

    - already-relative paths pass through unchanged (existing convention);
    - absolute paths under ``WORKSPACE_ROOT`` are stripped to the relative tail
      (``/home/ubuntu/sub/b.py`` -> ``sub/b.py``);
    - an absolute path OUTSIDE the workspace root (or the root itself) is a
      sandbox-escape attempt and raises ``CoordinatorPathContractError``.

    This is the single source of truth for the abs->rel mapping; the
    application-layer ``coordinator_child_runner._to_workspace_relative``
    delegates here (translating to its ``_OutOfLeaseWriteError`` so the child
    finalizer routes to NEEDS_AUTHORIZATION). The directory-component check is a
    SEPARATE concern (``validate_coordinator_path`` /
    ``validate_directory_qualified_relative_path``) — this helper only strips
    the prefix.
    """
    if not path.startswith("/"):
        return path  # already workspace-relative
    prefix = WORKSPACE_ROOT.rstrip("/") + "/"
    if path.startswith(prefix):
        rel = path[len(prefix):]
        if rel:
            return rel
    raise CoordinatorPathContractError(
        f"path {path!r} is absolute but not under the workspace root "
        f"{WORKSPACE_ROOT!r}"
    )


def validate_relative_path(value: str) -> str:
    """Pydantic AfterValidator — fail closed on dangerous path shapes.

    Rejects:
    - empty / non-string
    - NUL byte (truncates downstream filesystem APIs silently)
    - newline / carriage-return (corrupts audit log lines + shell quoting)
    - ``..`` as a path segment (traversal — ``foo..bar`` is legal, ``foo/../bar`` is not)

    Does NOT reject absolute paths — used by PathLease / ProposedPath where
    existing PR-3 dispatch fixtures + ChildScopeGate lease conventions use
    absolute paths. For the security-sensitive wire boundary (PatchManifest),
    use ``validate_relative_path_strict`` which additionally rejects absolute
    paths.

    Returns the value unchanged when valid. Raises ValueError on rejection.
    """
    if not isinstance(value, str):
        raise ValueError(f"path must be str, got {type(value).__name__}")
    if not value:
        raise ValueError("path must be non-empty")
    if "\x00" in value:
        raise ValueError(f"path must not contain NUL byte: {value!r}")
    if "\n" in value or "\r" in value:
        raise ValueError(f"path must not contain newline: {value!r}")
    # ``..`` as a path segment (not just substring — ``foo..bar`` is legal)
    segments = value.split("/")
    if ".." in segments:
        raise ValueError(
            f"path must not contain '..' segment (traversal attempt): {value!r}"
        )
    return value


def validate_relative_path_strict(value: str) -> str:
    """[r5 P1#2] Stricter variant — same checks as ``validate_relative_path``
    PLUS rejects absolute paths AND non-canonical relative forms.

    Used by ``FilePatchEntry.path`` where the path is a write target the
    child has authority to produce. A child publishing a manifest with
    ``/tmp/...`` or ``/etc/...`` is an attempted escape from the parent
    sandbox; the relative→absolute join now lives in the sandbox file service
    (``sandbox/app/core/workspace.py``, the Sandbox Workspace Isolation epic),
    anchoring the path under ``workspace_root`` (/home/ubuntu). Fail-closed
    here keeps the manifest wire schema honest about being sandbox-relative.

    [codex R3 P2#6 fix] Reject non-canonical relative forms (``./x.py``,
    ``x/./y.py``, ``x//y.py``) so the reducer's cross-worker conflict
    detection can compare paths as raw strings. Without this two
    workers could both write ``x.py`` and ``./x.py`` and bypass the
    CONFLICT detection — the applier would then issue two writes to
    the same final sandbox file.
    """
    value = validate_relative_path(value)
    if value.startswith("/"):
        raise ValueError(
            f"path must be relative (got absolute path {value!r}); "
            "PatchManifest entries are sandbox-relative — absolute paths "
            "would escape the parent sandbox boundary in PR-5 PatchApplier"
        )
    # posixpath.normpath strips redundant ``./``, ``/./``, ``//`` etc.
    # Use posixpath (not os.path) so the same canonicalization applies
    # on every platform — sandbox paths are always POSIX-style.
    import posixpath
    canonical = posixpath.normpath(value)
    if canonical != value:
        raise ValueError(
            f"path must be in canonical form (got {value!r}, "
            f"canonical={canonical!r}); reducer conflict detection "
            "compares paths as raw strings and ``./x.py`` vs ``x.py`` "
            "would both reach the same sandbox file"
        )
    return value


def validate_directory_qualified_relative_path(value: str) -> str:
    """[single-path contract — manifest boundary] STRICT relative PLUS a
    required directory component.

    Used by ``FilePatchEntry.path`` (the coordinator MANIFEST wire schema). The
    bare-filename rejection that previously lived ONLY in the host
    ``ParentSandboxAdapter.atomic_write_file`` guard (surfacing late as
    ``write_io_error`` after children ran + the reducer built an apply plan —
    the §14 live-repro) is moved forward to the wire schema here, so an invalid
    manifest path is caught at envelope construction. A coordinator manifest
    path must be directory-qualified workspace-relative (e.g.
    ``workspace/foo.py``, ``api/bar.py``) — a bare ``part_a.md`` is rejected.

    The adapter guard STAYS as a documented defense-in-depth backstop (spec
    §3.8 / D6); this validator makes it a should-never-fire last line.
    """
    value = validate_relative_path_strict(value)
    # ``validate_relative_path_strict`` already guarantees a canonical relative
    # path (no leading/trailing slash, no ``./``), so ``"/" in value`` is an
    # exact test for "carries a directory component".
    if "/" not in value:
        raise ValueError(
            f"bare filename rejected (no directory component): {value!r}; "
            f"coordinator manifest paths must be directory-qualified "
            f"workspace-relative (e.g. 'workspace/foo.py', 'api/bar.py')"
        )
    return value


def validate_coordinator_path(value: str) -> str:
    """[single-path contract — planner/lease boundary] Validate AND canonicalize
    a planner-proposed / lease path to the ONE canonical directory-qualified
    workspace-relative form.

    Used by ``_build_work_units_from_requests`` (the ProposedPath -> PathLease
    conversion, i.e. the lease-construction boundary) so a planner that proposes
    a bare or workspace-root path is rejected BEFORE any child spawns, rather
    than producing a bare manifest path that fails late at apply.

    Crucially this RETURNS the canonical form — the SAME form
    ``validate_directory_qualified_relative_path`` (the manifest validator)
    requires — so the lease can never be accepted in a shape (absolute, ``./``,
    ``//``, redundant segments) that the strict manifest validator or
    ChildScopeGate would later reject mid-flight, stranding the run. This is the
    spec §14 "pin ONE canonical relative form that child-write -> manifest ->
    parent atomic_write_file all agree on" resolution. The PathLease MODEL stays
    loose (N5) — only the coordinator's lease-construction is canonicalized here.

    Accepts (and normalizes): ``api/foo.py``, ``./api/foo.py``, ``api//foo.py``,
    ``/home/ubuntu/sub/b.py`` (-> ``sub/b.py``). Rejects (raises
    ``CoordinatorPathContractError``, a ``ValueError`` subclass): bare
    ``part_a.md`` / workspace-root ``/home/ubuntu/part_a.md`` (no directory
    component), absolute paths outside the workspace root, and the NUL / newline
    / ``..``-traversal shapes the loose hygiene validator rejects.
    """
    try:
        # Loose hygiene on the ORIGINAL: rejects NUL / newline / ``..`` segments
        # (BEFORE normpath, which could otherwise collapse a ``..``); permits
        # absolute + non-canonical (handled next). Re-raise as the contract
        # error so EVERY rejection from this entry point is one type.
        validate_relative_path(value)
    except CoordinatorPathContractError:
        raise
    except ValueError as exc:
        raise CoordinatorPathContractError(str(exc)) from exc
    # Canonicalize FIRST (collapse ``./`` / ``//`` / redundant segments) so an
    # absolute double-slash form strips cleanly, THEN map absolute-under-root to
    # its relative tail (raises for absolute-outside-root / the root itself).
    canonical = to_workspace_relative(posixpath.normpath(value))
    if "/" not in canonical:
        raise CoordinatorPathContractError(
            f"bare filename rejected (no directory component): {value!r}; "
            f"coordinator paths must be directory-qualified workspace-relative "
            f"(e.g. 'workspace/foo.py', 'api/bar.py'). Put a new file in a "
            f"subdirectory such as 'workspace/'."
        )
    return canonical
