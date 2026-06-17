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
