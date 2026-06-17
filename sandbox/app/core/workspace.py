"""Workspace path anchoring + service-tree write protection.

Single chokepoint shared by ``file.py`` and ``shell.py`` (DRY). RELATIVE paths
anchor to ``workspace_root`` (/home/ubuntu) instead of the process CWD
(/sandbox = the service install dir). ABSOLUTE paths pass through unchanged.
Writes/deletes resolving under ``service_install_dir`` (/sandbox) are denied.

Threat model: ACCIDENTAL misdirection (coordinator manifest-relative paths,
LLM-improvised relatives), NOT a malicious agent — it has a root shell. See the
design spec §1/§8.
"""
import os.path

from app.core.config import get_settings
from app.interfaces.errors.exceptions import BadRequestException


class OutsideWorkspaceError(BadRequestException):
    """A relative path escaped the workspace root (Guard A) → HTTP 400."""

    def __init__(self, msg: str = "路径超出工作区范围") -> None:
        super().__init__(msg=msg)


class ServiceTreeWriteDenied(BadRequestException):
    """A write/delete resolved under the protected service dir (Guard B) → 400."""

    def __init__(self, msg: str = "禁止写入服务目录") -> None:
        super().__init__(msg=msg)


def _within_or_equal(child: str, parent: str) -> bool:
    """True iff ``child`` is ``parent`` or a descendant — by PATH COMPONENT
    containment (os.path.commonpath), NOT string prefix (so ``/sandbox-foo`` is
    NOT within ``/sandbox``). Both args must be absolute."""
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:
        # mixed abs/rel or different drives -> not contained (fail-closed).
        return False


def resolve_in_workspace(path: str, *, follow_final: bool = False) -> str:
    """Anchor a RELATIVE ``path`` under ``workspace_root`` and confine it
    (Guard A). ABSOLUTE paths and ``""``/``"."`` are special-cased. Returns the
    absolute path to use for the filesystem operation."""
    workspace_root = get_settings().workspace_root

    # EMPTY / "." -> the workspace root itself. A WRITE to it fails naturally as
    # IsADirectory (not an escape).
    if path == "" or path == ".":
        return workspace_root

    # ABSOLUTE -> pass through unchanged (preserves /home/ubuntu, /tmp,
    # /workspace/.memory, skills, ...).
    if os.path.isabs(path):
        return path

    # RELATIVE -> anchor + confine.
    cand = os.path.normpath(os.path.join(workspace_root, path))
    root_real = os.path.realpath(workspace_root)
    if follow_final:
        # dir-traversal target (find_files dir_path): resolve the FINAL
        # component too — a dir_path that is itself a symlink to /sandbox must
        # be caught here.
        check = os.path.realpath(cand)
    else:
        # single-file / atomic-write target: follow ANCESTOR symlinks, keep the
        # final component lexical (matches the atomic "replace final in place"
        # semantics).
        check = os.path.join(
            os.path.realpath(os.path.dirname(cand)),
            os.path.basename(cand),
        )
    if not _within_or_equal(check, root_real):
        raise OutsideWorkspaceError(
            f"路径 {path!r} 超出工作区 {workspace_root!r} 范围"
        )
    return cand


def is_within_workspace(abs_path: str) -> bool:
    """True iff ``abs_path`` (after full realpath) is within-or-equal the
    workspace root — used by find_files result filtering."""
    root_real = os.path.realpath(get_settings().workspace_root)
    return _within_or_equal(os.path.realpath(abs_path), root_real)


def deny_service_tree_write(resolved_path: str, *, follows_final_symlink: bool) -> None:
    """Guard B — raise ``ServiceTreeWriteDenied`` if a write/delete to
    ``resolved_path`` would land under ``service_install_dir`` (/sandbox).

    ``follows_final_symlink=True`` (append/sudo — they FOLLOW the final symlink)
    resolves the final component; ``False`` (atomic overwrite / delete —
    replace-or-remove the final component in place) uses the parent realpath
    joined with the basename."""
    install_real = os.path.realpath(get_settings().service_install_dir)
    if follows_final_symlink:
        effective = os.path.realpath(resolved_path)
    else:
        effective = os.path.join(
            os.path.realpath(os.path.dirname(resolved_path)),
            os.path.basename(resolved_path),
        )
    if _within_or_equal(effective, install_real):
        raise ServiceTreeWriteDenied(
            f"禁止写入/删除服务目录 {get_settings().service_install_dir!r}: {resolved_path!r}"
        )
