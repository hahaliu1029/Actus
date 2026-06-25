"""S2 PR-1 — FileService.snapshot_workspace os.walk + caps + fail-closed.

CI-only: runs under the SANDBOX venv (`cd sandbox && uv run pytest`); needs a
real filesystem (tmp_path) but NO live container.
"""
from __future__ import annotations

import hashlib
import os

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ws(tmp_path, monkeypatch):
    from app.core import config as cfg

    ws = tmp_path / "home"
    ws.mkdir()
    monkeypatch.setenv("WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("SERVICE_INSTALL_DIR", "/sandbox")
    cfg.get_settings.cache_clear()
    return ws


async def test_regular_file_emits_sha_and_size(tmp_path, monkeypatch):
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace").mkdir()
        (ws / "workspace" / "a.py").write_bytes(b"hello")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        e = scan.entries["workspace/a.py"]
        assert e.kind == "regular"
        assert e.sha256 == hashlib.sha256(b"hello").hexdigest()
        assert e.size == 5
        assert e.link_target is None
        assert scan.truncated is False
    finally:
        cfg.get_settings.cache_clear()


async def test_directories_are_excluded(tmp_path, monkeypatch):
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace" / "sub").mkdir(parents=True)
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        # F19: directory-only — no entries emitted for dirs.
        assert scan.entries == {}
    finally:
        cfg.get_settings.cache_clear()


async def test_symlink_emits_link_target_no_sha(tmp_path, monkeypatch):
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace").mkdir()
        (ws / "workspace" / "t.py").write_bytes(b"x")
        os.symlink("t.py", ws / "workspace" / "link")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        link = scan.entries["workspace/link"]
        assert link.kind == "symlink"
        assert link.sha256 is None
        assert link.link_target == "t.py"
    finally:
        cfg.get_settings.cache_clear()


async def test_max_files_cap_truncates(tmp_path, monkeypatch):
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace").mkdir()
        for i in range(5):
            (ws / "workspace" / f"f{i}.txt").write_bytes(b"x")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=2,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        assert scan.truncated is True
    finally:
        cfg.get_settings.cache_clear()


async def test_dir_only_deep_tree_trips_max_paths(tmp_path, monkeypatch):
    # max_paths counts DIRECTORY ENTRIES + files (spec §3.1). A deep dir-only
    # tree with NO files must still trip the cap — the counter increments while
    # walking ``dirnames``, not only inside ``filenames``.
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        # 10 nested directories, ZERO files.
        deep = ws / "workspace"
        for i in range(10):
            deep = deep / f"d{i}"
        deep.mkdir(parents=True)
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=3, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        # No files emitted, but the directory-node count blew the cap.
        assert scan.entries == {}
        assert scan.truncated is True
    finally:
        cfg.get_settings.cache_clear()


async def test_symlink_to_directory_in_dirnames_emitted_not_descended(
    tmp_path, monkeypatch
):
    # F8-class: a symlink-to-a-directory shows up in os.walk's ``dirnames``
    # list (followlinks=False), NOT ``filenames``. It must be os.lstat'd like
    # any other entry -> emitted as kind="symlink" (with link_target) and NOT
    # descended into. A REAL directory is traversed-not-emitted.
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace" / "realdir").mkdir(parents=True)
        (ws / "workspace" / "realdir" / "inside.py").write_bytes(b"i")
        # symlink whose target is a directory -> appears in dirnames.
        os.symlink("realdir", ws / "workspace" / "dlink")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        # The dir-symlink is emitted as a symlink entry...
        dlink = scan.entries["workspace/dlink"]
        assert dlink.kind == "symlink"
        assert dlink.link_target == "realdir"
        assert dlink.sha256 is None
        # ...and NOT descended (no inside.py reached THROUGH the symlink).
        assert "workspace/dlink/inside.py" not in scan.entries
        # The real directory is traversed (its child emitted) but the dir
        # itself is never an entry.
        assert "workspace/realdir/inside.py" in scan.entries
        assert "workspace/realdir" not in scan.entries
    finally:
        cfg.get_settings.cache_clear()


async def test_classifiable_special_file_is_emitted_not_truncated(
    tmp_path, monkeypatch
):
    # P0 PR-1<->PR-4 contract: a CLASSIFIABLE special inode (fifo/socket/
    # block/char) is EMITTED as kind=<special> (sha256=None, link_target=None),
    # NOT skipped and NOT treated as truncation (spec §3.1: fail-CLOSED only on
    # the INDETERMINATE ``other`` kind). The downstream PR-1 differ (F8) and the
    # PR-4 differ (F9, reason ``special_file``) reject it — but ONLY if a real
    # scan actually surfaces the special entry, hence this assertion.
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace").mkdir()
        os.mkfifo(ws / "workspace" / "pipe")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        # Emitted as a fifo entry — NOT a truncated scan.
        assert scan.truncated is False
        pipe = scan.entries["workspace/pipe"]
        assert pipe.kind == "fifo"
        assert pipe.sha256 is None  # no content read for a special inode
        assert pipe.link_target is None  # only symlinks carry a target
    finally:
        cfg.get_settings.cache_clear()


async def test_indeterminate_kind_fails_closed_truncated(
    tmp_path, monkeypatch
):
    # Spec §3.1: an INDETERMINATE / unclassifiable inode (``kind == "other"``)
    # is the ONLY kind that fails CLOSED -> truncated, no entry. ``other`` types
    # (S_ISDOOR/PORT/WHT, …) are not portably creatable on Linux, so we
    # monkeypatch ``_classify`` to return ``"other"`` for the one file we drop
    # in, exercising the real fail-closed branch in ``_emit``.
    from app.core import config as cfg
    from app.services import file as file_mod
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace").mkdir()
        (ws / "workspace" / "weird").write_bytes(b"z")
        real_classify = file_mod._classify

        def _fake_classify(st):
            kind = real_classify(st)
            return "other" if kind == "regular" else kind

        monkeypatch.setattr(file_mod, "_classify", _fake_classify)
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        # fail-CLOSED on the indeterminate inode -> truncated, no entry.
        assert scan.truncated is True
        assert "workspace/weird" not in scan.entries
    finally:
        cfg.get_settings.cache_clear()


async def test_configured_memory_mount_subtree_excluded(tmp_path, monkeypatch):
    # P1-c: the read-only memory MOUNT is excluded by the CONFIGURED
    # ``memory_mount_target`` (NOT ``workspace_root/.memory``). To exercise the
    # exclusion in a tmp_path harness we point the mount target UNDER the test
    # workspace (``ws/.memory``); only THEN is that subtree dropped. ``/sandbox``
    # (service tree) is likewise excluded.
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    # Point the memory mount target at ``ws/.memory`` so the realpath-containment
    # prune actually matches inside this scan; re-clear the settings cache.
    monkeypatch.setenv("MEMORY_MOUNT_TARGET", str(ws / ".memory"))
    cfg.get_settings.cache_clear()
    try:
        (ws / ".memory").mkdir()
        (ws / ".memory" / "secret.md").write_bytes(b"m")
        (ws / "workspace").mkdir()
        (ws / "workspace" / "ok.py").write_bytes(b"k")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        assert "workspace/ok.py" in scan.entries
        # mount target == ws/.memory -> THAT subtree is dropped.
        assert ".memory/secret.md" not in scan.entries
    finally:
        cfg.get_settings.cache_clear()


async def test_workspace_dot_memory_captured_with_default_mount(
    tmp_path, monkeypatch
):
    # P1-c (the bug fix): with the DEFAULT mount target (``/workspace/.memory``,
    # OUTSIDE the test workspace — MEMORY_MOUNT_TARGET deliberately NOT set), a
    # child's own ``ws/.memory`` is the child's legitimate write directory, NOT
    # the read-only mount, so it must be CAPTURED. The OLD code excluded any
    # ``workspace_root/.memory`` (wrong path) -> this asserted the file was
    # excluded -> silent data-loss. RED before the fix, GREEN after.
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)  # default MEMORY_MOUNT_TARGET (/workspace/.memory)
    try:
        (ws / ".memory").mkdir()
        (ws / ".memory" / "hidden.txt").write_bytes(b"h")
        (ws / "workspace").mkdir()
        (ws / "workspace" / "ok.py").write_bytes(b"k")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        assert "workspace/ok.py" in scan.entries
        # default mount target is /workspace/.memory (outside this ws) -> the
        # child's own ws/.memory is CAPTURED, not the mount.
        assert ".memory/hidden.txt" in scan.entries
        assert scan.truncated is False
    finally:
        cfg.get_settings.cache_clear()


async def test_symlink_to_sandbox_is_emitted_not_pruned(tmp_path, monkeypatch):
    # P1-2: a symlink whose target resolves INTO the service tree (/sandbox)
    # must be EMITTED as kind="symlink", NOT silently pruned by the /sandbox
    # realpath/commonpath drop. Spec §3.1/L136: ANY symlink must surface so the
    # downstream differ group-zero-applies. The /sandbox prune is for REAL dirs
    # under the service tree only — a symlink POINTING at /sandbox is data, not a
    # subtree to descend.
    #
    # NOTE: the symlink target must be a REAL directory for the symlink to land
    # in os.walk's ``dirnames`` (where the buggy pre-lstat prune lives). A
    # symlink to a non-existent path lands in ``filenames`` and is unaffected.
    # So we point SERVICE_INSTALL_DIR at a real dir under tmp_path and symlink to
    # it — this exercises the exact ``commonpath(...) == install_real`` prune.
    from app.core import config as cfg
    from app.services.file import FileService

    install = tmp_path / "service_tree"
    install.mkdir()
    (install / "internal.py").write_bytes(b"secret")
    ws = _ws(tmp_path, monkeypatch)  # sets WORKSPACE_ROOT + SERVICE_INSTALL_DIR=/sandbox
    # Override SERVICE_INSTALL_DIR to the REAL tmp dir AFTER _ws, then re-clear
    # the settings cache so get_settings() picks up the override.
    monkeypatch.setenv("SERVICE_INSTALL_DIR", str(install))
    cfg.get_settings.cache_clear()
    try:
        (ws / "workspace").mkdir()
        # absolute symlink whose realpath IS the (real) service install dir.
        os.symlink(str(install), ws / "workspace" / "slink")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        slink = scan.entries["workspace/slink"]
        assert slink.kind == "symlink"
        assert slink.link_target == str(install)
        assert slink.sha256 is None
        # NOT descended: nothing from under the symlink target leaks in.
        assert not any(k.startswith("workspace/slink/") for k in scan.entries)
        assert "workspace/slink/internal.py" not in scan.entries
        assert scan.truncated is False
    finally:
        cfg.get_settings.cache_clear()


async def test_mode_is_permission_bits_only(tmp_path, monkeypatch):
    # P2-1: spec L110 — ``mode`` is st_mode PERMISSION bits (S_IMODE), not the
    # full st_mode (which includes the S_IFREG type bits 0o100000). A 0o750
    # regular file must report mode == 0o750, not 0o100750.
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace").mkdir()
        f = ws / "workspace" / "x.sh"
        f.write_bytes(b"#!/bin/sh\n")
        os.chmod(f, 0o750)
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        assert scan.entries["workspace/x.sh"].mode == 0o750
    finally:
        cfg.get_settings.cache_clear()


async def test_max_files_counts_regular_only(tmp_path, monkeypatch):
    # P2-2: spec L123 — max_files is the REGULAR-file count. A symlink (or any
    # non-regular emitted inode) must NOT consume a max_files slot. With
    # max_files=1 and ONE regular file + ONE symlink, both must be emitted and
    # the scan must NOT be truncated (non-regulars are bounded by max_paths).
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace").mkdir()
        (ws / "workspace" / "only.txt").write_bytes(b"r")
        os.symlink("only.txt", ws / "workspace" / "ln")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        assert scan.truncated is False
        assert "workspace/only.txt" in scan.entries
        assert "workspace/ln" in scan.entries
    finally:
        cfg.get_settings.cache_clear()


# P1-4 (os.walk fail-closed on a scandir error): a chmod-000 attempt is vacuous
# here because the walk runs as root (chmod 000 on a directory does NOT block
# root traversal, so the assertion would never go RED). R5 instead locks the
# behavior DETERMINISTICALLY by monkeypatching ``os.walk`` to raise mid-iteration
# — see ``test_os_walk_onerror_fails_closed`` below, which exercises the
# ``onerror=_on_err`` re-raise + the ``try/except OSError -> truncated=True``
# wrapper around the os.walk loop in ``snapshot_workspace``.


async def test_memory_symlink_emitted_not_pruned(tmp_path, monkeypatch):
    # R2-A: a symlink NAMED ``.memory`` pointing at a REAL directory lands in
    # os.walk's ``dirnames`` (followlinks=False). The ``.memory`` basename prune
    # must run ONLY for a REAL directory — a ``.memory`` SYMLINK must surface as
    # kind="symlink" (spec §3.1/L136: ANY symlink must surface so the differ
    # group-zero-applies), NOT be silently dropped by the basename prune. This is
    # the same class as the /sandbox pre-lstat prune fixed by P1-2.
    #
    # NOTE: the target must be a REAL directory so the symlink lands in
    # ``dirnames`` (where the buggy pre-lstat ``.memory`` prune lived). A symlink
    # to a non-existent path would land in ``filenames`` and be unaffected.
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        realtarget = ws / "realtarget"
        realtarget.mkdir()
        (realtarget / "inside.py").write_bytes(b"i")
        (ws / "workspace").mkdir()
        # symlink NAMED .memory -> a real dir, so it lands in dirnames.
        os.symlink(str(realtarget), ws / "workspace" / ".memory")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        mem = scan.entries["workspace/.memory"]
        assert mem.kind == "symlink"
        assert mem.link_target == str(realtarget)
        assert mem.sha256 is None
        # NOT descended through the symlink.
        assert "workspace/.memory/inside.py" not in scan.entries
        assert scan.truncated is False
    finally:
        cfg.get_settings.cache_clear()


async def test_in_hash_deadline_trips_truncates(tmp_path, monkeypatch):
    # R5-P1: the wall-clock deadline (max_seconds) is re-checked DURING the
    # chunked read+hash of a regular file, not only at the per-entry loop top.
    # A slow/large file whose read overshoots the cap must set truncated=True
    # and NOT be emitted.
    #
    # DETERMINISTIC (no real slow I/O): we monkeypatch ``time.monotonic`` in the
    # ``file`` module's namespace with a stateful counter-backed fake. The first
    # few calls (``started`` capture + the dirnames/filenames loop-top checks)
    # return ``base`` / ``base + 0.001`` so elapsed < max_seconds (the per-entry
    # gate PASSES and we reach the hash loop); a LATER call — the in-hash
    # deadline check — returns ``base + 999`` so elapsed > max_seconds, tripping
    # the mid-hash truncate. No dependence on real clock or file size.
    from app.core import config as cfg
    from app.services import file as file_mod
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace").mkdir()
        (ws / "workspace" / "slow.txt").write_bytes(b"content")

        base = 1000.0
        calls = {"n": 0}

        def _fake_monotonic() -> float:
            # call 0 = ``started`` capture -> base
            # calls 1..N (dirnames + filenames loop-top gates) -> base + 0.001
            #   (elapsed 0.001 < max_seconds=10 -> PASS, reach the hash loop)
            # the NEXT call is the in-hash deadline check -> base + 999
            #   (elapsed 999 > 10 -> truncate mid-hash, file NOT emitted)
            n = calls["n"]
            calls["n"] += 1
            if n == 0:
                return base
            if n <= 3:
                return base + 0.001
            return base + 999.0

        monkeypatch.setattr(file_mod.time, "monotonic", _fake_monotonic)
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        # mid-hash deadline breach -> truncated, file NOT emitted.
        assert scan.truncated is True
        assert "workspace/slow.txt" not in scan.entries
    finally:
        cfg.get_settings.cache_clear()


async def test_os_walk_onerror_fails_closed(tmp_path, monkeypatch):
    # R5-P2-2: P1-4's os.walk fail-closed (onerror re-raises + the walk loop is
    # wrapped in ``try/except OSError -> truncated=True``) is locked here by a
    # DETERMINISTIC regression test. We monkeypatch ``os.walk`` in the ``file``
    # module's namespace with a generator that raises ``OSError`` mid-iteration
    # (the same shape as a real scandir/lstat failure surfaced via onerror).
    # The scan must return with truncated=True and NOT propagate the exception.
    #
    # This is written as a GREEN lock (asserting the already-fixed behavior): a
    # RED-first version would require temporarily removing the try/except wrapper
    # in production code, which we do not do. With the wrapper in place the raise
    # is caught and converted to a truncated scan; without it the OSError would
    # propagate out of ``snapshot_workspace`` and this test would fail (the
    # mutation that drops the except clause makes it go RED).
    from app.core import config as cfg
    from app.services import file as file_mod
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "workspace").mkdir()
        (ws / "workspace" / "a.py").write_bytes(b"x")

        def _boom_walk(*args, **kwargs):
            # os.walk is a generator: yield nothing, raise on first __next__,
            # mirroring a scandir error re-raised by the onerror callback.
            raise OSError("simulated scandir failure")
            yield  # pragma: no cover  (makes this a generator)

        monkeypatch.setattr(file_mod.os, "walk", _boom_walk)
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        # fail CLOSED: the walk-level OSError is caught -> truncated, no raise.
        assert scan.truncated is True
        assert scan.entries == {}
    finally:
        cfg.get_settings.cache_clear()


async def test_nested_dot_memory_real_dir_is_descended_not_pruned(
    tmp_path, monkeypatch
):
    # R6-P1: ``.memory`` must be excluded by MOUNT-PATH CONTAINMENT, not by
    # basename. The read-only memory mount lives at exactly
    # ``workspace_root/.memory``; only THAT subtree is dropped. A NESTED real
    # directory that merely happens to be NAMED ``.memory`` (e.g.
    # ``proj/.memory``) is a child's legitimate write — it must be DESCENDED and
    # its files emitted, NOT silently skipped (the old basename prune
    # ``if d == ".memory": continue`` was a fail-open: the differ saw a clean
    # no-op and the child's writes were lost with truncated=False).
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "proj" / ".memory").mkdir(parents=True)
        (ws / "proj" / ".memory" / "hidden.txt").write_bytes(b"h")
        (ws / "proj" / "ok.py").write_bytes(b"k")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        # The NESTED .memory (NOT the root mount) is descended + emitted.
        assert "proj/.memory/hidden.txt" in scan.entries
        assert "proj/ok.py" in scan.entries
        assert scan.truncated is False
    finally:
        cfg.get_settings.cache_clear()


async def test_configured_skills_bundle_subtree_excluded(tmp_path, monkeypatch):
    # R10-2 (S4 PR-5): the foreground bundle sync of a fresh native MEMBER skill
    # writes its files under the CONFIGURED ``skill_sandbox_bundle_root`` (default
    # ``/home/ubuntu/workspace/.skills``) AFTER the S2 PRE snapshot is taken.
    # Without exclusion those ``.skills`` files surface as spurious ADD/MODIFY in
    # the S2 patch manifest. Exactly mirrors the read-only memory mount + service
    # tree prune: a REAL ``.skills`` directory CONTAINED in the configured bundle
    # root is dropped by realpath containment (``.skills`` is never a legitimate
    # work-unit output). To exercise the prune in a tmp_path harness we point the
    # bundle root UNDER the test workspace (``ws/.skills``); only THEN is that
    # subtree dropped.
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    # Point the bundle root at ``ws/.skills`` so the realpath-containment prune
    # actually matches inside this scan; re-clear the settings cache.
    monkeypatch.setenv("SKILL_SANDBOX_BUNDLE_ROOT", str(ws / ".skills"))
    cfg.get_settings.cache_clear()
    try:
        (ws / ".skills" / "my-skill").mkdir(parents=True)
        (ws / ".skills" / "my-skill" / "bundle.py").write_bytes(b"s")
        (ws / "workspace").mkdir()
        (ws / "workspace" / "ok.py").write_bytes(b"k")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        assert "workspace/ok.py" in scan.entries
        # bundle root == ws/.skills -> THAT subtree is dropped.
        assert ".skills/my-skill/bundle.py" not in scan.entries
        assert scan.truncated is False
    finally:
        cfg.get_settings.cache_clear()


async def test_skills_symlink_emitted_not_pruned(tmp_path, monkeypatch):
    # Security mirror of ``test_memory_symlink_emitted_not_pruned`` /
    # ``test_symlink_to_sandbox_is_emitted_not_pruned``: a symlink NAMED
    # ``.skills`` (or one resolving INTO the bundle root) that points at a REAL
    # directory lands in os.walk's ``dirnames`` (followlinks=False). The
    # ``.skills`` prune must run ONLY for a REAL directory CONTAINED in the bundle
    # root — a ``.skills`` SYMLINK must still surface as kind="symlink" (spec
    # §3.1/L136: ANY symlink must surface so the differ group-zero-applies), NEVER
    # be silently dropped. Same class as the /sandbox + .memory pre-lstat prunes.
    from app.core import config as cfg
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    # Default SKILL_SANDBOX_BUNDLE_ROOT (/home/ubuntu/workspace/.skills, outside
    # this ws) — the symlink basename is ``.skills`` but it is NOT contained in
    # the bundle root, so it must be emitted regardless.
    try:
        realtarget = ws / "realtarget"
        realtarget.mkdir()
        (realtarget / "inside.py").write_bytes(b"i")
        (ws / "workspace").mkdir()
        # symlink NAMED .skills -> a real dir, so it lands in dirnames.
        os.symlink(str(realtarget), ws / "workspace" / ".skills")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        sk = scan.entries["workspace/.skills"]
        assert sk.kind == "symlink"
        assert sk.link_target == str(realtarget)
        assert sk.sha256 is None
        # NOT descended through the symlink.
        assert "workspace/.skills/inside.py" not in scan.entries
        assert scan.truncated is False
    finally:
        cfg.get_settings.cache_clear()


async def test_nested_dot_skills_real_dir_is_descended_not_pruned(
    tmp_path, monkeypatch
):
    # R10-2 security mirror of ``test_nested_dot_memory_real_dir_is_descended_not_pruned``:
    # ``.skills`` must be excluded by BUNDLE-ROOT REALPATH CONTAINMENT, not by
    # basename. Only the CONFIGURED ``skill_sandbox_bundle_root`` subtree is
    # dropped. A NESTED real directory merely NAMED ``.skills`` (e.g.
    # ``proj/.skills``) is a child's legitimate write — it must be DESCENDED and
    # its files emitted, NOT silently skipped. A basename prune
    # (``if d == ".skills": continue``) would be fail-open: the differ sees a
    # clean no-op and the child's writes vanish with truncated=False.
    from app.core import config as cfg
    from app.services.file import FileService

    # Default SKILL_SANDBOX_BUNDLE_ROOT (/home/ubuntu/workspace/.skills) sits
    # OUTSIDE this tmp workspace, so a nested ws/proj/.skills is NOT contained in
    # the bundle root and must be descended (no setenv — exercise the default).
    ws = _ws(tmp_path, monkeypatch)
    try:
        (ws / "proj" / ".skills").mkdir(parents=True)
        (ws / "proj" / ".skills" / "keep.py").write_bytes(b"k")
        (ws / "proj" / "ok.py").write_bytes(b"k")
        scan = await FileService.snapshot_workspace(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        # The NESTED .skills (NOT the configured bundle root) is descended + emitted.
        assert "proj/.skills/keep.py" in scan.entries
        assert "proj/ok.py" in scan.entries
        assert scan.truncated is False
    finally:
        cfg.get_settings.cache_clear()


async def test_scan_root_inside_skills_bundle_root_rejected(tmp_path, monkeypatch):
    # P1-d fail-closed (R10-2): a scan ``root`` that resolves INSIDE the
    # CONFIGURED skill bundle root must be rejected — mirrors the memory-mount /
    # service-tree root-guard. The bundle root is never a legitimate scan target;
    # scanning it would surface every native skill bundle as spurious output.
    from app.core import config as cfg
    from app.core.workspace import OutsideWorkspaceError
    from app.services.file import FileService

    ws = _ws(tmp_path, monkeypatch)
    bundle = ws / "bundle-root"
    bundle.mkdir()
    # Point the bundle root at a real dir INSIDE the workspace so the P0
    # workspace-confinement passes and the P1-d skills-root guard is what fires.
    monkeypatch.setenv("SKILL_SANDBOX_BUNDLE_ROOT", str(bundle))
    cfg.get_settings.cache_clear()
    try:
        with pytest.raises(OutsideWorkspaceError):
            await FileService.snapshot_workspace(
                root=str(bundle), max_paths=1000, max_files=1000,
                max_total_bytes=1_000_000, max_seconds=10.0,
            )
    finally:
        cfg.get_settings.cache_clear()


async def test_outside_workspace_root_rejected(tmp_path, monkeypatch):
    # P0 fail-closed: an absolute ``root`` OUTSIDE the workspace must NOT be
    # scanned (spec §3.1 "Scope = /home/ubuntu only"). resolve_in_workspace
    # passes absolute paths through unchanged, so the service itself must
    # confine scan_root to within-or-equal workspace_root and fail closed.
    from app.core import config as cfg
    from app.core.workspace import OutsideWorkspaceError
    from app.services.file import FileService

    _ws(tmp_path, monkeypatch)  # WORKSPACE_ROOT = tmp_path/home, NOT /etc//sandbox
    try:
        # /etc — a real dir well outside the workspace.
        with pytest.raises(OutsideWorkspaceError):
            await FileService.snapshot_workspace(
                root="/etc", max_paths=1000, max_files=1000,
                max_total_bytes=1_000_000, max_seconds=10.0,
            )
        # /sandbox — the (write-denied) service install dir, also out of scope.
        with pytest.raises(OutsideWorkspaceError):
            await FileService.snapshot_workspace(
                root="/sandbox", max_paths=1000, max_files=1000,
                max_total_bytes=1_000_000, max_seconds=10.0,
            )
    finally:
        cfg.get_settings.cache_clear()
