"""C2 PR-5 Task 5.4 — LocalFSRollbackSnapshotStore tests.

Spec ref: §10.4 (process-stable per-pod snapshot store).

These tests pin:
- save/load roundtrip preserves byte content
- discard is idempotent (missing file, missing dir both OK)
- multi-run isolation via per-run-id subdirectory
- path-hex encoding handles deep paths, special chars, traversal attempts
"""
from __future__ import annotations

import hashlib
import os

import pytest

from app.application.services.rollback_snapshot_store import (
    FileSnapshot,
    LocalFSRollbackSnapshotStore,
)

pytestmark = pytest.mark.anyio


_SHA_A = hashlib.sha256(b"a").hexdigest()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_save_load_discard_roundtrip(tmp_path) -> None:
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    snap = await store.save(
        coordinator_run_id="r1",
        path="x/y.py",
        content=b"original content",
        original_digest=_SHA_A,
    )
    assert isinstance(snap, FileSnapshot)
    loaded = await store.load(snap)
    assert loaded == b"original content"
    await store.discard("r1", [snap])
    assert not os.path.exists(snap.snapshot_path)


async def test_discard_idempotent_missing_file(tmp_path) -> None:
    """A second discard on the same snapshot must not raise.

    Concurrent cleanup, pod restart, operator manual cleanup all are
    legitimate sources of "snapshot already gone" — the rollback path
    must remain robust."""
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    snap = await store.save(
        coordinator_run_id="r1", path="x.py",
        content=b"x", original_digest=_SHA_A,
    )
    await store.discard("r1", [snap])
    await store.discard("r1", [snap])


async def test_discard_handles_concurrent_rundir(tmp_path) -> None:
    """If another run still holds files in the parent dir, rmdir fails
    and we swallow — that's a best-effort cleanup, not a correctness
    invariant. This guards against a per-run rmdir failing the discard
    of the cleanup path itself."""
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    snap_a = await store.save(
        coordinator_run_id="r1", path="a.py",
        content=b"a", original_digest=_SHA_A,
    )
    snap_b = await store.save(
        coordinator_run_id="r1", path="b.py",
        content=b"b", original_digest=_SHA_A,
    )
    await store.discard("r1", [snap_a])
    assert os.path.exists(snap_b.snapshot_path)


async def test_different_runs_isolated(tmp_path) -> None:
    """Two coordinator runs writing the same path must use different
    subdirs so a cancel/rollback of one can't smear the other."""
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    snap1 = await store.save(
        coordinator_run_id="r1", path="x.py",
        content=b"v1", original_digest=_SHA_A,
    )
    snap2 = await store.save(
        coordinator_run_id="r2", path="x.py",
        content=b"v2", original_digest=_SHA_A,
    )
    assert snap1.snapshot_path != snap2.snapshot_path
    assert await store.load(snap1) == b"v1"
    assert await store.load(snap2) == b"v2"


async def test_deep_path_hex_encoded(tmp_path) -> None:
    """Path-traversal defense: the on-disk filename is hex-encoded so
    no FS path syntax (../ , slashes, NUL) survives. Confirms a deep
    nested path produces a single flat filename under the run dir."""
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    snap = await store.save(
        coordinator_run_id="r1",
        path="src/foo/bar/baz/qux.py",
        content=b"deep",
        original_digest=_SHA_A,
    )
    parent = os.path.dirname(snap.snapshot_path)
    basename = os.path.basename(snap.snapshot_path)
    assert "/" not in basename
    assert ".." not in basename
    assert os.path.basename(parent) == "r1"
    assert await store.load(snap) == b"deep"


async def test_path_traversal_attempt_hex_neutralized(tmp_path) -> None:
    """Defense in depth: even if a caller passed ``../../etc/passwd``
    here (which should be rejected upstream by
    validate_relative_path_strict), the hex encoding renders it inert
    on disk — the snapshot lives under the run dir, not at the
    attacker-chosen location."""
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    snap = await store.save(
        coordinator_run_id="r1",
        path="../../etc/passwd",
        content=b"trap",
        original_digest=_SHA_A,
    )
    real = os.path.realpath(snap.snapshot_path)
    base_real = os.path.realpath(str(tmp_path))
    assert real.startswith(base_real + os.sep)


async def test_load_missing_snapshot_raises(tmp_path) -> None:
    """[Contract] Programmer error if the applier rolls back after
    discard — the applier's _rollback catches FileNotFoundError and
    routes to rollback_partial + HealthEvent, but the failure mode must
    be observable, not silent."""
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    bogus = FileSnapshot(
        coordinator_run_id="r1",
        original_path="x.py",
        snapshot_path=str(tmp_path / "nonexistent"),
        original_digest=_SHA_A,
    )
    with pytest.raises(FileNotFoundError):
        await store.load(bogus)


async def test_save_rejects_traversal_run_id(tmp_path) -> None:
    """[codex R3 P2#4 fix] coordinator_run_id is used as a path
    segment under base_dir; a malicious / buggy value containing
    ``..`` or path separators must be rejected so snapshots can't
    escape the base directory."""
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    for bad_run_id in [
        "../etc",
        "../../tmp/escape",
        "run/with/slashes",
        "run\\with\\backslashes",
        "/absolute/run",
        "",
    ]:
        with pytest.raises(ValueError, match="coordinator_run_id"):
            await store.save(
                coordinator_run_id=bad_run_id,
                path="x.py",
                content=b"x",
                original_digest=_SHA_A,
            )


async def test_discard_rejects_traversal_run_id(tmp_path) -> None:
    """Same defense applies to discard's rmdir of the run subdir."""
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="coordinator_run_id"):
        await store.discard("../etc", [])


async def test_save_overwrite_same_path_same_run(tmp_path) -> None:
    """Pin behavior: re-saving the same path in the same run
    overwrites (the snapshot_path is determined by hex(path), not by a
    monotonic counter). The Redis lock prevents concurrent re-save in
    practice; this test pins the os-level overwrite semantics."""
    store = LocalFSRollbackSnapshotStore(base_dir=str(tmp_path))
    snap1 = await store.save(
        coordinator_run_id="r1", path="x.py",
        content=b"first", original_digest=_SHA_A,
    )
    snap2 = await store.save(
        coordinator_run_id="r1", path="x.py",
        content=b"second", original_digest=_SHA_A,
    )
    assert snap1.snapshot_path == snap2.snapshot_path
    assert await store.load(snap2) == b"second"
