import os

from app.models.file import FileCheckResult


async def _check(filepath: str) -> FileCheckResult:
    from app.services.file import FileService
    return await FileService.check_file_exists(filepath)


async def test_regular_file_kind(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x")
    r = await _check(str(p))
    assert r.exists is True
    assert r.kind == "regular"


async def test_fifo_kind(tmp_path):
    p = tmp_path / "pipe"
    os.mkfifo(p)
    r = await _check(str(p))
    assert r.exists is True
    assert r.kind == "fifo"


async def test_missing_kind(tmp_path):
    r = await _check(str(tmp_path / "nope"))
    assert r.exists is False
    assert r.kind == "missing"


async def test_broken_symlink_exists_false_kind_symlink(tmp_path):
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "target-does-not-exist")
    r = await _check(str(link))
    # os.path.exists follows the link → False for a broken symlink (unchanged);
    # os.lstat sees the link itself → kind="symlink".
    assert r.exists is False
    assert r.kind == "symlink"


async def test_lstat_soft_failure_returns_other_not_500(tmp_path, monkeypatch):
    # An lstat that raises a non-ENOENT OSError (EACCES/ELOOP/…) MUST NOT
    # raise out of check_file_exists — it degrades to kind="other", exists
    # keeps its os.path.exists value. Pins the R7#P2 soft-failure guard.
    from app.services import file as filemod
    real_lstat = os.lstat

    def boom(path, *a, **k):
        raise PermissionError(13, "EACCES")

    monkeypatch.setattr(filemod.os, "lstat", boom)
    monkeypatch.setattr(filemod.os.path, "exists", lambda p: True)
    r = await _check(str(tmp_path / "whatever"))
    assert r.exists is True
    assert r.kind == "other"
    monkeypatch.setattr(filemod.os, "lstat", real_lstat)
