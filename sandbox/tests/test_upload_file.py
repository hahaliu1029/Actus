"""S1 — upload_file (coordinator hot path) routes through _atomic_write_bytes."""
import pytest


async def test_upload_file_atomic_overwrite(tmp_path, make_upload):
    from app.services.file import FileService

    target = tmp_path / "u.bin"
    target.write_bytes(b"OLD-LONGER")
    result = await FileService.upload_file(make_upload(b"NEW"), str(target))
    assert result.success is True
    assert result.file_size == 3
    assert target.read_bytes() == b"NEW"


async def test_upload_file_no_truncation_on_read_error(tmp_path):
    """A mid-stream read failure must leave the original intact and orphan
    no temp file (upload wraps the error in AppException)."""
    from app.services import file as filemod
    from app.services.file import FileService
    from app.interfaces.errors.exceptions import AppException

    target = tmp_path / "u.bin"
    target.write_bytes(b"ORIGINAL")

    class _BadFile:
        def __init__(self):
            self._n = 0

        def read(self, size):
            self._n += 1
            if self._n == 1:
                return b"PART"
            raise RuntimeError("read failed mid-stream")

    uf = filemod.UploadFile(file=_BadFile(), filename="x")
    with pytest.raises(AppException):
        await FileService.upload_file(uf, str(target))

    assert target.read_bytes() == b"ORIGINAL"
    leftovers = [p for p in target.parent.iterdir()
                 if p.name.startswith(".actus-tmp-")]
    assert leftovers == []


async def test_upload_file_bare_path_rejected(tmp_path, make_upload, monkeypatch):
    """[spec test 9] A bare filename (no directory component) is rejected at
    the PUBLIC surface — parity with today (`os.makedirs("")` raises, wrapped
    in AppException). S1 does NOT widen bare-path acceptance."""
    from app.services.file import FileService
    from app.interfaces.errors.exceptions import AppException

    monkeypatch.chdir(tmp_path)  # ensure no stray "bare.py" lands in the repo
    with pytest.raises(AppException):
        await FileService.upload_file(make_upload(b"x"), "bare.py")
