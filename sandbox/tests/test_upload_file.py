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


async def test_upload_file_bare_name_anchors_under_workspace(tmp_path, make_upload, monkeypatch):
    """[§3.8] Bare name now anchors to workspace_root/<name> at the sandbox
    service surface (host adapter still rejects bare names — unchanged)."""
    from app.core import config as cfg
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    monkeypatch.setenv("WORKSPACE_ROOT", str(ws))
    cfg.get_settings.cache_clear()
    try:
        result = await FileService.upload_file(make_upload(b"x"), "bare.py")
        assert (ws / "bare.py").read_bytes() == b"x"
        assert result.success is True
    finally:
        cfg.get_settings.cache_clear()


async def test_upload_file_threads_refuse_special_to_atomic_writer(tmp_path, monkeypatch):
    """D5-A: the service-level refuse_special kwarg reaches _atomic_write_bytes."""
    from app.services import file as filemod
    from app.services.file import FileService

    seen = {}

    def fake_atomic(target, chunks, *, refuse_special=False):
        seen["target"] = target
        seen["refuse_special"] = refuse_special
        for _ in chunks:
            pass
        return 0

    monkeypatch.setattr(filemod, "_atomic_write_bytes", fake_atomic)

    class _F:
        filename = "x"

        class file:  # minimal UploadFile.file shim
            @staticmethod
            def read(_n):
                return b""

    await FileService.upload_file(_F(), str(tmp_path / "f"), refuse_special=True)
    assert seen["refuse_special"] is True

    await FileService.upload_file(_F(), str(tmp_path / "f2"))
    assert seen["refuse_special"] is False  # default unchanged for non-coordinator
