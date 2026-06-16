"""S1 — delete_file is idempotent (ENOENT -> success); other OSError propagate."""
import pytest


async def test_delete_present_file(tmp_path):
    from app.services.file import FileService

    target = tmp_path / "d.txt"
    target.write_text("x")
    result = await FileService().delete_file(str(target))
    assert result.deleted is True
    assert not target.exists()


async def test_delete_absent_file_is_idempotent(tmp_path):
    """Deleting an already-absent file is success (terminal-absent), no raise.
    Safe: the applier preflight owns FILE_MISSING via exists(); there is no
    agent delete tool."""
    from app.services.file import FileService

    result = await FileService().delete_file(str(tmp_path / "nope.txt"))
    assert result.deleted is True


async def test_delete_non_enoent_oserror_wrapped_in_appexception(tmp_path, monkeypatch):
    """A non-ENOENT OSError (e.g. EACCES) is wrapped in AppException with a
    SPECIFIC diagnostic message — NOT swallowed, NOT degraded to a generic 500.
    Consistent with write_file/upload_file (the delete endpoint does not wrap,
    so a raw OSError would hit the global handler's generic message). ENOENT
    stays idempotent success (test above). See Deviation D-2."""
    from app.services import file as filemod
    from app.services.file import FileService
    from app.interfaces.errors.exceptions import AppException

    target = tmp_path / "d.txt"
    target.write_text("x")

    def boom(path):
        raise PermissionError("EACCES denied")

    monkeypatch.setattr(filemod.os, "remove", boom)
    with pytest.raises(AppException) as exc:
        await FileService().delete_file(str(target))
    assert "EACCES denied" in exc.value.msg
    assert str(target) in exc.value.msg
