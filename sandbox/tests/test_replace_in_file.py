"""S1 — replace_in_file inherits atomic overwrite transitively via write_file."""
import pytest


async def test_replace_in_file_atomic_overwrite(tmp_path):
    from app.services.file import FileService

    target = tmp_path / "r.txt"
    target.write_text("alpha beta alpha")
    result = await FileService().replace_in_file(str(target), "alpha", "GAMMA")
    assert result.replaced_count == 2
    assert target.read_text() == "GAMMA beta GAMMA"


async def test_replace_in_file_failed_write_preserves_original(tmp_path, monkeypatch):
    """A failed atomic write leaves the original content intact (no truncation
    on a failed replace-write)."""
    from app.services import file as filemod
    from app.services.file import FileService

    target = tmp_path / "r.txt"
    target.write_text("alpha")

    def boom(t, source_chunks):
        raise OSError("disk full")

    monkeypatch.setattr(filemod, "_atomic_write_bytes", boom)
    with pytest.raises(Exception):  # write_file wraps OSError -> AppException
        await FileService().replace_in_file(str(target), "alpha", "BETA")

    assert target.read_text() == "alpha"


async def test_replace_in_file_noop_when_old_str_absent(tmp_path):
    """If old_str isn't present, no write happens (replaced_count == 0)."""
    from app.services.file import FileService

    target = tmp_path / "r.txt"
    target.write_text("nothing here")
    result = await FileService().replace_in_file(str(target), "MISSING", "X")
    assert result.replaced_count == 0
    assert target.read_text() == "nothing here"
