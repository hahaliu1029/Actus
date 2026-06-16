"""S1 harness smoke + low-level write primitive (`_write_all`) tests."""


def test_harness_imports_sandbox_app():
    """The harness resolves `app` to sandbox/app (not api/app) and the
    FileService module imports cleanly under the workspace venv."""
    import app.services.file as filemod

    assert hasattr(filemod, "FileService")


def test_write_all_handles_short_writes(tmp_path, monkeypatch):
    """`_write_all` must loop until every byte lands, even when the raw
    `os.write` returns a short count (the kernel is allowed to)."""
    from app.services import file as filemod
    import os as _os

    path = tmp_path / "wa.bin"
    fd = _os.open(str(path), _os.O_WRONLY | _os.O_CREAT, 0o644)
    real_write = _os.write
    state = {"first": True}

    def short_write(fd_, data):
        if state["first"]:
            state["first"] = False
            return real_write(fd_, data[:2])  # write only 2 of N bytes first
        return real_write(fd_, data)

    monkeypatch.setattr(filemod.os, "write", short_write)
    try:
        total = filemod._write_all(fd, b"abcdef")
    finally:
        _os.close(fd)

    assert total == 6
    assert path.read_bytes() == b"abcdef"


def test_new_file_mode_constant_matches_umask():
    """`_NEW_FILE_MODE` mirrors old `open(path, "wb")` new-file perms
    (0o666 & ~umask), snapshotted once at import."""
    from app.services import file as filemod

    assert filemod._NEW_FILE_MODE == (0o666 & ~filemod._UMASK)
