"""S1 — write_file(append=False, sudo=False) atomic overwrite + byte-count;
append stays non-atomic and must NOT route through the helper."""
import pytest


async def test_write_file_atomic_overwrite(tmp_path):
    from app.services.file import FileService

    target = tmp_path / "w.txt"
    target.write_text("OLD CONTENT LONGER")
    result = await FileService.write_file(filepath=str(target), content="NEW")
    assert result.bytes_written == 3
    assert target.read_text() == "NEW"


async def test_write_file_bytes_written_is_byte_count_for_non_ascii(tmp_path):
    """[H4] The new overwrite path returns the BYTE count (len(encode)),
    aligning non-sudo with the sudo branch. The field is documented '字节数'."""
    from app.services.file import FileService

    target = tmp_path / "u.txt"
    content = "héllo→中文"
    result = await FileService.write_file(filepath=str(target), content=content)
    assert result.bytes_written == len(content.encode("utf-8"))
    with open(target, "rb") as f:
        assert f.read() == content.encode("utf-8")


async def test_append_does_not_use_atomic_helper(tmp_path, monkeypatch):
    """[D8] append stays best-effort/non-atomic and must NOT route through
    `_atomic_write_bytes` (rename can't append)."""
    from app.services import file as filemod
    from app.services.file import FileService

    target = tmp_path / "log.txt"
    target.write_text("line1\n")
    called = {"n": 0}
    real = filemod._atomic_write_bytes

    def spy(*args, **kwargs):
        called["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(filemod, "_atomic_write_bytes", spy)
    await FileService.write_file(filepath=str(target), content="line2\n", append=True)

    assert called["n"] == 0  # append branch never calls the atomic helper
    assert target.read_text() == "line1\nline2\n"


async def test_write_file_bare_name_anchors_under_workspace(tmp_path, monkeypatch):
    """[§3.8] The SANDBOX service now ACCEPTS a bare name — it anchors to
    workspace_root/<name>. (The host ParentSandboxAdapter still rejects bare
    names as coordinator manifest hygiene — that guard is unchanged.)"""
    from app.core import config as cfg
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    monkeypatch.setenv("WORKSPACE_ROOT", str(ws))
    cfg.get_settings.cache_clear()
    try:
        result = await FileService.write_file(filepath="bare.py", content="x")
        assert (ws / "bare.py").read_text() == "x"
        assert result.filepath == "bare.py"  # echo-original
    finally:
        cfg.get_settings.cache_clear()


async def test_write_file_over_fifo_keeps_d12_write_through_nonblocking(tmp_path, monkeypatch):
    import os
    from app.services import file as filemod
    from app.services.file import FileService

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    wt_calls = []

    def spy_wt(target, source_chunks):
        chunks = list(source_chunks)
        wt_calls.append((target, chunks))
        return sum(len(c) for c in chunks)

    monkeypatch.setattr(filemod, "_direct_write_through", spy_wt)

    result = await FileService.write_file(str(fifo), "data")
    assert wt_calls == [(str(fifo), [b"data"])]  # D12 path, never the refuse branch
    assert result.bytes_written == 4
