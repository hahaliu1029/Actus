"""Sandbox Workspace Isolation — wired FileService endpoints (PR-2)."""
import os

import pytest


def _patch_roots(monkeypatch, workspace, install="/sandbox"):
    from app.core import config as cfg
    monkeypatch.setenv("WORKSPACE_ROOT", workspace)
    monkeypatch.setenv("SERVICE_INSTALL_DIR", install)
    cfg.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _restore_settings_cache():
    yield
    from app.core import config as cfg
    cfg.get_settings.cache_clear()


async def test_read_file_resolves_relative_under_workspace(monkeypatch, tmp_path):
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    (ws / "note.txt").write_text("HELLO")
    _patch_roots(monkeypatch, str(ws))
    result = await FileService.read_file(filepath="note.txt")
    assert result.content == "HELLO"
    assert result.filepath == "note.txt"  # echo-original


async def test_check_file_exists_resolves_relative(monkeypatch, tmp_path):
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    (ws / "present.txt").write_text("x")
    _patch_roots(monkeypatch, str(ws))
    present = await FileService.check_file_exists(filepath="present.txt")
    assert present.exists is True
    assert present.filepath == "present.txt"
    absent = await FileService.check_file_exists(filepath="absent.txt")
    assert absent.exists is False


async def test_ensure_file_resolves_relative(monkeypatch, tmp_path):
    from app.services.file import FileService
    from app.interfaces.errors.exceptions import NotFoundException

    ws = tmp_path / "home"
    ws.mkdir()
    (ws / "here.txt").write_text("x")
    _patch_roots(monkeypatch, str(ws))
    await FileService.ensure_file("here.txt")  # no raise
    with pytest.raises(NotFoundException):
        await FileService.ensure_file("nope.txt")


async def test_write_file_lands_under_workspace_echo_original(monkeypatch, tmp_path):
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    _patch_roots(monkeypatch, str(ws))
    result = await FileService.write_file(filepath="sub/out.txt", content="DATA")
    assert (ws / "sub" / "out.txt").read_text() == "DATA"
    assert result.filepath == "sub/out.txt"  # echo-original, NOT the resolved abs


async def test_write_absolute_service_dir_denied_4xx(monkeypatch, tmp_path):
    from app.services.file import FileService
    from app.core.workspace import ServiceTreeWriteDenied

    install = tmp_path / "sandbox"
    install.mkdir()
    (install / "app").mkdir()
    (install / "app" / "main.py").write_text("SERVICE")
    _patch_roots(monkeypatch, str(tmp_path / "home"), install=str(install))
    with pytest.raises(ServiceTreeWriteDenied):  # BadRequestException subclass -> 400
        await FileService.write_file(
            filepath=str(install / "app" / "main.py"), content="HACKED"
        )
    assert (install / "app" / "main.py").read_text() == "SERVICE"  # untouched


async def test_append_through_final_symlink_to_service_denied(monkeypatch, tmp_path):
    from app.services.file import FileService
    from app.core.workspace import ServiceTreeWriteDenied

    home = tmp_path / "home"
    home.mkdir()
    install = tmp_path / "sandbox"
    install.mkdir()
    victim = install / "main.py"
    victim.write_text("SERVICE")
    os.symlink(str(victim), str(home / "link"))
    _patch_roots(monkeypatch, str(home), install=str(install))
    with pytest.raises(ServiceTreeWriteDenied):
        await FileService.write_file(filepath="link", content="X", append=True)
    assert victim.read_text() == "SERVICE"  # untouched


async def test_write_dotdot_escape_4xx(monkeypatch, tmp_path):
    from app.services.file import FileService
    from app.core.workspace import OutsideWorkspaceError

    ws = tmp_path / "home"
    ws.mkdir()
    _patch_roots(monkeypatch, str(ws))
    with pytest.raises(OutsideWorkspaceError):  # 400, NOT 500
        await FileService.write_file(filepath="../escape.txt", content="X")


async def test_upload_lands_under_workspace_echo_original(monkeypatch, tmp_path, make_upload):
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    _patch_roots(monkeypatch, str(ws))
    result = await FileService.upload_file(make_upload(b"BYTES"), "sub/u.bin")
    assert (ws / "sub" / "u.bin").read_bytes() == b"BYTES"
    assert result.filepath == "sub/u.bin"  # echo-original


async def test_upload_absolute_service_dir_denied_4xx(monkeypatch, tmp_path, make_upload):
    from app.services.file import FileService
    from app.core.workspace import ServiceTreeWriteDenied

    install = tmp_path / "sandbox"
    install.mkdir()
    (install / "app").mkdir()
    (install / "app" / "main.py").write_bytes(b"SERVICE")
    _patch_roots(monkeypatch, str(tmp_path / "home"), install=str(install))
    with pytest.raises(ServiceTreeWriteDenied):  # 400, not wrapped to 500
        await FileService.upload_file(make_upload(b"HACK"), str(install / "app" / "main.py"))
    assert (install / "app" / "main.py").read_bytes() == b"SERVICE"


async def test_upload_tmp_fallback_passes_through(monkeypatch, tmp_path, make_upload):
    """[R3 P3] An absolute fallback path outside both roots is pass-through at
    the SERVICE-layer guard (does not over-block)."""
    from app.services.file import FileService

    _patch_roots(monkeypatch, str(tmp_path / "home"))
    dest = tmp_path / "scratch.bin"  # absolute, outside both roots -> allowed
    result = await FileService.upload_file(make_upload(b"OK"), str(dest))
    assert dest.read_bytes() == b"OK"
    assert result.success is True


async def test_upload_endpoint_fallback_synthesizes_tmp_path(monkeypatch):
    """[R1 P2 / §5] When the ENDPOINT receives no filepath, it synthesizes
    /tmp/{file.filename} (the real fallback the spec wants covered). Drive the
    endpoint with a fake service that captures the filepath it is handed."""
    import io

    from fastapi import UploadFile

    from app.interfaces.endpoints.file import upload_file
    from app.models.file import FileUploadResult

    captured = {}

    class _FakeService:
        # NOTE: the live endpoint (S1b) passes refuse_special= through; the fake
        # must accept it. The behavior under test is the /tmp/ fallback synthesis.
        async def upload_file(self, file, filepath, refuse_special=False):
            captured["filepath"] = filepath
            return FileUploadResult(filepath=filepath, file_size=0, success=True)

    uf = UploadFile(file=io.BytesIO(b"x"), filename="drop.bin")
    await upload_file(file=uf, filepath=None, file_service=_FakeService())
    assert captured["filepath"] == "/tmp/drop.bin"  # endpoint synthesized the fallback


async def test_delete_resolves_relative_and_echoes(monkeypatch, tmp_path):
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    (ws / "gone.txt").write_text("x")
    _patch_roots(monkeypatch, str(ws))
    result = await FileService().delete_file("gone.txt")
    assert not (ws / "gone.txt").exists()
    assert result.filepath == "gone.txt"
    assert result.deleted is True


async def test_delete_absolute_service_dir_denied(monkeypatch, tmp_path):
    from app.services.file import FileService
    from app.core.workspace import ServiceTreeWriteDenied

    install = tmp_path / "sandbox"
    install.mkdir()
    (install / "x").write_text("SERVICE")
    _patch_roots(monkeypatch, str(tmp_path / "home"), install=str(install))
    with pytest.raises(ServiceTreeWriteDenied):
        await FileService().delete_file(str(install / "x"))
    assert (install / "x").exists()  # untouched


async def test_delete_final_symlink_removes_link_only(monkeypatch, tmp_path):
    """[R3 P2] os.remove removes the LINK, not its target -> /sandbox/x intact."""
    from app.services.file import FileService

    home = tmp_path / "home"
    home.mkdir()
    install = tmp_path / "sandbox"
    install.mkdir()
    victim = install / "x"
    victim.write_text("SERVICE")
    link = home / "link"
    os.symlink(str(victim), str(link))
    _patch_roots(monkeypatch, str(home), install=str(install))
    await FileService().delete_file("link")  # follows_final_symlink=False -> allowed
    assert not link.exists()                 # link removed
    assert victim.read_text() == "SERVICE"   # target intact


async def test_replace_in_file_resolves_relative_and_echoes(monkeypatch, tmp_path):
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    (ws / "r.txt").write_text("foo bar foo")
    _patch_roots(monkeypatch, str(ws))
    result = await FileService().replace_in_file(
        filepath="r.txt", old_str="foo", new_str="baz"
    )
    assert (ws / "r.txt").read_text() == "baz bar baz"
    assert result.replaced_count == 2
    assert result.filepath == "r.txt"  # echo-original


async def test_search_in_file_resolves_relative_and_echoes(monkeypatch, tmp_path):
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    (ws / "s.txt").write_text("alpha\nbeta\nalphabet\n")
    _patch_roots(monkeypatch, str(ws))
    result = await FileService().search_in_file(filepath="s.txt", regex="alpha")
    assert result.matches == ["alpha", "alphabet"]
    assert result.filepath == "s.txt"  # echo-original


async def test_find_files_relative_dir_relative_results(monkeypatch, tmp_path):
    from app.services.file import FileService

    ws = tmp_path / "home"
    (ws / "proj").mkdir(parents=True)
    (ws / "proj" / "a.py").write_text("x")
    (ws / "proj" / "b.py").write_text("y")
    _patch_roots(monkeypatch, str(ws))
    result = await FileService.find_files(dir_path="proj", glob_pattern="*.py")
    assert result.dir_path == "proj"  # echo-original
    assert sorted(result.files) == ["proj/a.py", "proj/b.py"]  # relative-in -> relative-out


async def test_find_files_absolute_dir_passthrough(monkeypatch, tmp_path):
    """[R6] Absolute dir_path is NOT filtered against workspace_root."""
    from app.services.file import FileService

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "m.md").write_text("x")
    _patch_roots(monkeypatch, str(tmp_path / "home"))
    result = await FileService.find_files(dir_path=str(scratch), glob_pattern="*.md")
    assert result.dir_path == str(scratch)
    assert result.files == [str(scratch / "m.md")]  # absolute-in -> absolute-out


async def test_find_files_rejects_absolute_glob_pattern(monkeypatch, tmp_path):
    from app.services.file import FileService
    from app.interfaces.errors.exceptions import BadRequestException

    ws = tmp_path / "home"
    ws.mkdir()
    _patch_roots(monkeypatch, str(ws))
    with pytest.raises(BadRequestException):
        await FileService.find_files(dir_path="proj", glob_pattern="/sandbox/*")


async def test_find_files_rejects_dotdot_glob_pattern(monkeypatch, tmp_path):
    from app.services.file import FileService
    from app.interfaces.errors.exceptions import BadRequestException

    ws = tmp_path / "home"
    ws.mkdir()
    _patch_roots(monkeypatch, str(ws))
    with pytest.raises(BadRequestException):
        await FileService.find_files(dir_path="proj", glob_pattern="../*")


async def test_find_files_dir_is_final_symlink_to_outside_rejected(monkeypatch, tmp_path):
    """[R5] dir_path that IS a final symlink out of the workspace -> rejected
    by follow_final=True full-realpath confinement."""
    from app.services.file import FileService
    from app.core.workspace import OutsideWorkspaceError

    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.py").write_text("x")
    os.symlink(str(outside), str(home / "dlink"))
    _patch_roots(monkeypatch, str(home))
    with pytest.raises(OutsideWorkspaceError):
        await FileService.find_files(dir_path="dlink", glob_pattern="*.py")


async def test_find_files_inner_symlink_result_filtered_out(monkeypatch, tmp_path):
    """[R4 A1] A * pattern following an INNER directory symlink out of the
    workspace is dropped by the result filter (base = realpath(workspace_root))."""
    from app.services.file import FileService

    home = tmp_path / "home"
    realdir = home / "realdir"
    realdir.mkdir(parents=True)
    (realdir / "ok.txt").write_text("x")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "leak.txt").write_text("x")
    os.symlink(str(outside), str(realdir / "link"))
    _patch_roots(monkeypatch, str(home))
    # pattern matches realdir/ok.txt and realdir/link/leak.txt; the leak is filtered.
    result = await FileService.find_files(dir_path="realdir", glob_pattern="**/*.txt")
    assert "realdir/ok.txt" in result.files
    assert all("leak.txt" not in f for f in result.files)


async def test_find_files_empty_dir_lists_workspace_root(monkeypatch, tmp_path):
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    (ws / "top.txt").write_text("x")
    _patch_roots(monkeypatch, str(ws))
    result = await FileService.find_files(dir_path="", glob_pattern="*.txt")
    assert result.dir_path == ""
    assert result.files == ["top.txt"]


async def test_download_endpoint_resolves_filepath(monkeypatch, tmp_path):
    """[R1 P1] The download endpoint's FileResponse must point at the ANCHORED
    file, not the raw relative arg (round-trip identity, D8). Calls the endpoint
    function directly so it is RED until the endpoint resolves the path."""
    from app.interfaces.endpoints.file import download_file
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    _patch_roots(monkeypatch, str(ws))
    await FileService.write_file(filepath="a/b.txt", content="ROUND")

    response = await download_file(filepath="a/b.txt", file_service=FileService())
    assert response.path == str(ws / "a" / "b.txt")  # RED before patch (== "a/b.txt")
    with open(response.path) as f:
        assert f.read() == "ROUND"
