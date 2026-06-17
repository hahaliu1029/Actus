"""Sandbox Workspace Isolation — shell exec_dir anchoring (PR-3)."""
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


async def test_exec_command_anchors_relative_exec_dir(monkeypatch, tmp_path):
    from app.services.shell import ShellService

    ws = tmp_path / "home"
    (ws / "work").mkdir(parents=True)
    _patch_roots(monkeypatch, str(ws))
    svc = ShellService()
    result = await svc.exec_command(
        session_id="s-rel", exec_dir="work", command="pwd", wait_seconds=5
    )
    assert str(ws / "work") in result.output


async def test_exec_command_absolute_exec_dir_unchanged(monkeypatch, tmp_path):
    from app.services.shell import ShellService

    ws = tmp_path / "home"
    ws.mkdir()
    target = tmp_path / "abs"
    target.mkdir()
    _patch_roots(monkeypatch, str(ws))
    svc = ShellService()
    result = await svc.exec_command(
        session_id="s-abs", exec_dir=str(target), command="pwd", wait_seconds=5
    )
    assert str(target) in result.output


async def test_exec_command_empty_exec_dir_uses_home(monkeypatch, tmp_path):
    from app.services.shell import ShellService

    ws = tmp_path / "home"
    ws.mkdir()
    _patch_roots(monkeypatch, str(ws))
    svc = ShellService()
    # empty exec_dir -> os.path.expanduser("~") (unchanged behavior); just
    # assert it runs and produces output.
    result = await svc.exec_command(
        session_id="s-empty", exec_dir="", command="pwd", wait_seconds=5
    )
    assert result.output  # ran in ~ (HOME), no anchoring applied


async def test_ensure_pty_session_anchors_relative_exec_dir(monkeypatch, tmp_path):
    """[R1 P1 / spec §3.6] ensure_pty_session is the SECOND shell exec_dir path
    and must ALSO anchor a relative exec_dir. Capture the cwd passed to the
    subprocess and short-circuit before the real PTY spawn (raise a sentinel)."""
    import asyncio as _asyncio

    from app.services.shell import ShellService

    ws = tmp_path / "home"
    (ws / "work").mkdir(parents=True)
    _patch_roots(monkeypatch, str(ws))

    captured = {}

    async def _fake_exec(*args, cwd=None, **kwargs):
        captured["cwd"] = cwd
        raise RuntimeError("stop-before-spawn")  # only the resolved cwd matters

    # exec_dir is resolved (line ~241) BEFORE pty.openpty + create_subprocess_exec,
    # so by the time our fake runs the anchoring has already happened.
    monkeypatch.setattr(_asyncio, "create_subprocess_exec", _fake_exec)
    svc = ShellService()
    with pytest.raises(RuntimeError, match="stop-before-spawn"):
        await svc.ensure_pty_session(session_id="pty-rel", exec_dir="work")
    assert captured["cwd"] == str(ws / "work")
