# sandbox/tests/test_kill_all_shell_sessions.py  — CI-only (needs sandbox venv)
import asyncio
import os
import signal

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_kill_all_shell_sessions_killpgs_tracked_groups(monkeypatch):
    from app.services.shell import ShellService

    killed: list[tuple[int, int]] = []

    def _fake_killpg(pgid, sig):
        killed.append((pgid, sig))

    monkeypatch.setattr(os, "killpg", _fake_killpg)
    # The impl resolves the pgid via os.getpgid(pid) BEFORE os.killpg. The fake
    # pids below (1001/1002/2001) are not live processes, so a real os.getpgid
    # would raise ProcessLookupError — which the impl catches and `continue`s,
    # so os.killpg would never run and this test would falsely stay red. Make the
    # fake pid resolvable (start_new_session=True ⇒ pgid == pid) so the
    # getpgid→killpg order is preserved and killpg actually fires.
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    svc = ShellService()
    # The live ShellService tracks non-PTY sessions in `active_shells`
    # (Dict[str, Shell] — Shell.process is the asyncio subprocess) and PTY
    # sessions in `pty_shells` (Dict[str, PtyShellSession] — .process likewise).
    # start_new_session=True ⇒ each process's pgid == its pid. Populate BOTH
    # so the RPC reaps every tracked group.
    svc.active_shells = {  # type: ignore[assignment]
        "s1": _FakeShell(pid=1001),
        "s2": _FakeShell(pid=1002),
    }
    svc.pty_shells = {  # type: ignore[assignment]
        "p1": _FakeShell(pid=2001),
    }
    await svc.kill_all_shell_sessions()
    assert (1001, signal.SIGKILL) in killed
    assert (1002, signal.SIGKILL) in killed
    assert (2001, signal.SIGKILL) in killed


async def test_kill_all_shell_sessions_waits_for_sigkill_reaping(monkeypatch):
    """A just-killed process may remain in /proc briefly before it is reaped."""
    from app.services.shell import ShellService

    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    svc = ShellService()
    svc.active_shells = {"s1": _FakeShell(pid=1001)}  # type: ignore[assignment]
    svc.pty_shells = {}  # type: ignore[assignment]

    checks = iter([False, True])
    monkeypatch.setattr(
        svc,
        "_workspace_quiescent",
        lambda killed_pgids: next(checks),
    )
    sleeps: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    assert await svc.kill_all_shell_sessions() is True
    assert sleeps


class _FakeProc:
    def __init__(self, pid: int) -> None:
        self.pid = pid  # start_new_session=True ⇒ pgid == pid
        self.returncode = None

    def kill(self) -> None:  # pragma: no cover
        pass


class _FakeShell:
    """Mirrors the real Shell / PtyShellSession surface used by the RPC:
    both expose `.process` (an asyncio subprocess) carrying `.pid`."""

    def __init__(self, pid: int) -> None:
        self.process = _FakeProc(pid=pid)


async def test_kill_all_shell_sessions_reports_proc_survivor(monkeypatch, tmp_path):
    # [§3.2(c)] After killpg, a remaining descendant whose cwd (or open fd) is
    # under /home/ubuntu must be detected so quiesce reports NOT clean. Point the
    # check at a fake /proc tree with one survivor cwd → /home/ubuntu/work.
    from app.services import shell as shell_mod
    from app.services.shell import ShellService

    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    # Build a fake /proc: pid 4242 with cwd symlink under the workspace root.
    proc_root = tmp_path / "proc"
    p = proc_root / "4242"
    (p / "fd").mkdir(parents=True)
    workspace_file = tmp_path / "home" / "ubuntu" / "work" / "out.txt"
    workspace_file.parent.mkdir(parents=True)
    workspace_file.write_text("x")
    (p / "cwd").symlink_to(workspace_file.parent)

    # Override the workspace root + /proc location the survivor check reads.
    monkeypatch.setattr(
        shell_mod, "_QUIESCE_WORKSPACE_ROOT", str(tmp_path / "home" / "ubuntu"),
        raising=False,
    )
    monkeypatch.setattr(
        shell_mod, "_QUIESCE_PROC_ROOT", str(proc_root), raising=False,
    )

    svc = ShellService()
    svc.active_shells = {}  # type: ignore[assignment]
    svc.pty_shells = {}  # type: ignore[assignment]

    clean = await svc.kill_all_shell_sessions()
    # kill_all_shell_sessions returns False (not quiescent) when a survivor's
    # cwd/fd resolves under the workspace root.
    assert clean is False


def _write_fake_stat(proc_dir, pid: int, pgrp: int) -> None:
    # Minimal /proc/<pid>/stat: "pid (comm) state ppid pgrp ..." — field 5 is
    # pgrp. comm deliberately contains a space+paren to exercise the rpartition.
    (proc_dir / "stat").write_text(
        f"{pid} (wri ter) S 1 {pgrp} 0 0 -1 0 0 0 0 0 0 0\n"
    )


async def test_kill_all_shell_sessions_unreadable_OUR_survivor_fails_closed(
    monkeypatch, tmp_path
):
    # [codex PR-4 R2 P1] A survivor OF OUR KILLED SHELL GROUPS whose /proc cwd is
    # UNREADABLE (PermissionError — e.g. a sudo-spawned root-owned descendant a
    # non-root service cannot ptrace) cannot be confirmed quiescent → fail-CLOSED.
    # Scope is keyed on pgid (read from the world-readable /proc/<pid>/stat).
    from app.services import shell as shell_mod
    from app.services.shell import ShellService

    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    proc_root = tmp_path / "proc"
    p = proc_root / "9999"
    (p / "fd").mkdir(parents=True)
    (p / "cwd").symlink_to(tmp_path)  # real link; readlink is forced to raise
    _write_fake_stat(p, pid=9999, pgrp=9999)  # pgid 9999 == our tracked shell

    monkeypatch.setattr(
        shell_mod, "_QUIESCE_WORKSPACE_ROOT", str(tmp_path / "home" / "ubuntu"),
        raising=False,
    )
    monkeypatch.setattr(
        shell_mod, "_QUIESCE_PROC_ROOT", str(proc_root), raising=False,
    )

    _real_readlink = os.readlink

    def _fake_readlink(path, *args, **kwargs):
        if "9999" in str(path):
            raise PermissionError("operation not permitted (PTRACE)")
        return _real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", _fake_readlink)

    svc = ShellService()
    # A tracked shell with pid 9999 ⇒ killed_pgids = {9999} ⇒ the unreadable
    # survivor IS ours ⇒ fail-closed.
    svc.active_shells = {"s1": _FakeShell(pid=9999)}  # type: ignore[assignment]
    svc.pty_shells = {}  # type: ignore[assignment]

    clean = await svc.kill_all_shell_sessions()
    assert clean is False  # OUR uninspectable survivor → fail-closed


async def test_kill_all_shell_sessions_unreadable_unrelated_process_skipped(
    monkeypatch, tmp_path
):
    # [codex PR-4 R2 P1] An UNRELATED unreadable process (NOT in our killed pgids
    # — e.g. a root-owned system process under a non-root service) must be SKIPPED,
    # not failed: fail-closing on it would false-fail EVERY capture. Scope keyed
    # on pgid keeps the fail-close to genuine survivors of our shells only.
    from app.services import shell as shell_mod
    from app.services.shell import ShellService

    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    proc_root = tmp_path / "proc"
    p = proc_root / "7777"
    (p / "fd").mkdir(parents=True)
    (p / "cwd").symlink_to(tmp_path)
    _write_fake_stat(p, pid=7777, pgrp=7777)  # pgid 7777 — NOT one we killed

    monkeypatch.setattr(
        shell_mod, "_QUIESCE_WORKSPACE_ROOT", str(tmp_path / "home" / "ubuntu"),
        raising=False,
    )
    monkeypatch.setattr(
        shell_mod, "_QUIESCE_PROC_ROOT", str(proc_root), raising=False,
    )

    _real_readlink = os.readlink

    def _fake_readlink(path, *args, **kwargs):
        if "7777" in str(path):
            raise PermissionError("operation not permitted (PTRACE)")
        return _real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", _fake_readlink)

    svc = ShellService()
    # We killed pgid 9999 (our shell); the unreadable 7777 is unrelated ⇒ skipped.
    svc.active_shells = {"s1": _FakeShell(pid=9999)}  # type: ignore[assignment]
    svc.pty_shells = {}  # type: ignore[assignment]

    clean = await svc.kill_all_shell_sessions()
    assert clean is True  # unrelated unreadable process → skipped, no false-fail
