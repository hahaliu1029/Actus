"""C5d-1 surfaces ②③: positive smoke + NNP, on the REAL actus-sandbox image.

Boots the real image with the production-emitted hardening kwargs (default
profile, NNP off) and proves the workload SURVIVES (shell, offline pip, retained
CHOWN/FOWNER, Chromium CDP up, sudo) AND that the real hardened container itself
has the 4 caps dropped (Cap*/NoNewPrivs/pids.max on PID 1 + a root exec — codex
R3#1, closing the alpine→real-image inference). A separate NNP-on container
proves NoNewPrivs:1 and the documented sudo break (INV-6).

`-m "sandbox and sandbox_real_image"` (the heavy CI job builds the image).
INV-7: ALL docker access inside test functions / the fixture.
"""
from __future__ import annotations

import time

import pytest

from tests.sandbox._docker_helpers import (
    DROPPED_CAP_BITS,
    SANDBOX_IMAGE,
    _docker_client_or_skip,
    _require_image,
    decode_caps,
    hardening_kwargs,
    read_pids_max,
)

pytestmark = [pytest.mark.sandbox, pytest.mark.sandbox_real_image]

_READY_TIMEOUT = 180  # chrome under Xvfb in CI can be slow (codex R3#3)

# Retained-cap probe: prove CHOWN + FOWNER (the caps we do NOT drop) still work
# — create as root, chown to ubuntu (CHOWN), chmod the now-ubuntu file as root
# (FOWNER). Exits 0 on success (codex R4).
_RETAINED_CAP_PROBE = (
    "import os,pwd;"
    "p='/tmp/c5d1_retained';"
    "open(p,'w').close();"
    "u=pwd.getpwnam('ubuntu');"
    "os.chown(p,u.pw_uid,u.pw_gid);"
    "os.chmod(p,0o640);"
    "os.remove(p)"
)


def _exec(container, cmd, **kw):
    """exec_run returning (exit_code, stdout_bytes); stderr demuxed away (codex R4)."""
    res = container.exec_run(cmd, demux=True, **kw)
    stdout = (res.output[0] or b"") if res.output else b""
    return res.exit_code, stdout


def _caps_from_proc(container, pid_path: str) -> dict[str, set[int]]:
    code, out = _exec(container, ["sh", "-c", f"grep -E 'Cap(Eff|Prm|Bnd)' {pid_path}"])
    assert code == 0, f"reading {pid_path} failed: {out!r}"
    sets: dict[str, set[int]] = {}
    for line in out.decode().splitlines():
        key, _, val = line.partition(":")
        if key.strip() in ("CapEff", "CapPrm", "CapBnd"):
            sets[key.strip()] = decode_caps(val.strip())
    return sets


def _nonewprivs(container, pid_path: str) -> str:
    code, out = _exec(container, ["sh", "-c", f"grep NoNewPrivs {pid_path}"])
    assert code == 0, f"reading {pid_path} failed: {out!r}"
    return out.decode().split(":", 1)[1].strip()


_READY_PROCS = ("xvfb", "chrome", "socat", "app")


def _wait_ready(container) -> None:
    """Ready (spec §6): supervisord reports xvfb+chrome+socat+app all RUNNING, THEN
    chrome CDP (:8222) responds. `chrome RUNNING` precedes CDP readiness, so the
    supervisorctl gate is the pre-filter and the CDP curl is the real serving
    signal (codex R3#3 + plan-R2)."""
    deadline = time.time() + _READY_TIMEOUT
    while time.time() < deadline:
        container.reload()
        if container.status != "running":
            break
        _, out = _exec(container, ["sh", "-c", "supervisorctl -c /sandbox/supervisord.conf status"])
        # supervisord.conf groups all programs in `[group:services]`, so status names
        # are namespecs like `services:app` — strip the group prefix to match
        # `_READY_PROCS` (plan-R3 P1: bare-name parse would never match → timeout).
        lines = {ln.split()[0].rsplit(":", 1)[-1]: ln
                 for ln in out.decode(errors="replace").splitlines() if ln.split()}
        all_running = all(p in lines and "RUNNING" in lines[p] for p in _READY_PROCS)
        if all_running and _exec(container, ["sh", "-c", "curl -fsS http://127.0.0.1:8222/json/version >/dev/null"])[0] == 0:
            return
        time.sleep(2)
    _, sup = _exec(container, ["sh", "-c", "supervisorctl -c /sandbox/supervisord.conf status || true"])
    pytest.fail(f"sandbox not ready in {_READY_TIMEOUT}s; supervisor:\n{sup.decode(errors='replace')}")


@pytest.fixture
def hardened_sandbox():
    """Boot the REAL actus-sandbox image with the DEFAULT hardened profile (NNP off)."""
    client = _docker_client_or_skip()
    _require_image(client, SANDBOX_IMAGE)
    container = client.containers.run(
        SANDBOX_IMAGE, detach=True, **hardening_kwargs(hardening=True, nnp=False)
    )
    try:
        _wait_ready(container)
        yield container
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


def test_workload_survives_default_hardened_profile(hardened_sandbox):
    c = hardened_sandbox

    # shell
    code, out = _exec(c, ["sh", "-lc", "echo ok"])
    assert code == 0 and out.strip() == b"ok", out

    # offline pip (runtime install survives the profile, no network; codex R3#4/R4)
    setup = "from setuptools import setup; setup(name='c5d1probe', version='0.0.0', packages=['c5d1mod'])"
    mk = (
        "mkdir -p /tmp/pkg/c5d1mod && "
        f"printf '%s' \"{setup}\" > /tmp/pkg/setup.py && "
        "touch /tmp/pkg/c5d1mod/__init__.py"
    )
    assert _exec(c, ["sh", "-c", mk])[0] == 0
    code, out = _exec(c, ["sh", "-c", "python3 -m pip install --no-index --no-build-isolation --no-deps /tmp/pkg"])
    assert code == 0, f"offline pip failed: {out!r}"

    # retained caps: CHOWN + FOWNER still work (caps we did NOT drop)
    code, out = _exec(c, ["python3", "-c", _RETAINED_CAP_PROBE])
    assert code == 0, f"retained-cap probe failed: {out!r}"

    # sudo intact (NNP off → setuid sudo works) — run as the ubuntu NOPASSWD user
    code, out = _exec(c, ["sudo", "-n", "true"], user="ubuntu")
    assert code == 0, f"sudo -n true failed under default profile: {out!r}"

    # real-image hardening assertion (closes alpine→real-image inference, codex R3#1):
    # PID 1 (supervisord) AND a fresh root exec must have the 4 caps dropped in all
    # three sets, NoNewPrivs:0 (default), and pids.max==512.
    for pid_path in ("/proc/1/status", "/proc/self/status"):
        sets = _caps_from_proc(c, pid_path)
        for name in ("CapEff", "CapPrm", "CapBnd"):
            assert DROPPED_CAP_BITS.isdisjoint(sets[name]), f"{pid_path} {name} retains a dropped cap"
    # NNP must be OFF on BOTH PID 1 and a fresh exec under the default profile (plan-R2 P1).
    assert _nonewprivs(c, "/proc/1/status") == "0", "default profile must NOT set NoNewPrivs (PID 1)"
    assert _nonewprivs(c, "/proc/self/status") == "0", "default profile must NOT set NoNewPrivs (exec)"
    assert read_pids_max(c) == "512", "real-image pids.max != 512"


def test_no_new_privileges_opt_in_sets_nonewprivs_and_breaks_sudo():
    """NNP-on narrow proof (surface ③): NoNewPrivs:1 on PID 1 + exec; setuid sudo
    is expected to FAIL (the INV-6 rationale for keeping NNP a separate opt-in)."""
    client = _docker_client_or_skip()
    _require_image(client, SANDBOX_IMAGE)
    container = client.containers.run(
        SANDBOX_IMAGE, detach=True, **hardening_kwargs(hardening=True, nnp=True)
    )
    try:
        _wait_ready(container)
        assert _nonewprivs(container, "/proc/1/status") == "1", "NNP-on must set NoNewPrivs on PID 1"
        assert _nonewprivs(container, "/proc/self/status") == "1", "NNP-on must set NoNewPrivs on exec"
        # setuid sudo cannot escalate under no-new-privileges (run as non-root ubuntu).
        code, _ = _exec(container, ["sudo", "-n", "true"], user="ubuntu")
        assert code != 0, "sudo unexpectedly succeeded under no-new-privileges"
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass
