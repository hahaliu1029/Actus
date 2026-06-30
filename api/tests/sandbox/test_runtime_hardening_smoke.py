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

import errno
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

# C5d-2 strict mask (bits) — independent in-test literal (parity with the negative
# test's _EXPECTED_STRICT_CAP_BITS). Parity-10 fallback → add bit 18 (SYS_CHROOT).
_STRICT_MASK_BITS = frozenset({0, 1, 3, 4, 5, 6, 7, 8, 31})

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


@pytest.fixture
def strict_sandbox():
    """Boot the REAL actus-sandbox image with the STRICT profile (cap_drop=ALL +
    the 9-cap allowlist, NNP off)."""
    client = _docker_client_or_skip()
    _require_image(client, SANDBOX_IMAGE)
    container = client.containers.run(
        SANDBOX_IMAGE, detach=True, **hardening_kwargs(hardening=True, strict=True, nnp=False)
    )
    try:
        _wait_ready(container)
        yield container
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


def test_workload_survives_strict_profile(strict_sandbox):
    # C5d-2 surface ②: the 9-cap strict profile supports the CORE per-session workload
    # (startup + shell + offline pip + chown/chmod + sudo + Chromium CDP). SETFCAP/KILL
    # ride along un-exercised (safe over-grant; spec §9/§11).
    c = strict_sandbox

    code, out = _exec(c, ["sh", "-lc", "echo ok"])
    assert code == 0 and out.strip() == b"ok", out

    setup = "from setuptools import setup; setup(name='c5d2probe', version='0.0.0', packages=['c5d2mod'])"
    mk = (
        "mkdir -p /tmp/pkg/c5d2mod && "
        f"printf '%s' \"{setup}\" > /tmp/pkg/setup.py && "
        "touch /tmp/pkg/c5d2mod/__init__.py"
    )
    assert _exec(c, ["sh", "-c", mk])[0] == 0
    code, out = _exec(c, ["sh", "-c", "python3 -m pip install --no-index --no-build-isolation --no-deps /tmp/pkg"])
    assert code == 0, f"offline pip failed under strict: {out!r}"

    # retained caps: CHOWN + FOWNER still work
    code, out = _exec(c, ["python3", "-c", _RETAINED_CAP_PROBE])
    assert code == 0, f"retained-cap probe failed under strict: {out!r}"

    # sudo intact (NNP off → setuid sudo works) — SETUID/SETGID/SETPCAP in the allowlist
    code, out = _exec(c, ["sudo", "-n", "true"], user="ubuntu")
    assert code == 0, f"sudo -n true failed under strict profile: {out!r}"

    # real-image hardening: PID 1 + a fresh root exec carry EXACTLY the strict mask,
    # NoNewPrivs:0 (default), pids.max==512.
    for pid_path in ("/proc/1/status", "/proc/self/status"):
        sets = _caps_from_proc(c, pid_path)
        for name in ("CapEff", "CapPrm", "CapBnd"):
            assert sets[name] == _STRICT_MASK_BITS, f"{pid_path} {name} != strict mask: {sorted(sets[name])}"
    assert _nonewprivs(c, "/proc/1/status") == "0", "strict default profile must NOT set NoNewPrivs"
    assert read_pids_max(c) == "512", "real-image pids.max != 512"


def test_strict_with_no_new_privileges_breaks_sudo():
    # C5d-2 surface ③: strict + NNP → NoNewPrivs:1 + setuid sudo FAILS (INV-6). SETUID
    # in the allowlist is NOT enough — NNP is exactly the knob that forbids setuid escalation.
    client = _docker_client_or_skip()
    _require_image(client, SANDBOX_IMAGE)
    container = client.containers.run(
        SANDBOX_IMAGE, detach=True,
        **hardening_kwargs(hardening=True, strict=True, nnp=True),
    )
    try:
        _wait_ready(container)
        assert _nonewprivs(container, "/proc/1/status") == "1", "strict+NNP must set NoNewPrivs on PID 1"
        assert _nonewprivs(container, "/proc/self/status") == "1", "strict+NNP must set NoNewPrivs on exec"
        code, _ = _exec(container, ["sudo", "-n", "true"], user="ubuntu")
        assert code != 0, "sudo unexpectedly succeeded under strict + no-new-privileges"
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


_ALL_SERVICE_PROCS = ("xvfb", "chrome", "socat", "x11vnc", "websockify", "app")


def _all_six_running(container) -> bool:
    # The full [group:services] set — NOT the 4-proc `_wait_ready` filter, which would leave
    # x11vnc/websockify untested while green (codex R2 P2).
    _, out = _exec(container, ["sh", "-c", "supervisorctl -c /sandbox/supervisord.conf status"])
    lines = {ln.split()[0].rsplit(":", 1)[-1]: ln
             for ln in out.decode(errors="replace").splitlines() if ln.split()}
    return all(p in lines and "RUNNING" in lines[p] for p in _ALL_SERVICE_PROCS)


def _assert_port_accepts(container, port: int) -> None:
    # A crash-looping x11vnc/websockify can flash RUNNING transiently → a socket-accept is the
    # real liveness oracle (codex R2 P2/P3).
    code, out = _exec(container, ["python3", "-c",
        f"import socket; socket.create_connection(('127.0.0.1', {port}), 5).close()"])
    assert code == 0, f"port {port} not accepting connections: {out!r}"


def _uid_gid(container, pid_path: str) -> dict[str, list[str]]:
    # Return ALL FOUR columns of Uid:/Gid: (real, effective, saved, fs) — codex R2 P2:
    # asserting only the real-uid column would miss an effective/saved/fs uid drift the
    # spec §9① pins as `1000 1000 1000 1000`.
    code, out = _exec(container, ["sh", "-c", f"grep -E '^(Uid|Gid):' {pid_path}"])
    assert code == 0, f"reading {pid_path} failed: {out!r}"
    vals: dict[str, list[str]] = {}
    for line in out.decode().splitlines():
        key, _, val = line.partition(":")
        vals[key.strip()] = val.split()  # ["real", "effective", "saved", "fs"]
    return vals


@pytest.fixture
def nonroot_sandbox():
    """Boot the REAL actus-sandbox under `--user 1000:1000` (hardening ON + run_as_user ON,
    conservative caps, NNP off)."""
    client = _docker_client_or_skip()
    _require_image(client, SANDBOX_IMAGE)
    container = client.containers.run(
        SANDBOX_IMAGE, detach=True,
        **hardening_kwargs(hardening=True, run_as_user=True, nnp=False),
    )
    try:
        _wait_ready(container)
        yield container
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


def test_nonroot_profile_runs_as_uid_1000(nonroot_sandbox):
    # Surface ①: uid kernel-enforced on PID1 AND on an exec; all six services + sockets live;
    # sudo still reaches root (semi-rootless).
    c = nonroot_sandbox
    pid1 = _uid_gid(c, "/proc/1/status")
    # spec §9①: ALL FOUR uid/gid columns kernel-enforced to 1000 (real/effective/saved/fs).
    assert pid1["Uid"] == ["1000"] * 4 and pid1["Gid"] == ["1000"] * 4, (
        f"PID1 uid/gid not all-1000 (real/effective/saved/fs): {pid1}")
    assert _exec(c, ["id", "-u"])[1].strip() == b"1000", "exec id -u != 1000"
    assert _exec(c, ["id", "-g"])[1].strip() == b"1000", "exec id -g != 1000"
    assert _all_six_running(c), "not all 6 services RUNNING under non-root"
    for port in (8080, 5900, 5901):
        _assert_port_accepts(c, port)
    assert _exec(c, ["sh", "-c", "curl -fsS http://127.0.0.1:9222/json/version >/dev/null"])[0] == 0
    code, out = _exec(c, ["sudo", "-n", "id", "-u"])
    assert code == 0 and out.strip() == b"0", f"sudo -n id -u != 0 (semi-rootless broken): {out!r}"


@pytest.fixture
def rootdefault_sandbox():
    """Boot the SAME rebuilt image with NO `--user` (default) — the INV-0-for-image proof:
    the §4.4 image edits do NOT break the default root boot."""
    client = _docker_client_or_skip()
    _require_image(client, SANDBOX_IMAGE)
    container = client.containers.run(SANDBOX_IMAGE, detach=True)  # no kwargs, no --user
    try:
        _wait_ready(container)
        yield container
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


def test_root_default_boot_survives_image_edits(rootdefault_sandbox):
    # Surface ②: default boot is still ROOT (no USER directive) and the same six-service +
    # VNC-socket + CDP liveness holds (codex R2 P3 — RUNNING alone is not a health oracle).
    c = rootdefault_sandbox
    assert _uid_gid(c, "/proc/1/status")["Uid"] == ["0"] * 4, (
        "default boot must be root (all-0 uid: real/effective/saved/fs)")
    assert _all_six_running(c), "not all 6 services RUNNING under root default"
    for port in (8080, 5900, 5901):
        _assert_port_accepts(c, port)
    assert _exec(c, ["sh", "-c", "curl -fsS http://127.0.0.1:9222/json/version >/dev/null"])[0] == 0
    # chrome's new HOME=/home/ubuntu exists + is writable BY uid 1000 (the §4.4 edit). Probe AS
    # uid 1000 via `sudo -u ubuntu` (codex final-audit P3): a bare root `test -w` would pass even
    # if /home/ubuntu were root-owned 0700, so it would not prove chrome (uid 1000 under --user)
    # can write its profile/cache.
    assert _exec(c, ["sh", "-c", "sudo -u ubuntu test -w /home/ubuntu"])[0] == 0


def _assert_erofs_as_root(container, path: str) -> None:
    # Probe AS ROOT (user="0") — NOT sudo (codex R1 P2-3 / R2 P2-2): sudo writes its own timestamp
    # under the read-only /run,/var and would EROFS BEFORE the target write, confounding the source.
    # A uid-1000 write to /usr is EACCES even on a WRITABLE rootfs, so we MUST probe as root to
    # isolate EROFS (errno 30) for THIS path as the read-only proof.
    probe = (
        "import os,sys\n"
        f"try:\n"
        f"    open({path!r}, 'w').close()\n"
        f"    sys.exit(0)\n"
        f"except OSError as e:\n"
        f"    sys.exit(e.errno)\n"
    )
    code, out = _exec(container, ["python3", "-c", probe], user="0")
    assert code == errno.EROFS, (
        f"expected EROFS ({errno.EROFS}) writing {path} as root, got exit={code}: {out!r}")


@pytest.fixture
def readonly_sandbox(tmp_path):
    """Boot the REAL actus-sandbox with the FULL production-faithful read-only config (spec §9①):
    the translator's read_only+tmpfs+caps+user flat kwargs (run_as_user ON — the realistic
    non-root × read-only flip target) PLUS the /home/ubuntu anon volume PLUS a memory :ro bind from
    a temp host dir — exactly what _create_task assembles. A smoke passing only the flat kwargs
    (no anon volume) would put /home/ubuntu on the read-only rootfs → chrome dies → green-for-the-
    wrong-reason."""
    from docker.types import Mount
    client = _docker_client_or_skip()
    _require_image(client, SANDBOX_IMAGE)
    host_mem = tmp_path / "mem"
    host_mem.mkdir()
    (host_mem / "seed.md").write_text("seed", encoding="utf-8")  # prove the :ro bind is readable
    # mirror _create_task's assembly order EXACTLY (R1 P3): the memory :ro bind is built first
    # (docker_sandbox.py:236-238), then Task 4 appends the /home/ubuntu volume after the hardening
    # merge → [memory_bind, anon_volume] (the Task 4 test pins this same order). Targets are
    # disjoint so order is behaviourally irrelevant, but matching it keeps the smoke faithful.
    mounts = [
        Mount(target="/workspace/.memory", source=str(host_mem), type="bind", read_only=True),
        Mount(target="/home/ubuntu", source=None, type="volume", read_only=False),
    ]
    container = client.containers.run(
        SANDBOX_IMAGE, detach=True,
        **hardening_kwargs(hardening=True, run_as_user=True, read_only_rootfs=True),
        mounts=mounts,
    )
    try:
        _wait_ready(container)
        yield container
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass


def test_read_only_rootfs_profile(readonly_sandbox):
    # Surface ①: rootfs read-only (EROFS as root on a non-carve-out path) + /tmp & /home/ubuntu
    # writable + the memory :ro bind composes + all six services + sockets — every leg asserted to
    # defeat a half-boot-green (e.g. /tmp writable but /home/ubuntu not → chrome silently dies).
    c = readonly_sandbox
    # (1) core security claim: a ROOT write to a NON-carve-out system path → EROFS (errno 30).
    _assert_erofs_as_root(c, "/usr/_ro_proof")
    # (2) the /etc DIRECTORY is read-only (a NEW file on the image-backed /etc → EROFS as root). The
    # OCI default pseudo-fs & per-container mounts (/dev, /proc, the 3 /etc/* network files) stay
    # writable/standard by design and are NOT claimed confined (spec §10 INV-8 / §11).
    _assert_erofs_as_root(c, "/etc/_ro_proof")
    # (3) /tmp carve-out writable (tmpfs, 1777).
    assert _exec(c, ["sh", "-c", "touch /tmp/_rw_proof"])[0] == 0, "/tmp not writable"
    # (4) /home/ubuntu carve-out writable AS the non-root uid-1000 PID1 (anon volume + 1000:1000).
    # First pin that an exec runs as uid/gid 1000 (the container booted `--user 1000:1000`), so the
    # write below genuinely proves "writable AS the non-root uid" (spec §9①), not merely "writable"
    # (R2 P3 — mirrors the C5d-3 non-root smoke's id -u / id -g assertions).
    assert _exec(c, ["id", "-u"])[1].strip() == b"1000", "exec id -u != 1000 (run_as_user not enforced)"
    assert _exec(c, ["id", "-g"])[1].strip() == b"1000", "exec id -g != 1000"
    assert _exec(c, ["sh", "-c", "touch /home/ubuntu/_rw_proof"])[0] == 0, (
        "/home/ubuntu not writable by uid 1000 (anon volume ownership / mount)")
    # (5) memory :ro bind composes (the Task 7 mkdir proof): mounted + readable + EROFS even for root.
    assert _exec(c, ["sh", "-c", "cat /workspace/.memory/seed.md"])[1].strip() == b"seed", (
        "memory :ro bind not mounted/readable")
    _assert_erofs_as_root(c, "/workspace/.memory/_ro_proof")
    # (6) all SIX services RUNNING + sockets live + CDP up (RUNNING alone is not a health oracle).
    assert _all_six_running(c), "not all 6 services RUNNING under read-only rootfs"
    for port in (8080, 5900, 5901):
        _assert_port_accepts(c, port)
    assert _exec(c, ["sh", "-c", "curl -fsS http://127.0.0.1:9222/json/version >/dev/null"])[0] == 0
