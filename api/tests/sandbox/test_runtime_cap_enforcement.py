"""C5d-1 surface ①: negative runtime cap-enforcement proof.

Boots `python:3.12-alpine` with the EXACT production-emitted hardening kwargs
and proves the kernel drops NET_RAW/MKNOD/AUDIT_WRITE/NET_BIND_SERVICE (cleared
in CapEff/CapPrm/CapBnd) and caps pids at 512 — and that with hardening OFF the
same powers are present (the "teeth", INV-3). Cap enforcement is image-independent,
so alpine proves the kwargs are kernel-enforced cheaply (codex P2).

`-m sandbox` (excluded by default; CI `sandbox-adversarial` job runs it).
INV-7: ALL docker access is inside the test function.
"""
from __future__ import annotations

import json

import pytest

from tests.sandbox._docker_helpers import (
    DROPPED_CAP_BITS,
    EPERM,
    PYTHON_ALPINE_IMAGE,
    _docker_client_or_skip,
    _require_image,
    decode_caps,
    hardening_kwargs,
    run_container_probe,
)

pytestmark = [pytest.mark.sandbox]

# Probe: read Cap{Eff,Prm,Bnd}; attempt raw socket + mknod; bounded ALIVE-child
# fork loop (children block on pause() so the live count accumulates — codex
# R1#3 — killed in finally); read pids.max; emit JSON on stdout.
_PROBE = r"""
import json, os, socket, signal, errno

def caps():
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            for k in ("CapEff", "CapPrm", "CapBnd"):
                if line.startswith(k + ":"):
                    out[k] = line.split(":", 1)[1].strip()
    return out

def raw_socket_errno():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
        s.close()
        return 0
    except OSError as e:
        return e.errno

def mknod_errno():
    p = "/tmp/c5d1_mknod"
    try:
        if os.path.exists(p):
            os.remove(p)
        os.mknod(p, 0o600 | 0o020000, os.makedev(1, 3))  # S_IFCHR
        os.remove(p)
        return 0
    except OSError as e:
        return e.errno

def read_cgroup_diag():
    try:
        with open("/proc/self/cgroup") as f:
            return f.read()
    except OSError:
        return ""

def pids_max():
    # fast path (private cgroupns / v2 unified): the container's own cgroup is root
    fast = None
    try:
        fast = open("/sys/fs/cgroup/pids.max").read().strip()
        if fast == "512":
            return fast
    except OSError:
        pass
    # resolve the process's OWN pids controller (host cgroupns / non-root path)
    for line in read_cgroup_diag().splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hid, controllers, rel = parts
        if hid == "0":  # cgroup v2 unified
            cand = "/sys/fs/cgroup" + (rel.rstrip("/") or "") + "/pids.max"
        elif "pids" in controllers.split(","):  # cgroup v1 pids controller
            cand = "/sys/fs/cgroup/pids" + rel + "/pids.max"
        else:
            continue
        try:
            return open(cand).read().strip()
        except OSError:
            continue
    try:
        return open("/sys/fs/cgroup/pids/pids.max").read().strip()  # v1 fixed mount
    except OSError:
        return fast  # non-512 fast-path value or None — host asserts + dumps diag

def fork_eagain_at(limit):
    kids = []
    hit = None
    try:
        for i in range(limit):
            try:
                pid = os.fork()
            except OSError as e:
                if e.errno == errno.EAGAIN:
                    hit = i
                    break
                raise
            if pid == 0:
                signal.pause()      # block until SIGKILL
                os._exit(0)
            kids.append(pid)
        return hit
    finally:
        for pid in kids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        for pid in kids:
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass

limit = int(os.environ.get("PROBE_FORK_LIMIT", "700"))
print(json.dumps({
    "caps": caps(),
    "net_raw_errno": raw_socket_errno(),
    "mknod_errno": mknod_errno(),
    "pids_max": pids_max(),
    "cgroup": read_cgroup_diag(),   # diagnostics dumped via the assert messages (codex plan-R1 P2)
    "fork_eagain_at": fork_eagain_at(limit),
}))
"""


@pytest.mark.parametrize("hardening", [True, False], ids=["hardening_on", "hardening_off"])
def test_dropped_caps_and_pids_are_kernel_enforced(hardening: bool):
    client = _docker_client_or_skip()
    _require_image(client, PYTHON_ALPINE_IMAGE)

    kwargs = hardening_kwargs(hardening=hardening)
    # ON must exceed 512 to prove the cap; OFF only sanity-forks a small count
    # (codex P1 — don't require 700 forks succeed on the runner).
    fork_limit = "700" if hardening else "40"

    exit_code, stdout, stderr = run_container_probe(
        client,
        PYTHON_ALPINE_IMAGE,
        ["python3", "-c", _PROBE],
        environment={"PROBE_FORK_LIMIT": fork_limit},
        timeout=120,
        container_kwargs=kwargs,  # plan-R1 P1: thread cap_drop/pids_limit into containers.run
    )
    assert exit_code == 0, f"probe crashed (exit={exit_code}); stdout={stdout!r}; stderr={stderr!r}"
    data = json.loads(stdout.decode())

    eff = decode_caps(data["caps"]["CapEff"])
    prm = decode_caps(data["caps"]["CapPrm"])
    bnd = decode_caps(data["caps"]["CapBnd"])

    if hardening:
        # INV-3: dropped in ALL THREE sets (cannot use, raise, or regain).
        for name, s in (("CapEff", eff), ("CapPrm", prm), ("CapBnd", bnd)):
            assert DROPPED_CAP_BITS.isdisjoint(s), f"{name} still has a dropped cap: {data['caps']}"
        assert data["net_raw_errno"] == EPERM, data
        assert data["mknod_errno"] == EPERM, data
        assert data["pids_max"] == "512", data
        assert data["fork_eagain_at"] is not None, "pids cap not hit before fork limit"
    else:
        # Teeth: the SAME powers are present when unhardened (Docker default set).
        for name, s in (("CapEff", eff), ("CapPrm", prm), ("CapBnd", bnd)):
            assert DROPPED_CAP_BITS <= s, f"{name} missing a default cap: {data['caps']}"
        assert data["net_raw_errno"] == 0, data
        assert data["mknod_errno"] == 0, data
        assert data["pids_max"] != "512", data
        assert data["fork_eagain_at"] is None, "small fork sanity unexpectedly hit a cap"
