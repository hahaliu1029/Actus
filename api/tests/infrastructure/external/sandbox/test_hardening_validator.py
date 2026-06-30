from __future__ import annotations

import pytest
from docker.types import Mount

from app.infrastructure.external.sandbox.container_hardening import (
    SandboxHardeningConfigError,
    validate_hardening_config,
)

_STRICT_ADD = ["CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID",
               "SETUID", "SETGID", "SETPCAP", "SETFCAP", "KILL"]


# ---- passes: the real production configs ----------------------------------- #
def test_accepts_real_strict_config():
    validate_hardening_config({"cap_drop": ["ALL"], "cap_add": _STRICT_ADD, "security_opt": []})


def test_accepts_conservative_config():
    validate_hardening_config({
        "cap_drop": ["NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"],
        "pids_limit": 512,
    })


def test_accepts_no_new_privileges_security_opt():
    validate_hardening_config({
        "cap_drop": ["ALL"], "cap_add": _STRICT_ADD,
        "security_opt": ["no-new-privileges:true"],
    })


def test_accepts_parity10_fallback_with_sys_chroot():
    validate_hardening_config({"cap_drop": ["ALL"], "cap_add": _STRICT_ADD + ["SYS_CHROOT"]})


def test_accepts_empty_config():
    validate_hardening_config({})  # no hardening kwargs → nothing to reject


# ---- rejects: escape-enabling / malformed ---------------------------------- #
def test_rejects_privileged():
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"privileged": True})


def test_rejects_out_of_ceiling_cap_add():
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"cap_drop": ["ALL"], "cap_add": ["SYS_ADMIN"]})


def test_rejects_re_adding_a_c5c_dropped_cap():
    # NET_RAW is dropped by the conservative profile; the strict ceiling must NOT allow it back.
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"cap_drop": ["ALL"], "cap_add": ["NET_RAW"]})


def test_rejects_literal_all_in_cap_add():
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"cap_drop": ["ALL"], "cap_add": ["ALL"]})


def test_rejects_cap_add_without_drop_all():
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"cap_drop": ["NET_RAW"], "cap_add": ["CHOWN"]})


@pytest.mark.parametrize("opt", [
    "seccomp=unconfined", "apparmor=unconfined", "systempaths=unconfined", "label=disable",
])
def test_rejects_unvetted_security_opt(opt):
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"security_opt": [opt]})


@pytest.mark.parametrize("field,value", [
    ("cap_drop", "ALL"),                    # scalar str → would iterate as chars
    ("security_opt", "seccomp=unconfined"),
    ("cap_add", "CHOWN"),
])
def test_rejects_scalar_string_shapes(field, value):
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({field: value})


def test_rejects_non_str_elements():
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"cap_drop": ["ALL"], "cap_add": [123]})


@pytest.mark.parametrize("name", ["chown", "CAP_CHOWN", "Chown"])
def test_rejects_non_canonical_cap_name_deny_by_default(name):
    # Canonical form is UPPERCASE no-prefix; non-canonical → rejected (no case/prefix bypass).
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"cap_drop": ["ALL"], "cap_add": [name]})


# ---- C5d-3: the vetted-user guard ------------------------------------------ #
def test_accepts_vetted_run_as_user():
    validate_hardening_config({"user": "1000:1000"})


def test_accepts_absent_user():
    validate_hardening_config({"cap_drop": ["NET_RAW"]})  # no user key → root default, fine


@pytest.mark.parametrize("bad", [
    "0", "root", "1000:0", "0:1000", "1000", "ubuntu", "",          # root-equiv / passwd / empty
    "1000:1000:1000", " 1000:1000", "1000:1000 ", "01000:01000",    # malformed / padded / octal-ish
    "1000:1000\n",                                                  # trailing newline
])
def test_rejects_non_vetted_user(bad):
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"user": bad})


def test_rejects_non_str_user_without_typeerror():
    # codex R1 P2: an unhashable (list) `user` must raise the TYPED error, not a bare TypeError
    # from `user not in <frozenset>`. The isinstance guard precedes the membership test.
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"user": ["1000:1000"]})
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"user": 1000})


# ---- C5d-4: read-only-rootfs carve-out confinement ------------------------- #
def test_read_only_accepts_vetted_tmpfs_and_anon_volume():
    validate_hardening_config({
        "read_only": True,
        "tmpfs": {"/tmp": "rw,exec,nosuid,nodev,size=512m"},
        "mounts": [Mount(target="/home/ubuntu", source=None, type="volume", read_only=False)],
    })


def test_read_only_accepts_absent_tmpfs_and_no_mounts():
    # read_only WITHOUT the /tmp carve-out is a functional boot-failure, NOT an escape — outside
    # the validator's remit (spec §5). It accepts (the §9 smoke + prod timeout catch a broken boot).
    validate_hardening_config({"read_only": True})


def test_read_only_accepts_memory_ro_bind_alongside_anon_volume():
    # The realistic agent config: memory :ro bind (exempt — not a write surface) + the anon volume.
    validate_hardening_config({
        "read_only": True,
        "tmpfs": {"/tmp": "rw,exec,nosuid,nodev,size=512m"},
        "mounts": [
            Mount(target="/workspace/.memory", source="/host/mem", type="bind", read_only=True),
            Mount(target="/home/ubuntu", source=None, type="volume", read_only=False),
        ],
    })


@pytest.mark.parametrize("tmpfs", [
    {"/usr": "rw"},                  # system path → binary-replacement escape
    {"/tmp": "rw", "/etc": "rw"},    # one bad key among a good one
    {"/": "rw"},
])
def test_read_only_rejects_system_path_tmpfs(tmpfs):
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"read_only": True, "tmpfs": tmpfs})


def test_read_only_rejects_scalar_tmpfs():
    # A scalar would iterate as characters and silently bypass the membership check → fail closed.
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"read_only": True, "tmpfs": "/tmp"})


def test_read_only_rejects_writable_mount_at_wrong_target():
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({
            "read_only": True,
            "mounts": [Mount(target="/usr", source=None, type="volume", read_only=False)],
        })


def test_read_only_rejects_named_home_volume():
    # R2 P2-1: a NAMED volume (Source != None) = persistent/cross-container state, not the
    # ephemeral anonymous carve-out → rejected even at the right target.
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({
            "read_only": True,
            "mounts": [Mount(target="/home/ubuntu", source="namedvol", type="volume", read_only=False)],
        })


def test_read_only_rejects_writable_bind_at_home():
    # A writable bind (Type=bind, ReadOnly falsy) is not the vetted anonymous volume → rejected.
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({
            "read_only": True,
            "mounts": [Mount(target="/home/ubuntu", source="/host/x", type="bind", read_only=False)],
        })


def test_read_only_rejects_non_bool_readonly_on_system_mount():
    # R3 P3: a forged mount with a truthy NON-bool ReadOnly (e.g. the string "false") must NOT be
    # treated as an exempt read-only bind by a plain `if m.get("ReadOnly")` → fail closed on the
    # malformed shape. Use a raw dict (a forged/malformed config, not a real docker-py Mount).
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({
            "read_only": True,
            "mounts": [{"Target": "/usr", "Type": "bind", "ReadOnly": "false", "Source": "/host/x"}],
        })


def test_writable_rootfs_does_not_confine_carve_out():
    # The carve-out check is gated on read_only: without it the rootfs is writable anyway, so a
    # tmpfs/mount adds no new escape → the validator does NOT inspect tmpfs/mounts (INV-0-safe).
    validate_hardening_config({
        "tmpfs": {"/usr": "rw"},   # would be rejected UNDER read_only — but no read_only here
        "mounts": [Mount(target="/usr", source=None, type="volume", read_only=False)],
    })


@pytest.mark.parametrize("vkey,vval", [
    ("volumes", {"/host/usr": {"bind": "/usr", "mode": "rw"}}),
    ("volumes", ["/host/usr:/usr:rw"]),
    ("volumes_from", ["other_container"]),
])
def test_read_only_rejects_volumes_channels(vkey, vval):
    # R1 P2: `volumes`/`volumes_from` are un-vetted writable-mount channels → deny-by-default
    # under read_only (a writable /usr bind here would be a binary-replacement escape).
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"read_only": True, vkey: vval})


def test_read_only_accepts_vetted_carve_out_without_volumes():
    # The real carve-out (vetted tmpfs + anon /home/ubuntu volume via `mounts`, NO `volumes` kwarg)
    # still passes — the guard only bites the un-vetted `volumes`/`volumes_from` channels.
    validate_hardening_config({
        "read_only": True,
        "tmpfs": {"/tmp": "rw,exec,nosuid,nodev,size=512m"},
        "mounts": [Mount(target="/home/ubuntu", source=None, type="volume", read_only=False)],
    })


def test_writable_rootfs_does_not_confine_volumes():
    # INV-0: the volumes guard is gated on read_only — without it a `volumes` kwarg is not inspected.
    validate_hardening_config({"volumes": {"/host/usr": {"bind": "/usr", "mode": "rw"}}})


@pytest.mark.parametrize("opts", [
    "rw,exec,suid,dev,size=64g",        # suid+dev re-enabled, size unbounded
    "rw,exec,nosuid,nodev,size=64g",    # only size unbounded (DoS)
    "rw,exec",                          # nosuid/nodev dropped
    "",
])
def test_read_only_rejects_nonvetted_tmpfs_options(opts):
    # R2 P2-1: vetted /tmp key but altered option string → fail closed (value allowlisted, not just key).
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"read_only": True, "tmpfs": {"/tmp": opts}})


@pytest.mark.parametrize("extra", [
    {"VolumeOptions": {"NoCopy": True}},
    {"VolumeOptions": {"DriverConfig": {"Name": "local", "Options": {"type": "none", "o": "bind", "device": "/host"}}}},
    {"BindOptions": {"Propagation": "rshared"}},
])
def test_read_only_rejects_anon_volume_with_extra_keys(extra):
    # R2 P2-2: extra Mount keys on the "anonymous volume" (DriverConfig → host bind escape) → fail closed.
    m = {"Target": "/home/ubuntu", "Type": "volume", "Source": None, "ReadOnly": False, **extra}
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config({"read_only": True, "mounts": [m]})


def test_read_only_accepts_exact_real_mount_shape():
    # The REAL production Mount = exactly {Target,Source,Type,ReadOnly} → still passes (no over-reject).
    validate_hardening_config({
        "read_only": True,
        "tmpfs": {"/tmp": "rw,exec,nosuid,nodev,size=512m"},
        "mounts": [Mount(target="/home/ubuntu", source=None, type="volume", read_only=False)],
    })
