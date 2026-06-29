from __future__ import annotations

import pytest

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
