from __future__ import annotations

from app.application.services.sandbox_runtime_policy import compile_runtime_policy


class _Settings:
    sandbox_address = None
    sandbox_image = "actus/sandbox:latest"
    sandbox_network = None
    sandbox_mem_limit = "4g"
    sandbox_default_cwd = "/root"
    sandbox_https_proxy = None
    sandbox_http_proxy = None
    sandbox_no_proxy = None
    sandbox_memory_mount_target = "/workspace/.memory"
    sandbox_memory_mount_enabled = True
    sandbox_runtime_hardening_enabled = False
    sandbox_no_new_privileges_enabled = False


def test_returns_none_when_flag_off():
    assert compile_runtime_policy(_Settings()) is None


def test_returns_hardened_policy_when_flag_on():
    class _S(_Settings):
        sandbox_runtime_hardening_enabled = True

    pol = compile_runtime_policy(_S())
    assert pol is not None
    assert pol.cap_drop == ("NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE")
    assert pol.pids_limit == 512
    assert pol.security_opt == ()


def test_no_new_privileges_opt_in_via_flag():
    class _S(_Settings):
        sandbox_runtime_hardening_enabled = True
        sandbox_no_new_privileges_enabled = True

    assert compile_runtime_policy(_S()).security_opt == ("no-new-privileges:true",)


def test_missing_attr_defaults_off():
    class _Bare:
        pass

    assert compile_runtime_policy(_Bare()) is None
