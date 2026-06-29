from __future__ import annotations

from app.domain.models.sandbox_policy import build_settings_view
from app.domain.services.safety.sandbox_policy_compiler import SandboxPolicyCompiler
from core.config import Settings


def test_no_free_form_hardening_settings():
    # INV-4: hardening is a vetted profile behind booleans, NOT raw docker knobs.
    s = Settings(env="test")
    for forbidden in ("sandbox_security_opt", "sandbox_cap_drop", "sandbox_pids_limit"):
        assert not hasattr(s, forbidden), (
            f"INV-4 footgun: free-form {forbidden} must not exist on Settings"
        )


class _S:
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
    sandbox_runtime_hardening_enabled = True
    sandbox_no_new_privileges_enabled = False


def test_security_opt_only_source_is_no_new_privileges_flag():
    c = SandboxPolicyCompiler()
    # hardening on, no-new-priv off → NO security_opt
    pol = c.compile_container_runtime_policy(build_settings_view(_S()))
    assert pol.security_opt == ()

    class _On(_S):
        sandbox_no_new_privileges_enabled = True

    pol2 = c.compile_container_runtime_policy(build_settings_view(_On()))
    assert pol2.security_opt == ("no-new-privileges:true",)
