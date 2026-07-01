from __future__ import annotations

from app.domain.models.sandbox_policy import build_settings_view
from app.domain.services.safety.sandbox_policy_compiler import SandboxPolicyCompiler
from core.config import Settings


def test_no_free_form_hardening_settings():
    # INV-4: hardening is a vetted profile behind booleans, NOT raw docker knobs.
    # C5d-2 extends this to cap_add — the allowlist is a compiler constant, never a knob.
    s = Settings(env="test")
    for forbidden in (
        "sandbox_security_opt", "sandbox_cap_drop", "sandbox_cap_add", "sandbox_pids_limit",
    ):
        assert not hasattr(s, forbidden), (
            f"INV-4 footgun: free-form {forbidden} must not exist on Settings"
        )


def test_strict_cap_add_is_subset_of_vetted_ceiling():
    # INV-4: the two two-place literals must stay consistent. _STRICT_CAP_ADD (what the
    # compiler emits) ⊆ _VETTED_CAP_CEILING (what the validator permits); and never "ALL".
    from app.domain.services.safety.sandbox_policy_compiler import _STRICT_CAP_ADD
    from app.infrastructure.external.sandbox.container_hardening import _VETTED_CAP_CEILING

    assert set(_STRICT_CAP_ADD) <= _VETTED_CAP_CEILING
    assert "ALL" not in _STRICT_CAP_ADD
    assert len(_STRICT_CAP_ADD) == 9  # the documented allowlist size (parity-10 adds SYS_CHROOT)


def test_sandbox_policy_compiler_stays_pure_domain():
    # INV-5: the compiler is PURE DOMAIN — stdlib + app.domain siblings only. A future edit
    # importing app.infrastructure / core / docker / fastapi / sqlalchemy would pass every
    # behavioral test yet break Clean Architecture; this AST gate fails fast.
    import ast
    from pathlib import Path

    compiler = (
        Path(__file__).resolve().parents[2]   # api/tests/core/<file> → api/
        / "app" / "domain" / "services" / "safety" / "sandbox_policy_compiler.py"
    )
    tree = ast.parse(compiler.read_text(encoding="utf-8"))
    forbidden = ("app.infrastructure", "core", "docker", "fastapi", "sqlalchemy")
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        elif isinstance(node, ast.Import):
            mods = [alias.name for alias in node.names]
        else:
            continue
        bad += [m for m in mods if any(m == p or m.startswith(p + ".") for p in forbidden)]
    assert not bad, f"INV-5: sandbox_policy_compiler.py must stay pure domain; forbidden imports: {bad}"


def test_validator_lives_in_infrastructure_not_domain():
    # INV-5 other half: the fail-closed validator knows docker-py kwargs → it is INFRA, not
    # domain. It must be importable from the infra module (where the plan places it).
    from app.infrastructure.external.sandbox.container_hardening import (  # noqa: F401
        SandboxHardeningConfigError,
        validate_hardening_config,
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


def test_no_free_form_run_as_user_settings():
    # INV-4 (C5d-3): the uid is a vetted constant, NOT a raw operator knob.
    s = Settings(env="test")
    for forbidden in ("sandbox_run_as_user", "sandbox_uid", "sandbox_gid"):
        assert not hasattr(s, forbidden), (
            f"INV-4 footgun: free-form {forbidden} must not exist on Settings"
        )


def test_run_as_user_constant_is_in_vetted_set():
    # INV-4: the two two-place literals must stay consistent — _RUN_AS_USER (what the compiler
    # emits) ∈ _VETTED_RUN_AS_USER (what the validator permits). Mirrors
    # test_strict_cap_add_is_subset_of_vetted_ceiling.
    from app.domain.services.safety.sandbox_policy_compiler import _RUN_AS_USER
    from app.infrastructure.external.sandbox.container_hardening import _VETTED_RUN_AS_USER

    assert _RUN_AS_USER == "1000:1000"
    assert _RUN_AS_USER in _VETTED_RUN_AS_USER
    # spec §5: EXACTLY one vetted identity (R5 P3) — pin the WHOLE set, not just membership, so a
    # future over-broadening of _VETTED_RUN_AS_USER (e.g. adding "1000:0") fails this gate.
    assert _VETTED_RUN_AS_USER == frozenset({_RUN_AS_USER})


def test_no_free_form_read_only_rootfs_settings():
    # INV-4 (C5d-4): read-only is a vetted BOOLEAN tier, NOT a raw operator knob; the carve-out
    # paths are vetted infra constants, never Settings.
    s = Settings(env="test")
    for forbidden in (
        "sandbox_read_only_rootfs",      # the bool flag is sandbox_read_only_rootfs_ENABLED only
        "sandbox_tmpfs",
        "sandbox_tmpfs_size",
        "sandbox_read_only_carve_out",
        "sandbox_writable_paths",
    ):
        assert not hasattr(s, forbidden), (
            f"INV-4 footgun: free-form {forbidden} must not exist on Settings"
        )


def test_read_only_carve_out_literals_are_consistent():
    # INV-4: the translator emit constants and the validator permit literals must stay consistent
    # (a SEPARATE-literal two-place edit, mirroring _RUN_AS_USER ∈ _VETTED_RUN_AS_USER). Pin the
    # WHOLE vetted set so a future over-broadening (e.g. adding "/usr") fails this gate.
    from app.infrastructure.external.sandbox.container_hardening import (
        _READONLY_TMPFS,
        _READONLY_WORKSPACE_TARGET,
        _VETTED_READONLY_TMPFS_OPTIONS,
        _VETTED_READONLY_TMPFS_TARGETS,
        _VETTED_READONLY_WORKSPACE_TARGET,
    )

    assert set(_READONLY_TMPFS) == {"/tmp"}
    assert _VETTED_READONLY_TMPFS_TARGETS == frozenset(_READONLY_TMPFS)
    assert _VETTED_READONLY_TMPFS_TARGETS == frozenset({"/tmp"})
    assert _READONLY_WORKSPACE_TARGET == _VETTED_READONLY_WORKSPACE_TARGET == "/home/ubuntu"
    assert _VETTED_READONLY_TMPFS_OPTIONS == _READONLY_TMPFS["/tmp"]


def test_no_free_form_egress_settings():
    # INV-4 (C5d-5/6): egress is a vetted BOOLEAN tier + ONE vetted network NAME, NOT raw docker
    # network knobs. The only egress Settings are the two *_enabled flags + sandbox_egress_internal_network.
    s = Settings(env="test")
    for forbidden in (
        "sandbox_network_mode",
        "sandbox_egress_mode",
        "sandbox_network_disabled",
        "sandbox_networking_config",
        "sandbox_egress_allowlist",
        "sandbox_egress_proxy",
    ):
        assert not hasattr(s, forbidden), (
            f"INV-4 footgun: free-form {forbidden} must not exist on Settings"
        )


def test_egress_settings_are_exactly_the_two_flags_plus_network_name():
    # Positive pin: the three egress Settings DO exist (so the gate above can't pass vacuously by a
    # rename), and nothing else egress-shaped does.
    s = Settings(env="test")
    assert hasattr(s, "sandbox_egress_isolation_enabled")
    assert hasattr(s, "sandbox_child_egress_isolation_enabled")
    assert hasattr(s, "sandbox_egress_internal_network")


def test_validator_network_rejection_literals_present():
    # INV-4: the validator's network confinement is a deny-by-default literal set. Pin that the
    # validator REJECTS each forbidden kwarg (behavioral, not source-grep) so a future weakening
    # (dropping one rejection) fails this gate.
    from app.infrastructure.external.sandbox.container_hardening import (
        SandboxHardeningConfigError, validate_hardening_config,
    )
    import pytest
    for bad in (
        {"network_mode": "host"},
        {"network_disabled": True},
        {"networking_config": {}},
        {"ports": {"8080/tcp": 8080}},
        {"publish_all_ports": True},
    ):
        with pytest.raises(SandboxHardeningConfigError):
            validate_hardening_config({"cap_drop": ["NET_RAW"], "pids_limit": 512, **bad})


def test_egress_network_two_place_handling_is_consistent():
    # INV-4 two-place handling (mirrors test_run_as_user_constant_is_in_vetted_set): the translator
    # EMITS `network` from policy.egress_network and the validator CONFINES it via
    # expected_egress_network. Prove the round-trip: a policy's egress_network → translator kwarg →
    # validator accepts iff expected matches.
    from app.domain.models.sandbox_policy import ContainerRuntimePolicy
    from app.infrastructure.external.sandbox.container_hardening import (
        SandboxHardeningConfigError, container_hardening_kwargs, validate_hardening_config,
    )
    import pytest
    pol = ContainerRuntimePolicy(
        capture_kind="configured", creation_mode="docker_run",
        image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
        read_only_rootfs=False, egress_network="actus-sandbox-internal",
        cap_drop=("NET_RAW",), cap_add=(), security_opt=(), pids_limit=512, mounts=(),
    )
    kw = container_hardening_kwargs(pol)
    cfg = {"cap_drop": ["NET_RAW"], "pids_limit": 512, **kw}
    validate_hardening_config(cfg, expected_egress_network="actus-sandbox-internal")  # match → ok
    with pytest.raises(SandboxHardeningConfigError):
        validate_hardening_config(cfg, expected_egress_network="other-net")  # mismatch → reject
