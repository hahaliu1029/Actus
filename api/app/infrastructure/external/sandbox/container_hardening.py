"""C5c pure-infra translator: ContainerRuntimePolicy → docker run kwargs +
applied-policy projection.

INV-0 heart — an unhardened policy yields {} so the ON-unhardened and OFF paths
produce a byte-identical container_config. Lives in infrastructure (it knows
docker-py kwarg names); the domain compiler stays pure (INV-5).
"""
from __future__ import annotations

from app.domain.models.sandbox_policy import ContainerRuntimePolicy, MountView


def container_hardening_kwargs(policy: ContainerRuntimePolicy) -> dict:
    """Render ONLY the hardening kwargs docker-py consumes (cap_drop/cap_add/
    security_opt as list[str], pids_limit as int). Empty/None fields are OMITTED → an
    unhardened policy → {} (INV-0); an empty cap_add (conservative profile) is omitted so
    the ON-conservative container_config stays byte-identical to C5c (INV-0 tier-2).
    run_as_user/read_only_rootfs are intentionally NOT emitted (C5c hardcodes them
    None/False → C5d)."""
    kwargs: dict = {}
    if policy.cap_drop:
        kwargs["cap_drop"] = list(policy.cap_drop)
    if policy.cap_add:
        kwargs["cap_add"] = list(policy.cap_add)
    if policy.security_opt:
        kwargs["security_opt"] = list(policy.security_opt)
    if policy.pids_limit is not None:
        kwargs["pids_limit"] = policy.pids_limit
    return kwargs


# ── C5d-2 fail-closed pre-run validator (spec §5) ─────────────────────────── #
# The ONLY caps the strict tier may EVER grant = the 9-cap allowlist + the
# documented SYS_CHROOT parity-10 fallback. A SEPARATE literal, NOT derived from the
# compiler's _STRICT_CAP_ADD, so defeating the guard needs a reviewed two-place edit.
# Rejects the 4 C5c drops, every escape cap (SYS_ADMIN/PTRACE/NET_ADMIN/BPF/…), and
# "ALL". Matching is EXACT canonical form (UPPERCASE, no CAP_ prefix — the form the
# compiler emits and docker-py passes verbatim to the daemon); any non-canonical name
# is rejected deny-by-default → no case/prefix bypass.
_VETTED_CAP_CEILING: frozenset[str] = frozenset({
    "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "SETUID", "SETGID",
    "SETPCAP", "SETFCAP", "KILL",   # the 9 allowlist
    "SYS_CHROOT",                   # only as the documented parity-10 fallback (spec §4.2)
})
# Deny-by-default: the ONLY security_opt Actus emits is no-new-privileges. A vetted
# allowlist auto-rejects seccomp=unconfined / apparmor=unconfined / systempaths=
# unconfined AND any unknown opt — stronger than blocklisting "*=unconfined".
_VETTED_SECURITY_OPTS: frozenset[str] = frozenset({"no-new-privileges:true"})


class SandboxHardeningConfigError(ValueError):
    """Raised by validate_hardening_config on escape-enabling / malformed container
    config. A ValueError subclass; _create_task re-raises it intact (before the broad
    handler) so the typed contract survives for precise diagnosis + create-path tests."""


def _str_list(value, field: str) -> list[str]:
    """Shape guard: a scalar security_opt="seccomp=unconfined" / cap_drop="ALL" would
    iterate as CHARACTERS and silently bypass the membership checks. Require a real
    list/tuple of str (None → [])."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SandboxHardeningConfigError(
            f"{field} must be a list[str], got {type(value).__name__}")
    if any(not isinstance(x, str) for x in value):
        raise SandboxHardeningConfigError(f"{field} must contain only str")
    return list(value)


def validate_hardening_config(container_config: dict) -> None:
    """Fail-closed guard over the REAL pre-run kwargs (sees privileged/security_opt/
    cap_add as actually assembled). Raises SandboxHardeningConfigError on any escape-
    enabling / malformed config. Scope = the HARDENED path only (the caller invokes it
    ONLY when runtime_policy is not None → INV-0: the OFF path is untouched).

    Bounded (spec §5/§11): guards privileged + the cap kwargs + security_opt; does NOT
    validate devices / userns_mode / pid_mode / ipc_mode / network_mode=host / cgroupns
    (the base container_config sets none today; a future slice adding any owns extending
    this guard and its tests)."""
    if container_config.get("privileged"):
        raise SandboxHardeningConfigError("privileged=True is never allowed for the sandbox")
    cap_add = _str_list(container_config.get("cap_add"), "cap_add")
    cap_drop = _str_list(container_config.get("cap_drop"), "cap_drop")
    security_opt = _str_list(container_config.get("security_opt"), "security_opt")
    bad = [c for c in cap_add if c == "ALL" or c not in _VETTED_CAP_CEILING]
    if bad:
        raise SandboxHardeningConfigError(f"cap_add outside vetted ceiling: {bad}")
    if cap_add and "ALL" not in cap_drop:
        raise SandboxHardeningConfigError("cap_add requires cap_drop=['ALL'] (deny-by-default)")
    bad_opts = [o for o in security_opt if o not in _VETTED_SECURITY_OPTS]
    if bad_opts:
        raise SandboxHardeningConfigError(
            "security_opt outside vetted set (rejects seccomp/apparmor/systempaths="
            f"unconfined + any unvetted opt): {bad_opts}")


def build_applied_runtime_policy(
    *, container_config: dict, memory_mount, memory_mount_target: str,
) -> ContainerRuntimePolicy:
    """Build the honest applied ContainerRuntimePolicy from the REAL, fully-assembled
    container_config (read back the merged kwargs) + the real per-bind mount decision.
    capture_kind='applied'. mounts reflect reality: a MountView iff a Mount was created
    (option (b) — never echoes an intended-but-skipped mount)."""
    mounts: tuple[MountView, ...] = ()
    if memory_mount is not None:
        mounts = (
            MountView(target=memory_mount_target, source_kind="memory_bind",
                      read_only=True),
        )
    return ContainerRuntimePolicy(
        capture_kind="applied",
        creation_mode="docker_run",
        image=container_config.get("image"),
        mem_limit=container_config.get("mem_limit"),
        run_as_user=container_config.get("user"),                # None in C5c
        read_only_rootfs=bool(container_config.get("read_only", False)),  # False in C5c
        cap_drop=tuple(container_config.get("cap_drop", ())),
        cap_add=tuple(container_config.get("cap_add", ())),
        security_opt=tuple(container_config.get("security_opt", ())),
        pids_limit=container_config.get("pids_limit"),
        mounts=mounts,
    )
