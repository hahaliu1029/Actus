"""C5c pure-infra translator: ContainerRuntimePolicy → docker run kwargs +
applied-policy projection.

INV-0 heart — an unhardened policy yields {} so the ON-unhardened and OFF paths
produce a byte-identical container_config. Lives in infrastructure (it knows
docker-py kwarg names); the domain compiler stays pure (INV-5).
"""
from __future__ import annotations

from app.domain.models.sandbox_policy import ContainerRuntimePolicy, MountView


# ── C5d-4 read-only-rootfs writable carve-out (spec §4.2) ──────────────────── #
# The ONLY paths made writable under read_only=True. `exec` because chrome/build steps exec
# from /tmp; nosuid+nodev defense-in-depth (NNP also blocks suid escalation); bounded size
# caps RAM. A writable SYSTEM path here would be an escape vector (binary replacement under a
# read-only rootfs) → the validator (validate_hardening_config) confines the tmpfs keys to
# _VETTED_READONLY_TMPFS_TARGETS (a SEPARATE literal).
_READONLY_TMPFS: dict[str, str] = {"/tmp": "rw,exec,nosuid,nodev,size=512m"}
# The agent workspace + chrome profile + caches. An ANONYMOUS Docker volume (source=None), NOT a
# tmpfs: disk-backed (no mem_limit cost) + inherits the image's 1000:1000 ownership (Dockerfile
# useradd -u 1000) so the C5d-3 non-root uid writes it with no uid= option; --rm auto-removes it.
# Added in _create_task (docker_sandbox.py) because a volume must go through the `mounts` list,
# not the translator's flat kwarg dict.
_READONLY_WORKSPACE_TARGET: str = "/home/ubuntu"


def container_hardening_kwargs(policy: ContainerRuntimePolicy) -> dict:
    """Render ONLY the hardening kwargs docker-py consumes (cap_drop/cap_add/
    security_opt as list[str], pids_limit as int). Empty/None fields are OMITTED → an
    unhardened policy → {} (INV-0); an empty cap_add (conservative profile) is omitted so
    the ON-conservative container_config stays byte-identical to C5c (INV-0 tier-2).
    run_as_user is emitted as docker-py `user=` when set (C5d-3, omit-when-None → INV-0);
    read_only_rootfs is emitted as docker-py read_only=True + the /tmp tmpfs when set (C5d-4,
    omit-both-when-False → INV-0); the /home/ubuntu writable volume is added in _create_task."""
    kwargs: dict = {}
    if policy.cap_drop:
        kwargs["cap_drop"] = list(policy.cap_drop)
    if policy.cap_add:
        kwargs["cap_add"] = list(policy.cap_add)
    if policy.security_opt:
        kwargs["security_opt"] = list(policy.security_opt)
    if policy.pids_limit is not None:
        kwargs["pids_limit"] = policy.pids_limit
    if policy.run_as_user is not None:
        # `is not None`, NOT truthy (codex R1 P2): a truthy check would suppress an empty
        # run_as_user="" → no `user` kwarg → Docker silently defaults to ROOT (a silent
        # non-root downgrade). `is not None` emits user="" so the validator rejects it
        # fail-closed. The compiler only ever produces None or "1000:1000" — defense-in-depth.
        kwargs["user"] = policy.run_as_user
    if policy.read_only_rootfs:
        # `bool` field, so a plain truthy check is correct (no empty-string footgun like
        # run_as_user). docker-py maps read_only=True → HostConfig.ReadonlyRootfs and
        # tmpfs={path: opts} → HostConfig.Tmpfs. Fresh dict copy — never share the module
        # constant (a caller mutation of container_config["tmpfs"] must not corrupt it).
        kwargs["read_only"] = True
        kwargs["tmpfs"] = dict(_READONLY_TMPFS)
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
# C5d-3: the ONE vetted non-root identity the strict path may EVER emit. A SEPARATE literal,
# NOT imported from the compiler's _RUN_AS_USER, so defeating the guard needs a reviewed
# two-place edit. EXACT match (codex Q4): rejects "0"/"root" (root), "1000:0" (root gid),
# "0:1000", bare "1000"/names (passwd-dependent), "" and any non-str object — deny-by-default.
# No numeric parser (a parser is easier to over-broaden); add parsing only if multiple vetted
# identities ever exist.
_VETTED_RUN_AS_USER: frozenset[str] = frozenset({"1000:1000"})
# ── C5d-4 read-only-rootfs carve-out confinement (spec §5) ─────────────────── #
# The ONLY paths the read-only-rootfs carve-out may make writable. SEPARATE literals (NOT imported
# from the translator's _READONLY_TMPFS / _READONLY_WORKSPACE_TARGET) so widening needs a reviewed
# two-place edit. A writable SYSTEM path (e.g. /usr) under read_only=True = binary-replacement
# escape → deny-by-default.
_VETTED_READONLY_TMPFS_TARGETS: frozenset[str] = frozenset({"/tmp"})
_VETTED_READONLY_WORKSPACE_TARGET: str = "/home/ubuntu"   # the one writable volume target
# R2 P2-1/P2-2 (final audit): validate the carve-out by EXACT match, not necessary-conditions.
# A forged tmpfs keeping the /tmp key but stripping nosuid/nodev (suid/dev escalation) or unbounding
# size, or a forged anonymous-volume Mount carrying extra docker-py keys (VolumeOptions.DriverConfig
# → host bind-mount escape; NoCopy → breaks uid-1000 ownership), must fail closed.
_VETTED_READONLY_TMPFS_OPTIONS: str = "rw,exec,nosuid,nodev,size=512m"
_VETTED_READONLY_MOUNT_KEYS: frozenset[str] = frozenset({"Target", "Source", "Type", "ReadOnly"})


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
    user = container_config.get("user")
    if user is not None and (not isinstance(user, str) or user not in _VETTED_RUN_AS_USER):
        # `not isinstance(user, str)` PRECEDES the membership test (codex R1 P2): `user not in
        # <frozenset>` raises TypeError for an unhashable value (e.g. a list) instead of the
        # typed error; the isinstance guard converts any non-str — and the empty string "" —
        # into a clean fail-closed raise. Absent `user` (None) is the default root boot, fine.
        raise SandboxHardeningConfigError(
            f"user outside vetted non-root identity (must be one of "
            f"{sorted(_VETTED_RUN_AS_USER)}): {user!r}")
    if container_config.get("read_only"):
        # C5d-4 (spec §5 / INV-8): under a read-only rootfs the writable carve-out is the only
        # escape surface — a writable tmpfs OR volume over a SYSTEM path lets an attacker replace a
        # binary the root services exec. Confine ALL THREE surfaces deny-by-default (tmpfs + mounts
        # + volumes/volumes_from). Gated on read_only (without it the rootfs is writable anyway → a
        # carve-out adds no new escape → INV-0-safe).
        tmpfs = container_config.get("tmpfs")
        if tmpfs is not None:
            if not isinstance(tmpfs, dict):
                # a scalar would iterate as characters and bypass the membership loop → fail closed.
                raise SandboxHardeningConfigError(
                    f"tmpfs must be a dict, got {type(tmpfs).__name__}")
            bad = [t for t in tmpfs if t not in _VETTED_READONLY_TMPFS_TARGETS]
            if bad:
                raise SandboxHardeningConfigError(
                    f"tmpfs outside vetted read-only carve-out "
                    f"(must be one of {sorted(_VETTED_READONLY_TMPFS_TARGETS)}): {bad}")
            bad_opts = {t: tmpfs[t] for t in tmpfs if tmpfs[t] != _VETTED_READONLY_TMPFS_OPTIONS}
            if bad_opts:
                raise SandboxHardeningConfigError(
                    f"tmpfs options outside vetted read-only carve-out "
                    f"(must be {_VETTED_READONLY_TMPFS_OPTIONS!r}): {bad_opts}")
        for m in container_config.get("mounts", []) or []:  # docker-py Mount = dict: Target/Type/ReadOnly/Source
            read_only = m.get("ReadOnly", False)
            if read_only is True:  # a genuine read-only bind (the memory :ro mount) — not a write surface
                continue
            if read_only not in (False, None):
                # R3 P3 defense-in-depth: a plain `if m.get("ReadOnly")` would treat a truthy NON-bool
                # (e.g. the string "false") as read-only/exempt → a writable mount could slip the
                # confinement. Fail closed on a malformed ReadOnly shape (mirrors this file's
                # _str_list / isinstance(user) shape-checks). Prod always sets a real bool.
                raise SandboxHardeningConfigError(
                    f"mount ReadOnly must be a bool, got {read_only!r} (target={m.get('Target')!r})")
            extra = set(m) - _VETTED_READONLY_MOUNT_KEYS
            if extra:
                # R2 P2-2: the vetted anonymous volume is EXACTLY {Target,Source,Type,ReadOnly};
                # extra docker-py Mount keys (VolumeOptions.DriverConfig → host bind-mount escape,
                # NoCopy → breaks uid-1000 ownership) must not ride through the carve-out. Fail closed.
                raise SandboxHardeningConfigError(
                    f"writable mount carries non-vetted keys {sorted(extra)} (the carve-out volume "
                    f"must be EXACTLY {sorted(_VETTED_READONLY_MOUNT_KEYS)}): target={m.get('Target')!r}")
            if not (
                m.get("Type") == "volume"
                and m.get("Target") == _VETTED_READONLY_WORKSPACE_TARGET
                and m.get("Source") is None  # ANONYMOUS only (R2 P2-1): a NAMED volume = persistent/shared
            ):
                raise SandboxHardeningConfigError(
                    f"writable mount outside vetted read-only carve-out (only an ANONYMOUS volume "
                    f"at {_VETTED_READONLY_WORKSPACE_TARGET!r}): target={m.get('Target')!r} "
                    f"type={m.get('Type')!r} source={m.get('Source')!r}")
        for _vkey in ("volumes", "volumes_from"):
            # C5d-4 (final-audit R1 P2): docker-py exposes writable-mount channels BEYOND
            # `mounts`/`tmpfs` — `volumes` (host-path / named-volume binds) and `volumes_from`
            # (inherit another container's mounts). The vetted carve-out uses ONLY the `mounts`
            # Mount-list + the `/tmp` tmpfs; production sets neither of these. Under a read-only
            # rootfs they are an un-vetted writable-mount channel (e.g.
            # {"volumes": {"/host/usr": {"bind": "/usr", "mode": "rw"}}} → writable /usr →
            # binary-replacement escape) → deny-by-default (fail closed).
            if container_config.get(_vkey):
                raise SandboxHardeningConfigError(
                    f"{_vkey} is not permitted under a read-only rootfs (the vetted carve-out uses "
                    f"only the mounts Mount-list + the /tmp tmpfs): {container_config.get(_vkey)!r}")


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
        run_as_user=container_config.get("user"),  # C5d-3: the real --user (None when OFF)
        read_only_rootfs=bool(container_config.get("read_only", False)),  # C5d-4 populates it
        cap_drop=tuple(container_config.get("cap_drop", ())),
        cap_add=tuple(container_config.get("cap_add", ())),
        security_opt=tuple(container_config.get("security_opt", ())),
        pids_limit=container_config.get("pids_limit"),
        mounts=mounts,
    )
