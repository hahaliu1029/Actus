from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from core.config import Settings


def test_read_only_rootfs_flag_defaults_false():
    # Default-OFF dark-launch: a bare test Settings has read_only_rootfs OFF.
    assert Settings(env="test").sandbox_read_only_rootfs_enabled is False


def test_read_only_rootfs_flag_parses_via_field_name_when_hardening_on():
    # The field name itself is an alias; read-only needs hardening on to pass the guard.
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_read_only_rootfs_enabled=True,
    )
    assert s.sandbox_read_only_rootfs_enabled is True


def test_read_only_rootfs_flag_parses_via_screaming_alias_when_hardening_on():
    s = Settings(
        env="test",
        SANDBOX_RUNTIME_HARDENING_ENABLED=True,
        SANDBOX_READ_ONLY_ROOTFS_ENABLED=True,
    )
    assert s.sandbox_read_only_rootfs_enabled is True


def test_read_only_rootfs_without_hardening_raises():
    # INV-7 fail-closed: read-only alone would emit no --read-only AND apply NO hardening.
    with pytest.raises(ValidationError):
        Settings(env="test", sandbox_read_only_rootfs_enabled=True)


def test_read_only_rootfs_without_hardening_raises_even_in_external_mode():
    # Mode-INDEPENDENT: a contradictory security config must never be silently accepted,
    # even when sandbox_address is set (external mode).
    with pytest.raises(ValidationError):
        Settings(
            env="test",
            sandbox_read_only_rootfs_enabled=True,
            sandbox_address="http://remote-sandbox:8080",
        )


def test_hardening_alone_constructs_read_only_rootfs_off():
    # The guard fires ONLY on read-only-without-hardening; hardening alone is valid.
    s = Settings(env="test", sandbox_runtime_hardening_enabled=True)
    assert s.sandbox_read_only_rootfs_enabled is False


def test_read_only_rootfs_with_default_memory_target_constructs():
    # The default sandbox_memory_mount_target=='/workspace/.memory' is the prebuilt mountpoint
    # (Dockerfile mkdir, Task 7) → read-only + memory-on constructs (a /root-cwd warning fires,
    # see below — non-fatal).
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_read_only_rootfs_enabled=True,
    )
    assert s.sandbox_read_only_rootfs_enabled is True
    assert s.sandbox_memory_mount_target == "/workspace/.memory"


@pytest.mark.parametrize("target", ["/workspace/other", "/etc/x", "/workspace/.memory2"])
def test_read_only_rootfs_with_nondefault_memory_target_raises(target):
    # The image pre-creates ONLY /workspace/.memory; a custom target would fail to mount under
    # a read-only rootfs (runc cannot create a missing mountpoint) → fail fast at construction.
    with pytest.raises(ValidationError):
        Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_read_only_rootfs_enabled=True,
            sandbox_memory_mount_target=target,
        )


@pytest.mark.parametrize("target", ["/workspace/./.memory", "/workspace//.memory", "/workspace/.memory/"])
def test_read_only_rootfs_accepts_normalized_default_memory_target(target):
    # posixpath.normpath normalizes . // trailing-slash to the canonical default → accepted.
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_read_only_rootfs_enabled=True,
        sandbox_memory_mount_target=target,
    )
    assert s.sandbox_read_only_rootfs_enabled is True


def test_read_only_rootfs_rejects_dotdot_memory_target():
    # A `..` component is rejected fail-closed even if it lexically normalizes elsewhere.
    with pytest.raises(ValidationError):
        Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_read_only_rootfs_enabled=True,
            sandbox_memory_mount_target="/workspace/../.memory",
        )


def test_read_only_rootfs_nondefault_memory_target_ok_when_mount_disabled():
    # memory_mount_enabled=False ⇒ no bind ⇒ no mountpoint concern ⇒ the target guard does not fire.
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_read_only_rootfs_enabled=True,
        sandbox_memory_mount_enabled=False,
        sandbox_memory_mount_target="/workspace/other",
    )
    assert s.sandbox_read_only_rootfs_enabled is True


def test_read_only_rootfs_off_nondefault_memory_target_ok():
    # The target guard is gated on read_only_rootfs — OFF ⇒ a custom target is unconstrained.
    s = Settings(env="test", sandbox_memory_mount_target="/workspace/other")
    assert s.sandbox_read_only_rootfs_enabled is False


def test_read_only_rootfs_on_with_default_cwd_root_warns(caplog):
    # Broadened pre-flip nudge (C5d-4): read-only flipped ALONE (no run_as_user) must still warn —
    # /root sits on the read-only rootfs and is unwritable. Construction still succeeds.
    with caplog.at_level(logging.WARNING, logger="core.config"):
        s = Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_read_only_rootfs_enabled=True,
        )
    assert s.sandbox_read_only_rootfs_enabled is True
    assert any("sandbox_default_cwd='/root'" in r.message for r in caplog.records)
    assert any("pre-flip checklist" in r.message for r in caplog.records)


def test_read_only_rootfs_on_with_nonroot_cwd_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="core.config"):
        Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_read_only_rootfs_enabled=True,
            sandbox_default_cwd="/home/ubuntu",
        )
    assert not any("pre-flip checklist" in r.message for r in caplog.records)


def test_read_only_rootfs_on_external_mode_does_not_warn(caplog):
    # External mode: the flag is inert (no container) → no spurious /root-cwd warning. (The raise
    # guard does NOT fire here because hardening is also on — this is a Settings-VALID combo.)
    with caplog.at_level(logging.WARNING, logger="core.config"):
        Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_read_only_rootfs_enabled=True,
            sandbox_address="http://remote-sandbox:8080",
            sandbox_memory_mount_enabled=False,
        )
    assert not any("pre-flip checklist" in r.message for r in caplog.records)
