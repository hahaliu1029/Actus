from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings


# ---- defaults: both flags OFF, network name None -------------------------- #
def test_egress_flags_default_false_and_network_none():
    s = Settings(env="test")
    assert s.sandbox_egress_isolation_enabled is False
    assert s.sandbox_child_egress_isolation_enabled is False
    assert s.sandbox_egress_internal_network is None


# ---- global flag parses via both aliases (with hardening + network) ------- #
def test_global_egress_flag_parses_via_field_name():
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_egress_isolation_enabled=True,
        sandbox_egress_internal_network="actus-sandbox-internal",
    )
    assert s.sandbox_egress_isolation_enabled is True
    assert s.sandbox_egress_internal_network == "actus-sandbox-internal"


def test_global_egress_flag_parses_via_screaming_alias():
    s = Settings(
        env="test",
        SANDBOX_RUNTIME_HARDENING_ENABLED=True,
        SANDBOX_EGRESS_ISOLATION_ENABLED=True,
        SANDBOX_EGRESS_INTERNAL_NETWORK="actus-sandbox-internal",
    )
    assert s.sandbox_egress_isolation_enabled is True


def test_global_egress_flag_parses_via_actus_alias():
    s = Settings(
        env="test",
        ACTUS_C5_SANDBOX_RUNTIME_HARDENING_ENABLED=True,
        ACTUS_C5_SANDBOX_EGRESS_ISOLATION_ENABLED=True,
        ACTUS_C5_SANDBOX_EGRESS_INTERNAL_NETWORK="actus-sandbox-internal",
    )
    assert s.sandbox_egress_isolation_enabled is True


# ---- child flag parses (independent of the global flag) ------------------- #
def test_child_egress_flag_parses_with_hardening_and_network():
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_child_egress_isolation_enabled=True,
        sandbox_egress_internal_network="actus-sandbox-internal",
    )
    assert s.sandbox_child_egress_isolation_enabled is True
    assert s.sandbox_egress_isolation_enabled is False  # global still off


def test_child_egress_flag_parses_via_screaming_alias():
    s = Settings(
        env="test",
        SANDBOX_RUNTIME_HARDENING_ENABLED=True,
        SANDBOX_CHILD_EGRESS_ISOLATION_ENABLED=True,
        SANDBOX_EGRESS_INTERNAL_NETWORK="actus-sandbox-internal",
    )
    assert s.sandbox_child_egress_isolation_enabled is True


# ---- _egress_isolation_requires_hardening (global) ------------------------ #
def test_global_egress_without_hardening_raises():
    # INV-7 fail-closed: egress-alone would apply NO hardening (compile_runtime_policy
    # returns None when hardening off) → false sense of isolation.
    with pytest.raises(ValidationError):
        Settings(
            env="test",
            sandbox_egress_isolation_enabled=True,
            sandbox_egress_internal_network="actus-sandbox-internal",
        )


def test_global_egress_without_hardening_raises_even_in_external_mode():
    # Mode-INDEPENDENT: a contradictory security config is never silently accepted.
    with pytest.raises(ValidationError):
        Settings(
            env="test",
            sandbox_egress_isolation_enabled=True,
            sandbox_egress_internal_network="actus-sandbox-internal",
            sandbox_address="http://remote-sandbox:8080",
        )


# ---- _child_egress_requires_hardening ------------------------------------- #
def test_child_egress_without_hardening_raises():
    with pytest.raises(ValidationError):
        Settings(
            env="test",
            sandbox_child_egress_isolation_enabled=True,
            sandbox_egress_internal_network="actus-sandbox-internal",
        )


# ---- _egress_isolation_requires_internal_network (BOTH flags) ------------- #
def test_global_egress_on_without_network_name_raises():
    # INV-EG2: egress-on with no network name must NOT silently fall back to the
    # routable actus-net — fail fast at construction.
    with pytest.raises(ValidationError):
        Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_egress_isolation_enabled=True,
        )  # no sandbox_egress_internal_network


def test_child_egress_on_without_network_name_raises():
    with pytest.raises(ValidationError):
        Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_child_egress_isolation_enabled=True,
        )  # no sandbox_egress_internal_network


# ---- valid combos: hardening alone, and hardening+flags+network ----------- #
def test_hardening_alone_constructs_egress_off():
    s = Settings(env="test", sandbox_runtime_hardening_enabled=True)
    assert s.sandbox_egress_isolation_enabled is False
    assert s.sandbox_child_egress_isolation_enabled is False


def test_network_name_alone_without_flags_is_inert_and_constructs():
    # The name being set with both flags OFF is harmless (nothing consumes it) — constructs fine.
    s = Settings(env="test", sandbox_egress_internal_network="actus-sandbox-internal")
    assert s.sandbox_egress_internal_network == "actus-sandbox-internal"
    assert s.sandbox_egress_isolation_enabled is False
