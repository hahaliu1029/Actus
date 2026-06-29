"""Local unit tests for the pure (non-Docker) helpers in _docker_helpers.

These run in the DEFAULT suite (no `sandbox` marker, no docker daemon needed):
they only exercise bit decoding + the production-faithful kwargs builder.
"""
from __future__ import annotations

from tests.sandbox._docker_helpers import (
    DROPPED_CAP_BITS,
    EPERM,
    decode_caps,
    hardening_kwargs,
)


def test_dropped_cap_bits_are_the_four_baseline_caps():
    # NET_BIND_SERVICE=10, NET_RAW=13, MKNOD=27, AUDIT_WRITE=29
    assert DROPPED_CAP_BITS == frozenset({10, 13, 27, 29})
    assert EPERM == 1


def test_decode_caps_docker_default_mask_contains_all_four_dropped():
    # Docker default cap set => OFF baseline => all four bits SET.
    bits = decode_caps("00000000a80425fb")
    assert DROPPED_CAP_BITS <= bits


def test_decode_caps_zero_is_empty():
    assert decode_caps("0000000000000000") == set()


def test_decode_caps_handles_0x_prefix_and_uppercase():
    assert decode_caps("0xA80425FB") == decode_caps("a80425fb")


def test_hardening_kwargs_on_matches_production_profile():
    kwargs = hardening_kwargs(hardening=True, nnp=False)
    assert kwargs == {
        "cap_drop": ["NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"],
        "pids_limit": 512,
    }


def test_hardening_kwargs_on_with_nnp_adds_security_opt():
    kwargs = hardening_kwargs(hardening=True, nnp=True)
    assert kwargs["security_opt"] == ["no-new-privileges:true"]
    assert kwargs["cap_drop"] == ["NET_RAW", "MKNOD", "AUDIT_WRITE", "NET_BIND_SERVICE"]
    assert kwargs["pids_limit"] == 512


def test_hardening_kwargs_off_is_empty():
    # compile_runtime_policy returns None when the flag is off → kwargs == {}.
    assert hardening_kwargs(hardening=False) == {}
