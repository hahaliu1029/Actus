"""B5 C1: SystemPromptBudget + compute_effective_window tests."""
from __future__ import annotations

import pytest

from app.domain.services.prompts.budget import (
    SystemPromptBudget,
    compute_effective_window,
)


# ---- SystemPromptBudget dataclass --------------------------------------- #


def test_budget_defaults() -> None:
    b = SystemPromptBudget(max_tokens=3500)
    assert b.max_tokens == 3500
    assert b.critical_priority_min == 8
    assert b.warn_on_overflow is True


def test_budget_is_frozen() -> None:
    b = SystemPromptBudget(max_tokens=3500)
    with pytest.raises(Exception):  # FrozenInstanceError
        b.max_tokens = 4000  # type: ignore[misc]


def test_budget_custom_critical_min() -> None:
    b = SystemPromptBudget(max_tokens=3500, critical_priority_min=9)
    assert b.critical_priority_min == 9


# ---- compute_effective_window ------------------------------------------- #


def test_effective_window_normal_case() -> None:
    """8000 total - 3500 system - 1000 reserved = 3500."""
    ew = compute_effective_window(
        total_context_window=8000,
        system_prompt_max_tokens=3500,
        reserved_output_tokens=1000,
    )
    assert ew == 3500


def test_effective_window_floor_kicks_in_when_negative() -> None:
    """Misconfigured: system + reserved exceeds total. Floor at 10% of total."""
    ew = compute_effective_window(
        total_context_window=8000,
        system_prompt_max_tokens=9000,
        reserved_output_tokens=500,
    )
    # raw = 8000 - 9000 - 500 = -1500
    # floor = int(8000 * 0.1) = 800
    assert ew == 800


def test_effective_window_floor_kicks_in_when_exactly_zero() -> None:
    """Edge case: raw equals zero. Floor wins."""
    ew = compute_effective_window(
        total_context_window=8000,
        system_prompt_max_tokens=7000,
        reserved_output_tokens=1000,
    )
    # raw = 0, floor = 800
    assert ew == 800


def test_effective_window_zero_total_returns_zero() -> None:
    """Pathological: zero total_context_window."""
    ew = compute_effective_window(
        total_context_window=0,
        system_prompt_max_tokens=3500,
        reserved_output_tokens=1000,
    )
    assert ew == 0


def test_effective_window_negative_total_returns_zero() -> None:
    """Pathological: negative total_context_window."""
    ew = compute_effective_window(
        total_context_window=-100,
        system_prompt_max_tokens=3500,
        reserved_output_tokens=1000,
    )
    assert ew == 0


def test_effective_window_large_total() -> None:
    """200k context window, 3500 system, 4096 reserved."""
    ew = compute_effective_window(
        total_context_window=200_000,
        system_prompt_max_tokens=3500,
        reserved_output_tokens=4096,
    )
    assert ew == 200_000 - 3500 - 4096


@pytest.mark.parametrize(
    "min_ratio,expected_floor",
    [
        (0.05, 400),  # 8000 * 0.05
        (0.1, 800),  # default
        (0.2, 1600),
        (0.5, 4000),
    ],
)
def test_effective_window_custom_min_ratio(min_ratio: float, expected_floor: int) -> None:
    """min_ratio parameter controls the floor."""
    ew = compute_effective_window(
        total_context_window=8000,
        system_prompt_max_tokens=10_000,  # forces floor path
        reserved_output_tokens=1000,
        min_ratio=min_ratio,
    )
    assert ew == expected_floor


# ---- min_ratio validation (C1 review fix) ------------------------------- #


@pytest.mark.parametrize("bad_ratio", [-0.1, -1, 1.01, 1.5, 2.0])
def test_effective_window_rejects_invalid_min_ratio(bad_ratio: float) -> None:
    """min_ratio outside [0.0, 1.0] is nonsense and must raise ValueError."""
    with pytest.raises(ValueError, match="min_ratio must be in"):
        compute_effective_window(
            total_context_window=8000,
            system_prompt_max_tokens=3500,
            reserved_output_tokens=1000,
            min_ratio=bad_ratio,
        )


def test_effective_window_accepts_zero_min_ratio() -> None:
    """min_ratio=0.0 is valid (no floor)."""
    ew = compute_effective_window(
        total_context_window=8000,
        system_prompt_max_tokens=3500,
        reserved_output_tokens=1000,
        min_ratio=0.0,
    )
    assert ew == 3500  # raw, no floor


def test_effective_window_accepts_one_min_ratio() -> None:
    """min_ratio=1.0 is valid (floor = total)."""
    ew = compute_effective_window(
        total_context_window=8000,
        system_prompt_max_tokens=3500,
        reserved_output_tokens=1000,
        min_ratio=1.0,
    )
    assert ew == 8000  # floor wins because raw < floor
