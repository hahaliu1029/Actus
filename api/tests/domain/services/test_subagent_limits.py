"""Unit tests for C1a spawn cap module."""
from __future__ import annotations

import pytest

from app.domain.services.subagent_limits import (
    MAX_DESCENDANTS_PER_ROOT,
    MAX_SUBAGENT_DEPTH,
    SpawnCapExceeded,
)


def test_constants_are_aligned_with_cc_openclaw_hermes_defaults():
    assert MAX_SUBAGENT_DEPTH == 1
    assert MAX_DESCENDANTS_PER_ROOT == 10


def test_spawn_cap_exceeded_holds_kind_current_cap():
    exc = SpawnCapExceeded("depth", current=2, cap=1)
    assert exc.kind == "depth"
    assert exc.current == 2
    assert exc.cap == 1
    assert "depth" in str(exc)


def test_spawn_cap_exceeded_inherits_value_error():
    """Application layer maps to 429/422 - must subclass ValueError so generic
    catch-blocks still see it."""
    with pytest.raises(ValueError):
        raise SpawnCapExceeded("descendants", current=10, cap=10)


def test_spawn_cap_exceeded_kind_must_be_literal():
    """Defensive: only 'depth' and 'descendants' are valid."""
    with pytest.raises(ValueError, match="kind"):
        SpawnCapExceeded("typo", current=0, cap=0)  # type: ignore[arg-type]
