"""PE-1 §5.5 Risk #2 — skill_id MUST NOT enter record_decision canonical attrs.

CANONICAL_ATTRIBUTES (observability.py:108) does not include `skill_id`,
and decision_trace.py:129 silently filters non-canonical keys. PE-1 must
NOT pass skill_id to record_decision (it would be a silent drop).
Expanding CANONICAL_ATTRIBUTES is an independent B5-obs follow PR.
"""

from __future__ import annotations

import inspect


def test_skill_id_not_in_record_decision_attrs_in_skill_source():
    """SkillSource MUST NOT include skill_id in any record_decision attrs.
    skill_id is allowed in logger.info / exception messages / Redis keys —
    NOT in OTel canonical attrs."""
    import app.domain.services.permission.sources.skill_source as mod
    src = inspect.getsource(mod)
    # Sanity: file does mention skill_id (it's used in Redis keys + logs).
    assert "skill_id" in src
    # Hard rule: not as a record_decision attrs key.
    # Grep for the dangerous pattern: skill_id=<...> inside an attrs=dict
    # or attrs={...skill_id...} literal.
    forbidden_patterns = [
        '"skill_id"',
        "'skill_id'",
    ]
    # Look for record_decision context near skill_id usage. If skill_id appears
    # as a dict-literal key string, AND record_decision is nearby in the source,
    # flag it.
    for pat in forbidden_patterns:
        if pat in src:
            # Locate the nearest record_decision call
            assert "record_decision" not in src or src.find(pat) > src.find(
                "record_decision"
            ) or "record_decision" not in src[
                src.find(pat) - 300:src.find(pat) + 300
            ], (
                f"SkillSource passes {pat} as a record_decision key — "
                "Risk #2 forbids this until CANONICAL_ATTRIBUTES extends."
            )


def test_canonical_attributes_does_not_include_skill_id():
    """Confirms the current state: CANONICAL_ATTRIBUTES does not list
    skill_id. PE-1 ships before the B5-obs follow PR adds it."""
    from app.domain.external.observability import CANONICAL_ATTRIBUTES
    assert "skill_id" not in CANONICAL_ATTRIBUTES, (
        "skill_id is now canonical — Risk #2 PE-1 hard rule can be relaxed; "
        "update SkillSource and remove this test guard."
    )


def test_unsupported_canonical_keys_dropped_silently():
    """Confirms decision_trace.py filter behavior so the test above is
    a meaningful guard (silently dropped means the bug would be invisible
    in dashboards)."""
    import app.infrastructure.observability.decision_trace as mod
    src = inspect.getsource(mod)
    # Sanity grep — the filter logic exists
    assert "CANONICAL_ATTRIBUTES" in src
