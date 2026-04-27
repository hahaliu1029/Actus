"""Recovery rule registry.

Not thread-safe mutation — rules register at import time during app startup.
Read path (match_rule) is pure.
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.recovery._base import RewriteAction, RuleKey


class SpecError(RuntimeError):
    """Raised when rule table is inconsistent."""


RECOVERY_RULES: dict[RuleKey, tuple[RewriteAction, ...]] = {}


def register_rule(key: RuleKey, actions: tuple[RewriteAction, ...]) -> None:
    _profile_id, _api_mode, error_class, _fp = key
    if not isinstance(error_class, ErrorClass):
        raise SpecError(f"rule key error_class must be ErrorClass, got {error_class!r}")

    existing = RECOVERY_RULES.get(key)
    if existing is not None and existing != actions:
        raise SpecError(
            f"Recovery rule conflict for key={key}: "
            f"existing={[a.code for a in existing]!r}, new={[a.code for a in actions]!r}"
        )
    RECOVERY_RULES[key] = actions


def _key_priority(key: RuleKey) -> int:
    return sum(1 for v in (key[0], key[1], key[3]) if v == "*")


def _key_matches(key: RuleKey, query: RuleKey) -> bool:
    for k, q in zip(key, query):
        if k == "*":
            continue
        if k != q:
            return False
    return True


def match_rule(
    profile_id: str,
    api_mode: str,
    error_class: ErrorClass,
    fingerprint_code: str | None,
) -> tuple[RewriteAction, ...] | None:
    query_fp = fingerprint_code if fingerprint_code is not None else "*"
    query: RuleKey = (profile_id, api_mode, error_class, query_fp)

    candidates: list[tuple[int, RuleKey, tuple[RewriteAction, ...]]] = []
    for key, actions in RECOVERY_RULES.items():
        if key[2] != error_class:
            continue
        if not _key_matches(key, query):
            continue
        candidates.append((_key_priority(key), key, actions))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][2]


def _validate_rules_unique() -> None:
    by_priority: dict[int, list[RuleKey]] = {}
    for key in RECOVERY_RULES:
        by_priority.setdefault(_key_priority(key), []).append(key)
    for priority, keys in by_priority.items():
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                if _keys_may_both_match(k1, k2):
                    raise SpecError(
                        f"Ambiguous recovery rules at priority {priority}: "
                        f"{k1} and {k2} can both match some input"
                    )


def _keys_may_both_match(k1: RuleKey, k2: RuleKey) -> bool:
    if k1[2] != k2[2]:
        return False
    for a, b in zip((k1[0], k1[1], k1[3]), (k2[0], k2[1], k2[3])):
        if a == "*" or b == "*":
            continue
        if a != b:
            return False
    return True
