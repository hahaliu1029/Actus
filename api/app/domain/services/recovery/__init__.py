"""B2 Recovery chain package.

PR-1 version: re-exports core protocol + data types only.
PR-2 appends `from ._rules import register_all; register_all()`.
"""
from app.domain.services.recovery._base import (
    ApiMode,
    OnContextOverflow,
    RecoveryContext,
    RewriteAction,
    RuleKey,
)
from app.domain.services.recovery._registry import (
    RECOVERY_RULES,
    match_rule,
    register_rule,
)

__all__ = [
    "ApiMode",
    "OnContextOverflow",
    "RecoveryContext",
    "RewriteAction",
    "RuleKey",
    "RECOVERY_RULES",
    "match_rule",
    "register_rule",
]
