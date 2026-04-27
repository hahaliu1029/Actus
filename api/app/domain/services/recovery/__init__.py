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

# PR-2: register rules at package import time. Side-effect imports below are
# load-bearing — the wrapper expects RECOVERY_RULES to be populated as soon
# as the package is imported. _validate_rules_unique() runs once after to
# fail-loud on ambiguous rule keys.
from app.domain.services.recovery._rules import register_all as _register_all
_register_all()
from app.domain.services.recovery._registry import _validate_rules_unique as _validate_rules_unique  # noqa: E402
_validate_rules_unique()
