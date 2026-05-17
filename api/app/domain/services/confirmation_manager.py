"""Backward-compat shim — class moved to permission/ subpackage in PE-0.

Removed after Phase 9 once all callers have been migrated. INV-1b grep
gate ignores this file."""

from app.domain.services.permission.confirmation_queue import (  # noqa: F401
    _CAS_LUA,
    ConfirmationDetail,
    ConfirmationQueue as ConfirmationManager,
    SWEEP_LOCK_KEY,
    ZSET_KEY,
)
