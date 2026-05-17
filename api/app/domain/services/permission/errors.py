"""Exception hierarchy for PermissionEngine.

HTTP mapping (interfaces layer):
- PolicyConflict           → 409 (race / dedup / arg_digest mismatch)
- SessionModeViolation     → 410 (lifecycle terminal state)
- WriterIntegrityError     → 500 + correlation_id
- EscalationProviderError  → internal, captured + converted to Asked outcome
"""


class PermissionError(Exception):
    """All PE exceptions inherit from this.

    Note: This name intentionally shadows the builtin ``PermissionError``
    (POSIX EACCES) within this module. Outside callers always import this
    explicitly: ``from app.domain.services.permission.errors import
    PermissionError`` — so there is no runtime ambiguity. If you need the
    OS error type in a catch site, alias it: ``import builtins; except
    builtins.PermissionError``.
    """


class PolicyConflict(PermissionError):
    """ConfirmationQueue claim race / arg_digest mismatch / writer UNIQUE.

    Codes (string body): approval_already_claimed | arg_digest_mismatch |
    no_pending_confirmation | session_mode_changed_during_evaluate |
    claim_nonce_mismatch
    """


class SessionModeViolation(PermissionError):
    """PE saw a session_mode that does not permit evaluate / resume."""


class WriterIntegrityError(PermissionError):
    """ApprovalStateWriter downstream IntegrityError pass-through."""


class EscalationProviderError(PermissionError):
    """Provider raised; PE captures + converts to Asked (fall-through)."""


class EscalationTimeout(EscalationProviderError):
    pass


class EscalationUnavailable(EscalationProviderError):
    pass
