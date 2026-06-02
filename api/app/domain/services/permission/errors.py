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


class UnsupportedSource(PermissionError):
    """tool_source not registered in PE _sources Mapping.

    Graph path: caller should have gated via is_pe_enabled_for_source() — reaching
    PE here means caller bug; _pe_dispatch catches and emits AllowError + alert.
    HTTP preflight path: mapped to 422 by exception_handlers.py.
    """

    def __init__(self, source: str):
        self.source = source
        super().__init__(f"unsupported tool_source={source!r}")


class PEInfrastructureUnavailable(PermissionError):
    """Redis / queue / writer infrastructure unavailable mid-evaluate.

    Graph path: _pe_dispatch has an explicit catch BEFORE the broad except
    that emits AllowError(content="pe infra unavailable: ...", retryable=True);
    agent retry chain handles transient outages (NOT PolicyConflict — that path
    creates AllowError(retryable=False) at react_graph.py:1630 and would bypass
    retry).
    HTTP preflight path: mapped to 503 by exception_handlers.py.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"pe_infrastructure_unavailable: {reason}")


class PermissionConfigurationError(PermissionError):
    """PE source registry doesn't match PE_SUPPORTED_SOURCES claim.

    Raised at DI / factory time by validate_pe_source_registry() when the
    sources Mapping is missing entries that is_pe_enabled_for_source would
    gate-pass at runtime.

    HARD RULE (spec §3.2 Round 4 P1#2): DI sites MUST re-raise this BEFORE
    any broad ``except Exception`` in build_permission_engine / _create_task,
    otherwise misconfig silently falls back to legacy and defeats the
    registry contract.
    """

    def __init__(self, message: str):
        super().__init__(message)
