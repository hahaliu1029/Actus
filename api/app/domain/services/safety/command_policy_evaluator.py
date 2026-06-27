"""C5b Command Policy Evaluator — pure decision over a compiled CommandPolicy.

PURE DOMAIN: imports only app.domain.models.sandbox_policy + validator constants
+ stdlib. Never calls get_settings(); no FastAPI/SQLAlchemy/infrastructure.
Total: never raises; fail-closed on unknown codes (INV-2).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import get_args

from app.domain.models.sandbox_policy import CommandPolicy
from app.domain.services.safety.shell_ast_validator import (
    DENY_VALIDATION_CODES,
    MAX_COMMAND_BYTES,
    ValidationCode,
)

_KNOWN_VALIDATION_CODES: frozenset[str] = frozenset(get_args(ValidationCode))


# Bound the hot-path digest work: `exec_dir` is un-capped agent input (the shell tool
# declares `exec_dir: str = ""` with no max; the validator caps only `command` bytes).
# C5b hashes cwd on EVERY shell call, so we cap the digest INPUT (R5#P2). 8192 ≫ PATH_MAX
# (4096) → every real path is hashed in full (byte-identical to today); only a pathological
# multi-KB cwd is truncated (audit-only — the decision never reads the digest).
_MAX_CWD_DIGEST_CHARS = 8192


def _effective_cwd_digest(effective_cwd: object) -> str:
    """sha256 hex — TOTAL + BOUNDED over EVERY value the gates can pass (R1#P2 + R2#P2-1
    + R5#P2 + R6#P2). `effective_cwd` arrives as un-schema-validated tool args
    (`tc["args"]["exec_dir"]`); `validate()` only ANNOTATES `str` and a benign command
    echoes it back (shell_ast_validator.py:1638), so it can be any JSON type. We branch:
    a `str` is sliced to `_MAX_CWD_DIGEST_CHARS` (bounds encode/hash work; for `< cap` —
    all real paths — byte-identical to sandbox_policy.sha256_hexdigest, venv-verified, so
    all C5a goldens hold) and encoded `surrogatepass` (total over lone surrogates); an
    off-schema NON-`str` is digested as a bounded `<non-str:TYPE>` marker — we NEVER
    materialize a huge `str(collection)` repr (R6#P2). AUDIT-ONLY: the decision
    (evaluate_command) reads only the deny-set, never this value — so no input can alter
    the allow/deny outcome, raise, or do unbounded work on the always-on path (INV-0/INV-2)."""
    if isinstance(effective_cwd, str):
        material = effective_cwd[:_MAX_CWD_DIGEST_CHARS]
    else:
        material = f"<non-str:{type(effective_cwd).__name__}>"
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()


def build_command_policy(*, effective_cwd: object, is_default_cwd: bool) -> CommandPolicy:
    """The SINGLE CommandPolicy constructor — used by both the compiler (observe)
    and the gates (enforce) so the observed decision/rule-set == the enforced one (the
    per-call cwd digest is audit-only; INV-3). Settings-free and TOTAL + BOUNDED (never
    raises / no unbounded work) for ANY `effective_cwd` value — `_effective_cwd_digest`
    (above) handles str (sliced) and off-schema non-str (bounded type marker); every other
    field is a validator constant. (`effective_cwd: object` because the gates pass raw,
    un-schema-validated `tc["args"]["exec_dir"]`; R7#P3.)"""
    return CommandPolicy(
        validator="shell_ast_validator",
        max_command_bytes=MAX_COMMAND_BYTES,
        blocked_validation_codes=tuple(DENY_VALIDATION_CODES),  # coerced; field is tuple[str,...]
        effective_cwd_digest=_effective_cwd_digest(effective_cwd),
        is_default_cwd=is_default_cwd,
    )


@dataclass(frozen=True)
class CommandGateDecision:
    allowed: bool
    code: str
    blocked_by_policy: bool   # True ⇔ code ∈ policy.blocked_validation_codes


def evaluate_command(*, validation_code: str, policy: CommandPolicy) -> CommandGateDecision:
    """Decide allow/deny for a validated shell command, policy-driven.

    allowed ⇔ code is a KNOWN code AND not in the policy deny-set.
    FAIL-CLOSED (INV-2): an unknown / unrecognized code is denied. Never raises.
    """
    blocked = validation_code in policy.blocked_validation_codes
    unknown = validation_code not in _KNOWN_VALIDATION_CODES
    return CommandGateDecision(
        allowed=not (blocked or unknown),
        code=validation_code,
        blocked_by_policy=blocked,
    )
