"""C5b §8.1/8.2/8.3 — unit tests for the pure command-policy evaluator."""
from __future__ import annotations

from typing import get_args

from app.domain.models.sandbox_policy import sha256_hexdigest
from app.domain.services.safety.command_policy_evaluator import (
    CommandGateDecision,
    _MAX_CWD_DIGEST_CHARS,
    _effective_cwd_digest,
    build_command_policy,
    evaluate_command,
)
from app.domain.services.safety.shell_ast_validator import (
    DENY_VALIDATION_CODES,
    ValidationCode,
)


# ── §8.1 exhaustive oracle (INV-0) ───────────────────────────────────────── #
def test_exhaustive_oracle_allow_iff_ok():
    policy = build_command_policy(effective_cwd="/x", is_default_cwd=False)
    for code in get_args(ValidationCode):
        decision = evaluate_command(validation_code=code, policy=policy)
        assert isinstance(decision, CommandGateDecision)
        assert decision.allowed == (code == "ok"), code
        assert decision.code == code


# ── §8.2 fail-closed / total / bounded (INV-2) ───────────────────────────── #
def test_fail_closed_on_unknown_code():
    policy = build_command_policy(effective_cwd="/x", is_default_cwd=False)
    decision = evaluate_command(validation_code="totally_unknown", policy=policy)
    assert decision.allowed is False
    assert decision.blocked_by_policy is False  # denied because unknown, not deny-set


def test_build_command_policy_shape():
    policy = build_command_policy(effective_cwd="/root", is_default_cwd=True)
    assert policy.validator == "shell_ast_validator"
    assert policy.max_command_bytes == 8192
    assert set(policy.blocked_validation_codes) == set(DENY_VALIDATION_CODES)
    assert len(policy.effective_cwd_digest) == 64
    assert policy.is_default_cwd is True


def test_total_and_bounded_on_pathological_cwd():
    # lone surrogate, non-str int/list, oversized str, oversized non-str collection.
    # None must raise; each yields a 64-char digest promptly (no repr materialization).
    for cwd in ("/tmp/\ud800x", 123, ["x"], "x" * 100_000, list(range(1_000_000))):
        policy = build_command_policy(effective_cwd=cwd, is_default_cwd=False)
        assert len(policy.effective_cwd_digest) == 64


def test_oversized_str_cwd_collides_at_cap():
    assert _effective_cwd_digest("x" * 100_000) == _effective_cwd_digest("x" * _MAX_CWD_DIGEST_CHARS)


def test_non_str_cwd_uses_bounded_type_marker():
    assert _effective_cwd_digest(123) == sha256_hexdigest("<non-str:int>")
    assert _effective_cwd_digest(["x"]) == sha256_hexdigest("<non-str:list>")


def test_digest_matches_c5a_for_short_surrogate_free_str():
    # Byte-identical to the C5a compiler's strict sha256_hexdigest for any real
    # (< cap, surrogate-free) path — so every C5a golden holds after the compiler
    # switches to the shared builder.
    s = "/home/ubuntu/workspace"
    assert _effective_cwd_digest(s) == sha256_hexdigest(s)


# ── §8.3 policy-driven, not allowed-driven (INV-1 unit) ───────────────────── #
def test_policy_driven_decision_ignores_any_result_bool():
    policy = build_command_policy(effective_cwd="/x", is_default_cwd=False)
    deny = evaluate_command(validation_code="fs_destructive", policy=policy)
    allow = evaluate_command(validation_code="ok", policy=policy)
    assert deny.allowed is False and deny.blocked_by_policy is True
    assert allow.allowed is True and allow.blocked_by_policy is False


def test_evaluate_reads_policy_deny_set_not_hardcoded():
    # The decision must CONSUME policy.blocked_validation_codes (the user's "decided via
    # CommandPolicy" requirement) — NOT a hardcoded DENY_VALIDATION_CODES / `code == "ok"`.
    # Vary the deny-set so an impl that ignores the passed policy fails. [codex planR7 P2]
    base = build_command_policy(effective_cwd="/x", is_default_cwd=False)
    custom = base.model_copy(update={"blocked_validation_codes": ("process_control",)})
    # fs_destructive is NOT in the custom deny-set → allowed; process_control IS → denied.
    assert evaluate_command(validation_code="fs_destructive", policy=custom).allowed is True
    assert evaluate_command(validation_code="process_control", policy=custom).allowed is False
    # unknown code is still fail-closed regardless of the policy deny-set.
    assert evaluate_command(validation_code="totally_unknown", policy=custom).allowed is False


def test_non_str_cwd_never_materializes_repr():
    # Prove _effective_cwd_digest uses a TYPE MARKER, never str(obj)/repr(obj): a sentinel whose
    # __str__/__repr__ raise crashes a `str(obj)[:cap]` impl but is fine for the type-marker path
    # (which reads only type(obj).__name__). [codex planR8 P2]
    class _Boom:
        def __str__(self):
            raise AssertionError("must not call str() on non-str cwd")
        def __repr__(self):
            raise AssertionError("must not call repr() on non-str cwd")
    assert _effective_cwd_digest(_Boom()) == sha256_hexdigest("<non-str:_Boom>")


def test_build_command_policy_reads_validator_max_bytes(monkeypatch):
    # The builder must READ the validator constant MAX_COMMAND_BYTES (imported into the evaluator
    # module), not a hardcoded 8192. [codex planR8 P3]
    import app.domain.services.safety.command_policy_evaluator as cpe
    monkeypatch.setattr(cpe, "MAX_COMMAND_BYTES", 12345)
    assert build_command_policy(effective_cwd="/x", is_default_cwd=False).max_command_bytes == 12345
