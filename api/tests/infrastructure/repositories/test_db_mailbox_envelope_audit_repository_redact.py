"""Unit tests for ``_redact`` (no DB).

C3 PR-1 — case-insensitive recursive redaction. C3 PR-1 codex round 3 P2
expanded the exact-match set and added a substring-match list
(``secret`` / ``password`` / ``credentials``) without introducing false
positives on innocuous names like ``token_count``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic_core import to_jsonable_python

from app.infrastructure.repositories.db_mailbox_envelope_audit_repository import (
    _LAST_ERROR_MAX_LEN,
    _LAST_ERROR_TRUNC_MARKER,
    _REDACT_FIELDS,
    _normalize_key,
    _redact,
    _truncate_error_for_audit,
)


def test_redact_masks_known_lowercase_keys() -> None:
    out = _redact({"api_key": "abc", "token": "xyz", "password": "p", "secret": "s"})
    assert out == {
        "api_key": "<redacted>",
        "token": "<redacted>",
        "password": "<redacted>",
        "secret": "<redacted>",
    }


def test_redact_is_case_insensitive() -> None:
    """C3 PR-1 hardening — uppercase / mixed-case keys must also be redacted."""
    out = _redact({"API_KEY": "x", "Token": "y", "PASSWORD": "z", "Secret": "s"})
    assert out["API_KEY"] == "<redacted>"
    assert out["Token"] == "<redacted>"
    assert out["PASSWORD"] == "<redacted>"
    assert out["Secret"] == "<redacted>"


def test_redact_preserves_non_sensitive_keys() -> None:
    out = _redact({"summary": "done", "outcome": "success", "count": 42})
    assert out == {"summary": "done", "outcome": "success", "count": 42}


def test_redact_returns_new_dict() -> None:
    src = {"api_key": "abc", "summary": "ok"}
    out = _redact(src)
    assert out is not src
    # Source dict is untouched.
    assert src == {"api_key": "abc", "summary": "ok"}


def test_redact_fields_is_frozenset() -> None:
    """Defensive: registry must be immutable so callers can't mutate it."""
    assert isinstance(_REDACT_FIELDS, frozenset)


def test_redact_walks_nested_dict_and_redacts_inner_secret() -> None:
    """C3 PR-1 codex round 2 P1 — approval envelopes embed
    ``tool_args_snapshot`` whose inner keys may carry secrets. Top-level-only
    redaction would leak ``api_key`` straight into ``audit_payload``."""
    payload = {"tool_args_snapshot": {"api_key": "sk-abc", "model": "gpt-4"}}
    redacted = _redact(payload)
    assert redacted["tool_args_snapshot"]["api_key"] == "<redacted>"
    assert redacted["tool_args_snapshot"]["model"] == "gpt-4"


def test_redact_walks_nested_list() -> None:
    """List items containing sensitive dict keys must also be scrubbed."""
    payload = {"args_list": [{"token": "t1"}, {"safe": "ok"}]}
    redacted = _redact(payload)
    assert redacted["args_list"][0]["token"] == "<redacted>"
    assert redacted["args_list"][1]["safe"] == "ok"


def test_redact_handles_non_container_values() -> None:
    """Scalars / None must pass through unchanged (no AttributeError)."""
    assert _redact("string") == "string"
    assert _redact(42) == 42
    assert _redact(None) is None


def test_redact_is_case_insensitive_at_any_depth() -> None:
    """Nested upper / mixed-case keys are still in scope of the registry."""
    payload = {"outer": {"API_KEY": "x", "Token": "y"}}
    redacted = _redact(payload)
    assert redacted["outer"]["API_KEY"] == "<redacted>"
    assert redacted["outer"]["Token"] == "<redacted>"


def test_jsonable_payload_converts_datetime() -> None:
    """C3 PR-1 codex round 2 P2 — approval / spawn envelopes embed datetime
    objects in payload. The JSONB encoder used by SQLAlchemy will raise
    TypeError on naked datetime, so the repo runs payloads through
    ``pydantic_core.to_jsonable_python`` first. Lock in that contract here."""
    payload = {"requested_at": datetime(2026, 5, 21, tzinfo=timezone.utc)}
    safe = to_jsonable_python(payload)
    assert isinstance(safe["requested_at"], str)
    assert "2026-05-21" in safe["requested_at"]


# ── C3 PR-1 codex round 3 P2: expanded exact-match + substring redaction ──


def test_redact_handles_camelcase_apikey() -> None:
    """Approval payloads from JS-flavored callers carry ``apiKey``; the
    expanded exact-match set must cover that variant."""
    out = _redact({"apiKey": "sk-abc"})
    assert out["apiKey"] == "<redacted>"


def test_redact_handles_client_secret_substring() -> None:
    """``client_secret`` is a common OAuth field; substring-match on
    ``secret`` must catch it."""
    out = _redact({"client_secret": "shh"})
    assert out["client_secret"] == "<redacted>"


def test_redact_handles_authorization_header_field() -> None:
    """HTTP-style ``Authorization`` header values must be redacted."""
    out = _redact({"Authorization": "Bearer xyz"})
    assert out["Authorization"] == "<redacted>"


def test_redact_handles_aws_secret_access_key() -> None:
    """AWS credential field — covered by both exact-match and
    ``secret`` substring."""
    out = _redact({"aws_secret_access_key": "AKIA..."})
    assert out["aws_secret_access_key"] == "<redacted>"


def test_redact_does_not_match_innocuous_token_count() -> None:
    """``token`` is intentionally NOT in the substring list because it would
    catch operational counters like ``token_count``."""
    out = _redact({"token_count": 42})
    assert out["token_count"] == 42


def test_redact_does_not_match_innocuous_key_value() -> None:
    """``key`` is intentionally NOT in the substring list because it would
    catch innocuous map-entry names like ``key_value``."""
    out = _redact({"key_value": "ok"})
    assert out["key_value"] == "ok"


# ── C3 PR-1 P2 (suffix variants): catch prefixed secret field names ──


def test_redact_handles_openai_api_key() -> None:
    """Provider-prefixed API key fields (e.g. PR-4.5+ approval payloads
    embedding LLM credentials) must hit the ``_key`` suffix rule."""
    out = _redact({"openai_api_key": "sk-abc"})
    assert out["openai_api_key"] == "<redacted>"


def test_redact_handles_github_token() -> None:
    """Provider-prefixed token fields must hit the ``_token`` suffix rule."""
    out = _redact({"github_token": "ghp_xxx"})
    assert out["github_token"] == "<redacted>"


def test_redact_handles_ssh_private_key() -> None:
    """Compound key field names (e.g. ``ssh_private_key``) must hit the
    ``_key`` suffix rule even when the existing ``private_key`` exact-match
    doesn't apply."""
    out = _redact({"ssh_private_key": "-----BEGIN..."})
    assert out["ssh_private_key"] == "<redacted>"


def test_redact_handles_azure_client_secret() -> None:
    """Provider-prefixed secret fields are covered by both the ``secret``
    substring and the ``_secret`` suffix — either rule must fire."""
    out = _redact({"azure_client_secret": "xxx"})
    assert out["azure_client_secret"] == "<redacted>"


def test_redact_does_not_match_tokens_used_or_token_count() -> None:
    """Operational counters must not be redacted: ``tokens_used`` ends with
    ``_used`` (not ``_token``) and ``token_count`` ends with ``_count``."""
    out = _redact({"tokens_used": 100, "token_count": 50})
    assert out["tokens_used"] == 100
    assert out["token_count"] == 50


def test_redact_does_not_match_keyboard_or_key_value() -> None:
    """``keyboard`` has no underscore separator (so no ``_key`` suffix) and
    ``key_value`` ends with ``_value`` — both must pass through unredacted."""
    out = _redact({"keyboard": "qwerty", "key_value": "ok"})
    assert out["keyboard"] == "qwerty"
    assert out["key_value"] == "ok"


def test_redact_handles_camelcase_oauth_field() -> None:
    """CamelCase ``oauthToken`` lowercases to ``oauthtoken`` which has no
    underscore boundary, so the suffix rule misses it. Covered by an
    explicit entry in the exact-match set."""
    out = _redact({"oauthToken": "ya29..."})
    assert out["oauthToken"] == "<redacted>"


# ── C3 PR-1 codex round 3 P2: last_error truncation helper ──


def test_truncate_error_passes_through_short_string() -> None:
    """Short tracebacks fit the column and must pass through unchanged."""
    s = "boom: traceback line 1"
    assert _truncate_error_for_audit(s) == s


def test_truncate_error_truncates_long_string_with_marker() -> None:
    """Long tracebacks must be clipped to _LAST_ERROR_MAX_LEN + marker so the
    audit row write does not raise DataError on the String(2048) column."""
    long_error = "x" * (_LAST_ERROR_MAX_LEN + 500)
    truncated = _truncate_error_for_audit(long_error)
    assert truncated.endswith(_LAST_ERROR_TRUNC_MARKER)
    assert len(truncated) == _LAST_ERROR_MAX_LEN + len(_LAST_ERROR_TRUNC_MARKER)
    # And the result fits the column (2048 chars).
    assert len(truncated) <= 2048


# ── C3 PR-1 P1 (codex round 4): camelCase secret keys must redact ──
#
# Before the ``_normalize_key`` helper, JS/TS-flavored callers sending
# ``accessToken`` / ``refreshToken`` / ``privateKey`` / ``apiKey`` /
# ``clientSecret`` slipped past redaction because ``key.lower()`` collapsed
# them to single-token lowercase strings that miss BOTH the underscore-form
# exact-match set AND the underscore-boundary suffix rule.


def test_redact_handles_camelcase_access_token() -> None:
    """``accessToken`` normalizes to ``access_token`` (in exact-match)."""
    out = _redact({"accessToken": "x"})
    assert out["accessToken"] == "<redacted>"


def test_redact_handles_camelcase_refresh_token() -> None:
    """``refreshToken`` normalizes to ``refresh_token`` (in exact-match)."""
    out = _redact({"refreshToken": "y"})
    assert out["refreshToken"] == "<redacted>"


def test_redact_handles_camelcase_private_key() -> None:
    """``privateKey`` normalizes to ``private_key`` (in exact-match)."""
    out = _redact({"privateKey": "z"})
    assert out["privateKey"] == "<redacted>"


def test_redact_handles_camelcase_client_secret() -> None:
    """``clientSecret`` normalizes to ``client_secret`` (in exact-match and
    also caught by the ``secret`` substring rule)."""
    out = _redact({"clientSecret": "shh"})
    assert out["clientSecret"] == "<redacted>"


def test_redact_handles_pascalcase_oauthtoken() -> None:
    """``OAuthToken`` normalizes to ``oauth_token`` which ends with
    ``_token`` and so hits the suffix rule."""
    out = _redact({"OAuthToken": "ya29..."})
    assert out["OAuthToken"] == "<redacted>"


def test_normalize_key_examples() -> None:
    """Sanity-check the normalization contract that downstream rules rely on."""
    assert _normalize_key("apiKey") == "api_key"
    assert _normalize_key("alreadysnake") == "alreadysnake"
    assert _normalize_key("API_KEY") == "api_key"
    # Additional invariants the redact rules depend on.
    assert _normalize_key("accessToken") == "access_token"
    # C3 PR-1 P1 staged-diff follow-up: the two-pass acronym-aware regex
    # produces ``o_auth_token`` (initial single-capital ``O`` is treated as a
    # 1-char acronym before ``Auth``). Still ends with ``_token`` so the
    # suffix rule continues to fire for redaction — see
    # ``test_redact_handles_pascalcase_oauthtoken``.
    assert _normalize_key("OAuthToken") == "o_auth_token"


# ── C3 PR-1 P1 staged-diff follow-up: acronym-style keys (two-pass regex) ──
#
# The original single-pass ``(?<=[a-z0-9])(?=[A-Z])`` boundary regex only
# inserted ``_`` at lowercase→uppercase transitions, so acronym-led keys like
# ``openaiAPIKey`` collapsed to ``openai_apikey`` (no ``_key`` boundary) and
# ``AWSAccessKeyId`` collapsed to ``awsaccess_key_id`` (``_id`` not a redact
# suffix). The two-pass acronym-aware regex resolves both: acronyms get
# split off cleanly and downstream suffix rules (``_key`` / ``_token`` /
# ``_secret``) fire as intended.


def test_redact_handles_openai_api_key_acronym() -> None:
    """``openaiAPIKey`` (provider-prefixed acronym key) must redact: the
    two-pass regex normalizes to ``openai_api_key`` so ``_key`` suffix fires."""
    out = _redact({"openaiAPIKey": "sk-abc"})
    assert out["openaiAPIKey"] == "<redacted>"


def test_redact_handles_aws_access_key_id_acronym() -> None:
    """``AWSAccessKeyId`` (AWS-style PascalCase acronym key) must redact:
    normalizes to ``aws_access_key_id`` so ``_key_id`` ends with... wait,
    the suffix rule needs ``_key`` at the END. Trace: ``aws_access_key_id``
    ends with ``_id`` (not a redact suffix). BUT the exact-match set already
    contains ``aws_access_key_id`` (added in PR-1), so this redacts via the
    exact-match path."""
    out = _redact({"AWSAccessKeyId": "AKIA..."})
    assert out["AWSAccessKeyId"] == "<redacted>"


def test_redact_handles_db_secret_acronym() -> None:
    """``DBSecret`` (2-letter acronym + word) must redact: normalizes to
    ``db_secret`` which ends with ``_secret`` (suffix rule)."""
    out = _redact({"DBSecret": "shh"})
    assert out["DBSecret"] == "<redacted>"


def test_redact_handles_ssh_private_key_acronym() -> None:
    """``SSHPrivateKey`` (3-letter acronym + compound word) must redact:
    normalizes to ``ssh_private_key`` which ends with ``_key`` (suffix rule)."""
    out = _redact({"SSHPrivateKey": "-----BEGIN..."})
    assert out["SSHPrivateKey"] == "<redacted>"


def test_normalize_key_acronym_examples() -> None:
    """Lock in the acronym-aware normalization contract."""
    assert _normalize_key("openaiAPIKey") == "openai_api_key"
    assert _normalize_key("AWSAccessKeyId") == "aws_access_key_id"
    assert _normalize_key("SSHPrivateKey") == "ssh_private_key"
    assert _normalize_key("apiKey") == "api_key"  # still works for simple case
    assert _normalize_key("DBSecret") == "db_secret"


# ── C3 PR-1 codex round 12 P2: Authorization header variants ──
#
# Before adding ``authorization`` / ``bearer`` to ``_REDACT_SUBSTRINGS``, keys
# like ``authorizationHeader`` (→ ``authorization_header``), ``xAuthorization``
# (→ ``x_authorization``), ``authorizationBearer`` (→ ``authorization_bearer``)
# and ``bearerHeader`` (→ ``bearer_header``) bypassed redaction: not in the
# exact-match set, not in the previous substring set, and ending with
# ``_header`` / no underscore boundary so the suffix rule missed them too.


def test_redact_handles_authorization_header_variant() -> None:
    """``authorizationHeader`` normalizes to ``authorization_header`` and must
    hit the ``authorization`` substring rule."""
    assert _redact({"authorizationHeader": "Bearer xyz"})["authorizationHeader"] == "<redacted>"


def test_redact_handles_authorization_bearer_variant() -> None:
    """``authorization_bearer`` must hit the ``authorization`` substring rule
    (also ``bearer`` substring — both rules fire)."""
    assert _redact({"authorization_bearer": "xyz"})["authorization_bearer"] == "<redacted>"


def test_redact_handles_x_authorization() -> None:
    """``xAuthorization`` normalizes to ``x_authorization`` and must hit the
    ``authorization`` substring rule."""
    assert _redact({"xAuthorization": "xyz"})["xAuthorization"] == "<redacted>"


def test_redact_handles_bearer_header() -> None:
    """``bearerHeader`` normalizes to ``bearer_header`` and must hit the
    ``bearer`` substring rule (``_header`` is not a redact suffix)."""
    assert _redact({"bearerHeader": "xyz"})["bearerHeader"] == "<redacted>"


def test_redact_does_not_match_innocuous_bear_or_authority() -> None:
    """``bear`` and ``authority`` are distinct words from ``bearer`` and
    ``authorization`` — neither must trigger redaction (false-positive check
    for the broadened substring rule)."""
    assert _redact({"bear_count": 5, "authority_level": "admin"}) == {
        "bear_count": 5,
        "authority_level": "admin",
    }


# ── C3 PR-1 codex round 14 P1: kebab-case HTTP-header-style keys ──
#
# Before ``_normalize_key`` replaced ``-`` with ``_``, kebab keys like
# ``openai-api-key`` / ``github-token`` / ``ssh-private-key`` / ``client-secret``
# (common in HTTP headers and JS SDK payloads) bypassed redaction: lowercase
# was a no-op (already lowercase), they were absent from the exact-match set,
# and the suffix rule (``_key`` / ``_token`` / ``_secret``) requires an
# underscore anchor that hyphens don't satisfy. Substring rule ``secret``
# happened to catch ``client-secret`` only because ``secret`` is a substring,
# but the more dangerous ``openai-api-key`` / ``github-token`` / ``ssh-private-key``
# fell through entirely.


def test_redact_handles_kebab_case_openai_api_key() -> None:
    """``openai-api-key`` (Authorization-style header from JS SDKs) normalizes
    to ``openai_api_key`` and must hit the ``_key`` suffix rule."""
    out = _redact({"openai-api-key": "sk-abc"})
    assert out["openai-api-key"] == "<redacted>"


def test_redact_handles_kebab_case_github_token() -> None:
    """``github-token`` (GitHub Actions style) normalizes to ``github_token``
    and must hit the ``_token`` suffix rule."""
    out = _redact({"github-token": "ghp_xxx"})
    assert out["github-token"] == "<redacted>"


def test_redact_handles_kebab_case_ssh_private_key() -> None:
    """``ssh-private-key`` normalizes to ``ssh_private_key`` and must hit
    the ``_key`` suffix rule."""
    out = _redact({"ssh-private-key": "-----BEGIN..."})
    assert out["ssh-private-key"] == "<redacted>"


def test_redact_handles_kebab_case_client_secret() -> None:
    """``client-secret`` (OAuth-style) normalizes to ``client_secret`` and
    is covered by both the ``secret`` substring rule and the ``_secret``
    suffix rule — either fires."""
    out = _redact({"client-secret": "shh"})
    assert out["client-secret"] == "<redacted>"


def test_normalize_key_kebab_examples() -> None:
    """Lock in the kebab-case → snake_case normalization contract that the
    downstream exact-match / substring / suffix rules rely on."""
    assert _normalize_key("openai-api-key") == "openai_api_key"
    assert _normalize_key("github-token") == "github_token"
    assert _normalize_key("ssh-private-key") == "ssh_private_key"
    assert _normalize_key("client-secret") == "client_secret"
    # Authorization-style kebab (common HTTP header convention).
    assert _normalize_key("x-api-key") == "x_api_key"


def test_redact_does_not_match_innocuous_kebab() -> None:
    """``x-request-id`` normalizes to ``x_request_id``. No rule matches:
    not in exact set, no secret/password/credentials/authorization/bearer
    substring, suffix ``_id`` is not a redact suffix. False-positive check
    for the broadened kebab-normalization."""
    out = _redact({"x-request-id": "abc-123"})
    assert out["x-request-id"] == "abc-123"
