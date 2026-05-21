"""DB-backed MailboxEnvelopeAuditRepository (C3 spec §5.8).

Consumer-side terminal-dedup authority. PK (parent_session_id, envelope_id)
+ ``processed_at`` distinguishes inflight processing rows from final acks.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic_core import to_jsonable_python
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.models.mailbox_envelope import MailboxEnvelope
from app.domain.repositories.mailbox_envelope_audit_repository import (
    MailboxEnvelopeAuditRepository,
)
from app.infrastructure.models.mailbox_envelope_audit import MailboxEnvelopeAuditModel


class DbMailboxEnvelopeAuditRepository(MailboxEnvelopeAuditRepository):
    """DB-backed audit repo with PG UPSERT semantics.

    **Why this repo is NOT registered with DBUnitOfWork**: MailboxSupervisor
    runs as a background asyncio task outside any HTTP request's UoW span.
    Each call here uses its own short-lived session via ``session_factory``;
    the supervisor manages its own commit boundaries per-envelope.

    **Concurrency assumption — C3 PR-1 baseline**:
    ``mark_processed`` and ``increment_reclaim`` use the
    ``session.get() → mutate → commit`` pattern, which is NOT atomic against
    concurrent writers on the same ``(parent_session_id, envelope_id)`` row.
    This is acceptable in PR-1 because the C3 design pins a single supervisor
    consumer per ``actus:mailbox-supervisor:v1`` consumer group, so each audit
    row only ever has one in-flight mutator at a time.

    **PR-3b follow-up (XAUTOCLAIM PEL takeover)**: once dead-consumer PEL
    reclaim lands, multiple supervisor instances can race on the same row.
    Both mutators below MUST then switch to either:
      - ``SELECT ... FOR UPDATE`` row locking inside the session, or
      - a single-statement ``UPDATE ... RETURNING reclaim_count`` /
        ``UPDATE ... WHERE processed_at IS NULL`` so the DB enforces atomicity.
    Until then the single-consumer invariant carries the safety.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_processed(self, parent_session_id: str, envelope_id: str) -> bool:
        async with self._session_factory() as session:
            stmt = (
                select(MailboxEnvelopeAuditModel.processed_at)
                .where(MailboxEnvelopeAuditModel.parent_session_id == parent_session_id)
                .where(MailboxEnvelopeAuditModel.envelope_id == envelope_id)
            )
            row = (await session.execute(stmt)).first()
            return row is not None and row[0] is not None

    async def upsert_processing(
        self, envelope: MailboxEnvelope, *, processing_at: datetime
    ) -> None:
        """Insert row marking envelope as inflight; ON CONFLICT update processing_at only.

        Notes:
        - ``received_at`` uses table server_default NOW() on insert and is NOT
          touched on update (spec §5.8 — preserve first-seen timestamp).
        - ``processed_at`` is NOT touched here; ``mark_processed`` is the only
          path that sets it.
        """
        # C3 PR-1 (codex round 2 P2): JSONB column must receive JSON-safe
        # primitives. Approval / spawn envelopes embed datetime objects
        # (requested_at / expires_at / sandbox_ready_at) in payload — the
        # default psycopg JSON encoder raises TypeError on those. Convert
        # via pydantic_core.to_jsonable_python which handles datetime,
        # UUID, Decimal, and nested pydantic models uniformly.
        redacted_payload = _redact(envelope.payload)
        json_safe_payload = to_jsonable_python(redacted_payload)
        async with self._session_factory() as session:
            stmt = pg_insert(MailboxEnvelopeAuditModel).values(
                parent_session_id=envelope.parent_session_id,
                envelope_id=envelope.envelope_id,
                child_session_id=envelope.child_session_id,
                type=envelope.type.value,
                producer_role=envelope.producer_role.value,
                correlation_id=envelope.correlation_id,
                processing_at=processing_at,
                audit_payload=json_safe_payload,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["parent_session_id", "envelope_id"],
                set_={"processing_at": processing_at},
            )
            await session.execute(stmt)
            await session.commit()

    # TODO(C3 PR-3b): under XAUTOCLAIM PEL takeover this get+mutate+commit
    # pattern races against the takeover consumer. Switch to
    # ``SELECT ... FOR UPDATE`` or a single ``UPDATE ... RETURNING`` once
    # multi-consumer reclaim lands. Safe today only because the C3 design
    # pins one consumer per ``MAILBOX_CONSUMER_GROUP_NAME``.
    async def mark_processed(
        self,
        parent_session_id: str,
        envelope_id: str,
        *,
        processed_at: datetime,
    ) -> None:
        async with self._session_factory() as session:
            row = await session.get(
                MailboxEnvelopeAuditModel,
                {"parent_session_id": parent_session_id, "envelope_id": envelope_id},
            )
            if row is None:
                raise ValueError(
                    f"mark_processed on missing audit row "
                    f"({parent_session_id}, {envelope_id})"
                )
            row.processed_at = processed_at
            await session.commit()

    # TODO(C3 PR-3b): same race as ``mark_processed`` above. With multiple
    # supervisors active, two concurrent increments would race and lose updates.
    # Replace with a single-statement
    # ``UPDATE ... SET reclaim_count = reclaim_count + 1 ... RETURNING reclaim_count``
    # once PR-3b XAUTOCLAIM takeover lands.
    async def increment_reclaim(
        self,
        parent_session_id: str,
        envelope_id: str,
        last_error: str,
    ) -> int:
        async with self._session_factory() as session:
            row = await session.get(
                MailboxEnvelopeAuditModel,
                {"parent_session_id": parent_session_id, "envelope_id": envelope_id},
            )
            if row is None:
                raise ValueError(
                    f"increment_reclaim on missing audit row "
                    f"({parent_session_id}, {envelope_id})"
                )
            # C3 PR-1 (codex P2): capture the new count in a local before commit
            # — production session_factory uses SQLAlchemy default
            # ``expire_on_commit=True``, which expires ORM attributes after the
            # commit() below. Reading ``row.reclaim_count`` post-commit would
            # trigger an async lazy-load and raise MissingGreenlet outside the
            # session's greenlet context.
            new_count = row.reclaim_count + 1
            row.reclaim_count = new_count
            row.last_error = _truncate_error_for_audit(last_error)
            await session.commit()
            return new_count

    async def fetch_raw(
        self, parent_session_id: str, envelope_id: str
    ) -> dict[str, Any]:
        async with self._session_factory() as session:
            row = await session.get(
                MailboxEnvelopeAuditModel,
                {"parent_session_id": parent_session_id, "envelope_id": envelope_id},
            )
            if row is None:
                return {}
            return {
                "parent_session_id": row.parent_session_id,
                "envelope_id": row.envelope_id,
                "child_session_id": row.child_session_id,
                "type": row.type,
                "producer_role": row.producer_role,
                "correlation_id": row.correlation_id,
                "received_at": row.received_at,
                "processing_at": row.processing_at,
                "processed_at": row.processed_at,
                "audit_payload": row.audit_payload,
                "reclaim_count": row.reclaim_count,
                "last_error": row.last_error,
            }


# C3 PR-1 — case-insensitive recursive redaction for nested dict / list payloads.
# Exact-match set is the high-confidence allowlist; substring set catches
# *secret* / *password* / *credentials* / *authorization* / *bearer* variants
# without false-positives on innocuous names like `token_count` or `key_value`;
# suffix set catches prefixed variants like `openai_api_key`, `github_token`,
# `ssh_private_key`, `service_account_key`, `azure_client_secret`. The leading
# underscore on every suffix prevents false positives: `token_count` ends with
# `_count`, not `_token`; `key_value` ends with `_value`, not `_key`;
# `keyboard` has no underscore separator at all.
# TODO(C3 PR-4.5): replace with redaction.py module + per-tool registry pattern
# so each tool can declare its own redact_args explicitly.
_REDACT_FIELDS: frozenset[str] = frozenset({
    # Direct + common variants (compared lowercase)
    "api_key", "apikey", "api-key", "x-api-key", "x_api_key",
    "secret", "secret_key", "client_secret", "secret_access_key",
    "aws_access_key_id", "aws_secret_access_key",
    "token", "access_token", "refresh_token", "auth_token",
    "bearer_token", "session_token",
    "password", "db_password", "database_password",
    "authorization", "credentials", "private_key",
    # CamelCase / no-separator variants that the suffix rule can't catch
    # because they have no underscore boundary (`oauthtoken` -> no `_token`).
    "oauthtoken",
})

# Substring match (case-insensitive). High precision for tokens that ALWAYS
# indicate secrets even mid-word.
#
# C3 PR-1 codex round 12 P2: ``authorization`` and ``bearer`` added so that
# header-style variants (``authorizationHeader`` → ``authorization_header``,
# ``xAuthorization`` → ``x_authorization``, ``authorizationBearer`` →
# ``authorization_bearer``, ``bearerHeader`` → ``bearer_header``) — common in
# tool args coming from JS/TS MCP servers and OAuth libs — fall through both
# the exact-match set (only bare ``authorization`` / ``bearer_token``) AND the
# ``_key`` / ``_token`` / ``_secret`` suffix rules. The substring widening is
# safe because no innocuous identifier embeds ``authorization`` or ``bearer``
# as a mid-word fragment (``bear`` and ``authority`` are distinct words).
_REDACT_SUBSTRINGS: tuple[str, ...] = (
    "password", "secret", "credentials", "authorization", "bearer",
)

# Suffix match (case-insensitive). Catches prefixed variants like
# `openai_api_key`, `github_token`, `ssh_private_key`, `service_account_key`,
# `azure_client_secret`. The leading underscore prevents false positives:
# `token_count` ends with `_count`, not `_token`; `key_value` ends with
# `_value`, not `_key`; `tokens_used` ends with `_used`, not `_token`.
_REDACT_SUFFIXES: tuple[str, ...] = (
    "_key", "_token", "_secret", "_password", "_credentials",
)


# C3 PR-1 P1 (codex round 4): camelCase callers (JS/TS-flavored MCP servers,
# OAuth libs, browser tools) frequently send keys like ``accessToken`` /
# ``refreshToken`` / ``privateKey`` / ``apiKey`` / ``clientSecret``. Naive
# ``key.lower()`` collapses these to ``accesstoken`` / ``apikey`` etc., which
# the underscore-boundary suffix rule (``_token`` / ``_key``) can't see and the
# underscore-form exact-match set (``access_token`` / ``private_key``) doesn't
# contain. Normalize camelCase → snake_case BEFORE matching so the same key
# hits the same rule regardless of casing convention.
#
# C3 PR-1 P1 (staged-diff follow-up): the single-pass lowercase→uppercase
# boundary regex missed acronym-style keys like ``openaiAPIKey`` (collapsed to
# ``openai_apikey`` — no ``_key`` suffix boundary) and ``AWSAccessKeyId``
# (collapsed to ``awsaccess_key_id`` — ``_id`` not a redact suffix). Adopt the
# canonical Python two-pass camelCase→snake_case idiom which handles acronyms
# cleanly by inserting underscores both at acronym-then-Word boundaries and at
# lowercase/digit-then-uppercase boundaries.
# Pass 1: insert _ before a capital that starts a new word AFTER an acronym
#         (e.g. ``openaiAPIKey`` → ``openaiAPI_Key`` here, pass 2 finishes it).
_CAMEL_SPLIT_ACRONYM_RE = re.compile(r"(.)([A-Z][a-z]+)")
# Pass 2: insert _ between lowercase/digit and uppercase
#         (e.g. ``openaiAPI_Key`` → ``openai_API_Key``).
_CAMEL_SPLIT_LOWER_RE = re.compile(r"([a-z0-9])([A-Z])")


def _normalize_key(key: str) -> str:
    """Convert camelCase/PascalCase/kebab-case → snake_case + lowercase.

    Two-pass camelCase split handles acronyms cleanly:
        ``"apiKey"``          → ``"api_key"``
        ``"accessToken"``     → ``"access_token"``
        ``"AWSAccessKeyId"``  → ``"aws_access_key_id"``
        ``"openaiAPIKey"``    → ``"openai_api_key"``
        ``"OAuthToken"``      → ``"o_auth_token"``
        ``"SSHPrivateKey"``   → ``"ssh_private_key"``
        ``"alreadysnake"``    → ``"alreadysnake"``
        ``"API_KEY"``         → ``"api_key"``  (lowercase passthrough)
        ``"openai-api-key"``  → ``"openai_api_key"``  (kebab → snake)
        ``"client-secret"``   → ``"client_secret"``
    """
    s = _CAMEL_SPLIT_ACRONYM_RE.sub(r"\1_\2", key)
    s = _CAMEL_SPLIT_LOWER_RE.sub(r"\1_\2", s)
    # C3 PR-1 (codex round 14 P1): normalize kebab-case separators so
    # HTTP-header-style keys (``openai-api-key`` / ``github-token`` /
    # ``ssh-private-key`` / ``client-secret``) coming from JS SDKs hit the
    # same exact/suffix rules as snake_case. Without this, kebab keys
    # lowercase to themselves and bypass both the exact-match set (only
    # snake_case variants) AND the underscore-anchored suffix rules.
    return s.lower().replace("-", "_")


def _should_redact_key(key: str) -> bool:
    k = _normalize_key(key)
    if k in _REDACT_FIELDS:
        return True
    if any(sub in k for sub in _REDACT_SUBSTRINGS):
        return True
    return any(k.endswith(suffix) for suffix in _REDACT_SUFFIXES)


def _redact(payload: Any) -> Any:
    """Walk dicts/lists and replace values for sensitive keys (case-insensitive).

    Non-dict/list values pass through unchanged. The redacted output is shape-
    preserving so downstream JSONB read paths see the same key structure.
    """
    if isinstance(payload, dict):
        return {
            k: ("<redacted>" if _should_redact_key(k) else _redact(v))
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return [_redact(item) for item in payload]
    return payload


# C3 PR-1 (codex round 3 P2): ``last_error`` column is ``String(2048)``; Python
# tracebacks routinely exceed this and writing them raises
# ``DataError: value too long for type character varying(2048)``. Truncate at
# the write site (no migration needed). Keep a 48-char headroom for the
# ``...[truncated]`` marker so consumers can tell the string was clipped.
_LAST_ERROR_MAX_LEN: int = 2000
_LAST_ERROR_TRUNC_MARKER: str = "...[truncated]"


def _truncate_error_for_audit(s: str) -> str:
    """Truncate a last_error string so it fits the audit column.

    Short strings pass through unchanged; long strings are clipped to
    ``_LAST_ERROR_MAX_LEN`` and suffixed with ``_LAST_ERROR_TRUNC_MARKER`` so
    operators can identify clipped entries.
    """
    if len(s) <= _LAST_ERROR_MAX_LEN:
        return s
    return s[:_LAST_ERROR_MAX_LEN] + _LAST_ERROR_TRUNC_MARKER
