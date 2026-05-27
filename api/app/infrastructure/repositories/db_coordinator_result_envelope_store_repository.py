"""DB-backed CoordinatorResultEnvelopeStoreRepository (C2 PR-7 §13.x).

Short-lived sessions per call — the mailbox supervisor / coordinator
orchestrator persist terminal envelopes from background-task contexts
that have no enclosing UoW span. Each insert commits in its own
session; UniqueViolation on duplicate (run_id, wu_id) bubbles to the
caller as the idempotency signal.

Payload safeguards applied before insert:

- ``_filter_minimum_rehydrate`` strips everything except the
  ``_MIN_REHYDRATE_KEYS`` whitelist so free-text fields (assistant
  message, raw tool transcripts) don't land in a long-term recovery
  table.
- 64 KB truncation cap — payloads that serialise larger collapse to
  ``{outcome, _truncated: True}`` plus a WARN log so the operator can
  investigate why the rehydrate-essential fields ballooned.
- Email / phone regex redaction — best-effort PII guard; matches
  collapse to ``{outcome, _pii_redacted: True}`` plus a WARN log.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.repositories.coordinator_result_envelope_store_repository import (
    CoordinatorResultEnvelopeStoreRepository,
)
from app.infrastructure.models.coordinator_result_envelope_store import (
    CoordinatorResultEnvelopeStore,
)


logger = logging.getLogger(__name__)


# Whitelist of payload keys preserved into the persisted JSONB. Anything
# else (assistant message, raw tool transcripts, debug metadata) gets
# stripped at ``_filter_minimum_rehydrate``. PR-7 rehydrate only
# consumes these five fields when rebuilding completed_work_units.
_MIN_REHYDRATE_KEYS = frozenset(
    {
        "outcome",
        "patch_manifest",
        "cost_summary",
        "needs_authorization_details",
        "final_state",
    }
)


# Serialised payload size ceiling. Beyond this the row degrades to
# ``{outcome, _truncated: True}`` — the rehydrate path can still
# resume the run (knows the work-unit terminated) but loses the
# patch manifest / cost summary, which is the lesser evil over
# unbounded JSONB growth.
_MAX_PAYLOAD_BYTES = 64 * 1024


# Best-effort PII regexes — match common email / phone formats. Hits
# downgrade the row to ``{outcome, _pii_redacted: True}``. These are
# defense-in-depth; the application layer is the canonical PII
# scrubbing surface.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d -]{7,}")


def _filter_minimum_rehydrate(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict containing only keys in ``_MIN_REHYDRATE_KEYS``.

    Pure function — does not mutate ``payload``. Missing keys are
    simply absent in the output (no defaulting); the rehydrate consumer
    handles partial dicts.
    """
    return {k: v for k, v in payload.items() if k in _MIN_REHYDRATE_KEYS}


def _pii_guard(payload_json: str) -> bool:
    """Return True if either email or phone regex matches the serialised
    payload string. Used as a binary decision — the caller doesn't need
    to know which one fired.
    """
    return bool(_EMAIL_RE.search(payload_json) or _PHONE_RE.search(payload_json))


class DbCoordinatorResultEnvelopeStoreRepository(
    CoordinatorResultEnvelopeStoreRepository,
):
    """DB impl. Mirrors the short-session pattern used by
    ``DbCoordinatorApplyAuditRepository``."""

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._sf = session_factory

    async def persist_terminal(
        self,
        *,
        coordinator_run_id: str,
        work_unit_id: str,
        child_session_id: str,
        envelope_type: str,
        payload: dict[str, Any],
    ) -> None:
        # Step 1: strip to minimum-rehydrate whitelist.
        filtered = _filter_minimum_rehydrate(payload)

        # Step 2: serialise once and inspect for size + PII. Both
        # safeguards fire independently — truncation degrades to a
        # marker-only payload, and PII redaction does the same (the
        # later check would overwrite the earlier if both hit, which
        # is fine because either marker is sufficient signal for
        # operators).
        try:
            serialised = json.dumps(filtered, separators=(",", ":"))
        except (TypeError, ValueError):
            # Non-serialisable filtered dict — degrade to a marker so
            # the row is still inserted (rehydrate needs *some* row to
            # know the work-unit terminated).
            logger.warning(
                "persist_terminal: payload not JSON-serialisable for "
                "run_id=%s wu_id=%s; persisting marker only.",
                coordinator_run_id, work_unit_id,
            )
            filtered = {
                "outcome": str(payload.get("outcome", "unknown")),
                "_unserialisable": True,
            }
            serialised = json.dumps(filtered, separators=(",", ":"))

        # Truncation check — apply BEFORE PII so we don't burn CPU on
        # regex scanning a payload we're about to discard anyway.
        if len(serialised.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            logger.warning(
                "persist_terminal: payload exceeds %d bytes for "
                "run_id=%s wu_id=%s (size=%d); truncating to marker.",
                _MAX_PAYLOAD_BYTES,
                coordinator_run_id,
                work_unit_id,
                len(serialised.encode("utf-8")),
            )
            filtered = {
                "outcome": filtered.get("outcome", "unknown"),
                "_truncated": True,
            }
            serialised = json.dumps(filtered, separators=(",", ":"))

        # PII check — runs on the (possibly truncated) serialised form.
        if _pii_guard(serialised):
            logger.warning(
                "persist_terminal: PII regex matched payload for "
                "run_id=%s wu_id=%s; redacting to marker.",
                coordinator_run_id, work_unit_id,
            )
            filtered = {
                "outcome": filtered.get("outcome", "unknown"),
                "_pii_redacted": True,
            }

        async with self._sf() as s:
            row = CoordinatorResultEnvelopeStore(
                coordinator_run_id=coordinator_run_id,
                work_unit_id=work_unit_id,
                child_session_id=child_session_id,
                envelope_type=envelope_type,
                payload=filtered,
            )
            s.add(row)
            await s.commit()

    async def find_terminal_envelopes_by_run(
        self, coordinator_run_id: str,
    ) -> list[CoordinatorResultEnvelopeStore]:
        async with self._sf() as s:
            stmt = (
                select(CoordinatorResultEnvelopeStore)
                .where(
                    CoordinatorResultEnvelopeStore.coordinator_run_id
                    == coordinator_run_id
                )
                .order_by(CoordinatorResultEnvelopeStore.received_at.asc())
            )
            result = await s.execute(stmt)
            return list(result.scalars().all())
