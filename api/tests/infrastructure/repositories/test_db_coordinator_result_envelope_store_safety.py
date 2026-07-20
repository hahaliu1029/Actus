"""[codex R1 P2] Unit tests for the envelope store safety paths.

The integration test (``tests/integration/test_migration_c2pr7_envelope_store.py``)
covers schema + unique constraint. This file covers the orthogonal safety
contracts that fire BEFORE the INSERT runs:

  • whitelist filter (``_filter_minimum_rehydrate``) — only keys in
    ``_MIN_REHYDRATE_KEYS`` survive.
  • PII regex (``_pii_guard``) — email / phone patterns.
  • oversize truncation (>64KB serialised payload).
  • non-serialisable payload fallback (``json.dumps`` raises).

All four branches degrade gracefully to a marker-only ``{outcome, _flag}``
shape so the rehydrate path still sees an envelope row.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.infrastructure.repositories.db_coordinator_result_envelope_store_repository import (
    DbCoordinatorResultEnvelopeStoreRepository,
    _MAX_PAYLOAD_BYTES,
    _MIN_REHYDRATE_KEYS,
    _filter_minimum_rehydrate,
    _pii_guard,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ── Pure helpers ────────────────────────────────────────────────────────


class TestFilterMinimumRehydrate:
    def test_keeps_whitelisted_keys(self) -> None:
        payload = {
            "outcome": "success",
            "patch_manifest": {"file_count": 2},
            "patch_manifest_ref": "coordinator/r1/wu1/manifest/abc",
            "cost_summary": {"total_cost_usd": 0.5},
            "needs_authorization_details": {"reason": "denied"},
            "final_state": "cancelled",
        }
        out = _filter_minimum_rehydrate(payload)
        # [S2 §3.2 C1] Pin the new key explicitly. The bare
        # ``set(out.keys()) == _MIN_REHYDRATE_KEYS`` equality is symmetric and
        # would stay GREEN pre-impl (the key is dropped from BOTH sides); these
        # two asserts go RED until the frozenset is extended.
        assert "patch_manifest_ref" in out
        assert "patch_manifest_ref" in _MIN_REHYDRATE_KEYS
        assert set(out.keys()) == _MIN_REHYDRATE_KEYS

    def test_manifest_ref_in_whitelist(self) -> None:
        assert "patch_manifest_ref" in _MIN_REHYDRATE_KEYS

    def test_strips_free_text_keys(self) -> None:
        payload = {
            "outcome": "success",
            "summary": "user typed a long free-text message here",
            "rationale": "free-text reasoning leakage candidate",
            "patch_manifest": {"x": 1},
        }
        out = _filter_minimum_rehydrate(payload)
        assert "summary" not in out
        assert "rationale" not in out
        assert "outcome" in out
        assert "patch_manifest" in out

    def test_unknown_keys_dropped(self) -> None:
        payload = {"outcome": "success", "x_attacker_injected": "rm -rf /"}
        out = _filter_minimum_rehydrate(payload)
        assert "x_attacker_injected" not in out

    def test_empty_payload_yields_empty(self) -> None:
        assert _filter_minimum_rehydrate({}) == {}


class TestPiiGuard:
    def test_email_match_returns_true(self) -> None:
        assert _pii_guard('{"outcome":"success","leak":"alice@example.com"}')

    def test_phone_match_returns_true(self) -> None:
        assert _pii_guard('{"outcome":"failed","note":"+1 555 1234 9876"}')

    def test_clean_json_returns_false(self) -> None:
        assert not _pii_guard('{"outcome":"success","cost_summary":{}}')

    def test_short_number_does_not_match_phone(self) -> None:
        # _PHONE_RE requires \d[\d -]{7,} → at least 8 digit-ish chars
        assert not _pii_guard('{"outcome":"x","ts":"123"}')


# ── persist_terminal safety integration via fake session_factory ─────────


class _FakeSession:
    """Captures the row passed to ``s.add`` so tests can assert on payload."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *a: object) -> None:
        return None

    def add(self, row: Any) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        return None


class _FakeSessionFactory:
    """Yields a fresh ``_FakeSession`` on each ``()`` call.

    The repo holds the factory and calls ``self._sf()`` in each method to
    open a short-session — this fake mirrors that contract while letting
    tests inspect what was actually persisted.
    """

    def __init__(self) -> None:
        self.sessions: list[_FakeSession] = []

    def __call__(self) -> _FakeSession:
        s = _FakeSession()
        self.sessions.append(s)
        return s


class TestPersistTerminalSafety:
    async def test_oversize_payload_truncated_to_marker(self) -> None:
        factory = _FakeSessionFactory()
        repo = DbCoordinatorResultEnvelopeStoreRepository(factory)  # type: ignore[arg-type]
        big_dict = {
            "outcome": "success",
            "patch_manifest": {"files": ["f%d" % i for i in range(20000)]},
        }
        await repo.persist_terminal(
            coordinator_run_id="r1",
            work_unit_id="wu1",
            child_session_id="c1",
            envelope_type="RESULT_READY",
            payload=big_dict,
        )
        row = factory.sessions[0].added[0]
        # Truncation marker present, original list stripped.
        assert row.payload == {"outcome": "success", "_truncated": True}

    async def test_pii_email_payload_redacted_to_marker(self) -> None:
        factory = _FakeSessionFactory()
        repo = DbCoordinatorResultEnvelopeStoreRepository(factory)  # type: ignore[arg-type]
        # Use a whitelisted key that survives _filter_minimum_rehydrate
        # but carries PII in a nested value.
        payload_with_pii = {
            "outcome": "success",
            "needs_authorization_details": {
                "user_contact": "alice@example.com",
            },
        }
        await repo.persist_terminal(
            coordinator_run_id="r1",
            work_unit_id="wu_pii",
            child_session_id="c_pii",
            envelope_type="RESULT_READY",
            payload=payload_with_pii,
        )
        row = factory.sessions[0].added[0]
        assert row.payload == {
            "outcome": "success",
            "_pii_redacted": True,
        }

    async def test_unserialisable_payload_falls_back_to_marker(self) -> None:
        """A non-JSON-serialisable value in a whitelisted key triggers the
        ``json.dumps`` TypeError fallback."""
        factory = _FakeSessionFactory()
        repo = DbCoordinatorResultEnvelopeStoreRepository(factory)  # type: ignore[arg-type]
        # ``set`` is not JSON-serialisable; pass it under a whitelisted key.
        payload = {
            "outcome": "failed",
            "patch_manifest": {"non_serialisable_set": {1, 2, 3}},
        }
        await repo.persist_terminal(
            coordinator_run_id="r1",
            work_unit_id="wu_bad",
            child_session_id="c_bad",
            envelope_type="RESULT_READY",
            payload=payload,
        )
        row = factory.sessions[0].added[0]
        assert row.payload == {"outcome": "failed", "_unserialisable": True}

    async def test_clean_whitelisted_payload_persisted_verbatim(self) -> None:
        factory = _FakeSessionFactory()
        repo = DbCoordinatorResultEnvelopeStoreRepository(factory)  # type: ignore[arg-type]
        payload = {"outcome": "success", "cost_summary": {"total_cost_usd": 0.05}}
        await repo.persist_terminal(
            coordinator_run_id="r1",
            work_unit_id="wu_ok",
            child_session_id="c_ok",
            envelope_type="RESULT_READY",
            payload=payload,
        )
        row = factory.sessions[0].added[0]
        assert row.payload == payload

    async def test_inline_manifest_machine_ids_are_not_redacted_as_pii(self) -> None:
        """Digest/ref fields are structured identifiers, not phone numbers."""
        factory = _FakeSessionFactory()
        repo = DbCoordinatorResultEnvelopeStoreRepository(factory)  # type: ignore[arg-type]
        payload = {
            "outcome": "success",
            "patch_manifest": {
                "patch_id": "run-12345678:wu-12345678:p",
                "coordinator_run_id": "run-12345678",
                "work_unit_id": "wu-12345678",
                "files": [
                    {
                        "path": "workspace/result.txt",
                        "op": "add",
                        "base_digest": None,
                        "new_digest": "1234567890abcdef" * 4,
                        "content_ref": "coordinator/12345678/patch/1234567890",
                        "content_size": 12,
                    }
                ],
            },
        }
        await repo.persist_terminal(
            coordinator_run_id="r1",
            work_unit_id="wu_manifest",
            child_session_id="c_manifest",
            envelope_type="RESULT_READY",
            payload=payload,
        )
        row = factory.sessions[0].added[0]
        assert row.payload == payload

    async def test_payload_with_non_whitelisted_keys_filtered_before_insert(self) -> None:
        factory = _FakeSessionFactory()
        repo = DbCoordinatorResultEnvelopeStoreRepository(factory)  # type: ignore[arg-type]
        payload = {
            "outcome": "success",
            "summary": "very long free-text summary",
            "rationale": "more free-text leakage",
        }
        await repo.persist_terminal(
            coordinator_run_id="r1",
            work_unit_id="wu_strip",
            child_session_id="c_strip",
            envelope_type="RESULT_READY",
            payload=payload,
        )
        row = factory.sessions[0].added[0]
        assert "summary" not in row.payload
        assert "rationale" not in row.payload
        assert row.payload == {"outcome": "success"}


# ── _MAX_PAYLOAD_BYTES constant ─────────────────────────────────────────


class TestMaxPayloadBytesContract:
    def test_64kb_constant(self) -> None:
        assert _MAX_PAYLOAD_BYTES == 64 * 1024


# ── PII scan excludes the structured manifest ref ───────────────────────


import hashlib

from app.infrastructure.repositories.db_coordinator_result_envelope_store_repository import (  # noqa: E501
    _pii_scan_target,
)


class TestPiiExcludesManifestRef:
    def test_digest_like_manifest_ref_not_treated_as_pii(self) -> None:
        # A content-addressed MinIO ref ends in a 64-char sha256 hex digest,
        # which contains runs of 8+ digits → would trip the phone regex.
        digest = hashlib.sha256(b"x").hexdigest()
        ref = f"coordinator/r1/wu1/manifest/{digest}"
        filtered = {"outcome": "success", "patch_manifest_ref": ref}
        # The PII scan target must EXCLUDE the structured ref field, so the
        # digit-run inside the digest does not cause redaction.
        scan_target = _pii_scan_target(filtered)
        assert ref not in scan_target

    def test_email_in_non_excluded_field_still_caught(self) -> None:
        # Regression: real PII in a non-excluded field is still scanned.
        # The email here lives in ``needs_authorization_details.observed_evidence``
        # (a whitelisted free-text-bearing field); assert the PII excluder only
        # drops ``patch_manifest_ref``, nothing else.
        filtered = {
            "outcome": "success",
            "patch_manifest_ref": "coordinator/r1/wu1/manifest/abc",
            "needs_authorization_details": {"observed_evidence": "a@b.co"},
        }
        scan_target = _pii_scan_target(filtered)
        assert "a@b.co" in scan_target

    def test_email_in_manifest_path_is_still_scanned(self) -> None:
        filtered = {
            "outcome": "success",
            "patch_manifest": {
                "patch_id": "run-12345678:wu-12345678:p",
                "files": [{"path": "workspace/alice@example.com.txt", "op": "add"}],
            },
        }
        scan_target = _pii_scan_target(filtered)
        assert "alice@example.com" in scan_target
