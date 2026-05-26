"""C2 PR-4 Task 4.1 — PatchManifest + FilePatchEntry wire schema tests.

Spec ref: §6.2 新 schema 1.
- frozen + extra=forbid
- op ∈ {"add", "modify", "delete"} (Literal)
- base_digest required for "modify"/"delete" lineage; None for "add"
- content_ref/content_size optional (None for "delete")
- deterministic patch_id format: f"{coordinator_run_id}:{work_unit_id}:p"
"""
from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from app.domain.models.patch_manifest import FilePatchEntry, PatchManifest


# [r5 P2] Synthetic SHA-256 hex values for digest fields. Real hex is
# 64 lowercase hex chars; the literal "sha_a"/"s"/etc. used in earlier
# rounds would now fail the digest format validator.
_SHA_A = hashlib.sha256(b"a").hexdigest()
_SHA_B = hashlib.sha256(b"b").hexdigest()
_SHA_C = hashlib.sha256(b"c").hexdigest()


class TestFilePatchEntryModify:
    def test_modify_with_digests(self) -> None:
        e = FilePatchEntry(
            path="x", op="modify",
            base_digest=_SHA_A, new_digest=_SHA_B,
            content_ref="minio://ref", content_size=100,
        )
        assert e.op == "modify"
        assert e.base_digest == _SHA_A
        assert e.new_digest == _SHA_B
        assert e.content_ref == "minio://ref"
        assert e.content_size == 100
        assert e.diff_ref is None


class TestFilePatchEntryAdd:
    def test_add_no_base_digest(self) -> None:
        e = FilePatchEntry(
            path="x", op="add",
            new_digest=_SHA_B, content_ref="minio://ref", content_size=50,
        )
        assert e.base_digest is None
        assert e.new_digest == _SHA_B
        assert e.content_ref == "minio://ref"
        assert e.content_size == 50


class TestFilePatchEntryDelete:
    def test_delete_no_content(self) -> None:
        e = FilePatchEntry(path="x", op="delete", base_digest=_SHA_A)
        assert e.content_ref is None
        assert e.content_size is None
        assert e.new_digest is None


class TestFilePatchEntryInvariants:
    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            FilePatchEntry(  # type: ignore[call-arg]
                path="x", op="add", new_digest=_SHA_A, content_ref="r", content_size=1,
                unexpected="nope",
            )

    def test_frozen(self) -> None:
        e = FilePatchEntry(path="x", op="delete", base_digest=_SHA_A)
        with pytest.raises(ValidationError):
            e.op = "modify"  # type: ignore[misc]

    def test_invalid_op(self) -> None:
        with pytest.raises(ValidationError):
            FilePatchEntry(path="x", op="rename")  # type: ignore[arg-type]

    def test_path_required(self) -> None:
        with pytest.raises(ValidationError):
            FilePatchEntry(op="add")  # type: ignore[call-arg]


class TestFilePatchEntryPerOpValidators:
    """[r1 P1#3] Per-op constraints enforced at wire schema, NOT just producer."""

    def test_add_with_base_digest_rejected(self) -> None:
        with pytest.raises(ValidationError, match="op=add must have base_digest=None"):
            FilePatchEntry(
                path="x", op="add", base_digest=_SHA_A,
                new_digest=_SHA_A, content_ref="r", content_size=1,
            )

    def test_add_missing_new_digest_rejected(self) -> None:
        with pytest.raises(ValidationError, match="op=add must have new_digest"):
            FilePatchEntry(
                path="x", op="add", content_ref="r", content_size=1,
            )

    def test_add_missing_content_ref_rejected(self) -> None:
        with pytest.raises(ValidationError, match="op=add must have"):
            FilePatchEntry(
                path="x", op="add", new_digest=_SHA_A, content_size=1,
            )

    def test_modify_missing_base_digest_rejected(self) -> None:
        with pytest.raises(ValidationError, match="op=modify requires base_digest"):
            FilePatchEntry(
                path="x", op="modify",
                new_digest=_SHA_A, content_ref="r", content_size=1,
            )

    def test_modify_missing_new_digest_rejected(self) -> None:
        with pytest.raises(ValidationError, match="op=modify must have"):
            FilePatchEntry(
                path="x", op="modify", base_digest=_SHA_B,
                content_ref="r", content_size=1,
            )

    def test_delete_missing_base_digest_rejected(self) -> None:
        with pytest.raises(ValidationError, match="op=delete requires base_digest"):
            FilePatchEntry(path="x", op="delete")

    def test_delete_with_new_digest_rejected(self) -> None:
        with pytest.raises(ValidationError, match="op=delete must have"):
            FilePatchEntry(
                path="x", op="delete", base_digest=_SHA_B, new_digest=_SHA_B,
            )

    def test_delete_with_content_ref_rejected(self) -> None:
        with pytest.raises(ValidationError, match="op=delete must have"):
            FilePatchEntry(
                path="x", op="delete", base_digest=_SHA_B, content_ref="r",
            )

    def test_delete_with_content_size_rejected(self) -> None:
        with pytest.raises(ValidationError, match="op=delete must have"):
            FilePatchEntry(
                path="x", op="delete", base_digest=_SHA_B, content_size=0,
            )

    def test_negative_content_size_rejected(self) -> None:
        """Field(ge=0) — negative sizes are non-sensical and would mislead
        downstream consumers reading content_size for budget tracking."""
        with pytest.raises(ValidationError):
            FilePatchEntry(
                path="x", op="add",
                new_digest=_SHA_A, content_ref="r", content_size=-1,
            )


class TestFilePatchEntryDigestFormat:
    """[r5 P2] base_digest + new_digest validated as SHA-256 hex (64
    lowercase hex chars). Prevents a malicious child from publishing
    phantom digests that would bypass PR-5 reducer's compare logic."""

    def test_short_digest_rejected(self) -> None:
        with pytest.raises(ValidationError, match="SHA-256"):
            FilePatchEntry(
                path="x", op="add",
                new_digest="abc", content_ref="r", content_size=1,
            )

    def test_uppercase_digest_rejected(self) -> None:
        """Lowercase hex only — hashlib emits lowercase, and case-sensitive
        compare is faster than case-insensitive in the reducer hot path."""
        with pytest.raises(ValidationError, match="SHA-256"):
            FilePatchEntry(
                path="x", op="add",
                new_digest="A" * 64, content_ref="r", content_size=1,
            )

    def test_non_hex_chars_rejected(self) -> None:
        with pytest.raises(ValidationError, match="SHA-256"):
            FilePatchEntry(
                path="x", op="add",
                new_digest="g" * 64, content_ref="r", content_size=1,
            )

    def test_base_digest_format_also_validated(self) -> None:
        with pytest.raises(ValidationError, match="SHA-256"):
            FilePatchEntry(
                path="x", op="modify",
                base_digest="not-a-digest", new_digest=_SHA_B,
                content_ref="r", content_size=1,
            )

    def test_valid_hex_accepted(self) -> None:
        e = FilePatchEntry(
            path="x", op="modify",
            base_digest=_SHA_A, new_digest=_SHA_B,
            content_ref="r", content_size=1,
        )
        assert len(e.new_digest) == 64


class TestPatchManifestStructure:
    def test_deterministic_id_format(self) -> None:
        m = PatchManifest(
            patch_id="r1:wu1:p",
            coordinator_run_id="r1", work_unit_id="wu1",
            files=(FilePatchEntry(
                path="x", op="add",
                new_digest=_SHA_A, content_ref="minio://r", content_size=1,
            ),),
        )
        assert m.patch_id == "r1:wu1:p"
        assert len(m.files) == 1

    def test_empty_files_allowed(self) -> None:
        """Coordinator child may produce a manifest with no writes (e.g. exploration
        finalize fallback). Schema must accept empty tuple."""
        m = PatchManifest(
            patch_id="x", coordinator_run_id="r", work_unit_id="w", files=(),
        )
        assert m.files == ()


class TestPatchManifestInvariants:
    def test_frozen(self) -> None:
        m = PatchManifest(
            patch_id="x", coordinator_run_id="r", work_unit_id="w", files=(),
        )
        with pytest.raises(ValidationError):
            m.coordinator_run_id = "y"  # type: ignore[misc]

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            PatchManifest(  # type: ignore[call-arg]
                patch_id="x", coordinator_run_id="r", work_unit_id="w", files=(),
                extra="nope",
            )

    def test_files_is_tuple_not_list(self) -> None:
        """Tuple field is immutable; list field would let downstream mutate
        the manifest under the frozen contract."""
        m = PatchManifest(
            patch_id="x", coordinator_run_id="r", work_unit_id="w", files=(),
        )
        assert isinstance(m.files, tuple)


class TestPatchManifestWireRoundtrip:
    """model_dump → model_validate cycle preserves all fields.

    MailboxEnvelope re-validates payload via model_dump → model_validate;
    PatchManifest must survive the round-trip."""
    def test_roundtrip_complete(self) -> None:
        original = PatchManifest(
            patch_id="r1:wu1:p", coordinator_run_id="r1", work_unit_id="wu1",
            files=(
                FilePatchEntry(
                    path="a", op="add",
                    new_digest=_SHA_A, content_ref="ref_a", content_size=10,
                ),
                FilePatchEntry(
                    path="b", op="modify",
                    base_digest=_SHA_A, new_digest=_SHA_B,
                    content_ref="ref_b", content_size=20,
                ),
                FilePatchEntry(path="c", op="delete", base_digest=_SHA_C),
            ),
        )
        dumped = original.model_dump(mode="python")
        rehydrated = PatchManifest.model_validate(dumped)
        assert rehydrated.patch_id == "r1:wu1:p"
        assert len(rehydrated.files) == 3
        assert rehydrated.files[2].op == "delete"
        assert rehydrated.files[2].content_ref is None
