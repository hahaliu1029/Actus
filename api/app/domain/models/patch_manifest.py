"""C2 v1 PatchManifest schema (spec §6.2 新 schema 1).

Frozen + extra=forbid wire schema. The child coordinator runner produces a
PatchManifest summarizing its file writes; the parent reducer inspects it
(but does NOT apply files in PR-4 — apply lands in PR-5 PatchApplier).

[r5 P2] Digest fields validated to SHA-256 hex shape (64 lowercase hex
chars) so a malicious child can't smuggle phantom `base_digest`/`new_digest`
values that would silently bypass the PR-5 reducer's lease-vs-manifest
digest compare. content_ref/diff_ref remain opaque strings; their shape
is the producer's contract (MinIO ref) and PR-5 PatchApplier resolves
them via the artifact_storage port, which fails on missing keys — so an
adversarial ref name is observable but not exploitable.

content_ref holds a MinIO ref (content-addressed in PR-4 Task 4.9) to the
final post-write bytes for each entry. The reducer reads bytes-by-ref when
applying to the parent sandbox.

Schema invariants enforced at the type level:
- ``op`` is one of {"add", "modify", "delete"} (Literal)
- ``files`` is a tuple (immutable) under the frozen contract
- ``extra=forbid`` rejects unknown fields on the wire
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from app.domain.models.path_validation import validate_relative_path_strict


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_sha256_hex(value: str) -> str:
    """[r5 P2] Reject digests that aren't 64-char lowercase hex (SHA-256 shape).

    Used by base_digest + new_digest fields on FilePatchEntry. The check
    catches a malicious child publishing free-form strings like
    ``"valid_base_yeah_trust_me"`` that would silently bypass the PR-5
    reducer's lease-vs-manifest digest compare."""
    if not _SHA256_HEX_RE.match(value):
        raise ValueError(
            f"digest must be 64-char lowercase hex (SHA-256), got {value!r}"
        )
    return value


class FilePatchEntry(BaseModel):
    """Single file change in a PatchManifest.

    Per-op semantics (spec §6.2), enforced at the wire schema level so an
    invalid manifest can never reach RESULT_READY:

    - ``add``    : ``base_digest`` MUST be None (file didn't exist at preflight)
                   ``new_digest`` / ``content_ref`` / ``content_size`` required
    - ``modify`` : ``base_digest`` required (parent preflight SHA-256, used
                   by reducer for conflict detection in PR-5)
                   ``new_digest`` / ``content_ref`` / ``content_size`` required
    - ``delete`` : ``base_digest`` required (lineage identifier so reducer
                   knows which parent revision is being deleted)
                   ``content_ref`` / ``content_size`` / ``new_digest`` MUST
                   be None (no post-write content)

    ``content_size`` ≥ 0 invariant (negative sizes are non-sensical).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # [codex R11 P1] max_length=2048 matches
    # ``coordinator_apply_audit.failed_at_path`` String(2048); also
    # bounds NAME_MAX exposure in the snapshot store. Schema
    # validator is the single source of truth for path length so the
    # DB column length + on-disk snapshot filename + log strings stay
    # aligned.
    path: Annotated[
        str,
        Field(max_length=2048),
        AfterValidator(validate_relative_path_strict),
    ]
    op: Literal["add", "modify", "delete"]
    base_digest: Optional[Annotated[str, AfterValidator(_validate_sha256_hex)]] = None
    new_digest: Optional[Annotated[str, AfterValidator(_validate_sha256_hex)]] = None
    content_ref: Optional[str] = None
    content_size: Optional[int] = Field(default=None, ge=0)
    diff_ref: Optional[str] = None

    @model_validator(mode="after")
    def _enforce_per_op_constraints(self) -> "FilePatchEntry":
        if self.op == "add":
            if self.base_digest is not None:
                raise ValueError("op=add must have base_digest=None")
            if self.new_digest is None or self.content_ref is None or self.content_size is None:
                raise ValueError(
                    "op=add must have new_digest/content_ref/content_size set"
                )
        elif self.op == "modify":
            if self.base_digest is None:
                raise ValueError("op=modify requires base_digest")
            if self.new_digest is None or self.content_ref is None or self.content_size is None:
                raise ValueError(
                    "op=modify must have new_digest/content_ref/content_size set"
                )
        elif self.op == "delete":
            if self.base_digest is None:
                raise ValueError("op=delete requires base_digest (lineage id)")
            if self.new_digest is not None or self.content_ref is not None or self.content_size is not None:
                raise ValueError(
                    "op=delete must have new_digest/content_ref/content_size=None"
                )
        return self


class PatchManifest(BaseModel):
    """Aggregate of per-file changes produced by one coordinator child.

    The ``patch_id`` is deterministic: ``f"{coordinator_run_id}:{work_unit_id}:p"``.
    Determinism lets the parent dedupe replays from a crashed child (PR-7
    crash-recovery scope) without needing a separate idempotency table.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    patch_id: str
    coordinator_run_id: str
    work_unit_id: str
    files: tuple[FilePatchEntry, ...]
