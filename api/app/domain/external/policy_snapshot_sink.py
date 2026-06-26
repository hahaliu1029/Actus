"""C5a domain port for policy-snapshot observation.

PURE DOMAIN. The SandboxPolicySnapshot reference stays under TYPE_CHECKING so
this module's top-level imports remain stdlib-only (INV-3, §8.10).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.domain.models.sandbox_policy import SandboxPolicySnapshot


class PolicySnapshotSink(Protocol):
    async def record(self, snapshot: "SandboxPolicySnapshot") -> None: ...


class NoopPolicySnapshotSink:
    """Default sink: records nothing. Non-suspending (no await/IO) — INV-0."""

    async def record(self, snapshot: "SandboxPolicySnapshot") -> None:  # noqa: RUF029
        return None
