"""C3 PR-4.5 — _should_skip_mailbox_lifecycle helper (spec §11.4).

Tests only the pure predicate. Integration with AgentService suspend call
sites is exercised at the call-site level (existing AgentService tests +
T10 E2E).
"""

from __future__ import annotations

from app.domain.models.session import Session
from app.domain.services.mailbox_skip_helper import _should_skip_mailbox_lifecycle


def _session(
    worker_type: str = "subagent",
    subagent_control_plane: str | None = None,
) -> Session:
    return Session(
        worker_type=worker_type,  # type: ignore[arg-type]
        subagent_control_plane=subagent_control_plane,  # type: ignore[arg-type]
        # parent_session_id mirrors worker_type: child needs parent, root must not
        parent_session_id="parent-id" if worker_type == "subagent" else None,
        tool_filter_preset=(
            "subagent_research" if worker_type == "subagent" else None
        ),
    )


def test_subagent_mailbox_plane_skips_lifecycle() -> None:
    assert _should_skip_mailbox_lifecycle(_session("subagent", "mailbox")) is True


def test_subagent_legacy_plane_does_not_skip() -> None:
    assert _should_skip_mailbox_lifecycle(_session("subagent", "legacy")) is False


def test_subagent_null_plane_does_not_skip() -> None:
    """Pre-C3 / pre-PR-5 rows default to NULL ≡ legacy."""
    assert _should_skip_mailbox_lifecycle(_session("subagent", None)) is False


def test_root_session_never_skips_regardless_of_plane() -> None:
    """Root sessions are not subagent — AgentService keeps owning their suspend.

    Defense in depth: even if a stray row somehow carries
    ``worker_type='root'`` AND ``subagent_control_plane='mailbox'``
    (an invariant the application code never produces; a future DB
    CHECK could enforce it at the schema layer), the helper still
    returns False so AgentService takes ownership of the suspend
    via the legacy path.
    """
    assert _should_skip_mailbox_lifecycle(_session("root", "mailbox")) is False
    assert _should_skip_mailbox_lifecycle(_session("root", None)) is False
