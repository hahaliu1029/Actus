from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Mapping, NamedTuple, Optional, Protocol

from app.domain.models.event import BaseEvent
from app.domain.models.file import File
from app.domain.models.memory import Memory
from app.domain.models.session import Session, SessionStatus
from app.domain.repositories._sentinel import _UNSET, UnsetType

if TYPE_CHECKING:
    from app.domain.models.conversation_summary import ConversationSummary
    from app.domain.models.skill_creation_state import SkillCreationState
    from app.domain.models.skill_graph_state import SkillGraphState


class BgSessionRow(NamedTuple):
    """Lightweight row used by supervisor restart reconciliation."""

    session_id: str
    task_id: str | None
    user_id: str
    status: SessionStatus


class SessionRepository(Protocol):
    """会话仓库协议定义"""

    async def save(self, session: Session) -> None:
        """存储或更新传递进来的会话"""
        ...

    async def get_all(self) -> List[Session]:
        """获取所有会话列表信息"""
        ...

    async def get_all_by_user(self, user_id: str) -> List[Session]:
        """根据用户ID获取会话列表信息"""
        ...

    async def get_by_id(self, session_id: str) -> Optional[Session]:
        """根据传递的会话id查询会话"""
        ...

    async def get_by_id_for_update(self, session_id: str) -> Optional[Session]:
        """根据传递的会话id查询会话并加行锁"""
        ...

    async def delete_by_id(self, session_id: str) -> None:
        """根据传递的会话id删除会话"""
        ...

    async def update_title(self, session_id: str, title: str) -> None:
        """根据传递的会话id+标题更新会话信息"""
        ...

    async def update_latest_message(
        self, session_id: str, message: str, timestamp: datetime
    ) -> None:
        """根据传递的信息更新最新消息"""
        ...

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        """根据传递的信息更新未读消息数"""
        ...

    async def increment_unread_message_count(self, session_id: str) -> None:
        """根据传递的会话id新增未读消息数"""
        ...

    async def decrement_unread_message_count(self, session_id: str) -> None:
        """根据传递的会话id减少未读消息数"""
        ...

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        """Update non-terminal status; terminal writes use update_to_terminal."""
        ...

    async def find_running_background(self) -> list[BgSessionRow]:
        """Return background sessions in running/recovering supervisor phases."""
        ...

    async def find_running_mailbox_plane_root_ids(self) -> list[str]:
        """C3 PR-3c: return DISTINCT root session ids whose mailbox-plane
        subagents are still in flight.

        Used by ``SandboxLifecycleService.reconcile_orphans`` (after a pod
        restart) to re-spawn ``MailboxSupervisor`` tasks for roots that lost
        their per-pod supervisor when the process died.

        Selection criteria:
          * ``worker_type = 'subagent'`` (only mailbox sub-sessions need
            a supervisor; root-only sessions are out of scope).
          * ``subagent_control_plane = 'mailbox'`` (legacy plane has no
            mailbox supervisor).
          * ``status NOT IN ('completed', 'timed_out')`` — every other
            ``SessionStatus`` value (PENDING / RUNNING / TAKEOVER_PENDING /
            TAKEOVER / WAITING / FINISHING) is non-terminal and still
            needs a supervisor. codex r1 [HIGH CONTRACT] caught the
            original RUNNING/PENDING/FINISHING-only IN-list dropping
            WAITING+TAKEOVER* subagents (live but paused awaiting human).
            The NOT-IN-terminal phrasing also forward-protects against
            future non-terminal SessionStatus additions.

        Returns DISTINCT ``parent_session_id`` values. C1a memory pins
        ``parent_session_id`` as the sole lineage source; the C3 plan §6.5
        audit foci confirm mailbox subagents spawn only at depth=1, so
        ``parent_session_id`` IS the root id for these rows.
        """
        ...

    async def update_supervisor_fields(
        self,
        session_id: str,
        *,
        execution_mode: str | UnsetType = _UNSET,
        background_reason: str | None | UnsetType = _UNSET,
        expires_at: datetime | None | UnsetType = _UNSET,
        execution_phase: str | UnsetType = _UNSET,
        retry_budget_remaining: int | UnsetType = _UNSET,
        terminal_reason: str | None | UnsetType = _UNSET,
        suspended_reason: str | None | UnsetType = _UNSET,
        was_background: bool | UnsetType = _UNSET,
    ) -> None:
        """Patch supervisor fields; _UNSET skips, None writes NULL."""
        ...

    async def suspend_running_background_if_active(self, session_id: str) -> bool:
        """Atomically suspend an active background session.

        Returns False when a stale watchdog scan raced with terminalization or
        another phase transition.
        """
        ...

    async def promote_foreground_to_background(
        self,
        session_id: str,
        *,
        expires_at: datetime,
        retry_budget_remaining: int,
    ) -> int | None:
        """Atomically promote and return the persisted retry budget."""
        ...

    async def claim_background_retry_from_suspend(
        self,
        session_id: str,
        *,
        expires_at,
    ) -> int | None:
        """Atomically claim a suspended background retry and return remaining budget."""
        ...

    async def rollback_background_retry_claim_if_active(
        self,
        session_id: str,
        *,
        retry_budget_remaining: int,
        expires_at: datetime | None,
        suspended_reason: str | None,
    ) -> bool:
        """Restore a claimed retry only while it is still the active running phase."""
        ...

    async def update_to_terminal(
        self,
        session_id: str,
        status: SessionStatus,
        terminal_reason: str,
    ) -> bool:
        """Atomically write terminal status, phase and reason.

        Returns True when this call performed the terminal transition and
        False when the row was already terminal/terminating.
        """
        ...

    async def update_terminal_reason(
        self,
        session_id: str,
        terminal_reason: str,
    ) -> None:
        """Late-bind terminal_reason for existing terminal rows."""
        ...

    async def distinct_user_ids_with_running_bg(self) -> list[str]:
        """Return user ids with sweepable background sessions."""
        ...

    async def add_event(self, session_id: str, event: BaseEvent) -> None:
        """往会话中新增事件"""
        ...

    async def add_file(self, session_id: str, file: File) -> None:
        """往会话中新增文件"""
        ...

    async def remove_file(self, session_id: str, file_id: str) -> None:
        """根据传递的会话id+文件id移除文件"""
        ...

    async def get_file_by_path(self, session_id: str, filepath: str) -> Optional[File]:
        """查询会话中的文件信息"""
        ...

    async def save_memory(
        self, session_id: str, agent_name: str, memory: Memory
    ) -> None:
        """更新or创建会话中指定Agent的记忆"""
        ...

    async def get_memory(self, session_id: str, agent_name: str) -> Memory:
        """根据传递的会话id+Agent名字获取记忆"""
        ...

    async def get_summary(self, session_id: str) -> list[ConversationSummary]:
        """获取会话的对话摘要列表"""
        ...

    async def save_summary(
        self, session_id: str, summaries: list[ConversationSummary]
    ) -> None:
        """保存会话的对话摘要列表"""
        ...

    async def get_skill_creation_state(
        self, session_id: str
    ) -> SkillCreationState | None:
        """获取 Skill 创建链路的等待状态"""
        ...

    async def save_skill_creation_state(
        self, session_id: str, state: SkillCreationState
    ) -> None:
        """保存 Skill 创建链路的等待状态"""
        ...

    async def clear_skill_creation_state(self, session_id: str) -> None:
        """清理 Skill 创建链路的等待状态"""
        ...

    async def get_skill_graph_state(
        self, session_id: str
    ) -> SkillGraphState | None:
        """获取 Skill 创建子图的持久化状态"""
        ...

    async def save_skill_graph_state(
        self, session_id: str, state: SkillGraphState
    ) -> None:
        """保存 Skill 创建子图的持久化状态"""
        ...

    async def clear_skill_graph_state(self, session_id: str) -> None:
        """清理 Skill 创建子图的持久化状态"""
        ...

    async def transition_status(
        self,
        *,
        session_id: str,
        from_state: SessionStatus,
        to_state: SessionStatus,
        extra_values: Mapping[str, Any] | None = None,
    ) -> bool:
        """Atomic CAS: UPDATE sessions SET status=:to,
        mode_revision=mode_revision+1 WHERE id=:sid AND status=:from.
        Returns True iff row updated. Caller (SSM) owns commit via UoW.

        ``extra_values`` (optional) merges additional column writes into the
        same UPDATE statement so terminal metadata (``completed_at``,
        ``terminal_reason``, ``execution_phase``) can be written atomically
        with the status CAS. Keys MUST be SessionModel column names; values
        are passed through to SQLAlchemy unchanged. Reserved keys
        (``status``, ``mode_revision``, ``updated_at``) are owned by the
        repo and MUST NOT appear in ``extra_values``.
        """
        ...

    async def read_mode_revision(self, session_id: str) -> int:
        """Return the current mode_revision counter."""
        ...

    async def read_status_with_revision(
        self, session_id: str,
    ) -> tuple[SessionStatus, int]:
        """Return (status, mode_revision) as a single read."""
        ...

    async def find_descendants(
        self,
        ancestor_id: str,
        *,
        user_id: str,
        max_depth: int,
        limit: int,
    ) -> List[Session]:
        """C1a: return descendants of ancestor_id, depth-limited, user-scoped.

        `max_depth` is the inclusive depth limit (1 == direct children only).
        `limit` is the row cap; callers pass `cap + 1` to detect truncation.
        Order: depth ASC, id ASC. Excludes the ancestor itself.
        """
        ...

    async def count_descendants(
        self,
        ancestor_id: str,
        *,
        user_id: str,
        cap: int,
    ) -> int:
        """C1a: return count of descendants up to `cap + 1` (sentinel for >= cap).
        Implementation MUST use a LIMIT cap+1 subquery, not a full count."""
        ...

    async def lock_session_for_spawn(
        self, parent_id: str, *, user_id: str
    ) -> Optional[Session]:
        """C1a: SELECT ... FOR UPDATE on parent row inside an active transaction.
        Caller MUST be inside `async with uow:` - lock releases on UoW exit.

        Defense-in-depth: ``user_id`` is pushed into the SQL ``WHERE`` clause
        alongside ``id`` so a cross-tenant ``parent_id`` never acquires a row
        lock (returns ``None`` exactly like a missing id). This collapses
        cross-tenant + not-found into a single ``None`` result, mirroring
        ``find_by_id_for_user`` and defeating ID-enumeration via lock-timing.
        """
        ...

    async def find_by_id_for_user(
        self, session_id: str, *, user_id: str
    ) -> Optional[Session]:
        """C1a: owner-scoped fetch. Returns None for missing OR foreign-user
        (collapse 403/404 to a single 404 to defeat ID enumeration)."""
        ...

    # ── C2 PR-3 §7.5 P0-3 — coordinator attempt counter (JSONB) ────────────

    async def peek_coordinator_attempt(
        self, *, session_id: str, step_id: str,
    ) -> Optional[int]:
        """C2 PR-3 §7.5 P0-3 — READ current attempt_ix without bumping.

        Returns ``None`` when ``step_id`` has never been dispatched (key missing
        from ``coordinator_attempts`` JSONB) or session row absent. Otherwise
        returns the current attempt_ix (``>= 1``).

        ``dispatch_node`` uses this for crash recovery detection BEFORE deciding
        whether to bump (atomic write) or rehydrate (read existing run).
        """
        ...

    async def bump_coordinator_attempt(
        self, *, session_id: str, step_id: str,
    ) -> int:
        """C2 PR-3 §7.5 P0-3 — atomic JSONB increment; returns new attempt_ix
        (``>= 1``).

        Called by ``dispatch_node`` only when a first-time dispatch (peek
        returned None) or a replan-bump (no existing run found at peeked
        attempt_ix) is needed. SQL writes ``coordinator_attempts`` JSONB via
        ``jsonb_build_object`` + ``COALESCE`` for an atomic
        read-modify-write under PostgreSQL row-level lock.
        """
        ...

    async def find_children_by_coordinator_run(
        self, *, coordinator_run_id: str, parent_session_id: str,
    ) -> List[Session]:
        """C2 PR-7 §12.3 — rehydrate query.

        Return every child session row whose
        ``(coordinator_run_id, parent_session_id)`` matches the given pair,
        ordered by ``created_at`` (stable across pod restarts).

        Used by ``CoordinatorRehydrateService.detect_existing_run`` to
        reconstruct the (wu_id → child_session_id) map after a pod crash.
        The query is intentionally scoped to a single parent (not just
        ``coordinator_run_id``) to defeat the theoretical case where two
        unrelated parents could collide on a manually-crafted run_id; the
        live run_id format ``"{session}:{hash16}:a{N}"`` already includes
        the parent session, but the explicit predicate is a defense-in-depth
        guard for that invariant.

        Returns an empty list (NOT None) when no children exist — the
        rehydrate service treats ``[]`` as "first-time dispatch, no resume
        needed".
        """
        ...
