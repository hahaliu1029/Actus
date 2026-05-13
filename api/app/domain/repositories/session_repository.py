from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Protocol

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
