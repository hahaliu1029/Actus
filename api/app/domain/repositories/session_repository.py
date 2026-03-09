from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Protocol

from app.domain.models.event import BaseEvent
from app.domain.models.file import File
from app.domain.models.memory import Memory
from app.domain.models.session import Session, SessionStatus

if TYPE_CHECKING:
    from app.domain.models.conversation_summary import ConversationSummary
    from app.domain.models.skill_creation_state import SkillCreationState
    from app.domain.models.skill_graph_state import SkillGraphState


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
        """根据传递的会话id更新会话状态"""
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
