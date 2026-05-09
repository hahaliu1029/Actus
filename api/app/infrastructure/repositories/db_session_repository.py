import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.domain.models.event import BaseEvent
from app.domain.models.file import File
from app.domain.models.memory import Memory
from app.domain.models.session import Session, SessionStatus
from app.domain.models.skill_creation_state import SkillCreationState
from app.domain.models.skill_graph_state import SkillGraphState
from app.domain.repositories._sentinel import _UNSET, UnsetType
from app.domain.repositories.session_repository import BgSessionRow, SessionRepository
from app.infrastructure.models import SessionModel
from pydantic import ValidationError
from sqlalchemy import cast, delete, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _strip_null_bytes(data: Any) -> Any:
    """递归移除数据中的 \\u0000 空字节，PostgreSQL的jsonb/text类型不支持该字符"""
    if isinstance(data, str):
        return data.replace("\x00", "")
    if isinstance(data, dict):
        return {k: _strip_null_bytes(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_strip_null_bytes(item) for item in data]
    return data

_SKILL_CREATION_STATE_KEY = "__skill_creation_state_v1"
_SKILL_CREATION_STATE_LEGACY_KEY = "_skill_creator"
_SKILL_GRAPH_STATE_KEY = "_skill_graph"


class DBSessionRepository(SessionRepository):
    """基于Postgres数据库的会话仓库"""

    def __init__(self, db_session: AsyncSession) -> None:
        """构造函数，完成数据仓库的初始化"""
        self.db_session = db_session

    async def save(self, session: Session) -> None:
        """根据传递的领域模型更新或者新增会话"""
        # 1.根据id查询会话是否存在
        stmt = select(SessionModel).where(SessionModel.id == session.id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        # 2.如果会话不存在则新建会话
        if not record:
            record = SessionModel.from_domain(session)
            self.db_session.add(record)
            return

        # 3.会话存在则更新会话
        record.update_from_domain(session)

    async def get_all(self) -> List[Session]:
        """获取所有会话列表"""
        # 1.构建sql查询所有记录
        stmt = select(SessionModel).order_by(SessionModel.latest_message_at.desc())
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()

        # 2.将数据循环遍历成Session
        return [record.to_domain() for record in records]

    async def get_all_by_user(self, user_id: str) -> List[Session]:
        """根据用户ID获取会话列表"""
        # 1.构建sql查询用户记录
        stmt = (
            select(SessionModel)
            .where(SessionModel.user_id == user_id)
            .order_by(SessionModel.latest_message_at.desc())
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()

        # 2.将数据循环遍历成Session
        return [record.to_domain() for record in records]

    async def get_by_id(self, session_id: str) -> Optional[Session]:
        """根据id查询会话"""
        # 1.根据id查询会话是否存在
        stmt = select(SessionModel).where(SessionModel.id == session_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        # 2.判断会话记录是否存在并返回
        return record.to_domain() if record is not None else None

    async def get_by_id_for_update(self, session_id: str) -> Optional[Session]:
        """根据id查询会话并加行锁"""
        stmt = (
            select(SessionModel)
            .where(SessionModel.id == session_id)
            .with_for_update()
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record is not None else None

    async def delete_by_id(self, session_id: str) -> None:
        """根据传递的id删除会话"""
        # 1.构建删除语句
        stmt = delete(SessionModel).where(SessionModel.id == session_id)

        # 2.执行sql无需检查是否删除
        await self.db_session.execute(stmt)

    async def update_title(self, session_id: str, title: str) -> None:
        """更新会话标题"""
        # 1.构建更新语句并执行
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(title=title)
        )
        result = await self.db_session.execute(stmt)

        # 2.检查是否更新成功
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def update_latest_message(
        self, session_id: str, message: str, timestamp: datetime
    ) -> None:
        """更新会话最新消息"""
        # 1.构建更新语句并执行
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                latest_message=message,
                latest_message_at=timestamp,
            )
        )
        result = await self.db_session.execute(stmt)

        # 2.检查是否更新成功
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def add_event(self, session_id: str, event: BaseEvent) -> None:
        """往会话中新增事件"""
        # 1.将event序列化为json，并移除PostgreSQL不支持的\u0000空字节
        event_data = _strip_null_bytes(event.model_dump(mode="json"))

        # 2.构建原子更新语句并执行
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                events=func.coalesce(SessionModel.events, cast([], JSONB))
                + cast([event_data], JSONB),
            )
        )
        result = await self.db_session.execute(stmt)

        # 3.检查是否新增成功
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def add_file(self, session_id: str, file: File) -> None:
        """往会话中新增文件"""
        # 1.将file序列化为json
        file_data = file.model_dump(mode="json")

        # 2.构建原子更新语句并执行
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                files=func.coalesce(SessionModel.files, cast([], JSONB))
                + cast([file_data], JSONB),
            )
        )
        result = await self.db_session.execute(stmt)

        # 3.检查是否新增成功
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def remove_file(self, session_id: str, file_id: str) -> None:
        """移除会话中的指定文件"""
        # 1.查询会话记录并加锁
        stmt = (
            select(SessionModel).where(SessionModel.id == session_id).with_for_update()
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        # 2.检查会话记录是否存在
        if not record:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

        # 3.会话记录存在在，则在内存中过滤files
        if not record.files:
            return
        original_length = len(record.files)
        new_files = [file for file in record.files if file.get("id") != file_id]

        # 4.判断文件长度是否有变化
        if len(new_files) == original_length:
            return

        # 5.更新数据
        record.files = new_files

    async def get_file_by_path(self, session_id: str, filepath: str) -> Optional[File]:
        """根据文件路径获取文件信息"""
        # 1.构建语句查询文件列表
        stmt = select(SessionModel.files).where(SessionModel.id == session_id)
        result = await self.db_session.execute(stmt)
        files = result.scalar_one_or_none()

        # 2.判断是否为空，如果不存在则返回None
        if not files:
            return None

        # 3.遍历查找数据，如果最后没找到则返回空
        for file in files:
            if file.get("filepath", "") == filepath:
                return File(**file)

        return None

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        """更新会话状态"""
        if status in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT):
            raise ValueError("terminal statuses must use update_to_terminal")

        # 1.构建更新值
        values = {"status": status.value, "updated_at": datetime.now()}
        if status == SessionStatus.TAKEOVER_PENDING:
            # reopen 场景：从 completed 恢复时清空 completed_at，
            # 避免统计逻辑误判"非空即完成过"
            values["completed_at"] = None

        # 2.构建更新语句并执行
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(**values)
        )
        result = await self.db_session.execute(stmt)

        # 2.检查是否更新成功
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def find_running_background(self) -> list[BgSessionRow]:
        """Return background sessions that may need restart reconciliation."""
        stmt = (
            select(
                SessionModel.id,
                SessionModel.task_id,
                SessionModel.user_id,
                SessionModel.status,
            )
            .where(SessionModel.execution_mode == "background")
            .where(SessionModel.execution_phase.in_(("running", "recovering")))
            .where(
                SessionModel.status.in_(
                    (SessionStatus.RUNNING.value, SessionStatus.FINISHING.value)
                )
            )
        )
        result = await self.db_session.execute(stmt)
        return [
            BgSessionRow(
                session_id=str(row.id),
                task_id=str(row.task_id) if row.task_id is not None else None,
                user_id=str(row.user_id),
                status=SessionStatus(row.status),
            )
            for row in result.all()
        ]

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
        values: dict[str, object | None] = {}
        if execution_mode is not _UNSET:
            values["execution_mode"] = execution_mode
        if background_reason is not _UNSET:
            values["background_reason"] = background_reason
        if expires_at is not _UNSET:
            values["expires_at"] = expires_at
        if execution_phase is not _UNSET:
            values["execution_phase"] = execution_phase
        if retry_budget_remaining is not _UNSET:
            values["retry_budget_remaining"] = retry_budget_remaining
        if terminal_reason is not _UNSET:
            values["terminal_reason"] = terminal_reason
        if suspended_reason is not _UNSET:
            values["suspended_reason"] = suspended_reason
        if was_background is not _UNSET:
            values["was_background"] = was_background
        if not values:
            return
        values["last_activity_at"] = func.now()

        result = await self.db_session.execute(
            update(SessionModel).where(SessionModel.id == session_id).values(**values)
        )
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def update_to_terminal(
        self,
        session_id: str,
        status: SessionStatus,
        terminal_reason: str,
    ) -> None:
        if status not in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT):
            raise ValueError(f"non-terminal status: {status}")

        now = datetime.now()
        result = await self.db_session.execute(
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .where(
                ~SessionModel.status.in_(
                    (SessionStatus.COMPLETED.value, SessionStatus.TIMED_OUT.value)
                )
            )
            .where(~SessionModel.execution_phase.in_(("terminating", "terminated")))
            .values(
                status=status.value,
                completed_at=now,
                terminal_reason=terminal_reason,
                execution_phase="terminated",
                last_activity_at=now,
                updated_at=now,
            )
        )
        if result.rowcount == 0:
            existing = await self.db_session.execute(
                select(SessionModel.status, SessionModel.execution_phase).where(
                    SessionModel.id == session_id
                )
            )
            row = existing.one_or_none()
            if row is None:
                raise ValueError(f"会话[{session_id}]不存在，请核实后重试")
            if row.status in (
                SessionStatus.COMPLETED.value,
                SessionStatus.TIMED_OUT.value,
            ) or row.execution_phase in ("terminating", "terminated"):
                return
            raise ValueError(f"会话[{session_id}]终态写入失败，请重试")

    async def update_terminal_reason(
        self,
        session_id: str,
        terminal_reason: str,
    ) -> None:
        result = await self.db_session.execute(
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(terminal_reason=terminal_reason, last_activity_at=func.now())
        )
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def distinct_user_ids_with_running_bg(self) -> list[str]:
        result = await self.db_session.execute(
            select(SessionModel.user_id)
            .where(SessionModel.execution_mode == "background")
            .where(SessionModel.execution_phase.in_(("running", "suspended")))
            .where(SessionModel.status == SessionStatus.RUNNING.value)
            .distinct()
        )
        return [str(row.user_id) for row in result.all()]

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        """更新会话的未读消息数"""
        # 1.构建更新语句并执行
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(unread_message_count=count)
        )
        result = await self.db_session.execute(stmt)

        # 2.检查是否更新成功
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def increment_unread_message_count(self, session_id: str) -> None:
        """新增会话的未读消息数"""
        # 1.构建新增未读消息数语句并更新
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                unread_message_count=func.coalesce(SessionModel.unread_message_count, 0)
                + 1,
            )
        )
        result = await self.db_session.execute(stmt)

        # 2.检查是否更新成功
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def decrement_unread_message_count(self, session_id: str) -> None:
        """将会话中的未读消息数-1"""
        # 1.构建新增未读消息数语句并更新
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                # 2.核心逻辑：GREATEST((当前值-1), 0)避免出现负数
                unread_message_count=func.greatest(
                    func.coalesce(SessionModel.unread_message_count, 0) - 1, 0
                )
            )
        )
        result = await self.db_session.execute(stmt)

        # 3.检查是否更新成功
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def save_memory(
        self, session_id: str, agent_name: str, memory: Memory
    ) -> None:
        """存储或者更新会话中的记忆(字典直接覆盖)"""
        # 1.将memory转换成为json结构
        memory_data = memory.model_dump(mode="json")

        # 2.构建要打补丁的字典
        patch_data = {agent_name: memory_data}

        # 3.执行合并更新
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                memories=func.coalesce(SessionModel.memories, cast({}, JSONB))
                + cast(patch_data, JSONB),
            )
        )
        result = await self.db_session.execute(stmt)

        # 4.检查是否更新成功
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def get_memory(self, session_id: str, agent_name: str) -> Memory:
        """获取指定会话的agent记忆信息"""
        # 1.查询会话记忆信息
        stmt = select(SessionModel.memories[agent_name]).where(
            SessionModel.id == session_id
        )
        result = await self.db_session.execute(stmt)
        memory_data = result.scalar_one_or_none()

        # 2.如果存在记忆则直接返回
        if memory_data:
            return Memory(**memory_data)

        # 3.如果记忆不存在，则构建一个空记忆后返回
        return Memory(messages=[])

    async def get_summary(self, session_id: str) -> list:
        """获取会话的对话摘要列表"""
        from app.domain.models.conversation_summary import ConversationSummary

        stmt = select(SessionModel.memories["_summary"]).where(
            SessionModel.id == session_id
        )
        result = await self.db_session.execute(stmt)
        summary_data = result.scalar_one_or_none()

        if not summary_data:
            return []

        rounds = summary_data.get("rounds", [])
        return [ConversationSummary.model_validate(round) for round in rounds]

    async def save_summary(self, session_id: str, summaries: list) -> None:
        """保存会话的对话摘要列表"""
        rounds_data = [summary.model_dump(mode="json") for summary in summaries]
        patch_data = {"_summary": {"rounds": rounds_data}}

        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                memories=func.coalesce(SessionModel.memories, cast({}, JSONB))
                + cast(patch_data, JSONB),
            )
        )
        result = await self.db_session.execute(stmt)

        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def get_skill_creation_state(
        self, session_id: str
    ) -> SkillCreationState | None:
        """获取会话中的 Skill 创建等待状态"""
        stmt = select(
            SessionModel.memories[_SKILL_CREATION_STATE_KEY],
            SessionModel.memories[_SKILL_CREATION_STATE_LEGACY_KEY],
        ).where(SessionModel.id == session_id)
        result = await self.db_session.execute(stmt)
        row = result.one_or_none()
        if not row:
            return None

        state_data = row[0] or row[1]

        if not state_data:
            return None

        # Reject payloads that contain no recognized SkillCreationState fields
        if not (state_data.keys() & SkillCreationState.model_fields.keys()):
            return None

        try:
            return SkillCreationState.model_validate(state_data)
        except ValidationError as exc:
            logger.warning(
                "Skill 创建状态结构非法，忽略并按空状态处理: session_id=%s key=%s error=%s",
                session_id,
                _SKILL_CREATION_STATE_KEY if row[0] else _SKILL_CREATION_STATE_LEGACY_KEY,
                exc,
            )
            return None

    async def save_skill_creation_state(
        self, session_id: str, state: SkillCreationState
    ) -> None:
        """保存会话中的 Skill 创建等待状态"""
        patch_data = {_SKILL_CREATION_STATE_KEY: state.model_dump(mode="json")}
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                memories=func.coalesce(SessionModel.memories, cast({}, JSONB))
                + cast(patch_data, JSONB),
            )
        )
        result = await self.db_session.execute(stmt)

        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def clear_skill_creation_state(self, session_id: str) -> None:
        """清理会话中的 Skill 创建等待状态"""
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                memories=func.coalesce(SessionModel.memories, cast({}, JSONB))
                .op("-")(_SKILL_CREATION_STATE_KEY)
                .op("-")(_SKILL_CREATION_STATE_LEGACY_KEY)
            )
        )
        result = await self.db_session.execute(stmt)

        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    # -- Skill 创建子图状态 -------------------------------------------------- #

    async def get_skill_graph_state(
        self, session_id: str
    ) -> SkillGraphState | None:
        """获取会话中的 Skill 创建子图状态"""
        stmt = select(
            SessionModel.memories[_SKILL_GRAPH_STATE_KEY],
        ).where(SessionModel.id == session_id)
        result = await self.db_session.execute(stmt)
        row = result.one_or_none()
        if not row or not row[0]:
            return None

        try:
            return SkillGraphState.model_validate(row[0])
        except ValidationError as exc:
            logger.warning(
                "Skill 子图状态结构非法，忽略: session_id=%s error=%s",
                session_id,
                exc,
            )
            return None

    async def save_skill_graph_state(
        self, session_id: str, state: SkillGraphState
    ) -> None:
        """保存会话中的 Skill 创建子图状态"""
        dumped = state.model_dump(mode="json")
        patch_data = {_SKILL_GRAPH_STATE_KEY: dumped}
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                memories=func.coalesce(SessionModel.memories, cast({}, JSONB))
                + cast(patch_data, JSONB),
            )
        )
        result = await self.db_session.execute(stmt)

        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def clear_skill_graph_state(self, session_id: str) -> None:
        """清理会话中的 Skill 创建子图状态"""
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                memories=func.coalesce(SessionModel.memories, cast({}, JSONB))
                .op("-")(_SKILL_GRAPH_STATE_KEY)
            )
        )
        result = await self.db_session.execute(stmt)

        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")
