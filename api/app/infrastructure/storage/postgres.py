import asyncio
import logging
from functools import lru_cache
from typing import AsyncGenerator, Optional

from app.domain.repositories.uow import IUnitOfWork
from app.infrastructure.repositories.db_uow import DBUnitOfWork
from core.config import get_settings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)


class Postgres:
    """Postgres数据库客户端封装类，用于完成Postgres的连接和基本操作"""

    def __init__(self):
        """构造函数，完成Postgres数据库引擎,会话工厂的创建"""
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._settings = get_settings()

    async def init(self) -> None:
        """初始化Postgres数据库连接"""
        # 1. 判断是否已经初始化
        if self._engine is not None:
            logger.warning("Postgres数据库客户端已初始化，跳过重复初始化。")
            return
        # 2. 创建数据库引擎
        try:
            logger.info("正在初始化Postgres数据库客户端...")
            self._engine = create_async_engine(
                self._settings.sqlalchemy_database_url,
                echo=True if self._settings.env == "development" else False,
                pool_pre_ping=True,  # 每次从连接池获取连接前先检测连接是否有效，防止使用已关闭的连接
            )

            # 3. 创建会话工厂
            self._session_factory = async_sessionmaker(
                autocommit=False,  # 禁用自动提交
                autoflush=False,  # 禁用自动刷新
                bind=self._engine,
            )
            logger.info("Postgres数据库客户端初始化成功。")

            async with self._engine.begin() as async_conn:
                # 5.检查是否安装了uuid扩展，如果没有的话则安装
                await async_conn.execute(
                    text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
                )
                logger.info("Postgres数据库uuid-ossp扩展检查/安装完成。")
        except Exception as e:
            logger.error(f"Postgres数据库客户端初始化失败: {e}")
            raise

    async def shutdown(self) -> None:
        """关闭Postgres数据库连接"""
        if self._engine:
            await self._engine.dispose()
            logger.info("Postgres数据库客户端连接已关闭.")
        else:
            logger.warning("Postgres数据库客户端未初始化，无法关闭连接.")
        self._engine = None
        self._session_factory = None

        get_postgres.cache_clear()

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """获取Postgres数据库会话工厂

        Returns:
            async_sessionmaker[AsyncSession]: Postgres数据库会话工厂
        """
        if not self._session_factory:
            raise RuntimeError(
                "Postgres数据库客户端未初始化，请先调用init方法进行初始化。"
            )
        return self._session_factory


@lru_cache()
def get_postgres() -> Postgres:
    """获取获取Postgres实例"""
    return Postgres()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """获取Postgres数据库会话，用于依赖注入

    Yields:
        AsyncSession: Postgres数据库会话
    """
    # 1.获取引擎和会话工厂
    db = get_postgres()
    session_factory = db.session_factory

    # 2.创建会话并在finally中显式关闭，避免请求取消时连接未及时归还
    session = session_factory()
    try:
        yield session
    except asyncio.CancelledError:
        # 客户端中断/SSE断连时，尽量回滚后继续抛出取消
        try:
            await asyncio.shield(session.rollback())
        except Exception:
            logger.warning("请求取消后回滚数据库事务失败", exc_info=True)
        raise
    except Exception:
        await session.rollback()
        raise
    finally:
        try:
            await asyncio.shield(session.close())
        except Exception:
            logger.warning("关闭数据库会话失败", exc_info=True)


def get_session_factory():
    """获取数据库会话工厂"""
    db = get_postgres()
    return db.session_factory


def get_uow() -> IUnitOfWork:
    return DBUnitOfWork(session_factory=get_session_factory())
