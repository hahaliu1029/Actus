# import logging
# from functools import lru_cache

# from app.infrastructure.repositories.db_session_repository import DBSessionRepository
# from app.infrastructure.storage.postgres import get_db_session
# from fastapi import Depends
# from sqlalchemy.ext.asyncio import AsyncSession

# logger = logging.getLogger(__name__)


# @lru_cache()
# def get_db_session_repository(
#     db_session: AsyncSession = Depends(get_db_session),
# ) -> DBSessionRepository:
#     """基于数据库的会话数据仓库"""
#     logger.info("加载获取DBSessionRepository")
#     return DBSessionRepository(db_session=db_session)
