import logging
from functools import lru_cache

from core.config import Settings, get_settings
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis客户端封装类，用于完成redis的连接和基本操作"""

    def __init__(self):
        """构造函数，完成redis客户端的初始化"""
        self._client: Redis | None = None
        self._settings: Settings = get_settings()

    async def init(self) -> None:
        """初始化Redis客户端连接"""
        if self._client:
            logger.warning("Redis客户端已初始化，跳过重复初始化。")
            return

        try:
            self._client = Redis(
                host=self._settings.redis_host,
                port=self._settings.redis_port,
                db=self._settings.redis_db,
                password=self._settings.redis_password,
                decode_responses=True,  # 自动解码响应内容为字符串
            )
            # 测试连接
            await self._client.ping()
            logger.info("Redis客户端初始化成功。")
        except Exception as e:
            logger.error(f"Redis客户端初始化失败: {e}")
            raise

    async def shutdown(self) -> None:
        """关闭Redis客户端连接"""
        if self._client:
            # 使用 aclose() 代替已弃用的 close()
            await self._client.aclose()
            logger.info("Redis客户端连接已关闭.")
        else:
            logger.warning("Redis客户端未初始化，无法关闭连接.")
        self._client = None

        get_redis.cache_clear()

    @property
    def client(self) -> Redis:
        """获取Redis客户端实例

        Returns:
            Redis: Redis客户端实例
        """
        if not self._client:
            raise RuntimeError("Redis客户端未初始化，请先调用init方法进行初始化。")
        return self._client


@lru_cache()
def get_redis() -> RedisClient:
    """获取Redis客户端实例"""
    return RedisClient()
