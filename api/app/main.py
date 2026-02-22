import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from alembic import command
from alembic.config import Config
from app.infrastructure.logging import setup_logging
from app.infrastructure.storage.minio import get_minio
from app.infrastructure.storage.postgres import get_postgres
from app.infrastructure.storage.redis import get_redis
from app.interfaces.endpoints.routes import router as api_router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from app.interfaces.service_dependencies import get_agent_service
from core.config import get_settings
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# 加载配置信息
settings = get_settings()

# 初始化日志记录
setup_logging()
logger = logging.getLogger()

logger.info("应用程序启动中...")

# 定义FastApi路由tags标签
openapi_tags = [
    {
        "name": "状态模块",
        "description": "包含 **状态监测** 等API 接口，用于监测系统的运行状态。",
    }
]


def _build_alembic_database_url() -> str:
    """构建 Alembic 使用的数据库连接串（同步驱动 + 连接超时）"""
    db_url = settings.sqlalchemy_database_url
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)

    parsed = urlparse(db_url)
    query = dict(parse_qsl(parsed.query))
    query.setdefault("connect_timeout", "5")
    return urlunparse(parsed._replace(query=urlencode(query)))


def _mask_database_url(url: str) -> str:
    """脱敏数据库连接串中的密码"""
    parsed = urlparse(url)
    if parsed.password is None:
        return url
    user = parsed.username or ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{user}:***@{host}{port}"
    return urlunparse(parsed._replace(netloc=netloc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """创建FastAPI应用生命周期上下文管理器"""
    # 1.日志打印代码已经开始执行了
    logger.info("Manus应用正在初始化")

    # 2.运行数据库迁移(将数据同步到生产环境)
    _api_root = Path(__file__).resolve().parent.parent
    alembic_cfg = Config(str(_api_root / "alembic.ini"))
    alembic_db_url = _build_alembic_database_url()
    alembic_cfg.set_main_option("sqlalchemy.url", alembic_db_url)
    logger.info(f"数据库迁移开始，连接地址: {_mask_database_url(alembic_db_url)}")
    command.upgrade(alembic_cfg, "head")
    logger.info("数据库迁移完成")

    # 3.初始化Redis/Postgres/Cos客户端

    # 2. 初始化Redis客户端
    logger.info("开始初始化 Redis 客户端")
    redis_client = get_redis()
    await redis_client.init()
    logger.info("Redis 客户端初始化完成")

    # 3. 初始化Postgres数据库客户端
    logger.info("开始初始化 Postgres 客户端")
    postgres_client = get_postgres()
    await postgres_client.init()
    logger.info("Postgres 客户端初始化完成")

    # 4. 初始化MinIO对象存储客户端
    logger.info("开始初始化 MinIO 客户端")
    minio_client = get_minio()
    await minio_client.init()
    logger.info("MinIO 客户端初始化完成")

    try:
        # 3.lifespan分界点
        yield
    finally:
        try:
            # 4.等待agent服务关闭
            logger.info("Manus应用正在关闭")
            await asyncio.wait_for(get_agent_service().shutdown(), timeout=30.0)
            logger.info("Agent服务成功关闭")
        except asyncio.TimeoutError:
            logger.warning("Agent服务关闭超时, 强制关闭, 部分任务将被释放")
        except Exception as e:
            logger.error(f"Agent服务关闭期间出现错误: {str(e)}")

        # 5. 应用关闭前的清理工作
        await redis_client.shutdown()
        await postgres_client.shutdown()
        await minio_client.shutdown()
        logger.info("Manus应用关闭成功")


app = FastAPI(
    title="Actus通用智能体",
    description="Actus是一个通用的AI Agent系统，可以完全私有部署，使用A2A+MCP连接Agent/Tool，同时支持在沙箱中运行各种内置工具和操作",
    lifespan=lifespan,
    openapi_tags=openapi_tags,
    version="1.0.0",
)

# 配置CORS中间件，解决跨域问题

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 注册全局异常处理器
register_exception_handlers(app)

app.include_router(api_router, prefix="/api")

logger.info("FastAPI应用程序实例已创建。")
