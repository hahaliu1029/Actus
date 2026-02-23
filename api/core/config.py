from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用程序的配置设置，继承自Pydantic的BaseSettings。从.env或者环境变量中加载配置。"""

    # 项目基础配置
    env: str = "development"  # 应用环境，默认为'development'
    log_level: str = "INFO"  # 日志级别，默认为'INFO'
    app_config_filepath: str = "config.yaml"  # 应用配置文件路径

    # 数据库配置
    sqlalchemy_database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/manus"
    )

    # Redis缓存配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # 请求限流配置
    rate_limit_window_seconds: int = 60
    rate_limit_read_per_minute: int = 120
    rate_limit_write_per_minute: int = 60
    rate_limit_chat_per_minute: int = 60
    rate_limit_sse_concurrent: int = 10
    rate_limit_ws_concurrent: int = 5
    rate_limit_connection_ttl_seconds: int = 120
    rate_limit_heartbeat_seconds: int = 30

    # MinIO对象存储配置
    minio_endpoint: str = "s3.example.com"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_region: str | None = None
    minio_secure: bool = True
    minio_bucket_name: str = "a2a-mcp"

    # Sandbox配置
    sandbox_address: Optional[str] = None
    sandbox_image: Optional[str] = None
    sandbox_name_prefix: Optional[str] = None
    sandbox_ttl_minutes: Optional[int] = 60
    sandbox_network: Optional[str] = None
    sandbox_chrome_args: Optional[str] = ""
    sandbox_https_proxy: Optional[str] = None
    sandbox_http_proxy: Optional[str] = None
    sandbox_no_proxy: Optional[str] = None
    container_timezone: str = "UTC"

    # Skill v2 配置
    skills_root_dir: str = "/app/data/skills"
    skill_sandbox_bundle_root: str = "/home/ubuntu/workspace/.skills"
    skill_backend: str = "filesystem"
    skill_blocked_command_patterns: str = "rm -rf,:(){,mkfs.,shutdown,reboot"

    # JWT 配置
    jwt_secret_key: str = "change-me-in-env"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # 微信公众号配置
    wechat_app_id: str = ""
    wechat_app_secret: str = ""
    wechat_redirect_uri: str = ""  # 微信授权后回调地址
    wechat_frontend_redirect_uri: str = ""  # 前端接收 token 的页面地址

    # 使用pydantic v2的写法来完成环境变量信息的告知
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """获取应用程序的配置设置实例，使用lru_cache进行缓存以提高性能。

    Returns:
        Settings: 应用程序的配置设置实例。
    """
    return Settings()


# 全局配置实例
settings = get_settings()
