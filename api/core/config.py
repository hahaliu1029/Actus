import logging
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """应用程序的配置设置，继承自Pydantic的BaseSettings。从.env或者环境变量中加载配置。"""

    # 项目基础配置
    env: str = "development"  # 应用环境，默认为'development'
    log_level: str = "INFO"  # 日志级别，默认为'INFO'
    app_config_filepath: str = "config.yaml"  # 应用配置文件路径
    cors_origins: str = "http://localhost:3000,http://localhost"  # 逗号分隔的允许来源列表
    max_request_body_size: int = 500 * 1024 * 1024  # 500MB 请求体限制

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
    rate_limit_read_per_minute: int = 600
    rate_limit_write_per_minute: int = 300
    rate_limit_chat_per_minute: int = 300
    rate_limit_sse_concurrent: int = 20
    rate_limit_ws_concurrent: int = 10
    rate_limit_connection_ttl_seconds: int = 120
    rate_limit_heartbeat_seconds: int = 30
    rate_limit_auth_per_minute: int = 10  # 认证端点 IP 限流
    rate_limit_trust_proxy: bool = False  # 反代部署时设为 True，从 X-Forwarded-For 取真实 IP

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
    sandbox_mem_limit: str = "4g"  # Docker 容器内存上限，防止 Chromium OOM
    sandbox_https_proxy: Optional[str] = None
    sandbox_http_proxy: Optional[str] = None
    sandbox_no_proxy: Optional[str] = None
    sandbox_default_cwd: str = "/root"
    container_timezone: str = "UTC"

    # Skill v2 配置
    skills_root_dir: str = "/app/data/skills"
    skill_sandbox_bundle_root: str = "/home/ubuntu/workspace/.skills"
    skill_backend: str = "filesystem"
    skill_blocked_command_patterns: str = "rm -rf,:(){,mkfs.,shutdown,reboot"
    github_token: str = ""

    # Session 接管配置
    feature_takeover_enabled: bool = True
    feature_takeover_browser_enabled: bool = True
    feature_takeover_allowed_roles: str = "super_admin,user"
    feature_takeover_user_whitelist: str = ""
    feature_takeover_single_worker_only: bool = True
    feature_takeover_pending_ttl_seconds: int = 300
    feature_takeover_lease_ttl_seconds: int = 900
    feature_takeover_reopen_window_seconds: int = 300
    feature_takeover_lease_guard_interval_seconds: int = 15

    # 危险工具确认配置
    tool_confirmation_timeout_seconds: int = 300
    smart_approve_enabled: bool = False

    # Skill 创建子图灰度配置
    skill_graph_canary_percent: int = 100  # 0-100，按 user_id 哈希分桶

    # B5 C11: Prompt telemetry log directory
    # JsonlPromptTelemetry writes two JSONL files here:
    # - assembly.jsonl (PromptAssembler.assemble events)
    # - llm_invocation.jsonl (per-LLM-call metadata from the adapter hook)
    # Consumed by B5.5 for caching-viability analysis (stable system_prompt
    # and tools across a session → cache_control is worth enabling).
    prompt_telemetry_log_dir: str = "/app/data/telemetry/prompt"

    # Checkpointer 连接池配置
    checkpointer_pool_min_size: int = 2
    checkpointer_pool_max_size: int = 10
    checkpointer_pool_timeout: float = 30.0

    # Config 缓存 TTL (秒)
    config_cache_ttl: int = 60

    # JWT 配置
    jwt_secret_key: str = "change-me-in-env"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # Memory 系统（M1）——文件挂载、LLM gate、用户配额
    # 三条路径彼此独立、含义不同：
    # - ``memory_root_host``：**宿主机**上 memory 根目录（docker bind source）
    # - ``memory_root_container``：**api 容器**视角下 memory 根目录（用于 mkdir
    #   创建用户子目录；需要 docker-compose bind 把它映射到 memory_root_host，
    #   PR-6 落地）
    # - ``sandbox_memory_mount_target``：**sandbox 容器**视角下的挂载点，由
    #   M0 spike 选定为 ``/workspace/.memory``；所有 agent 工具在沙箱里按这个
    #   固定路径读取 memory 文件
    # 见 docs/superpowers/specs/2026-04-17-m0-sandbox-memory-mount-spike.md
    memory_root_host: str = Field("~/.actus/memory")
    memory_root_container: str = Field("/app/data/memory")
    sandbox_memory_mount_target: str = Field("/workspace/.memory")
    # PR-0 默认关闭——只有当 docker-compose 已把 memory_root_host 正确 bind 到
    # memory_root_container 时（PR-6），才应打开本开关实际挂载 memory 目录；
    # 否则 Docker 会因为 bind source 不存在而拒绝启动 sandbox。
    sandbox_memory_mount_enabled: bool = Field(False)
    actus_uid: int = Field(1000, ge=0)
    actus_gid: int = Field(1000, ge=0)
    # LLM 质量 gate：None 表示 gate 未启用（PR-0 默认关闭）。
    # 部署时应显式设置为 ``summary_llm`` / ``chat_llm`` 之一。
    memory_gate_llm: str | None = Field(None)
    memory_gate_threshold: float = Field(0.7, ge=0.0, le=1.0)
    # 单次 gate 批量评估的最大候选数。
    memory_gate_batch_cap: int = Field(20, gt=0)
    # **per-user** 每日 auto-promote（gate 通过）的上限。与下方 user_daily_quota
    # 正交：gate cap 限速 gate-approved 写，user quota 限速所有 memory 写入。
    memory_gate_daily_cap: int = Field(100, gt=0)
    # per-user 每日任意写入上限（含手动 + gate + 自动）。
    memory_user_daily_quota: int = Field(500, gt=0)
    # per-session memory_save 工具硬上限——防止 Agent 在一次任务内频繁保存。
    # 与 user_daily_quota 正交：前者防单 session flood，后者防跨 session 累积。
    memory_session_save_cap: int = Field(20, gt=0)

    # 微信公众号配置
    wechat_app_id: str = ""
    wechat_app_secret: str = ""
    wechat_redirect_uri: str = ""  # 微信授权后回调地址
    wechat_frontend_redirect_uri: str = ""  # 前端接收 token 的页面地址

    # 使用pydantic v2的写法来完成环境变量信息的告知
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("memory_root_host", "memory_root_container")
    @classmethod
    def _must_be_absolute(cls, v: str) -> str:
        """memory_root 必须是绝对路径（``/``）或家目录展开前缀（``~``）。

        相对路径在 Docker bind mount source 处会被 docker daemon 解释为卷名，
        后续写入 ``{root}/{user_id}`` 会在容器外拼错路径。
        """
        if not (v.startswith("/") or v.startswith("~")):
            raise ValueError(f"{v!r} 必须是绝对路径（以 / 或 ~ 开头）")
        return v

    @model_validator(mode="after")
    def _reject_default_jwt_secret(self) -> "Settings":
        if self.jwt_secret_key == "change-me-in-env" and self.env != "test":
            raise ValueError(
                "JWT_SECRET_KEY 仍为默认值 'change-me-in-env'，"
                "请在 .env 或环境变量中设置一个安全的随机密钥"
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """获取应用程序的配置设置实例，使用lru_cache进行缓存以提高性能。

    Returns:
        Settings: 应用程序的配置设置实例。
    """
    return Settings()


# 全局配置实例
settings = get_settings()
