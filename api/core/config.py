import logging
import socket
from datetime import datetime
from functools import lru_cache
from typing import Optional

from pydantic import (
    AliasChoices,
    AwareDatetime,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Source the Settings defaults from the canonical spec constants so changing
# spec §4.3 in one place propagates to both the in-process consumer loop AND
# the env-var-exposed defaults. Codex r8 [P2] fix — prior version hard-coded
# 1000 / 32 here, creating silent drift if the spec constants moved.
from app.domain.models.mailbox_envelope import (
    MAILBOX_XREADGROUP_BLOCK_MS as _MAILBOX_XREADGROUP_BLOCK_MS,
    MAILBOX_XREADGROUP_COUNT as _MAILBOX_XREADGROUP_COUNT,
)

_logger = logging.getLogger(__name__)


class SubagentLimitsConfig(BaseSettings):
    """C1a spawn caps. Loaded once from env via ACTUS_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="ACTUS_",
        extra="ignore",
    )

    max_subagent_depth: int = Field(default=1, ge=1, le=2)
    max_descendants_per_root: int = Field(default=10, ge=1, le=200)


class Settings(BaseSettings):
    """应用程序的配置设置，继承自Pydantic的BaseSettings。从.env或者环境变量中加载配置。"""

    # 项目基础配置
    env: str = "development"  # 应用环境，默认为'development'
    log_level: str = "INFO"  # 日志级别，默认为'INFO'
    # B5 PR-S1-4 (A5): rotating file handler 输出目录。
    # docker 路径 ``/app/data/logs`` 经 ``${LOG_ROOT_HOST}`` bind mount 到
    # 宿主，沿用 ``MEMORY_ROOT_HOST`` 模式（UID 1000 ownership + bind）。
    # 本地 dev/CI 没有该路径时，``_install_file_handlers`` 会捕获 OSError
    # 并降级到仅 stdout，stdout handler 永远会装上。
    log_dir: str = "/app/data/logs"
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
    # NOTE (N3 follow-up): this default (/root) drifts from the real shell
    # default (~ = /home/ubuntu) and the file workspace root (/home/ubuntu).
    # The Sandbox Workspace Isolation epic anchors relative FILE/SHELL paths to
    # /home/ubuntu but deliberately does NOT touch this AST-validator default
    # (security-sensitive component). Reconcile in a separate follow-up.
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

    # C5a Sandbox Policy Compiler — observe-only dark-launch. Default OFF:
    # when OFF the two enforcement seams skip compiler + sink entirely
    # (byte-identical, INV-0). The field name itself MUST be an alias
    # (validation_alias replaces the field name as a source), else
    # Settings(sandbox_policy_compiler_enabled=True) is ignored (R5#5).
    sandbox_policy_compiler_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "sandbox_policy_compiler_enabled",
            "SANDBOX_POLICY_COMPILER_ENABLED",
            "ACTUS_C5_SANDBOX_POLICY_COMPILER_ENABLED",
        ),
    )

    # Skill 创建子图灰度配置
    skill_graph_canary_percent: int = 100  # 0-100，按 user_id 哈希分桶

    # B5 C11: Prompt telemetry log directory
    # JsonlPromptTelemetry writes two JSONL files here:
    # - assembly.jsonl (PromptAssembler.assemble events)
    # - llm_invocation.jsonl (per-LLM-call metadata from the adapter hook)
    # Consumed by B5.5 for caching-viability analysis (stable system_prompt
    # and tools across a session → cache_control is worth enabling).
    prompt_telemetry_log_dir: str = "/app/data/telemetry/prompt"

    # B5 PR-S1-1 (A5): salt for hashing user_id when emitting telemetry.
    # The canonical attribute ``user_id_hash`` is computed as
    # ``sha256(salt + user_id).hexdigest()[:16]`` so downstream observability
    # storage holds a stable but non-reversible identifier.
    # Empty string DISABLES hashing — ``user_id_hash`` falls back to ``None``
    # in the canonical attributes, which is contract-legal (the field is
    # nullable). Production deployments should set a long random secret
    # (e.g., 32-byte hex from ``secrets.token_hex(32)``) so the same user_id
    # produces the same hash across pod restarts but cannot be reversed via
    # rainbow-table attack on the UUID space.
    user_id_hash_salt: str = ""

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
    # - ``memory_root_host``：**宿主机**上 memory 根目录（docker bind source，
    #   必须是绝对路径；不要使用 ``~``）
    # - ``memory_root_container``：**api 容器**视角下 memory 根目录（用于 mkdir
    #   创建用户子目录；需要 docker-compose bind 把它映射到 memory_root_host，
    #   PR-6 落地）
    # - ``sandbox_memory_mount_target``：**sandbox 容器**视角下的挂载点，由
    #   M0 spike 选定为 ``/workspace/.memory``；所有 agent 工具在沙箱里按这个
    #   固定路径读取 memory 文件
    # 见 docs/superpowers/specs/2026-04-17-m0-sandbox-memory-mount-spike.md
    # 仅作为非 compose / 测试环境下的保守 fallback；docker-compose.yml 已要求
    # 显式提供 ``MEMORY_ROOT_HOST`` 绝对路径，避免 ``~`` 在容器里误展开成
    # ``/root/...`` 后再传给宿主机 docker daemon。
    memory_root_host: str = Field("/tmp/actus-memory")
    memory_root_container: str = Field("/app/data/memory")
    sandbox_memory_mount_target: str = Field("/workspace/.memory")
    # M1 PR-6A 起默认 True：docker-compose.yml 已把 memory_root_host bind 到
    # memory_root_container，首次 session 创建时 DockerSandbox._build_memory_mount
    # 会在 api 容器内 mkdir 出 user 子目录，再把对应 host 路径只读挂进 sandbox。
    # 若部署时 host 侧 MEMORY_ROOT_HOST 路径/权限未就绪，可临时置 False 让 agent
    # 降级走 memory_search 路径（file_read 不可用）。
    sandbox_memory_mount_enabled: bool = Field(True)
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
    # M3-A codex fix P1：LLM gate 启用的时间边界（ISO 8601 **timezone-aware**）。
    # `DELETE /v2/memories/legacy` 用 ``source='session_flush' AND category IS
    # NULL AND auto_promoted_at IS NULL`` 作为"未经 gate 收录"的推断；但 gate
    # 关闭（memory_gate_llm=None）的 deployment 里 **新** 写入的 session_flush
    # 也满足这组谓词，会被误删。设置本字段后 SQL 额外加 ``AND created_at <
    # rollout_at``，把"上线前"语义显式编码。未设 = 保持旧行为 + 前端警告
    # （由调用方负责）。
    #
    # 类型 ``AwareDatetime``（codex fix P1 round-2）：强制 ``tzinfo`` 非空，
    # 拒绝裸 naive ``"2026-04-01T00:00:00"``。destructive delete 的时间边界
    # 如果按宿主机 local tz 解释，跨时区部署节点会产生不同的 cutoff，删错
    # 行的风险写不能留。合法输入：``"2026-04-01T00:00:00Z"`` /
    # ``"2026-04-01T08:00:00+08:00"``。
    memory_gate_rollout_at: AwareDatetime | None = Field(None)

    # 微信公众号配置
    wechat_app_id: str = ""
    wechat_app_secret: str = ""
    wechat_redirect_uri: str = ""  # 微信授权后回调地址
    wechat_frontend_redirect_uri: str = ""  # 前端接收 token 的页面地址

    # B5 PR-S2-1 / PR-S3-1: OTel SDK 出口路由。
    # 默认两条都为空字符串 = 装载 SDK 但不外发、不本地输出（spec line 436）。
    # ``otel_exporter="stdout"`` 显式 opt-in：dev 调试场景下 ConsoleSpan/Log/
    # Metric exporter 写本地 stdout，仍是零外发。
    # ``otlp_endpoint`` 非空（PR-S3-1 起）= OTLP 出口；BatchSpanProcessor /
    # BatchLogRecordProcessor / PeriodicExportingMetricReader wrap 各自
    # 的 OTLP exporter（``opentelemetry-exporter-otlp``）把数据发到
    # ``OTLP_ENDPOINT``。Phoenix / Jaeger / Loki+Prometheus 都支持 OTLP，
    # 同一份配置可以接不同后端。
    # ``otlp_protocol`` 选 ``"http/protobuf"``（4318 端口，Phoenix 默认）或
    # ``"grpc"``（4317 端口，OpenTelemetry Collector 默认）。
    otlp_endpoint: str = ""
    otel_exporter: str = ""
    otlp_protocol: str = "http/protobuf"

    # B5 PR-S3-3: Prometheus scrape 端点 token。
    # 空 = 端点禁用（GET /api/v1/metrics → 404，PrometheusMetricReader
    # 不挂上 MeterProvider），零额外内存 + 零暴露面。
    # 非空 = 端点开启，要求 ``Authorization: Bearer <token>``（constant-time
    # 比对）；PrometheusMetricReader 注册到 OTel MeterProvider，把所有
    # OtelMeter / OtelLLMMetricsCallback 写入的 instrument 暴露成
    # Prometheus exposition format（``text/plain; version=0.0.4``）。
    # 仅供内部 Prometheus / VictoriaMetrics 拉取，不暴露给终端用户。
    # Accept BOTH the canonical unprefixed name AND the ``ACTUS_``-prefixed name.
    # ``Settings`` has no ``env_prefix``, so the field's native env var is the
    # unprefixed ``METRICS_ENDPOINT_TOKEN`` (used by the endpoint/observability
    # tests). The C2 coordinator runbooks + perf CLI prescribe
    # ``ACTUS_METRICS_ENDPOINT_TOKEN`` for consistency with
    # ``ACTUS_C2_COORDINATOR_ENABLED`` / ``ACTUS_COORDINATOR_*``. Without the
    # alias the prefixed name was silently ignored → empty token → /api/v1/metrics
    # 404 → silent empty scrape + perf-CLI ``raise_for_status`` crash.
    metrics_endpoint_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "METRICS_ENDPOINT_TOKEN", "ACTUS_METRICS_ENDPOINT_TOKEN"
        ),
    )

    # C1a: subagent spawn caps. Nested config so the same env_prefix=ACTUS_
    # surface stays consistent regardless of how it's accessed.
    subagent_limits: SubagentLimitsConfig = Field(default_factory=SubagentLimitsConfig)

    # ─── C3 Mailbox Supervisor (PR-3a..PR-6) ──────────────────────────────
    # `mailbox_supervisor_enabled` was the deployment-time feature flag used
    # during the C3 mailbox rollout. PR-6 (spec §11.7) retires the legacy
    # plane: SessionService now unconditionally writes
    # ``subagent_control_plane='mailbox'`` regardless of this flag's value,
    # and main.py always builds the SupervisorRegistry. The flag default is
    # True so any existing wiring that still reads it (e.g.,
    # ``AgentTaskRunner._mailbox_supervisor_enabled``) keeps mailbox
    # behavior; setting it to False is no longer a supported rollback path
    # (the §11.6 runbook is decommissioned by the PR-6 alembic migration
    # ``c3pr6_retire_legacy_ctrl_plane`` that rewrites every existing
    # ``subagent_control_plane='legacy'`` row to ``'mailbox'``). Kept as a
    # Settings attribute purely for back-compat with test fixtures that
    # pass an explicit Settings stub.
    # `mailbox_pod_id` is the consumer-group consumer-name prefix. Empty value
    # degrades to socket.gethostname() via resolve_mailbox_pod_id() — for
    # production prefer injecting the k8s downward-API pod name.
    # `mailbox_xreadgroup_block_ms` / `mailbox_xreadgroup_count` mirror the
    # spec §4.3 constants (1000ms, batch=32); kept here only so ops can tune
    # without code changes. Unit tests pass block_ms=0 explicitly because
    # fakeredis async XREADGROUP doesn't wake on a concurrent XADD.
    mailbox_supervisor_enabled: bool = True
    mailbox_pod_id: str = ""
    mailbox_xreadgroup_block_ms: int = Field(_MAILBOX_XREADGROUP_BLOCK_MS, gt=0)
    mailbox_xreadgroup_count: int = Field(_MAILBOX_XREADGROUP_COUNT, gt=0)

    # 使用pydantic v2的写法来完成环境变量信息的告知
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("memory_root_host")
    @classmethod
    def _host_root_must_be_absolute(cls, v: str) -> str:
        """宿主机 memory_root 必须是绝对路径。

        ``memory_root_host`` 会被 Docker daemon 当作 bind mount source 解释；
        若写成 ``~/.actus/memory``，api 容器内 ``expanduser()`` 会把它误展开成
        ``/root/.actus/memory``，最终指向错误的宿主机路径。
        """
        if not v.startswith("/"):
            raise ValueError(
                f"{v!r} 不是宿主机绝对路径。"
                "memory_root_host 必须以 / 开头，不能使用 ~"
            )
        return v

    @field_validator("memory_root_container")
    @classmethod
    def _container_root_must_be_absolute(cls, v: str) -> str:
        """api 容器内 memory_root 也要求绝对路径，避免被当成相对目录。"""
        if not v.startswith("/"):
            raise ValueError(f"{v!r} 必须是绝对路径（以 / 开头）")
        return v

    @model_validator(mode="after")
    def _reject_default_jwt_secret(self) -> "Settings":
        if self.jwt_secret_key == "change-me-in-env" and self.env != "test":
            raise ValueError(
                "JWT_SECRET_KEY 仍为默认值 'change-me-in-env'，"
                "请在 .env 或环境变量中设置一个安全的随机密钥"
            )
        return self


def resolve_mailbox_pod_id(configured: str) -> str:
    """Resolve the effective MailboxSupervisor pod_id.

    Empty / whitespace-only ``configured`` degrades to ``socket.gethostname()``.
    PR-3b's XAUTOCLAIM uses pod_id to detect cross-pod PEL ownership; an empty
    string would collapse all pods into one consumer-name space and break
    crash-recovery detection, so the hostname fallback is the safe default.
    """
    if configured and configured.strip():
        return configured
    return socket.gethostname()


@lru_cache()
def get_settings() -> Settings:
    """获取应用程序的配置设置实例，使用lru_cache进行缓存以提高性能。

    Returns:
        Settings: 应用程序的配置设置实例。
    """
    return Settings()


# 全局配置实例
settings = get_settings()
