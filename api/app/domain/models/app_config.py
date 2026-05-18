import uuid
from enum import Enum
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class LLMConfig(BaseModel):
    """LLM提供商配置"""

    base_url: HttpUrl = "https://api.deepseek.com"  # 模型基础URL地址
    api_key: str = ""  # 模型API秘钥
    model_name: str = (
        "deepseek-reasoner"  # 模型名字，默认使用deepseek-reasoner带推理的模型，传递tools会自动切换到deepseek-chat
    )
    provider: str | None = None
    """A7: provider_id (see provider_profiles/ constants: kimi_k2 / kimi_k2_6 /
    deepseek_chat / deepseek_reasoner / dashscope_qwen / dashscope_qwen_vl /
    anthropic_compat / gemini_compat / minimax / glm / openai_official /
    generic_openai).

    A7 P1 (2026-04-23): DashScope 拆 text / vl 两个 profile；当 base_url 命中
    DashScope / Anthropic / Gemini 但 model_name 不在 allowlist (例如
    qwen-max / qwen-vl-max / qwen3.5-plus / claude-opus-4-7 / gemini-2.5-pro /
    gemini-3-*) 时，heuristic 直接返回 generic_openai (= 今天行为)。显式
    provider=<registered_id> 会绕过 allowlist 验证但不保证 profile 字段与真实
    模型能力匹配，仅建议在用户明确对齐时使用；typo 直接抛 ConfigError.

    Strictly separate from ActusChatModel.provider_name (B5 C0a, prompt rendering,
    Literal['openai','anthropic']). The two fields do not infer from each other,
    do not share a value space, and do not override each other (A7 C1)."""
    temperature: float = Field(0.7)  # 温度，默认设置为0.7
    max_tokens: int = Field(
        8192, ge=0
    )  # 最大输出token数，默认设置为deepseek-chat模型的最大输出限制
    timeout_seconds: float = Field(
        120.0,
        ge=0,
        le=3600,
        description=(
            "Per-call LLM hard timeout in seconds. Default 120s is an "
            "assumption — no production wall-time data backs it; adjust "
            "after wall-time sampling (see TODOS #24 B5.5 bench). "
            "0 disables the per-call wrap (use sparingly for debugging). "
            "Wrapped by asyncio.wait_for in each adapter's _agenerate/"
            "_astream; TimeoutError is converted to ServerRequestsError "
            "so LangGraph RetryPolicy handles retries. Also passed through "
            "to httpx as the read/write/pool timeout ceiling; see "
            "connect_timeout_seconds for the TCP+TLS handshake budget."
        ),
    )
    connect_timeout_seconds: float = Field(
        60.0,
        ge=1.0,
        le=300.0,
        description=(
            "httpx connect-phase timeout in seconds (TCP establish + TLS "
            "handshake). Separate from timeout_seconds because 'don't wait "
            "300s for a dead endpoint' is a different SLO from 'don't wait "
            "300s for a model to finish generating'. The OpenAI SDK default "
            "is 5s which is too tight for slow cross-border networks, DNS "
            "drift, or connection-pool churn — a single slow TLS handshake "
            "raises httpcore.ConnectTimeout before the outer asyncio.wait_for "
            "(timeout_seconds) window is reached. Default 60s covers typical "
            "slow-network scenarios while still failing fast on truly dead "
            "endpoints; the upper bound 300s is for extreme environments."
        ),
    )
    context_window: int | None = Field(
        default=None, ge=1024
    )  # 上下文窗口大小，空表示根据模型映射自动推断
    api_type: Literal["chat_completions", "responses", "auto"] = (
        "chat_completions"
        # API 类型: chat_completions / responses / auto。
        # auto 是跨协议升级（不是同协议 retry）：chat.completions 仅在抛出
        # 协议不兼容信号 (BadRequestError / UnprocessableEntityError) 时
        # 升级到 responses；瞬时错误由 ActusChatModel + _timeout_helpers
        # 翻译成 ServerRequestsError，交 LangGraph RetryPolicy 处理。
        # Provider 若未实现 Responses API（如智谱 /api/paas/v4），应使用
        # chat_completions 避免 fallback 路径 404。
    )
    supports_response_format: bool = True  # 是否支持 response_format 参数，部分兼容 API 不支持需设为 False
    supports_vision: bool = True  # 模型是否支持视觉/多模态输入（图片嵌入），关闭后强制使用 MCP 工具分析图片
    supports_pdf_input: bool = False  # 是否支持原生 PDF 文件输入
    context_overflow_guard_enabled: bool = False  # 是否开启上下文超限治理
    overflow_retry_cap: int = Field(2, ge=0, le=10)  # 超限治理自动重试次数上限
    soft_trigger_ratio: float = Field(
        0.85, gt=0, le=1
    )  # 软阈值比例，超过后优先进入预处理
    hard_trigger_ratio: float = Field(
        0.95, gt=0, le=1
    )  # 硬阈值比例，超过后强制进入压缩治理
    reserved_output_tokens: int = Field(4096, ge=0)  # 预留输出token预算
    reserved_output_tokens_cap_ratio: float = Field(
        0.25, gt=0, le=1
    )  # 预留输出token占上下文窗口最大比例
    token_estimator: Literal["hybrid", "char", "provider_api"] = (
        "hybrid"  # token估算策略
    )
    token_safety_factor: float = Field(
        1.15, ge=1.0
    )  # token估算安全系数，避免低估预算
    unknown_model_context_window: int = Field(
        32768, ge=1024
    )  # 未知模型的上下文窗口兜底值
    tool_result_max_chars: int = Field(
        8000, ge=100
    )  # 工具结果截断阈值（字符数），Tier 1 守卫
    tool_compress_trigger_ratio: float = Field(
        0.75, gt=0, le=1
    )  # Phase 1 工具结果压缩触发比例（占预算百分比）
    system_prompt_max_tokens: int = Field(
        10000, ge=0
    )  # B5 C9 / M2-PR0: system prompt token 预算上限。PromptAssembler 用这个硬封顶 section 装配结果，
    # compute_effective_window() 把它从 context_window 里扣掉，留给 history 的预算。
    # M2-PR0 bump 3500→10000 为 memory sections (user/rule/fact_index) 留出空间。

    @model_validator(mode="after")
    def validate_context_budget_ratio(self):
        """校验上下文预算比例配置"""
        if self.hard_trigger_ratio <= self.soft_trigger_ratio:
            raise ValueError("hard_trigger_ratio 必须大于 soft_trigger_ratio")
        return self


class SkillSelectionPolicy(BaseModel):
    """Skill选择稳定性策略配置。"""

    base_threshold: int = Field(3, ge=1, le=20)
    short_message_max_chars: int = Field(24, ge=1, le=200)
    llm_trigger_token_count: int = Field(4, ge=1, le=50)
    continuation_llm_enabled: bool = True
    continuation_llm_timeout_seconds: float = Field(3.0, gt=0, le=10)
    continuation_llm_cache_size: int = Field(128, ge=0, le=2048)
    ask_user_min_attempt_rounds_per_step: int = Field(1, ge=1, le=10)
    step_skill_lock_enabled: bool = True
    step_skill_reselect_unknown_tool_threshold: int = Field(3, ge=1, le=20)
    step_skill_reselect_max_per_step: int = Field(1, ge=0, le=5)
    available_tool_summary_token_budget: int = Field(500, ge=100, le=2000)
    unknown_tool_candidate_limit: int = Field(10, ge=1, le=50)
    continuation_phrases: List[str] = Field(
        default_factory=lambda: [
            "继续",
            "请继续",
            "继续一下",
            "继续吧",
            "接着",
            "下一步",
            "好的",
            "好",
            "行",
            "嗯",
            "收到",
            "ok",
            "okay",
            "go on",
            "next",
            "please continue",
            "proceed",
        ]
    )
    continuation_patterns: List[str] = Field(
        default_factory=lambda: [
            r"^(请)?继续(一下|吧|下去)?$",
            r"^(好的?[,，\s]*)?(继续|接着)$",
            r"^(ok|okay)([,\s]+(go on|continue))?$",
            r"^(好的?[,，\s]*)?继续([,，\s]+(一下|下|吧))?$",
        ]
    )

    @model_validator(mode="after")
    def validate_continuation_patterns(self):
        """校验续写判定正则表达式可编译。"""
        for index, pattern in enumerate(self.continuation_patterns):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"continuation_patterns[{index}] 无法编译: {pattern} ({exc})"
                ) from exc
        return self


class SkillEmbeddingConfig(BaseModel):
    """Skill 向量化检索配置。"""

    enabled: bool = False
    api_base: str = ""
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimensions: int = 256


class MemoryConfig(BaseModel):
    """对话记忆配置"""

    summary_enabled: bool = True
    summary_model: Optional[str] = None
    summary_timeout_seconds: float | None = Field(
        default=30.0,
        ge=0,
        le=3600,
        description=(
            "Per-call timeout for the summarizer LLM. If None, inherits "
            "llm_config.timeout_seconds. Default 30s is an assumption — "
            "summarizer has no tools and short prompts, so calls are "
            "expected to finish within 15s; 30s leaves 2x buffer."
        ),
    )
    summary_max_rounds: int = Field(5, ge=1, le=20)
    summary_token_budget: int = Field(2000, ge=200, le=10000)
    summary_min_steps: int = Field(1, ge=1, le=10)
    context_anchor_enabled: bool = True
    compact_keep_summary: bool = True
    # Flush 调度
    flush_enabled: bool = False
    flush_min_steps: int = Field(2, ge=1, le=10)
    flush_min_new_tokens: int = Field(3000, ge=500, le=20000)
    # Flush 容错
    flush_max_retries: int = Field(3, ge=0, le=10)
    flush_circuit_breaker_threshold: int = Field(3, ge=1, le=10)
    # Embedding（C1 基础设施 + C4 连接字段 + 容错）
    embedding_enabled: bool = False
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_dim: int = Field(512, ge=1, description="pgvector 列维度，须与 DB schema 一致")
    embedding_model: str = "text-embedding-3-small"
    embedding_circuit_breaker_threshold: int = Field(3, ge=1, le=10)
    embedding_circuit_breaker_recovery_seconds: float = Field(300.0, ge=10, le=3600)
    # C7: 混合检索与排序
    half_life_days: int = Field(30, ge=1, le=365)
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
    hybrid_alpha: float = Field(0.7, ge=0.0, le=1.0)  # C7 仅占位不参与计算, C8 生效


class ToolConfirmationConfig(BaseModel):
    """危险工具确认策略配置"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="是否启用危险工具确认（关闭后所有工具直接执行）")
    timeout_seconds: int = Field(default=300, ge=30, le=3600, description="确认超时秒数")
    smart_approve_enabled: bool = Field(default=False, description="启用 Smart Approve（LLM 辅助审批）")
    smart_approve_medium_only: bool = Field(default=False, description="Smart Approve 仅对 medium 工具生效")
    legacy_rule_fallback: bool = Field(
        default=True,
        description=(
            "R5 CS4 过渡期开关：当 ApprovalStateReader 在 grants 表查不到匹配时，"
            "是否回退读旧 tool_approval_rules 表。默认 True；运维手动跑 "
            "`uv run python -m app.cli.backfill_approval_grants` 完成迁移后切 False。"
            "Phase 2 PermissionEngine 落地后删除本字段。"
        ),
    )
    # PE-0 (2026-05-14): per-source kill switches for the new PermissionEngine
    # path. Default True for all five; flip a single flag in config.yaml to
    # fall back to legacy _run_policy_chain for that source.
    permission_engine_native_enabled: bool = Field(
        default=True,
        description="PE-0 native (file/shell/browser) path; False -> legacy _run_policy_chain",
    )
    permission_engine_skill_enabled: bool = Field(
        default=True,
        description=(
            "Drives the PE Skill source after PE-1 ship. When False, "
            "skill tool calls fall back to the legacy R3 path inside "
            "react_graph (line 2243-2395) until PE-1b deletes that branch. "
            "PE-2 widens the source registry; per-source flags retire in PE-3."
        ),
    )
    permission_engine_mcp_enabled: bool = Field(
        default=True,
        description="PE-2 MCP path",
    )
    permission_engine_a2a_enabled: bool = Field(
        default=True,
        description="PE-3 A2A path",
    )
    permission_engine_a4_events_enabled: bool = Field(
        default=True,
        description="A4-0 SessionModeChangedEvent SSE; False -> suppress emission",
    )


class ExecutionConfig(BaseModel):
    """执行健康监控配置"""

    total_timeout_seconds: float = Field(default=600.0, ge=0, description="总执行超时秒数（0=无限制）")
    idle_timeout_seconds: float = Field(default=120.0, ge=10, description="idle 无输出超时秒数")
    max_same_tool_failures: int = Field(default=3, ge=1, le=20, description="同签名工具最大连续失败数")


class AgentConfig(BaseModel):
    """Agent通用配置"""

    max_iterations: int = Field(default=100, gt=0, lt=1000)  # Agent最大迭代次数
    max_retries: int = Field(default=3, gt=1, lt=10)  # 最大重试次数
    max_search_results: int = Field(default=10, gt=1, lt=30)  # 最大搜索结果条数
    skill_selection: SkillSelectionPolicy = Field(default_factory=SkillSelectionPolicy)
    skill_embedding: SkillEmbeddingConfig = Field(default_factory=SkillEmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tool_confirmation: ToolConfirmationConfig = Field(default_factory=ToolConfirmationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


class MCPTransport(str, Enum):
    """MCP传输类型枚举"""

    STDIO = "stdio"  # 本地输入输出
    SSE = "sse"  # 流式事件
    STREAMABLE_HTTP = "streamable_http"  # 流式HTTP


class MCPServerConfig(BaseModel):
    """MCP服务配置"""

    # 通用配置字段
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP  # 传输协议
    enabled: bool = True  # 是否开启，默认为True
    description: Optional[str] = None  # 服务器描述
    env: Optional[Dict[str, Any]] = None  # 环境变量配置

    # stdio配置
    command: Optional[str] = None  # 启用命令
    args: Optional[List[str]] = None  # 命令参数

    # streamable_http&sse配置
    url: Optional[str] = None  # MCP服务URL地址
    headers: Optional[Dict[str, Any]] = None  # MCP服务请求头

    # 渐进加载: 始终 bind 到 LLM 的工具名（短名，不含 mcp_ 前缀）
    # None = 全走发现模式；["tool_a", "tool_b"] = 这些工具始终 bind
    always_bind: Optional[List[str]] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def validate_mcp_server_config(self):
        """校验mcp_server_config的相关信息，包含url+command"""
        # 1.判断transport是否为sse/streamable_http
        if self.transport in [MCPTransport.SSE, MCPTransport.STREAMABLE_HTTP]:
            # 2.这两种模式需要传递url
            if not self.url:
                raise ValueError("在sse或streamable_http模式下必须传递url")

        # 3.判断transport是否为stdio类型
        if self.transport == MCPTransport.STDIO:
            # 4.stdio类型必须传递command
            if not self.command:
                raise ValueError("在stdio模式下必须传递command")

        return self


class MCPConfig(BaseModel):
    """应用MCP配置"""

    mcpServers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class A2AServerConfig(BaseModel):
    """A2A服务配置"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # 唯一标识
    base_url: str  # 服务基础URL
    enabled: bool = True  # 服务是否开启


class A2AConfig(BaseModel):
    """A2A配置"""

    a2a_servers: List[A2AServerConfig] = Field(default_factory=list)


class SkillRiskMode(str, Enum):
    """Skill 风险控制模式"""

    OFF = "off"
    ENFORCE_CONFIRMATION = "enforce_confirmation"


class SkillRiskPolicy(BaseModel):
    """Skill 风险控制配置"""

    mode: SkillRiskMode = SkillRiskMode.OFF

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_bool_mode(cls, data: Any) -> Any:
        """兼容旧版 YAML 中将 off 解析为 False 的情况。"""
        if isinstance(data, dict) and isinstance(data.get("mode"), bool):
            normalized = dict(data)
            normalized["mode"] = (
                SkillRiskMode.ENFORCE_CONFIRMATION
                if data["mode"]
                else SkillRiskMode.OFF
            )
            return normalized
        return data


class VisionFallbackConfig(BaseModel):
    """视觉模型 fallback 配置（非多模态主模型时，用此模型描述图片/视频帧）"""

    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""
    api_type: Literal["chat_completions", "responses", "auto"] = "chat_completions"


class AudioProcessorConfig(BaseModel):
    """音频转录处理器配置"""

    provider: str = "disabled"  # sandbox_whisper | openai_api | disabled
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "whisper-1"


class VideoProcessorConfig(BaseModel):
    """视频处理器配置"""

    max_keyframes: int = Field(5, ge=1)
    extract_audio: bool = True
    frame_strategy: Literal["scene", "uniform"] = "scene"
    scene_threshold: float = Field(0.3, ge=0.0, le=1.0)


class FileUnderstandingConfig(BaseModel):
    """文件理解配置（file_view 工具）"""

    vision_fallback: VisionFallbackConfig = VisionFallbackConfig()
    audio: AudioProcessorConfig = AudioProcessorConfig()
    video: VideoProcessorConfig = VideoProcessorConfig()


class ToolRuntimeConfig(BaseModel):
    """R2 CS2 tool runtime limits.

    Plumbed from ``config.yaml → AppConfig → react_graph.build_react_graph
    → configurable`` so Layer 1 (``_run_policy_chain``) and Layer 2
    (``_invoke_wrapper``) can read the values at runtime instead of
    depending on the module-level ``_SMART_APPROVE_TIMEOUT_SECONDS`` /
    ``_MAX_WRAPPER_OUTPUT_BYTES`` constants.

    Defaults match the original module constants so zero-config
    deployments preserve existing behavior.
    """

    max_wrapper_output_bytes: int = Field(
        default=1 << 20,  # 1 MiB
        ge=1024,
        description=(
            "Layer 2 wrapper output length guard (bytes). Wrapper content "
            "larger than this is converted to AllowError("
            "wrapper_output_too_large) before reaching the LLM, preventing "
            "memory spikes from runaway shell output or large blobs."
        ),
    )
    smart_approve_timeout_seconds: int = Field(
        default=15,
        ge=1,
        le=300,
        description=(
            "Layer 1 Stage P.2 SmartApprove LLM call timeout (seconds). "
            "Timeout is fail-open — the policy chain proceeds as if "
            "SmartApprove said allow, so a hanging LLM doesn't block "
            "every tool call. Tune down for faster fail-open or up to "
            "give the model more headroom."
        ),
    )
    enabled_outcome_variants: list[str] = Field(
        default_factory=lambda: [
            "allow_success", "allow_error", "denied", "asked", "passthrough",
        ],
        description=(
            "R4 P2.1 executable guard 1: wrapper 产出新 variant 前必须 config 开启, "
            "runtime 未启用的 variant 在 _translate_outcome 内部 fail-fast. "
            "控制 coordinated rollout — 运维先灰度, 再启用 wrapper."
        ),
    )


class AppConfig(BaseModel):
    """应用配置信息，包含Agent配置、LLM提供商配置、MCP配置、A2A配置"""

    llm_config: LLMConfig  # 语言模型配置
    agent_config: AgentConfig  # Agent通用配置
    mcp_config: MCPConfig  # MCP服务配置
    a2a_config: A2AConfig  # A2A服务配置
    skill_risk_policy: SkillRiskPolicy = SkillRiskPolicy()
    file_understanding: FileUnderstandingConfig = FileUnderstandingConfig()
    # R2 CS2: Layer 1/2 runtime limits (defaults preserve pre-R2 constants)
    tool_runtime: ToolRuntimeConfig = Field(default_factory=ToolRuntimeConfig)

    # Pydantic配置，允许传递额外的字段初始化
    model_config = ConfigDict(extra="allow")
