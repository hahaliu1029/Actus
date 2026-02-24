import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class LLMConfig(BaseModel):
    """LLM提供商配置"""

    base_url: HttpUrl = "https://api.deepseek.com"  # 模型基础URL地址
    api_key: str = ""  # 模型API秘钥
    model_name: str = (
        "deepseek-reasoner"  # 模型名字，默认使用deepseek-reasoner带推理的模型，传递tools会自动切换到deepseek-chat
    )
    temperature: float = Field(0.7)  # 温度，默认设置为0.7
    max_tokens: int = Field(
        8192, ge=0
    )  # 最大输出token数，默认设置为deepseek-chat模型的最大输出限制
    context_window: int | None = Field(
        default=None, ge=1024
    )  # 上下文窗口大小，空表示根据模型映射自动推断
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

    @model_validator(mode="after")
    def validate_context_budget_ratio(self):
        """校验上下文预算比例配置"""
        if self.hard_trigger_ratio <= self.soft_trigger_ratio:
            raise ValueError("hard_trigger_ratio 必须大于 soft_trigger_ratio")
        return self


class AgentConfig(BaseModel):
    """Agent通用配置"""

    max_iterations: int = Field(default=100, gt=0, lt=1000)  # Agent最大迭代次数
    max_retries: int = Field(default=3, gt=1, lt=10)  # 最大重试次数
    max_search_results: int = Field(default=10, gt=1, lt=30)  # 最大搜索结果条数


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


class AppConfig(BaseModel):
    """应用配置信息，包含Agent配置、LLM提供商配置、MCP配置、A2A配置"""

    llm_config: LLMConfig  # 语言模型配置
    agent_config: AgentConfig  # Agent通用配置
    mcp_config: MCPConfig  # MCP服务配置
    a2a_config: A2AConfig  # A2A服务配置
    skill_risk_policy: SkillRiskPolicy = SkillRiskPolicy()

    # Pydantic配置，允许传递额外的字段初始化
    model_config = ConfigDict(extra="allow")
