import asyncio
import logging
from typing import Dict, Optional

from app.application.services.app_config_service import AppConfigService
from app.domain.models.app_config import AgentConfig, FileUnderstandingConfig, LLMConfig, MCPConfig
from app.interfaces.dependencies import AdminUser, CurrentUser, rate_limit_write
from app.interfaces.schemas.app_config import (
    ListA2AServerResponse,
    ListMCPServerResponse,
    LLMConnectionTestResult,
)
from app.interfaces.schemas.base import Response
from app.interfaces.service_dependencies import (
    get_app_config_service,
    get_extension_install_service,
    _build_llm,
)
from fastapi import APIRouter, Body, Depends, Query
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/app-config", tags=["设置模块"])


@router.post(
    "/llm/test",
    response_model=Response[LLMConnectionTestResult],
    dependencies=[Depends(rate_limit_write)],
    summary="测试模型基础请求（不保存配置）",
)
async def test_llm_connection(
    new_llm_config: LLMConfig,
    admin_user: AdminUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[LLMConnectionTestResult]:
    config = new_llm_config.model_copy(deep=True)
    AppConfigService.validate_llm_provider(config.provider)
    if not config.api_key.strip():
        saved = await app_config_service.get_llm_config()
        config.api_key = saved.api_key
    # Bound this explicit, billable probe; use the production adapter factory.
    config.max_tokens = min(config.max_tokens or 256, 256)
    config.timeout_seconds = min(config.timeout_seconds or 30, 30)
    config.connect_timeout_seconds = min(config.connect_timeout_seconds, 15)
    llm = _build_llm(config)
    provider = llm.profile.provider_id
    probe_kwargs = {}
    if not llm.profile.thinking_always_on:
        if llm.profile.thinking_toggle_style == "extra_body_thinking":
            probe_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        elif llm.profile.thinking_toggle_style == "extra_body_enable_thinking":
            probe_kwargs["extra_body"] = {"enable_thinking": False}
    try:
        async with asyncio.timeout(30):
            response = await llm.ainvoke([HumanMessage(content="Reply with OK only.")], **probe_kwargs)
        success = bool(response.content)
        message = (
            "基础请求通过；未验证工具调用、图片或长任务。配置尚未保存。"
            if success else "端点已响应，但未返回可识别内容；请检查模型与协议。"
        )
    except Exception as exc:
        # Provider exceptions can contain credentials or response bodies.
        # Only expose their class, never stringify the exception.
        success = False
        message = f"连接测试失败（{type(exc).__name__}）；请检查地址、密钥、模型与协议。"
    return Response.success(data=LLMConnectionTestResult(
        success=success, provider=provider, api_type=config.api_type, message=message,
    ))


@router.get(
    path="/llm",
    response_model=Response[LLMConfig],
    summary="获取LLM配置信息",
    description="包含LLM提供商的base_url、temperature、model_name、max_tokens",
)
async def get_llm_config(
    current_user: CurrentUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[LLMConfig]:
    """获取LLM配置信息"""
    llm_config = await app_config_service.get_llm_config()
    return Response.success(data=llm_config.model_dump(exclude={"api_key"}))


@router.post(
    path="/llm",
    response_model=Response[LLMConfig],
    summary="更新LLM配置信息",
    description="更新LLM配置信息，当api_key为空的时候表示不更新该字段（仅限管理员）",
)
async def update_llm_config(
    new_llm_config: LLMConfig,
    admin_user: AdminUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[LLMConfig]:
    """更新LLM配置信息（仅限管理员）"""
    updated_llm_config = await app_config_service.update_llm_config(new_llm_config)
    return Response.success(
        msg="更新LLM信息配置成功",
        data=updated_llm_config.model_dump(exclude={"api_key"}),
    )


@router.get(
    path="/agent",
    response_model=Response[AgentConfig],
    summary="获取Agent通用配置信息",
    description="包含最大迭代次数、最大重试次数、最大搜索结果数",
)
async def get_agent_config(
    current_user: CurrentUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[AgentConfig]:
    """获取Agent通用配置信息"""
    agent_config = await app_config_service.get_agent_config()
    return Response.success(data=agent_config.model_dump())


@router.post(
    path="/agent",
    response_model=Response[AgentConfig],
    summary="更新Agent通用配置信息",
    description="更新Agent通用配置信息（仅限管理员）",
)
async def update_agent_config(
    new_agent_config: AgentConfig,
    admin_user: AdminUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[AgentConfig]:
    """更新Agent配置信息（仅限管理员）"""
    updated_agent_config = await app_config_service.update_agent_config(
        new_agent_config
    )
    return Response.success(
        msg="更新Agent信息配置成功", data=updated_agent_config.model_dump()
    )


@router.get(
    path="/file-understanding",
    response_model=Response[FileUnderstandingConfig],
    summary="获取文件理解配置",
)
async def get_file_understanding_config(
    current_user: CurrentUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[FileUnderstandingConfig]:
    """获取文件理解配置"""
    config = await app_config_service.get_file_understanding_config()
    return Response.success(
        data=config.model_dump(exclude={"vision_fallback": {"api_key"}, "audio": {"openai_api_key"}})
    )


@router.post(
    path="/file-understanding",
    response_model=Response[FileUnderstandingConfig],
    summary="更新文件理解配置（仅限管理员）",
)
async def update_file_understanding_config(
    config: FileUnderstandingConfig,
    admin_user: AdminUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[FileUnderstandingConfig]:
    """更新文件理解配置（仅限管理员）"""
    updated = await app_config_service.update_file_understanding_config(config)
    return Response.success(
        msg="文件理解配置已保存",
        data=updated.model_dump(exclude={"vision_fallback": {"api_key"}, "audio": {"openai_api_key"}}),
    )


@router.get(
    path="/mcp-servers",
    response_model=Response[ListMCPServerResponse],
    summary="获取MCP服务器工具列表",
    description="获取当前系统的MCP服务器列表，包含MCP服务名字、工具列表、启用状态等",
)
async def get_mcp_servers(
    current_user: CurrentUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[ListMCPServerResponse]:
    """获取当前系统的MCP服务器工具列表"""
    mcp_servers = await app_config_service.get_mcp_servers()
    return Response.success(
        msg="获取mcp服务器列表成功", data=ListMCPServerResponse(mcp_servers=mcp_servers)
    )


@router.post(
    path="/mcp-servers",
    response_model=Response[Optional[Dict]],
    summary="新增MCP服务配置，支持传递一个或者多个配置",
    description="传递MCP配置信息为系统新增MCP工具（仅限管理员）",
)
async def create_mcp_servers(
    mcp_config: MCPConfig,
    admin_user: AdminUser,
    dry_run: bool = Query(False, description="dry-run 预检（不落盘）"),
    acknowledge: bool = Query(False, description="确认 caution-tier 扫描结果"),
    force: bool = Query(False, description="强制安装 dangerous-tier 扩展"),
    app_config_service: AppConfigService = Depends(get_app_config_service),
    install_service=Depends(get_extension_install_service),
) -> Response[Optional[Dict]]:
    """根据传递的配置信息创建mcp服务（仅限管理员）。

    治理关闭（install_service=None）→ 旧直通（INV-D1-0 零行为变化）；开启 → dry_run
    预检 / commit 两阶段管道（逐个安装；warnings 加性附 data.warnings，旧 FE 零感知）。"""
    if install_service is None:                       # mode=off
        # dry_run 是治理新参——off 下对其透明（pre-D1a：未知参被忽略走正常创建）。
        # 只在治理开启时才 honor dry_run（下方 preview 分支）。
        await app_config_service.update_and_create_mcp_servers(mcp_config)
        return Response.success(msg="新增MCP服务配置成功")
    # 治理模式逐个安装（>1 → BatchNotAllowedError → 422 single_server_required）
    server_name, server_config = install_service.ensure_single_mcp(mcp_config)
    if dry_run:
        preview = await install_service.preview_mcp(server_name, server_config)
        return Response.success(msg="dry-run 预检完成", data=preview.model_dump())
    _new, warnings = await install_service.commit_mcp(
        server_name, server_config,
        actor_id=admin_user.id, acknowledged=acknowledge, forced=force)
    data = {"warnings": warnings} if warnings else None
    return Response.success(msg="新增MCP服务配置成功", data=data)


@router.post(
    path="/mcp-servers/{server_name}/delete",
    response_model=Response[Optional[Dict]],
    summary="删除MCP服务配置",
    description="根据传递的MCP服务名字删除指定的MCP服务（仅限管理员）",
)
async def delete_mcp_server(
    server_name: str,
    admin_user: AdminUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[Optional[Dict]]:
    """根据服务名字删除MCP服务器（仅限管理员）"""
    # actor 透传关闭 T16 transient（mode≠off 时 delete delta 需 actor 构造 UninstallContext）
    await app_config_service.delete_mcp_server(server_name, actor_id=admin_user.id)
    return Response.success(msg="删除MCP服务配置成功")


@router.post(
    path="/mcp-servers/{server_name}/enabled",
    response_model=Response[Optional[Dict]],
    summary="更新MCP服务的全局启用状态",
    description="根据传递的server_name+enabled更新指定MCP服务的全局启用状态（仅限管理员）",
)
async def set_mcp_server_enabled(
    server_name: str,
    admin_user: AdminUser,
    enabled: bool = Body(..., embed=True),
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[Optional[Dict]]:
    """根据传递的server_name+enabled更新服务的全局启用状态（仅限管理员）"""
    await app_config_service.set_mcp_server_enabled(
        server_name, enabled, actor_id=admin_user.id)
    return Response.success(msg="更新MCP服务启用状态成功")


@router.get(
    path="/a2a-servers",
    response_model=Response[ListA2AServerResponse],
    summary="获取a2a服务器列表",
    description="获取Actus项目中的所有已配置的a2a服务列表",
)
async def get_a2a_servers(
    current_user: CurrentUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[ListA2AServerResponse]:
    """获取a2a服务列表"""
    a2a_servers = await app_config_service.get_a2a_servers()
    return Response.success(
        msg="获取a2a服务列表成功", data=ListA2AServerResponse(a2a_servers=a2a_servers)
    )


@router.post(
    path="/a2a-servers",
    response_model=Response[Optional[Dict]],
    summary="新增a2a服务器",
    description="为Actus项目新增a2a服务器（仅限管理员）",
)
async def create_a2a_server(
    admin_user: AdminUser,
    base_url: str = Body(..., embed=True),
    dry_run: bool = Query(False, description="dry-run 预检（不落盘）"),
    acknowledge: bool = Query(False, description="确认 caution-tier 扫描结果"),
    force: bool = Query(False, description="强制安装 dangerous-tier 扩展"),
    app_config_service: AppConfigService = Depends(get_app_config_service),
    install_service=Depends(get_extension_install_service),
) -> Response[Optional[Dict]]:
    """新增a2a服务器（仅限管理员）。

    治理关闭 → 旧直通（INV-D1-0）；开启 → dry_run 预检 / commit 两阶段管道
    （a2a 单 base_url 无批量语义；warnings 加性附 data.warnings）。"""
    if install_service is None:                       # mode=off
        # dry_run 是治理新参——off 下对其透明（pre-D1a：未知参被忽略走正常创建）。
        await app_config_service.create_a2a_server(base_url)
        return Response.success(msg="新增A2A服务配置成功")
    if dry_run:
        preview = await install_service.preview_a2a(base_url)
        return Response.success(msg="dry-run 预检完成", data=preview.model_dump())
    _new, warnings = await install_service.commit_a2a(
        base_url, actor_id=admin_user.id, acknowledged=acknowledge, forced=force)
    data = {"warnings": warnings} if warnings else None
    return Response.success(msg="新增A2A服务配置成功", data=data)


@router.post(
    path="/a2a-servers/{a2a_id}/delete",
    response_model=Response[Optional[Dict]],
    summary="删除a2a服务器",
    description="根据A2A服务id标识删除指定的A2A服务（仅限管理员）",
)
async def delete_a2a_server(
    a2a_id: str,
    admin_user: AdminUser,
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[Optional[Dict]]:
    """删除a2a服务器（仅限管理员）"""
    await app_config_service.delete_a2a_server(a2a_id, actor_id=admin_user.id)
    return Response.success(msg="删除a2a服务器成功")


@router.post(
    path="/a2a-servers/{a2a_id}/enabled",
    response_model=Response[Optional[Dict]],
    summary="更新A2A服务的全局启用状态",
    description="启动or禁用A2A服务的全局状态（仅限管理员）",
)
async def set_a2a_server_enabled(
    a2a_id: str,
    admin_user: AdminUser,
    enabled: bool = Body(..., embed=True),
    app_config_service: AppConfigService = Depends(get_app_config_service),
) -> Response[Optional[Dict]]:
    """更新A2A服务的全局启用状态（仅限管理员）"""
    await app_config_service.set_a2a_server_enabled(
        a2a_id, enabled, actor_id=admin_user.id)
    return Response.success(msg="更新a2a服务器启用状态成功")
