import asyncio
import copy
import logging
import uuid
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, List
from urllib.parse import urlparse

from app.application.errors.exceptions import NotFoundError
from app.domain.external.extension_admission import UninstallContext
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    AgentConfig,
    AppConfig,
    FileUnderstandingConfig,
    LLMConfig,
    MCPConfig,
    SkillRiskPolicy,
)
from app.domain.models.extension_governance import ManagedByPluginError
from app.domain.repositories.app_config_repository import AppConfigRepository
from app.domain.services.tools.a2a import A2AClientManager
from app.domain.services.tools.mcp import MCPClientManager
from app.interfaces.schemas.app_config import ListA2AServerItem, ListMCPServerItem

if TYPE_CHECKING:
    from app.application.services.extension_reconciler import ConfigDeltaReconciler
    from app.domain.external.extension_admission import InstallContext

logger = logging.getLogger(__name__)

A2A_DISCOVERY_TIMEOUT_SECONDS = 8


def _is_fatal_error(error: BaseException) -> bool:
    return isinstance(error, (KeyboardInterrupt, SystemExit))


async def _run_probe_with_timeout(
    probe: Awaitable[None], timeout_seconds: int
) -> None:
    """在当前Task中执行探测，避免跨Task清理导致cancel scope上下文错配。"""
    async with asyncio.timeout(timeout_seconds):
        await probe


class AppConfigService:
    """应用配置服务"""

    def __init__(
        self,
        app_config_repository: AppConfigRepository,
        reconciler: "ConfigDeltaReconciler | None" = None,
        registry_read_port: Any = None,
    ) -> None:
        """构造函数，完成应用配置服务的初始化。

        reconciler / registry_read_port：D1a §6.1 治理挂钩——off（默认）=双 None
        → 六个写方法行为与治理接线前 byte 级一致（不构造 delta、不查成员守卫）。
        """
        self.app_config_repository = app_config_repository
        self._reconciler = reconciler                    # D1a §6.1：off=None → 零治理记账
        self._registry_read_port = registry_read_port    # R1#17/R3#15：delete 成员守卫读面

    async def _load_app_config(self) -> AppConfig:
        """加载获取所有的应用配置"""
        return self.app_config_repository.load()

    async def get_llm_config(self) -> LLMConfig:
        """获取LLM提供商配置"""
        app_config = await self._load_app_config()
        return app_config.llm_config

    async def update_llm_config(self, llm_config: LLMConfig) -> LLMConfig:
        """根据传递的llm_config更新语言模型提供商配置"""
        self.validate_llm_provider(llm_config.provider)
        # 1.获取应用配置
        app_config = await self._load_app_config()

        # 2.判断api_key是否为空
        if not llm_config.api_key.strip():
            llm_config.api_key = app_config.llm_config.api_key

        # 3.调用函数更新app_config
        app_config.llm_config = llm_config
        self.app_config_repository.save(app_config)

        return app_config.llm_config

    @staticmethod
    def validate_llm_provider(provider: str | None) -> None:
        from app.application.errors.exceptions import ConfigError, ValidationError
        from app.domain.services.provider_profiles import get_profile

        if provider and provider.strip():
            try:
                get_profile(provider.strip())
            except ConfigError as exc:
                raise ValidationError("未知的模型兼容策略，请重新选择 provider") from exc

    async def get_agent_config(self) -> AgentConfig:
        """获取Agent通用配置"""
        app_config = await self._load_app_config()
        return app_config.agent_config

    async def get_skill_risk_policy(self) -> SkillRiskPolicy:
        """获取 Skill 风险策略配置。"""
        app_config = await self._load_app_config()
        return app_config.skill_risk_policy

    async def update_skill_risk_policy(
        self, policy: SkillRiskPolicy
    ) -> SkillRiskPolicy:
        """更新 Skill 风险策略配置。"""
        app_config = await self._load_app_config()
        app_config.skill_risk_policy = policy
        self.app_config_repository.save(app_config)
        return app_config.skill_risk_policy

    async def get_file_understanding_config(self) -> FileUnderstandingConfig:
        """获取文件理解配置"""
        app_config = await self._load_app_config()
        return app_config.file_understanding

    async def update_file_understanding_config(
        self, config: FileUnderstandingConfig
    ) -> FileUnderstandingConfig:
        """更新文件理解配置"""
        self.validate_llm_provider(config.vision_fallback.provider)
        app_config = await self._load_app_config()
        # 保留已有 api_key（前端提交空字符串表示不更新）
        if not config.vision_fallback.api_key.strip():
            config.vision_fallback.api_key = app_config.file_understanding.vision_fallback.api_key
        if not config.audio.openai_api_key.strip():
            config.audio.openai_api_key = app_config.file_understanding.audio.openai_api_key
        app_config.file_understanding = config
        self.app_config_repository.save(app_config)
        return app_config.file_understanding

    async def update_agent_config(self, agent_config: AgentConfig) -> AgentConfig:
        """根据传递的agent_config更新Agent通用配置"""
        # 1.获取应用配置
        app_config = await self._load_app_config()

        # 2.调用函数更新app_config
        app_config.agent_config = agent_config
        self.app_config_repository.save(app_config)

        return app_config.agent_config

    async def get_mcp_servers(self) -> List[ListMCPServerItem]:
        """获取MCP服务器列表"""
        # 1.获取当前应用配置
        app_config = await self._load_app_config()

        # 2.先基于配置构建回退列表，保证接口不会因为探测失败而空白
        mcp_servers = [
            ListMCPServerItem(
                server_name=server_name,
                enabled=server_config.enabled,
                transport=server_config.transport,
                tools=[],
            )
            for server_name, server_config in app_config.mcp_config.mcpServers.items()
        ]

        # 3.创建mcp客户端管理器，对配置信息不进行过滤
        mcp_client_manager = MCPClientManager(mcp_config=app_config.mcp_config)

        try:
            # 4.初始化mcp客户端管理器
            #   不使用外层asyncio.timeout，以避免超时取消后需要
            #   asyncio.shield清理资源（shield会创建新Task，导致跨Task关闭elit_stack
            #   引发anyio cancel scope错误，从而资源泄漏CPU飙高）。
            #   每个MCP服务器已有独立5s超时，不会无限挂起。
            await mcp_client_manager.initialize()

            # 5.获取mcp客户端管理器的工具列表并回填
            tools = mcp_client_manager.tools
            errors = mcp_client_manager.errors
            for server in mcp_servers:
                server.tools = [
                    tool.name for tool in tools.get(server.server_name, [])
                ]
                if server.server_name in errors:
                    server.error = errors[server.server_name]
        except BaseException as e:
            if _is_fatal_error(e):
                raise
            logger.warning(f"加载MCP工具列表失败，降级返回基础配置: {str(e)}")
        finally:
            # 6.清除MCP客户端管理器的相关资源
            #   不使用asyncio.shield，确保在同一个Task中清理，
            #   避免跨Task关闭anyio cancel scope导致资源泄漏。
            try:
                await mcp_client_manager.cleanup()
            except BaseException as e:
                if _is_fatal_error(e):
                    raise
                logger.warning(f"清理MCP客户端管理器失败: {str(e)}")

        return mcp_servers

    async def update_and_create_mcp_servers(
        self,
        mcp_config: MCPConfig,
        *,
        actor_id: str | None = None,
        install_context: "InstallContext | None" = None,
        target_server: str | None = None,
    ) -> MCPConfig:
        """根据传递的数据新增或更新MCP配置"""
        # 1.获取应用配置
        app_config = await self._load_app_config()

        # 2.使用新的mcp_config更新原始的配置（F6：merge 原地——post-save delta 前深拷贝
        #   old，否则治理 diff 恒空）
        old_mcp = copy.deepcopy(app_config.mcp_config)
        app_config.mcp_config.mcpServers.update(mcp_config.mcpServers)

        # 3.调用数据仓库完成存储or更新
        self.app_config_repository.save(app_config)

        # 4.治理 delta（§6.1-1：post-save；off=None 时零记账）
        if self._reconciler is not None:
            await self._reconciler.reconcile_mcp_delta(
                old_mcp, app_config.mcp_config, actor_id=actor_id,
                install_context=install_context, target_ext_id=target_server)
        return app_config.mcp_config

    async def delete_mcp_server(
        self,
        server_name: str,
        *,
        actor_id: str | None = None,
        uninstall_context: "UninstallContext | None" = None,
        missing_ok: bool = False,
    ) -> MCPConfig:
        """根据名字删除MCP服务"""
        # 1.获取应用配置
        app_config = await self._load_app_config()

        # 2.standalone 成员守卫（§8.4）：uninstall_context 为 None 或 correlation 为 None
        #   → 在任何 mutation/save 之前查 read_port；行带 parent_plugin_ext_id 拒删。
        #   saga 路（correlation 非 None）跳过——成员随 plugin 卸载。
        await self._guard_member_delete("mcp", server_name, uninstall_context)

        # 3.查询对应服务的名字是否存在
        if server_name not in app_config.mcp_config.mcpServers:
            if missing_ok:
                return app_config.mcp_config     # saga 幂等（R19#F2）：目标不存在=成功
            raise NotFoundError(f"该MCP服务[{server_name}]不存在，请核实后重试")

        # 4.如果存在则删除字典中对应的服务
        old_mcp = copy.deepcopy(app_config.mcp_config)
        del app_config.mcp_config.mcpServers[server_name]
        self.app_config_repository.save(app_config)

        # 5.治理 delta（standalone → 构造 UninstallContext；R26#1）
        if self._reconciler is not None:
            ctx = uninstall_context
            if ctx is None and actor_id is not None:
                ctx = UninstallContext(correlation_id=None, actor_user_id=actor_id)
            await self._reconciler.reconcile_mcp_delta(
                old_mcp, app_config.mcp_config, actor_id=actor_id, uninstall_context=ctx)
        return app_config.mcp_config

    async def set_mcp_server_enabled(
        self, server_name: str, enabled: bool, *, actor_id: str | None = None
    ) -> MCPConfig:
        """更新MCP服务的启用状态"""
        # 1.获取应用配置
        app_config = await self._load_app_config()

        # 2.查询对应服务的名字是否存在
        if server_name not in app_config.mcp_config.mcpServers:
            raise NotFoundError(f"该MCP服务[{server_name}]不存在，请核实后重试")

        # 3.如果存在则更新该MCP服务的启用状态
        old_mcp = copy.deepcopy(app_config.mcp_config)
        app_config.mcp_config.mcpServers[server_name].enabled = enabled
        self.app_config_repository.save(app_config)

        # 4.治理 delta（enabled 翻转 → fingerprint 排除 enabled，T15 自然空转）
        if self._reconciler is not None:
            await self._reconciler.reconcile_mcp_delta(
                old_mcp, app_config.mcp_config, actor_id=actor_id)
        return app_config.mcp_config

    async def create_a2a_server(
        self,
        base_url: str,
        *,
        actor_id: str | None = None,
        install_context: "InstallContext | None" = None,
        preallocated_id: str | None = None,
    ) -> A2AConfig:
        """根据传递的配置新增a2a服务器"""
        # 1.获取当前的应用配置
        app_config = await self._load_app_config()

        # 2.往数据中新增a2a服务(在新增之前其实可以检测下当前Agent是否存在)
        old_a2a = copy.deepcopy(app_config.a2a_config)
        a2a_server_config = A2AServerConfig(
            id=preallocated_id or str(uuid.uuid4()),   # R19#F1：saga 预分配 id
            base_url=base_url,
            enabled=True,
        )
        logger.info(f"Creating A2A server with config: {a2a_server_config}")
        app_config.a2a_config.a2a_servers.append(a2a_server_config)

        # 3.调用数据仓库更新
        self.app_config_repository.save(app_config)

        # 4.治理 delta（target=生成/预分配的 id）
        if self._reconciler is not None:
            await self._reconciler.reconcile_a2a_delta(
                old_a2a, app_config.a2a_config, actor_id=actor_id,
                install_context=install_context, target_ext_id=a2a_server_config.id)
        return app_config.a2a_config

    async def get_a2a_servers(self) -> List[ListA2AServerItem]:
        """获取A2A服务列表"""
        # 1.获取当前的应用配置
        app_config = await self._load_app_config()

        # 2.基于配置构建回退列表，确保新增后即使探测失败也可渲染
        a2a_servers = [
            ListA2AServerItem(
                id=server.id,
                name=self._derive_a2a_name(server.base_url),
                description=server.base_url,
                input_modes=[],
                output_modes=[],
                streaming=False,
                push_notifications=False,
                enabled=server.enabled,
            )
            for server in app_config.a2a_config.a2a_servers
        ]

        # 3.构建a2a客户端管理器，对配置信息不过滤
        a2a_client_manager = A2AClientManager(app_config.a2a_config)

        try:
            # 4.初始化a2a客户端管理器（限制探测超时，避免设置页卡住）
            await _run_probe_with_timeout(
                a2a_client_manager.initialize(),
                timeout_seconds=A2A_DISCOVERY_TIMEOUT_SECONDS,
            )

            # 5.获取Agent卡片列表并回填数据
            agent_cards = a2a_client_manager.agent_cards

            for server in a2a_servers:
                agent_card = agent_cards.get(server.id)
                if not agent_card:
                    continue
                server.name = agent_card.get("name", server.name)
                server.description = agent_card.get("description", server.description)
                server.input_modes = agent_card.get("defaultInputModes", [])
                server.output_modes = agent_card.get("defaultOutputModes", [])
                server.streaming = agent_card.get("capabilities", {}).get(
                    "streaming", False
                )
                server.push_notifications = agent_card.get("capabilities", {}).get(
                    "push_notifications", False
                )
                server.enabled = agent_card.get("enabled", server.enabled)
        except BaseException as e:
            if _is_fatal_error(e):
                raise
            logger.warning(f"加载A2A卡片失败，降级返回基础配置: {str(e)}")
        finally:
            # 6.清除客户端管理器资源
            try:
                await asyncio.shield(a2a_client_manager.cleanup())
            except BaseException as e:
                if _is_fatal_error(e):
                    raise
                logger.warning(f"清理A2A客户端管理器失败: {str(e)}")

        return a2a_servers

    async def set_a2a_server_enabled(
        self, a2a_id: str, enabled: bool, *, actor_id: str | None = None
    ) -> A2AConfig:
        """根据传递的id+enabled更新服务启用状态"""
        # 1.获取当前的应用配置
        app_config = await self._load_app_config()

        # 2.计算需要更新位置的索引并判断是否存在
        idx = None
        for item_idx, item in enumerate(app_config.a2a_config.a2a_servers):
            if item.id == a2a_id:
                idx = item_idx
                break
        if idx is None:
            raise NotFoundError(f"该A2A服务[{a2a_id}]不存在，请核实后重试")

        # 3.如果存在则更新数据
        old_a2a = copy.deepcopy(app_config.a2a_config)
        app_config.a2a_config.a2a_servers[idx].enabled = enabled
        self.app_config_repository.save(app_config)

        # 4.治理 delta（enabled 翻转 → fingerprint 排除 enabled，T15 自然空转）
        if self._reconciler is not None:
            await self._reconciler.reconcile_a2a_delta(
                old_a2a, app_config.a2a_config, actor_id=actor_id)
        return app_config.a2a_config

    async def delete_a2a_server(
        self,
        a2a_id: str,
        *,
        actor_id: str | None = None,
        uninstall_context: "UninstallContext | None" = None,
        missing_ok: bool = False,
    ) -> A2AConfig:
        """根据传递的id删除指定的a2a服务"""
        # 1.获取当前的应用配置
        app_config = await self._load_app_config()

        # 2.standalone 成员守卫（§8.4）——镜像 delete_mcp_server
        await self._guard_member_delete("a2a", a2a_id, uninstall_context)

        # 3.计算需要操作位置的索引并判断是否存在
        idx = None
        for item_idx, item in enumerate(app_config.a2a_config.a2a_servers):
            if item.id == a2a_id:
                idx = item_idx
                break
        if idx is None:
            if missing_ok:
                return app_config.a2a_config     # saga 幂等
            raise NotFoundError(f"该A2A服务[{a2a_id}]不存在，请核实后重试")

        # 4.删除a2a服务器
        old_a2a = copy.deepcopy(app_config.a2a_config)
        del app_config.a2a_config.a2a_servers[idx]
        self.app_config_repository.save(app_config)

        # 5.治理 delta（standalone → 构造 UninstallContext；R26#1）
        if self._reconciler is not None:
            ctx = uninstall_context
            if ctx is None and actor_id is not None:
                ctx = UninstallContext(correlation_id=None, actor_user_id=actor_id)
            await self._reconciler.reconcile_a2a_delta(
                old_a2a, app_config.a2a_config, actor_id=actor_id, uninstall_context=ctx)
        return app_config.a2a_config

    async def _guard_member_delete(
        self, kind: str, ext_id: str, uninstall_context: "UninstallContext | None"
    ) -> None:
        """standalone 成员单独卸载守卫（§8.4，R1#17）：行带 parent_plugin_ext_id →
        ManagedByPluginError。saga 路（correlation 非 None）或 off（read_port=None）跳过。"""
        if self._registry_read_port is None:
            return
        if uninstall_context is not None and uninstall_context.correlation_id is not None:
            return   # saga 路——成员随 plugin 卸载，跳过守卫
        row = await self._registry_read_port.get_row(kind, ext_id)
        if row is not None and row.parent_plugin_ext_id is not None:
            raise ManagedByPluginError(
                f"{kind}[{ext_id}] 由 plugin 管理，禁 standalone 卸载（§8.4）")

    @staticmethod
    def _derive_a2a_name(base_url: str) -> str:
        """根据base_url生成可读的A2A名称"""
        parsed = urlparse(base_url)
        hostname = parsed.hostname or ""
        path = (parsed.path or "").strip("/")

        if hostname and path:
            return f"{hostname}/{path}"
        if hostname:
            return hostname
        if path:
            return path
        return base_url or "未命名 Agent"
