import asyncio
import io
import logging
import mimetypes
import re
import uuid
from typing import AsyncGenerator, BinaryIO, Callable, List

from app.domain.external.browser import Browser
from app.domain.external.file_storage import FileStorage
from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.external.task import Task, TaskRunner
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig, SkillRiskPolicy
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.event import (
    A2AToolContent,
    BaseEvent,
    BrowserToolContent,
    DoneEvent,
    ErrorEvent,
    Event,
    FileToolContent,
    MCPToolContent,
    MessageEvent,
    SearchToolContent,
    ShellToolContent,
    SkillToolContent,
    TitleEvent,
    ToolEvent,
    ToolEventStatus,
    WaitEvent,
)
from app.domain.models.file import File
from app.domain.models.message import Message
from app.domain.models.search import SearchResults
from app.domain.models.session import SessionStatus
from app.domain.models.skill import Skill
from app.domain.models.tool_result import ToolResult
from app.domain.models.user_tool_preference import ToolType

# from app.domain.repositories.file_repository import FileRepository
# from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.application.services.skill_index_service import SkillIndexService
from app.application.services.skill_selector import SkillSelector
from app.domain.services.flows.planner_react import PlannerReActFlow
from app.domain.services.tools.a2a import A2ATool
from app.domain.services.tools.mcp import MCPTool
from app.domain.services.tools.skill import SkillTool
from app.domain.services.tools.skill_bundle_sync import SkillBundleSyncManager
from app.infrastructure.repositories.db_user_tool_preference_repository import (
    DBUserToolPreferenceRepository,
)
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository
from app.infrastructure.storage.postgres import get_postgres
from core.config import get_settings
from fastapi import UploadFile
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)

MESSAGE_STREAM_CHUNK_SIZE = 24
MESSAGE_STREAM_CHUNK_DELAY_SEC = 0.03
SKILL_CONTEXT_MAX_SKILLS = 6
SKILL_CONTEXT_MAX_TOTAL_CHARS = 8000
SKILL_CONTEXT_MAX_SNIPPET_CHARS = 1200


class AgentTaskRunner(TaskRunner):
    """基于Agent智能体的任务运行器"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: LLM,  # 大语言模型
        agent_config: AgentConfig,  # 智能体配置
        mcp_config: MCPConfig,  # mcp配置
        a2a_config: A2AConfig,  # a2a配置
        session_id: str,  # 会话id
        user_id: str | None,  # 用户id
        # session_repository: SessionRepository,  # 会话仓库
        file_storage: FileStorage,  # 文件存储桶
        # file_repository: FileRepository,  # 文件数据仓库
        json_parser: JSONParser,  # json解析器
        browser: Browser,  # 浏览器
        search_engine: SearchEngine,  # 搜索引擎
        sandbox: Sandbox,  # 沙箱
        skill_risk_policy: SkillRiskPolicy | None = None,  # skill风险策略
        overflow_config: ContextOverflowConfig | None = None,  # 上下文治理配置
    ) -> None:
        """构造函数，完成Agent任务运行器的创建"""
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._session_id = session_id
        self._user_id = user_id
        # self._session_repository = session_repository
        self._sandbox = sandbox
        self._mcp_config = mcp_config
        self._mcp_tool = MCPTool()
        self._a2a_config = a2a_config
        self._a2a_tool = A2ATool()
        settings = get_settings()
        self._skill_repository = FileSkillRepository(settings.skills_root_dir)
        self._skill_bundle_sync = SkillBundleSyncManager(
            sandbox=sandbox,
            skills_root_dir=settings.skills_root_dir,
            sandbox_skill_root=settings.skill_sandbox_bundle_root,
        )
        self._skill_tool = SkillTool(
            sandbox=sandbox,
            mcp_tool=self._mcp_tool,
            a2a_tool=self._a2a_tool,
            risk_mode=(skill_risk_policy or SkillRiskPolicy()).mode.value,
            bundle_sync_manager=self._skill_bundle_sync,
            skill_sandbox_bundle_root=settings.skill_sandbox_bundle_root,
        )
        self._skill_index_service = SkillIndexService(
            skill_repository=self._skill_repository,
            skills_root=settings.skills_root_dir,
        )
        self._skill_selector = SkillSelector(default_top_k=12)
        self._session_skill_pool: list[Skill] = []
        self._file_storage = file_storage
        self._overflow_config = overflow_config or ContextOverflowConfig()
        # self._file_repository = file_repository
        self._browser = browser
        self._flow = PlannerReActFlow(
            uow_factory=uow_factory,
            llm=llm,
            agent_config=agent_config,
            session_id=session_id,
            # session_repository=session_repository,
            json_parser=json_parser,
            browser=browser,
            sandbox=sandbox,
            search_engine=search_engine,
            mcp_tool=self._mcp_tool,
            a2a_tool=self._a2a_tool,
            skill_tool=self._skill_tool,
            overflow_config=self._overflow_config,
        )

    async def _put_and_add_event(
        self, task: Task, event: Event, persist: bool = True
    ) -> None:
        """往指定任务的消息队列中添加事件"""
        # 1.往任务的输出消息队列中新增事件
        event_id = await task.output_stream.put(event.model_dump_json())
        event.id = event_id

        # 2.按需将事件添加到会话中（流式中间片段不落库）
        if persist:
            async with self._uow:
                await self._uow.session.add_event(self._session_id, event)

    async def _stream_assistant_message_event(
        self, event: MessageEvent
    ) -> AsyncGenerator[MessageEvent, None]:
        """将助手消息切片成可增量渲染的事件流，提升前端流式观感"""
        text = event.message or ""
        if not text or len(text) <= MESSAGE_STREAM_CHUNK_SIZE:
            event.stream_id = event.stream_id or str(uuid.uuid4())
            event.partial = False
            yield event
            return

        stream_id = event.stream_id or str(uuid.uuid4())
        total = len(text)
        for end in range(MESSAGE_STREAM_CHUNK_SIZE, total + MESSAGE_STREAM_CHUNK_SIZE, MESSAGE_STREAM_CHUNK_SIZE):
            current_end = min(end, total)
            is_final = current_end >= total

            yield MessageEvent(
                role=event.role,
                message=text[:current_end],
                stream_id=stream_id,
                partial=not is_final,
                attachments=event.attachments if is_final else [],
                created_at=event.created_at,
            )

            if not is_final:
                await asyncio.sleep(MESSAGE_STREAM_CHUNK_DELAY_SEC)

    @classmethod
    async def _pop_event(cls, task: Task) -> Event:
        """从任务的输入流中获取事件信息"""
        # 1.从任务task中读取数据
        event_id, event_str = await task.input_stream.pop()
        if event_str is None:
            logger.warning(f"AgentTaskRunner接收到空消息")
            return

        # 2.使用pydantic+type类型将字符串转换成事件
        event = TypeAdapter(Event).validate_json(event_str)
        event.id = event_id

        return event

    async def _sync_file_to_sandbox(self, file_id: str) -> File:
        """根据文件id将文件同步到沙箱中"""
        try:
            # 1.调用文件存储下载文件信息
            file_data, file = await self._file_storage.download_file(file_id)

            # 2.组装沙箱文件路径
            filepath = f"/home/ubuntu/upload/{file.filename}"

            # 3.调用沙箱将文件上传至沙箱
            tool_result = await self._sandbox.upload_file(
                file_data=file_data, filepath=filepath, filename=file.filename
            )

            # 4.判断是否上传成功
            if tool_result.success:
                file.filepath = filepath
                async with self._uow:
                    await self._uow.file.save(file)  # 可以更新也可以不更新
                return file
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步文件[{file_id}]失败: {str(e)}")

    async def _sync_message_attachments_to_sandbox(self, event: MessageEvent) -> None:
        """将消息事件中的附件同步到沙箱中"""
        # 1.定义附件列表
        attachments: List[str] = []

        try:
            # 2.判断消息中是否存在附件
            if event.attachments:
                # 3.循环遍历所有的消息附件
                for attachment in event.attachments:
                    # 4.根据同步文件的id将数据同步到沙箱中
                    file = await self._sync_file_to_sandbox(attachment.id)

                    # 5.文件是否同步成功
                    if file:
                        attachments.append(file)
                        async with self._uow:
                            await self._uow.session.add_file(self._session_id, file)

            # 6.更新消息事件中的attachments
            event.attachments = attachments
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步消息附件到沙箱失败: {str(e)}")

    @classmethod
    def _get_stream_size(cls, f: BinaryIO) -> int:
        """根据传递的文件流，获取计算文件的大小"""
        # 1.记录当前文件指针位置
        current_pos = f.tell()

        # 2.将指针移动到文件末尾, seek，0: 偏移量、2: 相对文件末尾
        f.seek(0, 2)

        # 3.获取当前位置，也就是文件大小
        size = f.tell()

        # 4.恢复指针到原始位置
        f.seek(current_pos)

        return size

    async def _sync_file_to_storage(self, filepath: str) -> File:
        """将沙箱中指定的文件路径数据同步到存储桶中"""
        try:
            # 1.根据文件路径从会话中查找文件数据
            async with self._uow:
                file = await self._uow.session.get_file_by_path(
                    self._session_id, filepath
                )

            # 2.从沙箱中下载文件
            file_data = await self._sandbox.download_file(filepath)

            # 3.判断会话中的文件是否存在
            if file:
                async with self._uow:
                    await self._uow.session.remove_file(self._session_id, file.filepath)

            # 4.提取文件名字、文件信息并更新文件路径
            filename = filepath.split("/")[-1]
            content_type, _ = mimetypes.guess_type(filename)
            upload_file = UploadFile(
                file=file_data,
                filename=filename,
                size=self._get_stream_size(file_data),
                headers={"content-type": content_type or "application/octet-stream"},
            )

            # 5.上传文件到文件存储桶
            file = await self._file_storage.upload_file(upload_file)
            file.filepath = filepath

            # 6.往会话中新增一个文件信息
            async with self._uow:
                await self._uow.session.add_file(self._session_id, file)
            return file
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步消息附件到文件存储桶失败: {str(e)}")

    async def _sync_message_attachments_to_storage(self, event: MessageEvent) -> None:
        """将消息事件的附件同步到文件存储桶中"""
        # 1.定义附件列表存储数据
        attachments: List[File] = []

        try:
            # 2.判断消息中是否存在附件
            if event.attachments:
                # 3.循环遍历所有附件
                for attachment in event.attachments:
                    # 4.根据文件路径将数据同步到文件存储桶
                    file = await self._sync_file_to_storage(attachment.filepath)
                    if file:
                        attachments.append(file)

            # 5.更新时间中的附件列表资源
            event.attachments = attachments
        except Exception as e:
            logger.exception(f"AgentTaskRunner同步消息附件到存储桶失败: {str(e)}")

    async def _get_browser_screenshot(self) -> str:
        """获取浏览器截图并返回截图文件对应的在线URL"""
        # 1.调用浏览器完成截图
        screenshot = await self._browser.screenshot()

        # 2.将浏览器截图上传到文件存储中
        file = await self._file_storage.upload_file(
            UploadFile(
                file=io.BytesIO(screenshot),
                filename=f"{str(uuid.uuid4())}.png",
                size=self._get_stream_size(io.BytesIO(screenshot)),
            )
        )
        try:
            async with self._uow:
                await self._uow.session.add_file(self._session_id, file)
        except Exception as e:
            logger.warning(f"保存截图文件到会话失败: {str(e)}")

        # 3.优先返回预签名URL，避免私有桶直链403
        try:
            minio_store = getattr(self._file_storage, "minio_store", None)
            bucket = getattr(self._file_storage, "bucket", None)
            if minio_store and bucket and file.key:
                return await minio_store.presigned_get_url(
                    bucket_name=bucket,
                    object_name=file.key,
                    expiry_seconds=24 * 60 * 60,
                )
        except Exception as e:
            logger.warning(f"生成截图预签名URL失败，回退为原始链接: {str(e)}")

        # 4.预签名失败则回退为文件原始路径
        return file.filepath

    async def _load_enabled_skills(self) -> list[Skill]:
        """加载启用中的 Skill 列表。"""
        try:
            return await self._skill_index_service.list_enabled_skills()
        except Exception as e:
            logger.warning(f"加载Skill列表失败，降级为空列表: {str(e)}")
            return []

    async def _load_selected_skills(
        self,
        user_message: str,
        preference_map: dict[str, bool],
    ) -> list[Skill]:
        skills = await self._load_enabled_skills()
        filtered = self._filter_skills_by_user_preferences(skills, preference_map)
        return self._select_skills_from_pool(filtered, user_message)

    def _select_skills_from_pool(self, skill_pool: list[Skill], user_message: str) -> list[Skill]:
        """从给定技能池中选择本轮激活的技能子集。"""
        if not skill_pool:
            return []
        return self._skill_selector.select(skill_pool, user_message)

    @staticmethod
    def _strip_skill_frontmatter(skill_md: str) -> str:
        """移除 SKILL.md 的 YAML frontmatter，仅保留正文。"""
        raw = (skill_md or "").strip()
        if not raw.startswith("---"):
            return raw

        lines = raw.splitlines()
        if len(lines) < 3:
            return raw

        if lines[0].strip() != "---":
            return raw

        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :]).strip()
        return raw

    def _build_skill_context_prompt(self, skills: list[Skill]) -> str:
        """将已选中的 Skill 构建为运行时系统上下文。"""
        if not skills:
            return ""

        sections: list[str] = [
            "## Active Skills",
            "Follow these selected SKILL.md guides when they are relevant to the current task.",
        ]

        total_chars = sum(len(item) for item in sections)
        for skill in skills[:SKILL_CONTEXT_MAX_SKILLS]:
            manifest = skill.manifest if isinstance(skill.manifest, dict) else {}
            context_blob = str(manifest.get("context_blob") or "").strip()
            if context_blob:
                body = context_blob
            else:
                skill_md = str(manifest.get("skill_md") or "").strip()
                body = self._strip_skill_frontmatter(skill_md)
            body = re.sub(r"\n{3,}", "\n\n", body).strip()
            if len(body) > SKILL_CONTEXT_MAX_SNIPPET_CHARS:
                body = body[:SKILL_CONTEXT_MAX_SNIPPET_CHARS].rstrip() + "\n...(truncated)"

            if not body:
                body = (skill.description or "").strip()
            if not body:
                body = "No additional guide content."

            block = f"### {skill.name} ({skill.slug})\n{body}"
            if total_chars + len(block) > SKILL_CONTEXT_MAX_TOTAL_CHARS:
                break

            sections.append(block)
            total_chars += len(block)

        return "\n\n".join(sections)

    async def _load_user_preferences_map(self, tool_type: ToolType) -> dict[str, bool]:
        """加载用户在指定工具类型下的偏好映射。"""
        if not self._user_id:
            return {}

        try:
            postgres = get_postgres()
            async with postgres.session_factory() as session:
                pref_repository = DBUserToolPreferenceRepository(session)
                preferences = await pref_repository.get_by_user_id(self._user_id, tool_type)
                return {preference.tool_id: preference.enabled for preference in preferences}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "加载用户工具偏好失败，降级为默认启用(user_id=%s, tool_type=%s): %s",
                self._user_id,
                tool_type.value,
                str(e),
            )
            return {}

    @staticmethod
    def _apply_user_preferences_to_mcp_config(
        mcp_config: MCPConfig, preference_map: dict[str, bool]
    ) -> MCPConfig:
        """将用户偏好应用到 MCP 配置并返回副本。"""
        return MCPConfig(
            mcpServers={
                server_name: server_config.model_copy(
                    deep=True,
                    update={
                        "enabled": bool(
                            server_config.enabled
                            and preference_map.get(server_name, True)
                        )
                    },
                )
                for server_name, server_config in mcp_config.mcpServers.items()
            }
        )

    @staticmethod
    def _apply_user_preferences_to_a2a_config(
        a2a_config: A2AConfig, preference_map: dict[str, bool]
    ) -> A2AConfig:
        """将用户偏好应用到 A2A 配置并返回副本。"""
        return A2AConfig(
            a2a_servers=[
                server.model_copy(
                    deep=True,
                    update={"enabled": bool(server.enabled and preference_map.get(server.id, True))},
                )
                for server in a2a_config.a2a_servers
            ]
        )

    @staticmethod
    def _filter_skills_by_user_preferences(
        skills: list[Skill], preference_map: dict[str, bool]
    ) -> list[Skill]:
        """基于用户偏好过滤 Skill 列表。"""
        return [skill for skill in skills if preference_map.get(skill.id, True)]

    async def _handle_tool_event(self, event: ToolEvent) -> None:
        """额外处理工具消息，使其前端交互更友好"""
        try:
            # 1.如果事件状态为已调用则执行以下代码
            if event.status == ToolEventStatus.CALLED:
                # 2.工具为浏览器则补全工具浏览器工具内容
                if event.tool_name == "browser":
                    event.tool_content = BrowserToolContent(
                        screenshot=await self._get_browser_screenshot(),
                    )
                elif event.tool_name == "search":
                    # 3.工具为搜索则添加搜索工具内容
                    search_results: ToolResult[SearchResults] = event.function_result
                    logger.info(f"搜索工具结果: {search_results}")
                    event.tool_content = SearchToolContent(
                        results=search_results.data.results
                    )
                elif event.tool_name == "shell":
                    # 4.工具为shell则生成shell工具内容
                    if "session_id" in event.function_args:
                        shell_result = await self._sandbox.read_shell_output(
                            event.function_args["session_id"],
                            console=True,
                        )
                        event.tool_content = ShellToolContent(
                            console=(shell_result.data or {}).get("console_records", [])
                        )
                    else:
                        event.tool_content = ShellToolContent(console="(No console)")
                elif event.tool_name == "file":
                    # 5.工具为file则将文件同步到对象存储
                    if "filepath" in event.function_args:
                        filepath = event.function_args["filepath"]
                        file_read_result = await self._sandbox.read_file(filepath)
                        file_content: str = (file_read_result.data or {}).get(
                            "content", ""
                        )
                        event.tool_content = FileToolContent(content=file_content)
                        await self._sync_file_to_storage(filepath)
                    else:
                        event.tool_content = FileToolContent(content="(No Content)")
                elif event.tool_name in ["mcp", "a2a"]:
                    # 6.工具为mcp/a2a则处理调用结果
                    logger.info(
                        f"处理MCP/A2A工具事件, function_result: {event.function_result}"
                    )
                    if event.function_result:
                        # 7.如果结果包含data则提取data
                        if (
                            hasattr(event.function_result, "data")
                            and event.function_result.data
                        ):
                            logger.info(
                                f"MCP/A2A工具调用结果: {event.function_result.data}"
                            )
                            event.tool_content = (
                                MCPToolContent(result=event.function_result.data)
                                if event.tool_name == "mcp"
                                else A2AToolContent(
                                    a2a_result=event.function_result.data
                                )
                            )
                        elif (
                            hasattr(event.function_result, "success")
                            and event.function_result.success
                        ):
                            # 8.mcp/a2a工具调用正常，但是无结果产生
                            logger.info(
                                f"MCP/A2A工具调用成功返回，但无结果: {event.function_result}"
                            )
                            result_data = (
                                event.function_result.model_dump()
                                if hasattr(event.function_result, "model_dump")
                                else str(event.function_result)
                            )
                            event.tool_content = (
                                MCPToolContent(result=result_data)
                                if event.tool_name == "mcp"
                                else A2AToolContent(a2a_result=result_data)
                            )
                        else:
                            # 9.其他情况将结果转换成字符串进行传递
                            logger.info(f"MCP/A2A工具结果: {event.function_result}")
                            event.tool_content = (
                                MCPToolContent(result=str(event.function_result))
                                if event.tool_name == "mcp"
                                else A2AToolContent(
                                    a2a_result=str(event.function_result)
                                )
                            )
                    else:
                        logger.warning("MCP/A2A工具调用结果未发现")
                        event.tool_content = (
                            MCPToolContent(result="(MCP工具无可用结果)")
                            if event.tool_name == "mcp"
                            else A2AToolContent(a2a_result="(A2A智能体无可用结果)")
                        )
                elif event.tool_name == "skill":
                    if event.function_result and event.function_result.data is not None:
                        event.tool_content = SkillToolContent(
                            skill_result=event.function_result.data
                        )
                    elif event.function_result and event.function_result.message:
                        event.tool_content = SkillToolContent(
                            skill_result=event.function_result.message
                        )
                    else:
                        event.tool_content = SkillToolContent(
                            skill_result="(Skill工具无可用结果)"
                        )
        except Exception as e:
            logger.exception(f"AgentTaskRunner生成工具内容失败: {str(e)}")

    async def _run_flow(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        """根据消息对象运行PlannerReActFlow"""
        # 1.判断传递的消息是否为空
        if not message.message:
            logger.warning(f"AgentTaskRunner接收了一条空消息")
            yield ErrorEvent(error="空消息错误")
            return

        # 2.调用流并运行获取事件信息
        async for event in self._flow.invoke(message):
            # 3.判断是否为工具事件，如果是则额外处理
            if isinstance(event, ToolEvent):
                await self._handle_tool_event(event)
            elif isinstance(event, MessageEvent):
                # 4.如果是消息事件则将AI消息事件中的附件同步到存储中
                await self._sync_message_attachments_to_storage(event)

            # 5.将事件直接返回
            yield event

    async def _cleanup_tools(self) -> None:
        """清理MCP和A2A工具资源，确保在同一任务上下文中释放

        注意：该方法必须在初始化MCP/A2A的同一个asyncio Task中调用，
        否则anyio的cancel scope会检测到任务上下文切换并抛出RuntimeError。
        """
        try:
            if self._mcp_tool:
                await self._mcp_tool.cleanup()
        except Exception as e:
            logger.warning(f"清理MCP工具资源时出错: {e}")
        try:
            if self._a2a_tool and self._a2a_tool.manager:
                await self._a2a_tool.manager.cleanup()
        except Exception as e:
            logger.warning(f"清理A2A工具资源时出错: {e}")
        try:
            if self._skill_bundle_sync:
                await self._skill_bundle_sync.cleanup()
        except Exception as e:
            logger.warning(f"清理Skill bundle同步任务时出错: {e}")
        try:
            if self._skill_tool:
                await self._skill_tool.cleanup()
        except Exception as e:
            logger.warning(f"清理Skill工具资源时出错: {e}")
        self._session_skill_pool = []

    async def invoke(self, task: Task) -> None:
        """根据传递的任务处理agent消息队列并运行agent流"""
        try:
            # 1.任务一启动先推进会话状态，避免前端长期显示pending
            async with self._uow:
                await self._uow.session.update_status(
                    self._session_id, SessionStatus.RUNNING
                )

            # 2.确保沙箱、mcp、a2a均初始化完成
            logger.info(f"AgentTaskRunner任务处理开始")
            await self._sandbox.ensure_sandbox()
            mcp_preference_map = await self._load_user_preferences_map(ToolType.MCP)
            a2a_preference_map = await self._load_user_preferences_map(ToolType.A2A)
            skill_preference_map = await self._load_user_preferences_map(ToolType.SKILL)
            filtered_mcp_config = self._apply_user_preferences_to_mcp_config(
                self._mcp_config,
                mcp_preference_map,
            )
            filtered_a2a_config = self._apply_user_preferences_to_a2a_config(
                self._a2a_config,
                a2a_preference_map,
            )
            await self._mcp_tool.initialize(filtered_mcp_config)
            await self._a2a_tool.initialize(filtered_a2a_config)
            enabled_skills = await self._load_enabled_skills()
            self._session_skill_pool = self._filter_skills_by_user_preferences(
                enabled_skills,
                skill_preference_map,
            )
            initial_skills = self._select_skills_from_pool(
                self._session_skill_pool,
                "",
            )
            await self._skill_bundle_sync.prepare_startup_sync(
                skill_pool=self._session_skill_pool,
                initial_selected=initial_skills,
            )
            await self._skill_bundle_sync.await_initial_sync()
            await self._skill_tool.initialize(initial_skills)
            initial_skill_context = self._build_skill_context_prompt(initial_skills)
            if hasattr(self._flow, "set_skill_context"):
                self._flow.set_skill_context(initial_skill_context)
            self._skill_bundle_sync.start_background_sync()

            # 3.循环读取任务中的输入消息队列
            while not await task.input_stream.is_empty():
                # 4.从输入流中获取数据
                event = await self._pop_event(task)
                if event is None:
                    continue
                message = ""

                # 5.判断事件类型是否为消息事件，如果是则处理消息并将附件同步到沙箱中
                if isinstance(event, MessageEvent):
                    message = event.message or ""
                    await self._sync_message_attachments_to_sandbox(event)
                    logger.info(f"AgentTaskRunner接收到新消息: {message[:50]}...")

                # 6.将消息事件转换称消息对象
                message_obj = Message(
                    message=message,
                    attachments=(
                        [attachment.filepath for attachment in event.attachments]
                        if isinstance(event, MessageEvent)
                        else []
                    ),
                )

                selected_skills = self._select_skills_from_pool(
                    self._session_skill_pool,
                    message_obj.message,
                )
                await self._skill_tool.initialize(selected_skills)
                skill_context = self._build_skill_context_prompt(selected_skills)
                if hasattr(self._flow, "set_skill_context"):
                    self._flow.set_skill_context(skill_context)

                # 7.传递消息对象并运行PlannerReActFlow
                async for event in self._run_flow(message_obj):
                    emitted_events: List[Event] = []
                    if isinstance(event, MessageEvent) and event.role == "assistant":
                        async for chunked_event in self._stream_assistant_message_event(
                            event
                        ):
                            emitted_events.append(chunked_event)
                    else:
                        emitted_events.append(event)

                    for emitted_event in emitted_events:
                        # 8.将得到的事件添加到消息队列中
                        should_persist = not (
                            isinstance(emitted_event, MessageEvent)
                            and emitted_event.partial
                        )
                        await self._put_and_add_event(
                            task, emitted_event, persist=should_persist
                        )

                        # 9.如果事件类型为标题事件则更新会话标题
                        if isinstance(emitted_event, TitleEvent):
                            async with self._uow:
                                await self._uow.session.update_title(
                                    self._session_id, emitted_event.title
                                )
                        elif isinstance(emitted_event, MessageEvent) and not emitted_event.partial:
                            # 10.如果事件为最终消息事件，则更新最新消息并新增未读消息数
                            async with self._uow:
                                await self._uow.session.update_latest_message(
                                    self._session_id,
                                    emitted_event.message,
                                    emitted_event.created_at,
                                )
                                await self._uow.session.increment_unread_message_count(
                                    self._session_id
                                )
                        elif isinstance(emitted_event, WaitEvent):
                            # 11.如果事件为等待，则更新会话状态并终止程序
                            async with self._uow:
                                await self._uow.session.update_status(
                                    self._session_id, SessionStatus.WAITING
                                )
                            return

                # 12.判断如果输入消息队列为空则跳出循环
                if not await task.input_stream.is_empty():
                    break

            # 13.更新会话状态为已完成
            async with self._uow:
                await self._uow.session.update_status(
                    self._session_id, SessionStatus.COMPLETED
                )
        except asyncio.CancelledError:
            # 14.异步任务被取消，推送结束事件并更新状态
            logger.info(f"AgentTaskRunner任务运行取消")
            await self._put_and_add_event(task, DoneEvent())
            async with self._uow:
                await self._uow.session.update_status(
                    self._session_id, SessionStatus.COMPLETED
                )
            raise
        except Exception as e:
            # 15.记录日志并往任务队列/消息队列中写入异常事件并更新会话状态
            logger.exception(f"AgentTaskRunner运行出错: {str(e)}")
            await self._put_and_add_event(
                task, ErrorEvent(error=f"AgentTaskRunner出错: {str(e)}")
            )
            async with self._uow:
                await self._uow.session.update_status(
                    self._session_id, SessionStatus.COMPLETED
                )
        finally:
            # 16.在同一个asyncio Task上下文中清理MCP/A2A工具资源
            # 这是关键：streamablehttp_client内部使用anyio.create_task_group()，
            # 要求在同一个Task中进入和退出cancel scope，
            # 所以必须在invoke()的finally块（即初始化MCP的同一个Task）中清理
            await self._cleanup_tools()

    async def destroy(self) -> None:
        """销毁任务运行器并释放资源"""
        # 1.清除沙箱
        logger.info(f"开始清除销毁AgentTaskRunner资源")
        if self._sandbox:
            logger.info("销毁AgentTaskRunner中的沙箱环境")
            await self._sandbox.destroy()

        # 2.清除mcp和a2a工具（幂等操作，如果invoke()中已清理则不会重复执行）
        await self._cleanup_tools()

    async def on_done(self, task: Task) -> None:
        """任务结束时执行的回调函数"""
        logger.info(f"AgentTaskRunner任务执行结束")
