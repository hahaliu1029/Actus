# Changelog

本文件记录项目的版本变更。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

_最新批次：2026-07-16（沙箱供给模式收口）_

### 2026-07 核心能力更新

- **三档沙箱供给模式**：新增 env-only 的
  `SANDBOX_PROVISION_MODE=always|on_demand|off`。`on_demand` 延迟到首个沙箱或
  浏览器工具调用再单飞建容器；`off` 同时收缩工具、提示词、会话端点和前端工作台
  表面，并提供 `docker-compose.sandbox-off.yml` 与运维 runbook。
- **扩展与 Plugin 治理（D1a）**：统一 MCP / A2A / Skill 运行时清单，新增
  `off|shadow|enforce` 治理模式、准入与隔离、审计、pin/reapprove，以及 Plugin
  安装/卸载 saga；管理面路由位于 `/api/v1/runtime/extensions`、
  `/api/v2/extensions`、`/api/v2/plugins`。
- **本地 MinIO Compose 栈**：标准 Compose 默认启动 loopback-only MinIO，
  `minio-init` 幂等创建 bucket；内部 endpoint 与浏览器可访问的 public endpoint
  分离，并增加结构测试与 `scripts.verify_local_minio` 验证入口。
- **斜杠命令（B11）**：前端新增 `/help`、`/mcp`、`/skills`、`/cost`、
  `/permissions`、`/takeover`、`/compact`，并允许已启用 Skill 作为受校验的命令入口。
- **多模态文件链路（B12）**：图片/视频 `media_type`、PDF/文档结构化预览、文件处理
  缓存与并行页处理进入默认路径，工具事件保留对应的结构化渲染元数据。
- **统一生命周期事件（C7）**：新增 plan / step / tool / task / subagent 封闭事件词表、
  单一构造入口、SSE dual-emit 与前端 typed reducer；后端与前端开关继续默认关闭，按
  同版本 pod 灰度开启。
- **多智能体会话面**：补齐子会话树、成本树、合并时间线、研究子智能体入口和
  subagent run 可选持久化观测面。

### 2026-06 权限与协调器更新

- **Permission Engine 完成主路径接管**：native / Skill / MCP / A2A 来源统一进入
  `DefaultPermissionEngine`；审批持久化收敛到 ApprovalState Reader/Writer，旧
  `tool_approval_rules` fallback、legacy flags 与缓存旁路已经退役。
- **CoordinatorTaskRunner**：并行 work unit、mailbox supervisor、child scope、预算、
  取消/回收、shell-capable task、层级治理与 agent-team bundle 已接入；生产开关和
  canary/rollback 仍按 `CONTRIBUTING.md` 与 runbook 执行。

### Permission Engine (PE-1)

- **Skill source 内化进 `pe.evaluate()`**：新增 `permission/sources/` 子包（`PermissionSource` ABC + `NativeSource` 透传 + `SkillSource` 重算），把历史上 `react_graph.py:2243-2395` 的 Skill Stage P 旁路接入 PE step 5.5。Spec: `docs/superpowers/specs/2026-05-18-pe-1-skill-source-internalize-design.md`
- **Per-source 特性开关**：`permission_engine_skill_enabled` 现在真正驱动 PE Skill 路径；`permission_engine_native_enabled` 保持原语义；`is_pe_enabled_for_source(source, tc)` + `PE_SUPPORTED_SOURCES_AFTER_PE_1 = {"native","skill"}` 替换 10 处 native-only callsite
- **三个新异常**：`UnsupportedSource` (HTTP 422), `PEInfrastructureUnavailable` (HTTP 503), `PermissionConfigurationError` (DI fail-fast)。`agent_service` / `react_graph` 在 broad-except 之前显式重抛
- **Redis 单飞 (single-flight) 风险刷新**：`SkillSource` 用 Redis NX + Lua compare-and-delete + loser poll + 45s 失败缓存，防 5 路并发同一 (skill_id, content_hash) 的 confirm storm
- **`ConfirmationDetail` 兼容**：复用 `resolve_tool_source(pending_detail.tool_name).source` 推导，不扩 wire schema
- **INV-6 grep gate（warning-only）**：扫描 `risk_level_meta` / `SkillRiskAssessor` / `risk_enforce` 在白名单外的引用；PE-1b 翻成 hard-fail 并删除 R3 legacy
- **Grafana dashboard**：`monitoring/dashboards/permission_engine.json`（5 panels：skill_evaluate_count by outcome / skill_refresh status / singleflight_lost / unsupported_source / skill vs native deny）
- **SmartApprove eval 套件**：`api/tests/eval/permission_smart_approve/`（opt-in via `ACTUS_RUN_SLOW_EVALS=1`），10 条人工标注 corpus（5 skill + 5 native）

> Legacy R3 skill bypass `react_graph.py:2243-2395` 留 14 天 grace；PE-1b 删除。

### 新增

- **工具审批与确认系统**：当前实现已经收敛到 Permission Engine
  - **用户偏好**：`user_tool_approval_policy.py` 按 canonical tool name 保存 `auto` / `ask` / `deny`，覆盖 native / MCP / A2A / Skill
  - **Grant 持久化**：`approval_grant.py` + ApprovalState Reader/Writer 保存 session / always 范围的 approve/deny 决策；旧 `tool_approval_rules` 与 Redis `approval_cache` 旁路已退役
  - **运行时决策**：`permission/default_engine.py` 组合来源、用户偏好、child scope、风险与 escalation；Smart Approve 超时或基础设施错误时 fail-safe 回落人工确认
  - **暂停与恢复**：`permission/confirmation_queue.py` 承载 durable confirmation 与 resume preflight，前端 `tool-confirmation-card.tsx` 展示用户决策
- **会话事件恢复（SSE State Recovery）**：基于 Redis Stream 的 `infrastructure/external/event_recovery/redis_event_recovery.py`，支持刷新或断线重连后从最后位点恢复事件流；前端 `session-recovery.test.ts` 覆盖端到端恢复路径（对应 TODOS #23 E2）
- **记忆系统下沉到 Agent 工具层**：
  - 新增 `domain/services/tools/memory_tools.py`：`memory_search`（embedding 召回 + ranker 流水线）和 `memory_get`（按 chunk_id 取详情）两个 Agent 可调工具
  - 新增 `domain/services/memory_ranker.py`：cosine 相似度 → 时间衰减（半衰期，带 `evergreen` 元数据豁免）→ MMR 多样性重排的检索流水线
  - 新增 `domain/models/memory_chunk.py`、`memory_chunk_repository.py`、`infrastructure/models/memory_chunk_orm.py`、`db_memory_chunk_repository.py`
  - 新增 alembic 迁移 `f1a2b3c4d5e6_add_memory_chunks.py`
  - 对应 TODOS #14（Agent 记忆工具）和 #15（混合检索与排序）
- **Embedding 熔断器**：`infrastructure/external/embedding/circuit_breaker_embedding_provider.py` 包装 OpenAI Embedding，连续失败时打开熔断
- **会话 FINISHING 状态**：在 `RUNNING` 与 `COMPLETED` 之间引入 `FINISHING` 中间态，承载最终摘要、附件归档、未读计数刷新等异步收尾；`agent_task_runner.py` 加入 finishing 分支与 lifespan 清理逻辑（对应 TODOS #16 E1）
- **后台摘要生成**：`graphs/background_summary.py` 与 `graphs/step_metadata.py` 把摘要工作从主 graph 拆出来，避免阻塞主流程
- **执行健康监控（D5）**：新增 `execution_watchdog.py`（步骤级超时看门狗）+ `execution_metrics.py`（执行指标采集），对应 TODOS #19
- **提示词模块化（B5）**：新建 `domain/services/prompts/` 子包，将单文件 prompt 拆成可组合单元
  - `assembler.py` / `section.py` / `render_context.py` / `budget.py` / `invariants.py` / `errors.py`
  - `sections/`：`identity`、`behavior_core`、`output_format`、`planner_identity`、`planner_tool_summary_legacy`、`sandbox_state`、`skill_context`、`tools_guide_dynamic`、`tools_guide_stable`、`conversation_summaries`
  - `bundles/`：中英文 prompt bundle
  - `reminders/`：可注册的提醒（`plan_mode`、`file_truncated`、`skill_install_confirm`）
  - 强约束：`updater_node` 是 **唯一** 允许写 `state.skill_context` 的节点；CI gate `test_executor_no_skill_context_writeback.py` 通过 AST 扫描禁止 `executor_node` 写回（对应 TODOS #20）
- **工具失败追踪器**：`tools/tool_failure_tracker.py` 跟踪每个工具的连续失败次数，触发降级或回退
- **JSON Envelope**：`domain/services/json_envelope.py` 统一工具调用结果的封装格式，便于前后端解析对齐
- **Telemetry hooks**：`infrastructure/external/llm/_telemetry_mixin.py` + `infrastructure/telemetry/prompt_telemetry.py`，记录 LLM 调用与 prompt 组装的关键指标；新增 `domain/external/telemetry.py` 协议
- **多语言贯通**：`Message.language` 字段贯穿，`AgentService` / `AgentTaskRunner` 现在显式传递初始语言；prompt 按 language 派发 bundle（对应 TODOS #29）
- **`shell_execute` 同步等待参数**：沙箱 `shell.py` 与 schema 新增 `wait_seconds`，调用方可指定明知短命令的同步等待时长，缩短"还要异步轮询"链路
- **会话健康事件**：在状态机和事件流中新增 health 事件类型，前端可展示后端探活/降级状态
- **前端 Markdown 渲染升级**：改用 `react-markdown` + `shiki` 实现稳定 markdown 渲染与代码语法高亮
- **前端 Settings 视觉/音频面板**：`manus-settings.tsx` 文件理解配置区域重设计
- **认证与限流重构**：`interfaces/dependencies/rate_limit.py` 重写；`AppConfig` 加入读取缓存层（`test_service_dependencies_cache.py` 覆盖）

### 变更

- `agent_task_runner.py` 进一步增强：集成审批确认、事件恢复、记忆工具、FINISHING 状态、telemetry、language plumbing；总行数从约 700 涨到约 2000
- `planner_react.py` 重构：DI 门控 (`test_planner_react_di_gate.py`)、deferred tool 收集、skill context 局部化消费
- `react_graph.py` / `main_graph.py`：集成 prompt assembler 输出、step_metadata 局部传递、updater 单写入约束
- `service_dependencies.py`：加入 `_llm_fingerprint` 与 `_build_config_snapshot` 缓存，避免重复构造 LLM/Embedding 适配器
- `actus_chat_model` / `actus_responses_model` clone 路径补齐 `connect_timeout_seconds` 透传
- `docker-compose.yml`：加入 `PYTHONUNBUFFERED=1`，确保容器日志实时输出；config.yaml 挂载路径调整
- 路由层：`session_routes.py` 新增 `GET /sessions/{id}/events?since=...` 事件恢复端点；`auth_routes.py` 的 `register` / `login` / `refresh_token` / `wechat_authorize` / `wechat_callback` 全部接入 `rate_limit_auth` 依赖；`schemas/auth.py` 把 `RegisterRequest.password` 长度约束从 `min=6, max=128` 调整为 `min=8, max=72`
- 测试体量：单 PR 新增 ~70 个测试文件，覆盖 prompt sections、reminders、bundles、approval、watchdog、recovery、memory chunks 集成、telemetry hooks、timeout hooks、prompt language dispatch、skill context provider wiring 等

### 修复

- **codex review 残留项（commit `17c5a03`）**：
  - `_build_config_snapshot` 构造 `vision_llm_config` 时显式继承主 config 的 `timeout_seconds`，修复视觉 fallback adapter 永远跑在 Pydantic 默认 120s 的回归
  - TODOS #27 SkillTool.initialize 原子性的 rollback recovery 路径
  - LLM fallback adapter `provider_name` 命名一致性
- planner 与 react 之间的 skill_context 漂移（`test_executor_prompt_assembler_parity.py` 锁住）
- `executor_node` 在恢复路径中错误注入"用户已完成接管"消息的旧分支（`test_executor_resume_path.py`）

## [Pre-release 2026-04-04]

### 新增

- **上下文溢出治理**：两级渐进压缩（85% LLM 摘要 / 95% 硬截断）+ 同步三阶段裁剪（TokenEstimator、ContextAssembler、GradualCompactor）
- **多模态文件理解**：音频转录（Whisper API / sandbox faster-whisper）、PDF 解析（原生 / pymupdf4llm）、图片处理、视频关键帧提取 + 视觉模型分析
- **基于 Embedding 的 Skill 语义选择**：numpy 向量索引 + OpenAI Embedding + Redis 缓存
- **渐进式 MCP 工具发现**：`list_mcp_tools` / `get_mcp_tool` 两阶段加载
- **SKILL.md 解析与导出**：支持 YAML frontmatter + markdown body 双向转换
- **Checkpointer 连接池**：psycopg AsyncConnectionPool，应用生命周期管理
- **Memory Flush 服务**：后台异步记忆刷新，指数退避 + 熔断器（3 次失败 → 300s 冷却）
- **消息清洗器**：LLM 调用前自动过滤不合规多模态内容（图片 >5MB、无效 MIME、PDF >50MB）
- **SSH 隧道**：可选 autossh 反向隧道，Docker profile 启用
- **文件传输面板**：前端上传/下载进度跟踪（EMA 测速、取消、重试）
- **Axios 客户端**：文件传输专用 HTTP 客户端，支持 401 自动刷新 token
- **CORS 配置与请求体大小限制**
- **sandbox_exec 辅助工具**：处理长时间运行的沙箱命令

### 变更

- `agent_task_runner.py` 大幅增强：集成上下文治理、文件理解、Embedding 选择
- `planner_react.py` 重构：支持动态工具收集、MCP 发现、Skill 指南注入
- `react_graph.py` 扩展：集成上下文裁剪和压缩流程
- `main_graph.py` 优化：支持中断节点、改进路由逻辑
- `actus_chat_model.py` 增强：支持视觉模式、PDF 输入、消息清洗
- 前端设置页扩展：新增文件理解配置面板
- **[D5.1] LLM per-call hard timeout** (`TODOS.md #22`)：
  1. `LLMConfig.timeout_seconds` default 120s and `MemoryConfig.summary_timeout_seconds` default 30s are added as assumption-based defaults. No production wall-time data supports these values; they are expected to be re-evaluated after wall-time sampling (see TODOS #24 B5.5 bench). Existing `config.yaml` without the new fields will inherit the Pydantic defaults.
  2. `ActusFallbackChatModel` (when `api_type=auto`) behavior changes: each child adapter now enforces its own hard timeout. Worst case fallback path wall-time is now bounded (~240s at the default 120/120 budget) where previously it was unbounded.
  3. Escape hatch: `timeout_seconds: 0` disables the per-call `asyncio.wait_for` wrap for debugging. This reverts to the pre-D5.1 behavior but **still** disables SDK retries (see point 5).
  4. `AsyncOpenAI` client construction now passes `max_retries=0`, disabling SDK-level retry. LangGraph `RetryPolicy(max_attempts=3)` at `react_graph.llm_node` and `main_graph.planner_node` is now the single retry authority. This prevents a worst case of `3 (graph) × 3 (SDK) = 9` HTTP attempts per logical call.
  5. `service_dependencies._build_llm` logs a budget warning when `api_type=auto` and `primary + fallback > 200s`（derived as `ExecutionWatchdog.total_timeout_seconds / 3 graph retries = 600 / 3 = 200`，即 `primary + fallback > 200s`）. Warning only — no hard raise. Extracted as the module-level constant `_FALLBACK_BUDGET_WARNING_THRESHOLD_SECONDS`.
  6. All three LLM adapters (`ActusChatModel` / `ActusResponsesModel` / `ActusFallbackChatModel`) and their `bind_tools` / `with_structured_output` clone paths now propagate `timeout_seconds` to the cloned instance. Without this, `react_graph.py:180` and `planner_react.py:501-503` would silently drop user-configured timeouts.
  7. Shared helper: `api/app/infrastructure/external/llm/_timeout_helpers.py` with a free function `with_llm_timeout(adapter, coro)` — mirrors the existing `_telemetry_mixin.py` idiom.
  8. `_build_config_snapshot` 构造 `vision_llm_config` 时现在也继承
     `app_config.llm_config.timeout_seconds`(Codex review 发现的漏传 ——
     之前 vision fallback adapter 永远使用 `LLMConfig.timeout_seconds` 的
     Pydantic 默认 120s，不响应用户在主 config 里的覆盖；下游 `image.py` /
     `video.py` 的视觉描述/帧抽取路径因此一直跑在 120s adapter timeout 下)。
     `VisionFallbackConfig` 没有独立 `timeout_seconds` 字段 —— 默认行为是
     继承主 config，和 `summary_llm` 在 `summary_timeout_seconds=None` 时
     的继承语义对齐。如果未来需要 vision 独立 timeout，可以在
     `VisionFallbackConfig` 加可选字段。
- **[D5.2] httpx connect-phase timeout**：补齐 D5.1 漏掉的连接阶段超时预算。
  1. **背景**：D5.1 只给 `_agenerate` / `_astream` 包了外层 `asyncio.wait_for(timeout_seconds)`，但 `AsyncOpenAI(...)` 客户端构造时没传 `timeout=` 参数，因此 httpx 仍然使用 SDK 默认 `Timeout(connect=5.0, read=600, write=600, pool=600)`。慢网 / VPN / 连接池复用失效 / DNS 漂移等场景下一次 TLS 握手稍慢就会直接抛 `httpcore.ConnectTimeout → httpx.ConnectTimeout → openai.APITimeoutError`，外层 `asyncio.wait_for` 根本来不及生效 —— 提 `timeout_seconds` 对这类失败完全无效。D5.1 spec 第 13 行把该缺口描述成「仅 600s read timeout」，遗漏了 5s connect 这条独立天花板；spec 第 155/620 行据此把 `AsyncOpenAI(timeout=httpx.Timeout(...))` 当作「双保险 / 过度设计」否掉，结论只对 read 阶段成立，对 connect 阶段被本修复反证。
  2. **新增字段** `LLMConfig.connect_timeout_seconds: float = Field(60.0, ge=1.0, le=300.0)`。与 `timeout_seconds` 独立是因为「不要为死端点等 300s」和「不要为一次模型生成等 300s」是两个 SLO。默认 60s 比 SDK 默认大一个数量级，覆盖典型慢网 / 跨境 TLS / 连接池 churn 场景；上界 300s 留给极端环境；真正死端点仍能在 ≤ 1 分钟失败。
  3. **`ActusChatModel._get_client` / `ActusResponsesModel._get_client`** 构造 `AsyncOpenAI` 时新增 `timeout=httpx.Timeout(<default>, connect=self.connect_timeout_seconds)`，其中 `<default>` 用 `self.timeout_seconds if self.timeout_seconds > 0 else None`：`timeout_seconds == 0` 的 escape hatch 下 read/write/pool 回到无限制，但 **connect 依然被 `connect_timeout_seconds` 硬切**，保证「禁用主 timeout 包装」不会倒退到「连死端点都要等 5s」的原始行为。
  4. **Clone 路径**：`ActusChatModel.bind_tools` / `ActusResponsesModel.bind_tools` 的 clone 构造同步加 `connect_timeout_seconds=self.connect_timeout_seconds`。否则 `react_graph.py` 和 `planner_react.py` 里 `llm.bind_tools(tools)` / `with_structured_output(...)` 产生的克隆实例会静默丢失这个字段，跟 D5.1 `timeout_seconds` clone hazard 的语义完全一致。
  5. **`service_dependencies`**：`_llm_fingerprint` 加入 `str(llm_config.connect_timeout_seconds)` 参与哈希（不同 connect 预算不能共享缓存的 adapter 实例）；`_build_llm` 对 `ActusChatModel` / `ActusResponsesModel` 透传新字段；`_build_config_snapshot` 构造 `vision_llm_config` 时显式继承 `app_config.llm_config.connect_timeout_seconds`（与 D5.1 对 `timeout_seconds` 的继承逻辑对称）。`summary_llm_config` 走 `model_copy(update=...)`，新字段自动携带无需改动。
  6. **迁移路径**：旧 `config.yaml` 不需要改 —— 不写 `connect_timeout_seconds` 会自动继承 Pydantic 默认 60s；已有 `timeout_seconds: 300` 的配置从此同时获得 60s connect 保护。`config.yaml.example` 新增注释示例说明什么场景该提高。

### 修复

- `exec_and_wait` 函数处理未返回状态的情况
- 执行子步骤提示词模板中的字符串引号格式
- `MemoryConfig` 中 `summary_min_steps` 默认值调整为 1
- `file_processor_lookup` 参数类型注释修正

## [Pre-release 2026-03-07]

### 文档

- 全量刷新仓库核心 Markdown 文档，使其与当前代码结构、Compose 服务、会话接管能力、Skill v2 路由和本地开发方式保持一致。
- 修正文档中关于 `api/config.yaml`、`sandbox-image`、本地后端环境变量、前端构建时 API 地址注入方式的过时描述。
- 重写中英文 README、部署说明、架构说明、子项目 README 和中英文 API 参考文档。

## [Pre-release 2026-02-24]

### 变更

- 沙箱 Shell 执行链路优化：长时间命令（如 `npm start`、`pip install`、`apt-get install`）不再阻塞主流程，`shell_execute` 会在短等待后返回 `running`，可通过读取输出/等待进程继续跟踪。
- Shell 输出读取任务增加会话级生命周期管理（启动前回收旧 reader，结束后清理 task），降低并发读输出与后台任务泄漏风险。
- Shell 输出统一增加最大长度截断策略，降低长命令持续输出导致的内存增长风险。
- 安装类命令自动非交互化增强：对 `apt/apt-get`、`yum/dnf`、`apk`、`pip`、`npm/yarn/pnpm/npx`、`conda`、`poetry` 注入常见非交互参数，减少卡在确认提示的概率。
- 设置页「模型提供商」补全全部 LLM 配置项的中文说明文案，便于理解参数语义。

### 文档

- 更新中英文 API 文档中的 `LLMConfig` 字段定义，补充上下文溢出治理相关配置项。
- 补充部署文档与 README：说明 `sandbox-image` 为 Compose 服务名，并提供沙箱代码变更后的正确重建命令。

## [0.1.0] - 2025-XX-XX

### 新增

- ReAct Agent 引擎（Reasoning + Acting 推理循环）
- MCP (Model Context Protocol) 工具协议集成，支持动态接入外部工具服务器
- A2A (Agent-to-Agent) 智能体间通信协议支持
- Planner + ReAct 两阶段 Agent 流程编排
- Docker 沙箱隔离代码执行环境
- Playwright + DOM 索引提取方案浏览器自动化，通过 CDP 连接沙箱 Chromium，实时提取可交互元素并通过选择器精确操作
- 多模型支持（兼容 OpenAI API 格式：DeepSeek、Kimi 等）
- 流式输出与思考过程展示
- MinIO/S3 兼容的文件上传下载
- JWT 认证与角色权限管理
- Next.js 16 + React 19 现代前端
- Docker Compose 一键部署
- PostgreSQL + Redis 数据存储
- Skill 生态系统：独立的 Skill 扩展层，基于文件系统存储（`/app/data/skills`），支持 GitHub / 本地目录双来源安装
- SKILL.md 驱动的安装规范，支持 frontmatter 自动解析，manifest 为可选兼容字段
- Skill 安全机制：命令黑名单、风险策略（off / enforce_confirmation）、路径穿越防护、bundle 大小限制
- Skill 选择器：基于关键词评分的渐进式候选 Skill 推荐
- Skill 索引服务：基于目录 mtime 的缓存失效机制
- Skill v2 API（`/api/v2/skills/*`），v1 API 返回 410 引导迁移
