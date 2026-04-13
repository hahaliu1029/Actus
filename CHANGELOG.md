# Changelog

本文件记录项目的版本变更。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased] - 2026-04-04

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

### 修复

- `exec_and_wait` 函数处理未返回状态的情况
- 执行子步骤提示词模板中的字符串引号格式
- `MemoryConfig` 中 `summary_min_steps` 默认值调整为 1
- `file_processor_lookup` 参数类型注释修正

## [Unreleased] - 2026-03-07

### 文档

- 全量刷新仓库核心 Markdown 文档，使其与当前代码结构、Compose 服务、会话接管能力、Skill v2 路由和本地开发方式保持一致。
- 修正文档中关于 `api/config.yaml`、`sandbox-image`、本地后端环境变量、前端构建时 API 地址注入方式的过时描述。
- 重写中英文 README、部署说明、架构说明、子项目 README 和中英文 API 参考文档。

## [Unreleased] - 2026-02-24

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
