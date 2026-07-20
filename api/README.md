# Actus API

`api/` 是 Actus 的 FastAPI 后端，负责会话编排、Agent 执行、认证授权、文件管理、运行时配置、Skill 生态以及 Docker 沙箱调度。

## 当前能力

- 用户注册、登录、刷新令牌、个人资料维护
- 超级管理员用户管理
- 会话创建、SSE 对话流、任务停止、文件读取
- 会话事件恢复（Redis Stream 单调 `seq` 游标）、前后台执行快照、取消/挂起重试与额度查询
- `shell` / `browser` 接管、续期、结束、补救
- 会话树与子智能体：后代列表、成本树、1-3 个只读研究子任务、mailbox control plane
- 归一化 lifecycle 事件投影：`task` / `plan` / `step` / `tool` / `subagent`，支持独立灰度开关
- LLM / MCP / A2A / Skill 风险策略配置
- 运行时扩展聚合：MCP / A2A / Skill / Plugin 的启停、健康探测、liveness、调用统计和推荐目录
- 扩展治理：`off | shadow | enforce` 三态、pin/观测/隔离/reapprove/审计，以及 Plugin 安装/补偿/卸载 saga
- Skill v2 安装（GitHub / 本地 / SKILL.md）、启用、删除、详情查看、AI 创建
- 多模态文件理解：音频转录、PDF 解析、图片处理、视频帧分析
- 上下文溢出治理：两级渐进压缩 + 同步三阶段裁剪
- 模块化提示词系统（B5）：sections / bundles / reminders / assembler / budget
- Agent 记忆系统：`memory_search` / `memory_get` 工具 + 检索流水线（cosine 相似度 → 时间衰减 → MMR 多样性重排）
- 工具审批与确认系统：风险评估、智能批准、明确确认、审批缓存、持久化日志
- 执行健康监控（D5）：步骤级 watchdog + 执行指标 + 工具失败追踪 + 统一 JSON Envelope
- LLM 调用预算：连接 / 读取阶段独立 timeout，与 LangGraph RetryPolicy 对齐
- 基于 Embedding 的 Skill 语义选择、渐进式 MCP 工具发现
- Embedding 熔断器（避免级联失败）
- Checkpointer 连接池（psycopg AsyncConnectionPool）
- 后台记忆刷新（Memory Flush，含指数退避 + 熔断器）
- 多语言贯通：`Message.language` 派发中英文 prompt bundle
- `always | on_demand | off` 三档沙箱供给；`on_demand` 纯聊天不创建容器，`off` 关闭沙箱能力面
- 文件上传、下载、删除
- 健康检查与 MinIO 自检；对象读写 endpoint 与公开签名 URL endpoint 分离

## 运行依赖

- Python 3.12
- PostgreSQL
- Redis
- MinIO / S3 兼容对象存储
- Docker（用于创建会话沙箱）

容器镜像额外内置：

- Docker CLI（支持 stdio 类 MCP 服务）
- Node.js 22（满足部分工具运行需求）

## 本地开发

### 1. 选择运行拓扑

完整联调优先使用仓库根目录的 Compose：

```bash
cp .env.example .env
cp api/config.yaml.example api/config.yaml
# 编辑 .env：设置强密码/JWT_SECRET_KEY，并把 MEMORY_ROOT_HOST 改为已创建的宿主机绝对路径
docker compose --env-file .env up -d --build
```

只在宿主机运行 FastAPI 时要注意：标准 Compose 的 PostgreSQL/Redis 只接入
`actus-net`，默认不发布到 host。仅执行 `docker compose up -d postgres redis` 后，
`127.0.0.1:5432/6379` 仍不可访问。请使用已经可从 host 访问的开发 PostgreSQL/Redis，
或由开发者明确选择并维护一套独立的端口发布拓扑；不要把集成测试指向 live `manus` 库。

MinIO 是例外：标准 Compose 将 S3 API/控制台仅绑定到 loopback，因此可单独启动：

```bash
docker compose --env-file .env up -d minio minio-init
docker compose --env-file .env build sandbox-image
```

`sandbox-image` 只构建镜像；真正的会话沙箱由 API 按供给模式动态创建。

### 2. MinIO 的内部/公开 endpoint

标准 Docker Compose 中：

- API 容器通过 `MINIO_ENDPOINT=minio:9000` 读写对象
- 宿主机/浏览器通过 `http://127.0.0.1:${MINIO_API_PORT:-9000}` 访问 S3 API；默认地址为 `http://127.0.0.1:9000`
- 控制台为 `http://127.0.0.1:${MINIO_CONSOLE_PORT:-9001}`；默认地址为 `http://127.0.0.1:9001`
- `minio-init` 幂等创建 `${MINIO_BUCKET_NAME:-a2a-mcp}`
- 预签名 URL 使用 `MINIO_PUBLIC_ENDPOINT`；未显式覆盖时 Compose 自动取 `localhost:${MINIO_API_PORT:-9000}`

`MINIO_ENDPOINT` 只服务后端内部 I/O，`MINIO_PUBLIC_ENDPOINT` 只决定公开 URL 的
host/port（不带 scheme/path），TLS 由 `MINIO_PUBLIC_SECURE` 控制。显式设置公开 endpoint
时必须同时设置 `MINIO_REGION`。`tunnel` profile 只转发 API，不转发 MinIO；远程消费者
必须能直接访问 public endpoint。Compose 固定的归档 MinIO 镜像用于可复现本地开发，
不作为生产基线；生产环境应使用部署者维护的远程 S3 或受保护的 TLS endpoint。

登录后可调用 `GET /api/status/minio?smoke=true` 做 put/get/remove 自检，管理员也可调用
`POST /api/status/minio/upload` 验证上传和预签名 URL。`scripts.verify_local_minio` 是 CI/
隔离验收工具，只接受 `localhost:19000` 的严格环境并以 `write`、`verify` 两阶段验证
重建后的持久化，不是通用生产探测脚本。

### 3. 准备宿主机后端配置

在 `api/` 下创建业务配置和独立的 `.env`。不要直接软链仓库根 `.env`：Compose 内部主机名
与 host-run 的 loopback 地址不同，混用会把本地后端连到错误目标。

```bash
cd api
cp config.yaml.example config.yaml
```

```dotenv
ENV=development
LOG_LEVEL=INFO
APP_CONFIG_FILEPATH=config.yaml
SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/manus
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
MINIO_ENDPOINT=localhost:9000
MINIO_PUBLIC_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_REGION=us-east-1
MINIO_SECURE=false
MINIO_PUBLIC_SECURE=false
MINIO_BUCKET_NAME=a2a-mcp
JWT_SECRET_KEY=replace-with-a-strong-random-string
SANDBOX_IMAGE=actus-sandbox:latest
SANDBOX_NAME_PREFIX=actus-sb
SANDBOX_PROVISION_MODE=always
EXTENSION_GOVERNANCE_MODE=off
```

### 4. 安装依赖并启动

后端所有 Python 命令必须走项目 venv；推荐在仓库根安装依赖，再由 `uv run` 启动：

```bash
uv sync
cd api
uv run bash dev.sh
```

启动后：

- API 文档：`http://localhost:8000/docs`
- OpenAPI：`http://localhost:8000/openapi.json`

### 5. 沙箱供给模式

`SANDBOX_PROVISION_MODE` 是 env-only 部署开关，修改后需重启 API：

- `always`（默认）：每个父会话预置沙箱
- `on_demand`：首次沙箱工具/VNC 等真实需求才创建；纯聊天会话不创建容器
- `off`：不注册沙箱相关工具，AI Skill 创建返回 `409 SANDBOX_DISABLED`，VNC/终端接管
  WebSocket 返回状态后以 `4409` 关闭

`off` 的 canonical Compose 部署必须使用仓库根的 `docker-compose.sandbox-off.yml`，它会
固定 `SANDBOX_PROVISION_MODE=off`、移除 `sandbox-image` 和 Docker socket。切换前需要 drain
存量沙箱，并把三个 coordinator flags 全部关闭，否则 API 启动会 fail-fast。完整步骤见
`docs/runbooks/sandbox-off-runbook.md`；该 override 需要 Docker Compose 2.24.4 及以上。

### 6. 扩展治理与 Plugin 灰度

`EXTENSION_GOVERNANCE_MODE` 是启动时读取的 env-only 开关：

- `off`：不构造 registry/admission、治理服务和 Plugin 服务，也不执行启动 reconcile/saga
  收尾。它不是卸载操作：已有 Plugin registry 行、bundle 和已物化成员不会删除；Plugin
  管理/父项投影不可见，MCP/A2A/Skill 成员仍按原 config/Skill store 工作，且不执行
  `parent_blocked` 等治理阻断。
- `shadow`：检测类原因（`unknown`、`unpinned`、`pin_stale`、`config_drift`、
  `pin_mismatch`、`registry_unavailable`）只记录并放行；行政/结构类原因
  （`quarantined`、`disabled`、`deleted`、`parent_blocked`）仍会阻断。
- `enforce`：检测类与行政/结构类原因都阻断。

推荐按 `off -> shadow -> enforce` 切换：先在 `shadow` 下查看
`GET /api/v2/extensions/governance` 和 `/api/v2/extensions/audit`，刷新观测并 approve
计划启用的 pin，再进入 `enforce`。每次修改仓库根 `.env` 后都要重建 API 容器，单纯编辑
文件不会刷新进程级 settings 和 lifespan 装配：

```bash
docker compose --env-file .env up -d --force-recreate api
```

Plugin 安装请求只支持 `source_type=local|github`；`local` 必须指向绝对目录，历史值
`mcp_registry` 与压缩包输入会返回 `422`。建议先传 `dry_run=true` 获取零写入 preview；
preview 不做碰撞查询和最终安装门，真实安装仍会重新判定。在 `enforce` 下，`caution` 需要
`acknowledge=true`，`dangerous` 需要 `force=true`；MCP/A2A probe 失败也只有
`force=true` 可放行，声明 hash 不匹配和身份碰撞不能被两个标志绕过。

安装成功只返回 `plugin_ext_id`、`operation_id`、`status`，不返回 revision。启停或卸载前
调用 `GET /api/v2/plugins` 取得 `row_revision`。`/v2/plugins/{id}/enabled` 与
`/v2/extensions/plugin/{id}/governance-enable|governance-disable` 复用同一个父治理行和 CAS
状态迁移，不是两层开关；停用父项只让成员变为 `parent_blocked`，不会改写各成员配置。

## 配置来源

### 本地运行

- 环境变量：`api/.env`
- 业务运行时配置：`api/config.yaml`
- `SANDBOX_PROVISION_MODE` / `EXTENSION_GOVERNANCE_MODE` 属于 env-only 安全开关，不写入 `config.yaml`
- lifecycle 灰度开关位于 `config.yaml` 的 `lifecycle_runtime`；总开关与 subagent 开关默认均为 `false`

### Docker Compose 运行

- 环境变量：仓库根目录 `.env`
- 业务运行时配置：宿主机 `api/config.yaml` 绑定到 `/app/data/config.yaml`
- Skill 目录：`/app/data/skills`

`FileAppConfigRepository` 在无配置文件时可以按默认值创建，但标准 Compose 已显式绑定
`./api/config.yaml`，首次部署应先从 `api/config.yaml.example` 复制，避免 Docker 把缺失的
宿主机文件路径创建成目录。

## 测试

```bash
cd api
uv run pytest
```

默认命令只跑 unit；`tests/integration/` 需要隔离的 `manus_test` PostgreSQL。标准 Compose
的 PostgreSQL 不发布 host 端口，且禁止把集成测试 URL 指向 live `manus` 数据库；本地未
准备隔离数据库时，可以只跑 focused unit/contract/structure tests，并由对应 CI job 验证
integration。`backend-test` 不覆盖 `coordinator_recovery`、`slow`、`sandbox`、
`browser_eval`；coordinator 与 sandbox 分别由专用 job 负责，slow/browser eval 需按需运行。

## 目录结构

```text
api/
├── app/
│   ├── application/      # 用例编排服务（Agent, Session, Skill, Extension/Plugin, Sandbox 等）
│   ├── domain/           # 领域模型、工具、流程、Prompt、上下文治理
│   │   ├── models/       # 领域模型（app_config, skill, memory_chunk, approval_grant, session 等）
│   │   ├── external/     # 外部依赖协议（file_processor, embedding, memory_flusher, event_recovery, telemetry）
│   │   ├── services/
│   │   │   ├── graphs/   # LangGraph 图（main_graph, react_graph, compaction, context_assembler,
│   │   │   │             #               token_estimator, background_summary, step_metadata, message_utils）
│   │   │   ├── flows/    # 流程编排（planner_react, skill_creation_graph）
│   │   │   ├── tools/    # LangChain 工具（file, shell, browser, mcp_discovery, dynamic_skill,
│   │   │   │             #               memory_tools, tool_failure_tracker）
│   │   │   ├── prompts/  # 模块化 Prompt 子包（assembler, section, render_context, budget,
│   │   │   │             #                    invariants, errors, sections/, bundles/, reminders/）
│   │   │   ├── permission/ # Permission Engine、来源适配、确认队列、Smart Approve provider
│   │   │   ├── approval_state_reader.py
│   │   │   ├── lifecycle_emit.py / lifecycle_projector.py / execution_watchdog.py
│   │   │   ├── memory_ranker.py / json_envelope.py
│   │   │   └── skill_md_parser.py / skill_md_exporter.py
│   │   └── repositories/ # 仓库接口 (ABC)，含 memory_chunk / approval_grant / user_tool_approval_policy
│   ├── infrastructure/   # 仓储实现、外部服务、存储客户端
│   │   ├── models/       # ORM（memory、approval、extension registry/plugin saga、session 等）
│   │   ├── repositories/ # 仓储实现（含 db_memory_chunk_repository, db_approval_grant_repository, db_user_tool_approval_policy_repository）
│   │   ├── telemetry/    # Prompt + LLM telemetry 实现
│   │   ├── external/
│   │   │   ├── llm/      # LLM 适配器 + 消息清洗器 + telemetry mixin + timeout helpers
│   │   │   ├── embedding/ # Embedding 提供者、缓存、向量索引、熔断器
│   │   │   ├── file_processors/ # 文件处理器（audio, image, pdf, video）
│   │   │   ├── event_recovery/  # Redis Stream 会话事件恢复
│   │   │   └── ...       # sandbox, file_storage, task
│   │   └── checkpointer_pool.py  # LangGraph 检查点连接池
│   └── interfaces/       # FastAPI 路由、Schema、依赖注入、限流
│       └── endpoints/    # session/runtime extension/governance/plugin/memory/cost 等路由
├── core/                 # 环境变量与安全配置
├── alembic/              # 数据库迁移（含 memory_chunks、tool_approval_tables）
├── scripts/              # 管理脚本
├── tests/                # 后端测试
├── config.yaml.example   # 本地运行时配置模板
├── dev.sh                # 本地开发启动脚本
├── run.sh                # 生产启动脚本
└── requirements.txt      # Python 依赖
```

## 关键路由模块

- `auth_routes.py`
- `session_routes.py`
- `app_config_routes.py`
- `file_routes.py`
- `skill_v2_routes.py`
- `runtime_extension_routes.py`
- `extension_governance_routes.py`
- `plugin_routes.py`
- `memory_routes.py`
- `notification_routes.py`
- `session_compaction_routes.py`
- `cost_routes.py`
- `user_tool_policies_routes.py`
- `user_routes.py`
- `user_tools_v2_routes.py`
- `admin_routes.py`
- `status_routes.py`

完整路由清单见：

- [../api_zhcn.md](../api_zhcn.md)
- [../api.md](../api.md)

## 管理脚本

- `scripts/create_super_admin.py`
- `scripts/reset_admin_password.py`
- `scripts/minio_smoke_test.py`
- `scripts/minio_upload_file.py`
- `scripts/verify_local_minio.py`（隔离验收专用：固定 `localhost:19000`，两阶段）
- `scripts/migrate_skills_db_to_fs.py`
- `scripts/rollback_skills_fs_to_db.py`

## 开发注意事项

- 会话聊天和会话列表流使用 SSE，不是普通轮询
- 研究子智能体接口同样使用 SSE；后代列表和成本树使用普通 GET
- 接管终端和 VNC 使用 WebSocket
- `GET /sessions/{id}` 返回部署级 `sandbox_mode`；客户端不得只根据按钮可见性猜测供给档位
- lifecycle 是现有事件之上的加性投影；开启 master flag 前确保所有 API pod 版本一致
- Extension governance 为 env-only；Plugin 路由在治理 `off` 时统一返回 `409 governance_disabled`
- `Skill` 旧接口保留为 `410` 迁移提示，新接口统一在 `/api/v2/skills/*`
- 修改 `sandbox/` 后，记得重建 `sandbox-image` 和 `api`
