# Actus API

`api/` 是 Actus 的 FastAPI 后端，负责会话编排、Agent 执行、认证授权、文件管理、运行时配置、Skill 生态以及 Docker 沙箱调度。

## 当前能力

- 用户注册、登录、刷新令牌、个人资料维护
- 超级管理员用户管理
- 会话创建、SSE 对话流、任务停止、文件读取
- 会话事件恢复（基于 Redis Stream 的 SSE 状态续传）
- `shell` / `browser` 接管、续期、结束、补救
- LLM / MCP / A2A / Skill 风险策略配置
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
- 文件上传、下载、删除
- 健康检查与 MinIO 自检

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

### 1. 启动基础依赖

```bash
docker compose up -d postgres redis
docker compose build sandbox-image
```

说明：

- Compose 不会启动 MinIO，你需要单独准备对象存储
- `sandbox-image` 只是构建镜像；真正的会话沙箱由 API 运行时动态创建

### 2. 准备本地配置

在 `api/` 下创建本地运行时配置文件：

```bash
cd api
cp config.yaml.example config.yaml
```

再创建 `api/.env`。后端本地运行读取的是这里，而不是根目录 Compose 的 `.env`：

```dotenv
ENV=development
LOG_LEVEL=INFO
APP_CONFIG_FILEPATH=config.yaml
SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/manus
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
MINIO_ENDPOINT=s3.example.com
MINIO_ACCESS_KEY=replace-me
MINIO_SECRET_KEY=replace-me
MINIO_SECURE=true
MINIO_BUCKET_NAME=replace-me
JWT_SECRET_KEY=replace-with-a-strong-random-string
SANDBOX_IMAGE=actus-sandbox:latest
SANDBOX_NAME_PREFIX=actus-sb
```

### 3. 安装依赖并启动

```bash
cd api
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash dev.sh
```

启动后：

- API 文档：`http://localhost:8000/docs`
- OpenAPI：`http://localhost:8000/openapi.json`

## 配置来源

### 本地运行

- 环境变量：`api/.env`
- 业务运行时配置：`api/config.yaml`

### Docker Compose 运行

- 环境变量：仓库根目录 `.env`
- 业务运行时配置：`/app/data/config.yaml`
- Skill 目录：`/app/data/skills`

如果 `/app/data/config.yaml` 不存在，后端会按默认值自动创建。

## 测试

```bash
cd api
pytest
```

## 目录结构

```text
api/
├── app/
│   ├── application/      # 用例编排服务（Agent, Session, Skill, Memory Flush 等）
│   ├── domain/           # 领域模型、工具、流程、Prompt、上下文治理
│   │   ├── models/       # 领域模型（app_config, skill, memory_chunk, tool_approval_rule, session 等）
│   │   ├── external/     # 外部依赖协议（file_processor, embedding, memory_flusher, event_recovery, telemetry）
│   │   ├── services/
│   │   │   ├── graphs/   # LangGraph 图（main_graph, react_graph, compaction, context_assembler,
│   │   │   │             #               token_estimator, background_summary, step_metadata, message_utils）
│   │   │   ├── flows/    # 流程编排（planner_react, skill_creation_graph）
│   │   │   ├── tools/    # LangChain 工具（file, shell, browser, mcp_discovery, dynamic_skill,
│   │   │   │             #               memory_tools, tool_failure_tracker）
│   │   │   ├── prompts/  # 模块化 Prompt 子包（assembler, section, render_context, budget,
│   │   │   │             #                    invariants, errors, sections/, bundles/, reminders/）
│   │   │   ├── risk_assessor.py / smart_approve.py / confirmation_manager.py / approval_cache.py
│   │   │   ├── execution_watchdog.py / execution_metrics.py
│   │   │   ├── memory_ranker.py / json_envelope.py
│   │   │   └── skill_md_parser.py / skill_md_exporter.py
│   │   └── repositories/ # 仓库接口 (ABC)，含 memory_chunk / tool_approval_rule / tool_approval_log
│   ├── infrastructure/   # 仓储实现、外部服务、存储客户端
│   │   ├── models/       # ORM（含 memory_chunk_orm, tool_approval_log, tool_approval_rule）
│   │   ├── repositories/ # 仓储实现（含 db_memory_chunk_repository, db_tool_approval_*）
│   │   ├── telemetry/    # Prompt + LLM telemetry 实现
│   │   ├── external/
│   │   │   ├── llm/      # LLM 适配器 + 消息清洗器 + telemetry mixin + timeout helpers
│   │   │   ├── embedding/ # Embedding 提供者、缓存、向量索引、熔断器
│   │   │   ├── file_processors/ # 文件处理器（audio, image, pdf, video）
│   │   │   ├── event_recovery/  # Redis Stream 会话事件恢复
│   │   │   └── ...       # sandbox, file_storage, task
│   │   └── checkpointer_pool.py  # LangGraph 检查点连接池
│   └── interfaces/       # FastAPI 路由、Schema、依赖注入、限流
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
- `scripts/migrate_skills_db_to_fs.py`
- `scripts/rollback_skills_fs_to_db.py`

## 开发注意事项

- 会话聊天和会话列表流使用 SSE，不是普通轮询
- 接管终端和 VNC 使用 WebSocket
- `Skill` 旧接口保留为 `410` 迁移提示，新接口统一在 `/api/v2/skills/*`
- 修改 `sandbox/` 后，记得重建 `sandbox-image` 和 `api`
