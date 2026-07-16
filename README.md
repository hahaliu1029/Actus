<p align="center">
  <h1 align="center">Actus</h1>
  <p align="center">
    自托管的通用 AI Agent 平台，覆盖规划、推理、执行与人工接管全流程
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Next.js-16-black.svg" alt="Next.js">
  </p>
  <p align="center">
    <a href="README_EN.md">English</a> · 中文
  </p>
</p>

---

## 项目概览

Actus 由三个核心运行时组成：

- `api/`：FastAPI 后端，负责会话、Agent、权限、文件、配置、Skill 生态与沙箱调度
- `ui/`：Next.js 16 前端，提供聊天、任务摘要、工作台、设置页和管理界面
- `sandbox/`：按会话动态拉起的 Docker 沙箱，内置 Shell、文件系统、Chromium、VNC/noVNC

系统基于 **LangGraph** 状态机实现 `Planner + ReAct` 双阶段流程：先规划任务，再逐步执行，并在执行过程中通过 SSE 持续推送计划、步骤、工具调用、消息和接管事件。

## 核心能力

- **LangGraph Agent 编排**：两层图架构（main_graph 规划调度 + react_graph 工具执行循环），支持规划、步骤执行、等待用户输入、最终总结，并通过 `FINISHING` 中间态承载异步收尾
- **LangChain 工具体系**：文件、Shell、浏览器、搜索工具通过 `@tool` 装饰器统一注册
- **MCP / A2A / Skill 扩展**：统一纳入 Agent 工具选择与运行时编排，支持渐进式 MCP 工具发现和基于 Embedding 的 Skill 语义选择
- **Skill v2 文件系统存储**：Skill 保存在 `/app/data/skills`，支持 GitHub、本地目录和 SKILL.md 格式安装
- **多模态文件理解**：音频转录（Whisper API / 沙箱 faster-whisper）、PDF 解析（原生 / pymupdf4llm）、图片处理、视频关键帧提取 + 视觉模型分析
- **上下文溢出治理**：两级渐进压缩（85% LLM 摘要 / 95% 硬截断）+ 同步三阶段裁剪，自动保护上下文窗口
- **模块化提示词系统（B5）**：sections / bundles / reminders / assembler / budget 子模块组合，支持中英文 bundle 和按情境注入的 reminders
- **Agent 记忆系统（M1 Memory Redesign 完成）**：三分类（`user` / `rule` / `fact`）+ `memory_search` / `memory_get` / `memory_save` 工具 + 检索流水线（cosine 相似度 → 时间衰减 → MMR 多样性重排）+ Embedding 熔断器；文件为事实源（host bind-mount 到 sandbox 只读）+ `FsReconciler` 后台修复 DB/fs 一致性；LLM 质量闸（独立 CircuitBreaker + per-user daily cap）+ 系统通知（gate paused / quota exceeded / fs failure）
- **工具审批与确认系统**：用户级永久允许/拒绝规则（`always_allow` / `always_deny`，按 command/dir glob 匹配）+ 会话级允许缓存（仅对 approve 生效）+ Smart Approve 智能批准 + 显式确认（前端 approve 支持 once/session/always 三档，deny 为一次性），含风险评估和持久化审计日志
- **会话事件恢复**：基于 Redis Stream 的 SSE 状态恢复，刷新或断线重连后从最后位点继续
- **执行健康监控**：步骤级 watchdog + 执行指标采集 + 工具失败追踪 + 统一 JSON Envelope
- **LLM 调用预算**：连接阶段 / 读取阶段独立 timeout 预算，与 LangGraph RetryPolicy 对齐，避免 9 次 HTTP 重试放大
- **人工接管**：支持 `shell` 和 `browser` 两类接管，包含申请、续期、结束、补救流程
- **工作台视图**：终端预览、浏览器预览、VNC 画面、时间线回放、文件预览
- **流式交互**：会话列表与对话执行均支持 SSE；接管终端和 VNC 使用 WebSocket
- **容器化沙箱**：每个会话独立 Docker 容器，内置 Chromium、Xvfb、x11vnc、websockify
- **对象存储与附件**：上传文件落到 MinIO/S3 兼容存储，并与会话关联；文件传输支持进度跟踪、断点续传
- **用户与管理**：JWT 鉴权、超级管理员、用户管理、工具偏好、应用设置
- **SSH 隧道**：可选的 autossh 反向隧道，将本地 API 暴露到云服务器
- **多语言贯通**：`Message.language` 字段贯穿 prompt assembler，按用户语言派发中英文 bundle

## 架构概览

![Actus Architecture](architecture.png)

<details>
<summary>ASCII 版本</summary>

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ UI (Next.js) │────▶│ API (FastAPI)│────▶│ PostgreSQL   │
└──────────────┘     │              │     └──────────────┘
                     │ Agent / Auth │────▶│ Redis        │
                     │ Files / Skill│     └──────────────┘
                     │ Settings     │────▶│ MinIO / S3   │
                     │ Sandbox Ctrl │     └──────────────┘
                     └──────┬───────┘
                            │ Docker
                            ▼
                     ┌──────────────┐
                     │ Sandbox      │
                     │ Shell/File   │
                     │ Chromium/VNC │
                     └──────────────┘

(可选) Phone ──HTTP──▶ Cloud:18082 ──SSH Tunnel──▶ API:8000
```
</details>

后端分层与关键模块见 [项目架构文档](项目架构.md)。源文件：[architecture.excalidraw](architecture.excalidraw)。

## Docker Compose 快速开始

### 前置条件

- Docker Engine + Docker Compose v2
  - off 档部署需 Docker Compose ≥ 2.24.4（`!override` YAML 标签，见 `docs/runbooks/sandbox-off-runbook.md`）
- 至少 6 GB 可用内存
- 一个可用的 LLM API Key（启动后在设置页填写，或预写入运行时配置）

### 启动步骤

```bash
git clone https://github.com/hahaliu1029/Actus.git
cd Actus

cp .env.example .env
# 编辑 .env，至少填写：
# POSTGRES_PASSWORD
# JWT_SECRET_KEY
# MINIO_ACCESS_KEY
# MINIO_SECRET_KEY
# NEXT_PUBLIC_API_BASE_URL
# 可选：如需覆盖默认 Python 包镜像，设置 PYTHON_PACKAGE_INDEX_URL

# Memory 系统首次部署（Memory Redesign M1 起必做）：
# host 端 memory 根目录必须存在，否则 docker compose up 会在 api 容器挂载阶段失败
mkdir -p ~/.actus/memory
# 如你在 .env 里启用了 ACTUS_UID 非 root 模式，请额外执行：
# sudo chown -R ${ACTUS_UID:-1000}:${ACTUS_GID:-1000} ~/.actus/memory

docker compose --env-file .env up -d --build

# 可选：创建超级管理员
docker compose exec api python scripts/create_super_admin.py

# 可选：运维手动全库扫一次 memory 一致性（通常由 api lifespan + session hook 自动）
# docker compose exec api python -m app.cli.memory_reconcile
```

启动完成后访问：

- 前端：`http://localhost`（默认 `UI_PORT=80`）
- API 文档：`http://localhost:8000/docs`

标准 Docker Compose 是本地开发拓扑：默认启动仅绑定 loopback 的本地 MinIO，API
启动前会幂等创建 `a2a-mcp` bucket。S3 API 为 `http://127.0.0.1:9000`，管理控制台为
`http://127.0.0.1:9001`，凭据来自 `.env` 的 `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`。
修改 `MINIO_API_PORT` 后，对外 endpoint 会自动变为 `localhost:<port>`；只有需要覆盖
主机名或 TLS 时才设置 `MINIO_PUBLIC_ENDPOINT` / `MINIO_PUBLIC_SECURE`。

Compose 固定的归档 MinIO release 镜像用于可复现的本地开发，不作为生产基线。生产
部署或需要远程 URL 消费者时，应由部署者维护远程 S3 或受保护的 TLS endpoint。现有
`tunnel` profile 只转发 API，不转发本地 MinIO；MCP 或其他远程 URL 直接拉取方必须能
访问上述 remote/public endpoint。

### 运行时配置说明

- Compose 模式下，后端运行时配置文件实际位于 `api-data` volume 内的 `/app/data/config.yaml`
- 如果该文件不存在，后端会按代码默认值自动创建
- 推荐在首次启动后，通过前端 `设置 -> 模型提供商 / MCP 服务器 / A2A Agent 配置 / Skill 生态` 完成配置
- `api/config.yaml.example` 主要用于**本地后端开发**或你需要手工预填配置文件时参考

### 修改 `sandbox/` 后的正确重建方式

Compose 中的服务名是 `sandbox-image`，不是 `sandbox`。当你修改沙箱代码后，应使用：

```bash
docker compose --env-file .env build sandbox-image api
docker compose --env-file .env up -d --force-recreate api

# 可选：清理旧的临时沙箱容器
docker ps --format '{{.Names}}' | grep '^actus-sb-' | xargs -r docker rm -f
```

## 本地开发

### 前端本地开发

```bash
cd ui
npm install
npm run dev
```

前端默认访问 `NEXT_PUBLIC_API_BASE_URL`，开发时通常指向 `http://localhost:8000/api`。

### 后端本地开发

后端本地运行与 Compose 使用的根目录 `.env` 不是一套变量。`api/core/config.py` 读取的是 `api/.env` 中的运行时变量，例如：

```bash
cd api
cp config.yaml.example config.yaml

cat > .env <<'EOF'
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
EOF

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash dev.sh
```

本地后端开发通常还需要：

- 启动 PostgreSQL、Redis
- 预先构建 `sandbox-image`
- 启动标准 Docker Compose 提供的本地 MinIO，或准备可访问的远程 S3 bucket

更详细说明见 [api/README.md](api/README.md)。

## 测试

```bash
# 后端
cd api
pytest

# 前端
cd ui
npm run test
```

## 目录结构

```text
Actus/
├── api/                  # FastAPI 后端
│   ├── app/
│   │   ├── application/  # 用例编排（Skill、Memory Flush）
│   │   ├── domain/       # 领域模型、流程、工具、Prompt、上下文治理
│   │   ├── infrastructure/ # 数据库/存储/外部实现、文件处理器、Embedding
│   │   └── interfaces/   # 路由、Schema、依赖注入
│   ├── core/             # 环境配置、安全
│   ├── scripts/          # 管理脚本
│   └── tests/            # 后端测试
├── ui/                   # Next.js 前端
├── sandbox/              # Docker 沙箱镜像源码
├── tunnel/               # SSH 反向隧道配置（可选）
├── docker-compose.yml    # 容器编排
├── DEPLOY.md             # 部署说明
├── api_zhcn.md           # 中文 API 文档
├── api.md                # English API reference
└── 项目架构.md             # 架构说明
```

## 文档索引

- [部署指南](DEPLOY.md)
- [中文 API 文档](api_zhcn.md)
- [English API Reference](api.md)
- [后端架构说明](项目架构.md)
- [后端 README](api/README.md)
- [前端 README](ui/README.md)
- [沙箱 README](sandbox/README.md)
- [SSH 隧道](tunnel/README.md)
- [贡献指南](CONTRIBUTING.md)

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | FastAPI、Uvicorn、Pydantic v2 |
| 数据库 | PostgreSQL 17、SQLAlchemy 2.0 async、Alembic |
| 缓存 / 限流 | Redis |
| 对象存储 | MinIO / S3 兼容 |
| Agent | LangGraph StateGraph、LangChain BaseChatModel、PlannerReActFlow |
| 上下文治理 | TokenEstimator、ContextAssembler、GradualCompactor |
| 文件理解 | Whisper (OpenAI/sandbox)、pymupdf4llm、视觉模型帧分析 |
| Embedding | OpenAI Embeddings、Redis 缓存、numpy 向量索引 |
| 扩展协议 | MCP（含渐进式发现）、A2A、Skill（含 SKILL.md 格式） |
| 前端 | Next.js 16、React 19、Tailwind CSS 4、Zustand |
| 浏览器执行 | Chromium、CDP、Playwright 风格 DOM 操作 |
| 沙箱 | Docker、Supervisor、Xvfb、x11vnc、websockify |
| 测试 | pytest、Vitest、Testing Library |

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。
