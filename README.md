<p align="center">
  <h1 align="center">Actus</h1>
  <p align="center">
    自托管的通用 AI Agent 平台 — 规划、推理、执行，一站完成
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

## 功能特性

- **ReAct Agent** — 基于 Reasoning + Acting 模式的智能体，支持多轮推理与工具调用
- **MCP 工具协议** — 通过 [Model Context Protocol](https://modelcontextprotocol.io/) 动态接入外部工具服务器
- **A2A 协议** — 支持 [Agent-to-Agent](https://google.github.io/A2A/) 智能体间通信与协作
- **Skill 生态层** — 统一管理 native / MCP / A2A Skill，支持 Manifest + SKILL.md 安装
- **Planner + ReAct 流程** — 先规划再执行的两阶段 Agent 流程编排
- **沙箱执行** — Docker 隔离的代码执行环境，安全运行用户代码
- **远程桌面** — 基于 noVNC 的沙箱桌面实时预览，浏览器内直接查看 Agent 操作画面
- **浏览器自动化** — 基于 Playwright + DOM 索引方案的网页操作能力，通过 CDP 连接沙箱 Chromium，实时提取可交互元素并精确操作
- **多模型支持** — 兼容 OpenAI API 格式（DeepSeek、Kimi 等）
- **流式输出** — 实时流式响应，支持思考过程展示
- **文件管理** — MinIO/S3 兼容的对象存储，支持文件上传下载
- **用户系统** — JWT 认证、角色权限管理
- **现代前端** — Next.js 16 + React 19 + shadcn/ui 组件库，Geist 字体，深色 / 浅色双主题

## 架构概览

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   UI (Next.js)│────▶│  API (FastAPI)│────▶│  PostgreSQL  │
└──────────────┘     │              │     └──────────────┘
                     │   ┌────────┐ │     ┌──────────────┐
                     │   │ Agent  │ │────▶│    Redis      │
                     │   │ Engine │ │     └──────────────┘
                     │   └───┬────┘ │     ┌──────────────┐
                     │       │      │────▶│  MinIO / S3   │
                     │   ┌───▼────┐ │     └──────────────┘
                     │   │MCP/A2A │ │     ┌──────────────┐
                     │   │/Skill  │ │
                     │   │Tools   │ │────▶│  Sandbox      │
                     │   └────────┘ │     │  (Docker)     │
                     └──────────────┘     └──────────────┘
```

后端采用 **整洁架构 (Clean Architecture) + DDD** 分层设计，详见 [项目架构文档](项目架构.md)。

## 快速开始

### 环境要求

- Docker Engine + Docker Compose v2
- 至少 6 GB 可用内存
- 一个可用的 MinIO / S3 兼容对象存储服务

### 一键部署

```bash
# 1. 克隆项目
git clone https://github.com/your-org/Actus.git
cd Actus

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的数据库密码、JWT 密钥、MinIO 凭证等

# 3. 配置 Agent 运行时参数
cp api/config.yaml.example api/config.yaml
# 编辑 api/config.yaml，填入你的 LLM API Key 和 MCP/A2A 配置
# Skill 生态配置通过「设置 -> Skill 生态」写入数据库管理

# 4. 启动所有服务
docker compose --env-file .env up -d --build

# 5. 创建管理员账号（推荐）
docker compose exec api python scripts/create_super_admin.py
```

启动后访问：

- 前端界面：http://localhost:3000
- API 文档：http://localhost:8000/docs

### 本地开发

```bash
# 后端
cd api
pip install -r requirements.txt
bash dev.sh

# 前端
cd ui
npm install
npm run dev
```

## 项目结构

```
Actus/
├── api/                 # 后端服务 (FastAPI)
│   ├── app/             # 应用代码（整洁架构分层）
│   │   ├── domain/      #   领域层（模型、Agent、工具、Prompt）
│   │   ├── application/ #   应用层（用例编排）
│   │   ├── infrastructure/ # 基础设施层（数据库、存储、外部服务）
│   │   └── interfaces/  #   接口层（路由、Schema、依赖注入）
│   ├── core/            # 核心配置与安全
│   ├── alembic/         # 数据库迁移
│   ├── scripts/         # 运维脚本
│   └── tests/           # 测试
├── ui/                  # 前端 (Next.js)
├── sandbox/             # 沙箱服务 (代码执行环境)
├── docker-compose.yml   # 容器编排
├── DEPLOY.md            # 部署指南
└── 项目架构.md            # 架构详细文档
```

## 文档

- [部署指南](DEPLOY.md) — Docker Compose 部署详细步骤
- [API 文档](api_zhcn.md) — 完整的 REST API 接口文档
- [API Docs (English)](api.md) — REST API reference
- [架构文档](项目架构.md) — 后端架构设计与编码规范
- [贡献指南](CONTRIBUTING.md) — 如何参与项目开发

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 前端框架 | Next.js 16 + React 19 |
| UI / 样式 | Tailwind CSS v4 + shadcn/ui + Geist 字体 |
| 状态管理 | Zustand |
| 数据库 | PostgreSQL 17 + SQLAlchemy 2.0 (async) |
| 缓存/队列 | Redis (Redis Streams) |
| 对象存储 | MinIO / S3 兼容 |
| LLM 接入 | OpenAI SDK（DeepSeek、Kimi 等） |
| Agent 协议 | MCP SDK + A2A + Skill Layer |
| 浏览器自动化 | Playwright + 自研 DOM 索引提取 |
| 沙箱 | Docker 容器隔离 |
| 认证 | JWT + bcrypt |
| 测试 | pytest + Vitest |

## 参与贡献

欢迎提交 Issue 和 Pull Request！请阅读 [贡献指南](CONTRIBUTING.md) 了解详情。

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 许可证开源。
