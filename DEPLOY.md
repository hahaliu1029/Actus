# Deployment Guide

本指南对应当前 `docker-compose.yml` 的实际运行方式，适用于单机开发、评估或受控内网部署。
它不是完整的公网生产 runbook；公网 TLS、反向代理、备份恢复、密钥轮换和远程对象存储
需要由部署者另行设计。

## 部署拓扑

![Deployment Topology](deploy-topology.png)

Compose 会启动或构建以下组件：

- `postgres`：持久化会话、用户、文件元数据
- `redis`：限流、状态、Embedding 缓存与部分运行时协调
- `minio`：仅绑定 loopback 的本地对象存储
- `minio-init`：幂等创建 API 使用的 `a2a-mcp` bucket
- `sandbox-image`：标准 Compose 中只负责构建沙箱镜像，不常驻运行；`off` override 会停用它
- `api`：FastAPI 后端
- `ui-app`：Next.js 运行时
- `ui`：nginx 网关，对外暴露前端入口
- `tunnel`：（可选，需显式启用 profile）autossh 反向隧道，将 API 暴露到云服务器

## 1. 前置条件

- Docker Engine + Docker Compose v2
- 如部署 `off` 档，Docker Compose 必须为 2.24.4 或更高版本（override 使用 `!override`）
- 至少 6 GB Docker 可用内存
- 可访问的 LLM 提供商 API（不阻止基础服务启动；Agent 可用前必须在设置页或
  `api/config.yaml` 中完成模型配置）

## 2. 配置环境变量

```bash
git clone https://github.com/hahaliu1029/Actus.git
cd Actus
cp .env.example .env
cp api/config.yaml.example api/config.yaml
```

至少需要修改这些值：

- `POSTGRES_PASSWORD`
- `JWT_SECRET_KEY`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MEMORY_ROOT_HOST`（宿主机绝对路径，且必须先创建）
- `NEXT_PUBLIC_API_BASE_URL`
- `ACTUS_C2_COORDINATOR_ENABLED=false`（除非本次部署专门执行 coordinator 验收）

例如：

```bash
mkdir -p "$HOME/.actus/memory"
# 然后把 .env 中 MEMORY_ROOT_HOST 改为该目录展开后的绝对路径（不要写 ~）
```

补充说明：

- `NEXT_PUBLIC_API_BASE_URL` 是**构建时注入**的前端 API 地址，必须是浏览器可访问的 URL；
  默认本机 Compose 可使用 `.env.example` 的 `http://localhost:8000/api`
- `SANDBOX_IMAGE` 默认是 `actus-sandbox:latest`
- 如需调整浏览器接管能力，可额外配置 `SANDBOX_CHROME_ARGS`
- `MEMORY_ROOT_HOST` 不能写 `~` 或容器内路径；它同时供 API 写入并由运行时只读挂载给沙箱
- `.env.example` 当前为 CI/评估方便保留 `ACTUS_C2_COORDINATOR_ENABLED=true`，但生产
  rollout gate 和部分累计预算护栏尚未完成；普通部署必须显式改为 `false`。启用前按
  `CONTRIBUTING.md` 的 C2 rollout checklist 完成 live-provider、监控和预算检查

标准 Docker Compose 是本地开发拓扑：默认启动本地 MinIO，`minio-init` 会在 API
启动前幂等创建 `a2a-mcp`。S3 API 为 `http://127.0.0.1:9000`，管理控制台为
`http://127.0.0.1:9001`，二者仅绑定 loopback，凭据来自 `MINIO_ACCESS_KEY` /
`MINIO_SECRET_KEY`。修改 `MINIO_API_PORT` 后，用于生成预签名 URL 的 public endpoint
自动变为 `localhost:<port>`；这里的 "public" 指 API 返回给浏览器/外部消费者的地址，
不会把 MinIO 从 loopback 自动暴露到远端。高级场景可设置 `MINIO_PUBLIC_ENDPOINT` /
`MINIO_PUBLIC_SECURE`。根目录 `.env` 中旧的 `MINIO_ENDPOINT` 不控制标准 Compose；容器内
API 始终通过 `minio:9000` 访问本地服务。

从远程对象存储切换到本地 MinIO 会得到新的空数据集，不自动迁移；旧附件仍保留在原
远程 S3，需要另行规划数据迁移。Compose 固定的归档 MinIO release 镜像不作为生产基线。
生产部署及远程 URL 消费者应使用部署者维护的远程 S3 或受保护的 TLS endpoint。

### 沙箱供给模式

`SANDBOX_PROVISION_MODE` 是 env-only 配置，修改后必须通过 `docker compose up -d`
**重新创建** API 容器；`docker compose restart api` 不会应用新的环境变量。当前支持：

| 模式 | 供给行为 | Compose 方式 |
|------|----------|--------------|
| `always` | 默认值；每个 session 预置父沙箱，保持原有行为 | 标准 `docker-compose.yml` |
| `on_demand` | 首次沙箱工具、VNC、接管或 coordinator 父沙箱 I/O 触发时才创建；纯聊天 session 不创建容器 | 标准 Compose，并在 `.env` 设置 `SANDBOX_PROVISION_MODE=on_demand` |
| `off` | 不创建沙箱，沙箱工具、VNC/接管和 skill-create 面关闭 | 必须叠加 `docker-compose.sandbox-off.yml` |

`on_demand` 的 create / ready / hooks 三阶段共享
`SANDBOX_PROVISION_TIMEOUT_SECONDS` 总预算，默认 90 秒。

`off` 不能只把 `.env` 改成 `off`：canonical override 还会移除 API 的 Docker Socket
挂载、删除 `sandbox-image` 依赖并停用镜像服务。切换前必须先清理存量沙箱，同时将
`ACTUS_C2_COORDINATOR_ENABLED`、`ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED`、
`ACTUS_C2_AGENT_TEAMS_ENABLED` 三个 coordinator flag 全部设为 `false`，否则 API
启动会 fail-fast。完整的 drain、部署、值级验证和回退命令以
[`docs/runbooks/sandbox-off-runbook.md`](docs/runbooks/sandbox-off-runbook.md) 为准；部署命令为：

```bash
docker compose -f docker-compose.yml -f docker-compose.sandbox-off.yml \
  --env-file .env up -d --build
```

## 3. 启动全部服务

`always` / `on_demand` 使用标准 Compose：

```bash
docker compose --env-file .env up -d --build
```

`off` 使用上一节给出的双文件命令，并**跳过**这条标准 Compose 命令；后续任何
`up` / `recreate` 操作都必须继续同时携带两个 `-f` 参数，否则会把 Docker Socket 与
`sandbox-image` 依赖重新带回 API。

首次启动时，后端会：

- 自动执行 Alembic 迁移
- 初始化 PostgreSQL / Redis / MinIO 客户端；`minio-init` 完成后才启动 API
- 从 host bind-mount 的 `api/config.yaml` 读取运行时配置

## 4. 初始化管理员（可选）

如果需要进入管理员设置和用户管理界面，创建超级管理员：

```bash
docker compose exec api python scripts/create_super_admin.py
```

## 5. 访问入口

- UI：`http://localhost`（默认 `UI_PORT=80`）
- API 文档：`http://localhost:8000/docs`
- OpenAPI JSON：`http://localhost:8000/openapi.json`

容器模式下，前端链路是：

```text
browser -> ui (nginx:80) -> ui-app (Next.js:3000) -> api
```

## 6. 配置运行时参数

Compose 模式下，Actus 把仓库中的 `api/config.yaml` bind-mount 为：

```text
/app/data/config.yaml
```

推荐做法：

1. 首次启动前执行 `cp api/config.yaml.example api/config.yaml`
2. 完成启动并以管理员登录
3. 在 `设置` 中配置（写回同一 host 文件）：
   - 模型提供商
   - MCP 服务器
   - A2A Agent 配置
   - Skill 风险策略
   - Skill 安装与启用

Skill 目录默认保存在：

```text
/app/data/skills
```

## 7. 验证部署

```bash
docker compose ps
docker compose logs --tail=200 api
docker compose logs --tail=200 ui
```

建议至少检查：

- `api` 健康检查通过
- `ui-app` 和 `ui` 健康检查通过
- `http://localhost:8000/docs` 可打开
- 前端登录后可以进入首页和设置页

如需验证对象存储连通性，可以调用：

- `GET /api/status/minio`
- `GET /api/status/minio?smoke=true`

或在容器内运行：

```bash
docker compose exec api python scripts/minio_smoke_test.py
```

该 smoke test 使用当前部署配置，适合日常连通性检查。`api/scripts/verify_local_minio.py`
则是隔离 acceptance 专用脚本，固定校验 `localhost:19000`，并按 `write` → 容器重建 →
`verify` 两阶段验证持久化和 presigned URL；不要把它当成任意生产 endpoint 的通用探针。

## 8. 停止与清理

```bash
docker compose down
```

删除持久化数据卷：

```bash
docker compose down -v
```

默认持久化卷：

- `postgres-data`
- `redis-data`
- `minio-data`
- `api-data`

## 9. 更新沙箱代码后的重建方式

如果你修改了 `sandbox/` 中的代码，不要运行不存在的 `sandbox` 服务。以下流程只适用于
`always` / `on_demand`；`off` 不选择沙箱镜像：

```bash
docker compose --env-file .env build sandbox-image api
docker compose --env-file .env up -d --force-recreate api
```

如需确保新会话不复用旧容器，可清理历史临时沙箱：

```bash
docker ps --format '{{.Names}}' | grep '^actus-sb-' | xargs -r docker rm -f
```

## 10. 可选：启用 SSH 隧道

如需将本地 API 暴露到外部网络（例如手机访问），可启用 tunnel 服务：

```bash
# 生成 SSH 密钥并配置，详见 tunnel/README.md
docker compose --profile tunnel up -d tunnel
```

`tunnel` profile 只转发 API，不转发本地 MinIO。MCP 或其他远程 URL 直接拉取方需要
能够访问单独配置的 remote/public endpoint。

## 11. 常见注意事项

- **⚠️ API 服务必须单实例部署。** Sandbox 生命周期管理依赖进程内锁，多 worker/多实例会导致状态竞争。不要修改 `docker-compose.yml` 中 `api` 的 `deploy.replicas`，不要在 uvicorn 命令中加 `--workers`，不要 `docker compose up --scale api=N`。多实例需求请参考 `CONTRIBUTING.md` 中 "Sandbox Lifecycle 不变式" 章节。
- 前端 API 地址变化后，需要重新构建 `ui-app`
- `always` / `on_demand` 下，`api` 通过挂载的 Docker Socket 动态创建会话沙箱
- 沙箱镜像本身不常驻；`always` / `on_demand` 由 API 按所选供给模式动态创建临时容器，
  `off` override 不挂载 Docker Socket，也不选择 `sandbox-image`
- 如果 Redis 不可用，限流相关接口会返回 `503`
- 旧版 Skill API 会返回 `410`，请使用 `/api/v2/skills/*`
- 文件理解功能（音频转录、PDF 解析、视频分析）需要在前端设置页的「文件理解」中配置
