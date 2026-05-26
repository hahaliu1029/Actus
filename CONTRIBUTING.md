# 贡献指南

本文档基于当前仓库结构和运行方式编写，适用于 [Actus](https://github.com/hahaliu1029/Actus)。

## 你可以如何贡献

- 报告 Bug
- 提交功能建议
- 修复代码问题
- 完善测试
- 更新文档

## 开始前

建议先阅读：

- [行为准则](CODE_OF_CONDUCT.md)
- [部署指南](DEPLOY.md)
- [项目架构](项目架构.md)

同时请先查看现有 Issue：

- <https://github.com/hahaliu1029/Actus/issues>

## 环境要求

- Python 3.12（后端）
- Node.js 22+（前端）
- Docker + Docker Compose v2
- PostgreSQL、Redis
- 可访问的 MinIO / S3 兼容对象存储

## 开发方式建议

### 方式一：优先使用 Docker Compose 验证完整链路

适合联调、部署验证、回归测试。

```bash
cp .env.example .env
docker compose --env-file .env up -d --build
```

### 方式二：本地运行前端或后端

适合快速迭代某个子项目，但要注意前后端与 Compose 使用的配置来源不同。

## 后端开发

### 1. 启动依赖

你至少需要 PostgreSQL、Redis，以及一个已经构建好的 `sandbox-image`。最简单做法是：

```bash
docker compose up -d postgres redis
docker compose build sandbox-image
```

MinIO/S3 仍需自行准备，Compose 不会启动它。

### 2. 配置本地后端环境

`api/core/config.py` 读取的是 `api/.env`，不是根目录 Compose 用的 `.env`。建议在 `api/` 下创建：

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
MINIO_BUCKET_NAME=a2a-mcp
JWT_SECRET_KEY=replace-with-a-strong-random-string
SANDBOX_IMAGE=actus-sandbox:latest
SANDBOX_NAME_PREFIX=actus-sb
```

同时创建本地运行时配置：

```bash
cd api
cp config.yaml.example config.yaml
```

### 3. 安装依赖并启动

```bash
cd api
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash dev.sh
```

### 4. 后端测试

```bash
cd api
pytest
```

## Prompt Assembly 不变式（B5 两时钟架构）

修改 `api/app/domain/services/graphs/main_graph.py` 时必须了解 `state.skill_context`
的**两时钟架构**（详见 B5 设计文档 "Two-Clock Architecture" 章节，
位于 `~/.gstack/projects/hahaliu1029-Actus/liuyixuan-develop-design-*.md`）：

- **`updater_node` 是唯一允许写 `state.skill_context` 的节点。** 它通过
  LangGraph `configurable` 字典里注入的 `skill_context_refresher` callable 写入。
- **`executor_node`**（以及未来任何节点）必须把 `StepMetadata.skill_context`
  作为**当前 step 的局部变量**消费，用于 prompt 组装。它**禁止**通过
  `Command(update=...)` 写回 `state["skill_context"]`。
- **`planner_node` 和 `summarizer_node`** 同样禁止写 `skill_context`。

CI gate 位于
`api/tests/domain/services/graphs/test_executor_no_skill_context_writeback.py`，
通过 AST 扫描 `executor_node` 函数体检测
`Command(update={"skill_context": ...})` 直接写回。

**间接写回**（helper 函数、变量间接、`dict(**base, skill_context=...)` 等）
AST 扫描**无法捕获**——依赖 code review 保障。

如果你认为确实需要在 executor 侧写 `skill_context`：

1. 重新阅读 B5 设计文档 "Two-Clock Architecture" 章节
2. 仍有需要时在 PR 里说明原因并 @ 核心 reviewer
3. 考虑是否应该写一个不同的 state 字段（比如 `system_prompt_version_hash`）

### 为什么要这样

B5 引入了 `skill_context` 的两个数据源：

| 数据源 | 写入者 | 读取者 | 用途 |
|--------|--------|--------|------|
| `state.skill_context` | `updater_node`（通过 `skill_context_refresher` 回调） | `executor_node`（仅当 `react_graph_provider` 为 None 的老 fallback 路径） | 老架构兼容 |
| `StepMetadata.skill_context`（即 `self._last_skill_context`） | `_build_step_react_graph` 通过 `_apply_refreshed_skills` | `executor_node`（`react_graph_provider` 存在时的主路径） | B5 step 级权威源 |

executor_node 在默认生产路径下**局部**消费 `StepMetadata.skill_context`，
**不**走 state。如果它也写回 state，两个数据源会竞争写入，updater 的值会
在下一 loop 被 executor 覆盖。

## Sandbox Lifecycle 不变式（单 Worker 部署契约）

Actus 的沙箱生命周期由 `SandboxLifecycleService` 管理，采用 K8s 风格的
terminal-state 状态机（详见 `docs/superpowers/specs/2026-04-15-sandbox-lifecycle-design.md`）。

### 单 Worker 硬约束

**当前 sandbox lifecycle 仅支持单 worker 部署。** 状态转换通过进程内
`asyncio.Lock` 串行化，跨 worker 会产生真实 race condition。

部署要求：
- `docker-compose.yml` 中 api 服务必须保持 `deploy.replicas: 1`
- uvicorn 启动参数不得传 `--workers N`（N > 1）
- `docker compose up --scale api=N`（N > 1）禁止使用
- 运行时防护：`WEB_CONCURRENCY` 环境变量必须为 `"1"`（或不设置）

多 worker 部署需要同时推进 §12 Q5 的跨进程协调 spec（Postgres advisory lock + Redis pub/sub invalidation），不得只扩 worker 不扩协调。

### CI gates

| Gate | 文件 | 保护的不变式 |
|------|------|-------------|
| Gate 1 | `tests/domain/test_no_raw_sandbox_references.py` | I3: 所有 sandbox 访问走 lifecycle service |
| Gate 2 | `tests/domain/test_no_session_sandbox_id_access.py` | I8: domain/application 层走 `sandbox_binding.*` |
| Gate 3 | `tests/domain/test_no_raw_sandbox_attribute_access.py` | I7: 不绕过 SandboxHandle generation 检查 |

## Memory System 不变式（M1 Redesign，2026-04）

Memory 是 **三视图一致** 的系统：DB（检索权威）、文件（sandbox 可见的事实源）、
Prompt snapshot（session-scoped 注入）。三者的同步边界：

- **DB-first 写入**：`MemoryManagementService.create_memory` 先 INSERT 再
  `file_store.write`；任何写入路径必须先把 `fs_synced=false` 落地，由
  writer 成功后回写 `fs_synced=true`，避免 "file 有 DB 无" 的逻辑不对称
- **Canonical frontmatter 单源**：`infrastructure/external/memory/frontmatter.py`
  里的 `build_memory_frontmatter` / `serialize_memory_file` 是应用层与
  `FsReconciler` 重建孤儿 DB 行共用的序列化入口；不要在别处构造
  frontmatter dict，会导致"写出 vs 重建"格式漂移
- **`move_category` 原子性**：`file_store.move_category` 必须先写新路径再删旧路径；
  step 3（旧路径删除）失败时保留旧文件，由 `FsReconciler._scan_fs_orphans`
  的 path-canonicality 检查把 `chunk.category != entry.parent.name` 的残留
  移入 `.orphans/`——**禁止**先 delete 后 write，会丢数据
- **`executor_node` 只读快照**：Memory 注入 prompt 的那份数据是 session-scoped
  快照，session 内后续写入的 memory 不反映到当前 prompt；要等下次 session
  （与 B5 Two-Clock Architecture 同样的设计原则）
- **fs 写入异常降级**：`file_store.write` OSError / SecurityError 被 writer 侧吞掉
  并保持 `fs_synced=false`，由 lifespan 背景任务 `scan_pending_fs_sync`
  重试；domain 不应该对这类错误做二次处理
- **FsReconciler 仅 writer 可变 fs**：reconciler 自己不直接 os.write，所有
  重建走 `FsMemoryWriter`——保证 symlink/path-traversal 防御链单一

**首次部署 checklist**（`README.md` 同步要求）：
```bash
mkdir -p ${MEMORY_ROOT_HOST:-~/.actus/memory}     # host bind source 必须先存在
# 如果用 ACTUS_UID 非 root 跑 api，先 chown 把所有权翻过去
```

运维路径：`python -m app.cli.memory_reconcile [--user-id UID]` 手动全库扫
DB/fs 一致性（覆盖 pending backlog + per-user fs walk + orphan 隔离）。

## C2 CoordinatorTaskRunner Deployment (spec §6.4)

Rollout order — STRICT, do not reorder:

1. Upgrade ALL MailboxSupervisor pods to new envelope schema version
   (含 `SpawnRequestPayload.agent_kind=coordinator_step` + new
   `ResultReadyOutcome` values + new `ResultReadyPayload` fields parser).
2. Verify supervisor majority quorum on new version (ops check).
3. THEN set `ACTUS_C2_COORDINATOR_ENABLED=true`.
4. Coordinator producer hard-gates on flag; false → planner prompt does
   not teach `parallel_work_units` + executor does not enter `_run_parallel_backend`.

Rollback: set flag back to false → producer stops; supervisors stay on
new version (forward compat).

## 前端开发

```bash
cd ui
npm install
npm run dev
```

常用命令：

```bash
cd ui
npm run lint
npm run test
npm run build
```

前端使用的主要环境变量：

- `NEXT_PUBLIC_API_BASE_URL`

该变量是**构建时注入**的；如果改了部署地址，需要重新构建前端。

## 文档更新

如果你的改动影响了以下任意内容，请同步更新文档：

- API 路由或响应格式
- 会话状态机 / 接管流程
- Skill / MCP / A2A 配置方式
- 文件理解 / 上下文治理 / Embedding 配置
- Docker Compose 服务名与部署步骤
- 本地开发命令

## 分支与提交

推荐从 `develop` 创建功能分支，完成后合并回 `develop`，再统一合并到 `main`：

```bash
git checkout develop
git checkout -b feature/<short-description>
```

提交信息建议使用 Conventional Commits：

```text
feat(api): add takeover reopen endpoint
fix(ui): handle image proxy failures
docs: refresh deployment and API docs
```

常见类型：

- `feat`
- `fix`
- `docs`
- `refactor`
- `test`
- `chore`

## Pull Request 建议

提交 PR 前请尽量完成以下检查：

- 后端相关改动已运行 `pytest`
- 前端相关改动已运行 `npm run test`
- 如涉及构建链路，已运行 `npm run build`
- 文档与代码一致
- 未提交敏感信息、`.env` 或临时文件

## Bug 与安全问题

- 普通问题请使用 Issue：<https://github.com/hahaliu1029/Actus/issues>
- 安全问题请不要公开提交，请参考 [SECURITY.md](SECURITY.md)
