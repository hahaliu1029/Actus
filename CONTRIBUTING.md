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

### C2 Coordinator env override review

打开 `ACTUS_C2_COORDINATOR_ENABLED=true` 之前，请逐项 review 以下硬上限 env：

**PR-6 enforcement status标签：**
- ✅ **gated in PR-6** — dispatch / orchestrator 路径在本 PR 已读取该 env 并按其值触发拒绝/超时。
- 🚧 **deferred (out-of-scope of the rollout-readiness pipeline)** — 本 PR 仅落地 service / callback 实现，对应 dispatch / orchestrator callsite 尚未读取该 env；flag flip 前调整该 env 值无运行时效果。（per-CHILD budget/wallclock 接入已由 commit `7f2f853` 完成，见下表 ✅ rows。）

| Env var | 默认值 | PR-6 状态 | Override 注意事项 |
|---|---|---|---|
| `ACTUS_COORDINATOR_MAX_WORK_UNITS_PER_RUN` | `5` | ✅ gated | `_first_time_dispatch` preflight（`parallel_execution_subgraph.py:297`）按 `>` 拒绝。设置 > 7 易触发 mailbox supervisor 退化（fan-out 放大 + 单 root_session 流量集中）；> 10 会撞 `MAX_DESCENDANTS_PER_ROOT=10` 静态上限并被 descendants cap 拒绝。 |
| `ACTUS_COORDINATOR_MAX_TOTAL_TOKEN_COST_USD_PER_RUN` | `2.00` | 🚧 deferred（out-of-scope of the rollout-readiness pipeline） | per-RUN token cap 仍无 callsite：`parallel_execution_subgraph.py:365` 仅有 commented TODO，dispatch 从未读取该 limit（注意 per-CHILD token cap 已 wired，见下一行）。语义是 "整个 coordinator run 的累计 token 成本硬上限"。与 LLM provider 余额 / 速率限制协调，过大会让 budget watchdog 在 LLM 限流之后才触发，浪费 token。 |
| `ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_CHILD` | `0.50` | ✅ wired | per-child token cap 已接入（commit `7f2f853`）：`BudgetEnforcementCallback` 经 starter 后置注入（`coordinator_child_runner_starter.py:220/308`）→ adapter → 子任务 LLM callbacks 链，累计 USD `>= cap` 触发 `request_stop(StopReason.TOKEN_BUDGET)` → `NEEDS_AUTHORIZATION(reason="budget_exhausted")`。建议保持 `child * max_work_units_per_run >= total_run`，避免某些 child 提前被切但 per-run total 未到。 |
| `ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD` | `300` | ✅ wired | per-child wallclock watchdog 已接入（commit `7f2f853`）：`CoordinatorChildRunner.run_work_unit` 调用 `start_wallclock_watchdog`（`coordinator_child_runner.py:334`），超时触发 `StopReason.WALLCLOCK_BUDGET`。`load_coordinator_limits_from_env()` 强制不变式 `< SUBAGENT_RESULT_READY_TIMEOUT_SECONDS (600s)`：违反时 silent fallback 到默认（避免 child 错过内部 cap → 被 supervisor backstop 杀掉错配 `TIMED_OUT`）。 |
| `ACTUS_COORDINATOR_MAX_TOTAL_WALLCLOCK_SECONDS_PER_RUN` | `900` | 🚧 deferred（out-of-scope of the rollout-readiness pipeline） | per-RUN wallclock cap 仍无 callsite：`CoordinatorRunOrchestrator` 硬编码 `timeout_seconds=600`（`coordinator_run_orchestrator.py:235`），**从未**从 `CoordinatorLimits` 读取该 900 limit（注意 per-CHILD wallclock cap 已 wired，见上一行）。值的语义是 "整个 coordinator run（含所有 work_unit + reducer）的总墙钟硬上限"；调高时确认 SSE 连接 + 客户端超时配置同步放宽；调低会让长任务的 reducer 整合阶段被强制截断。 |
| `ACTUS_COORDINATOR_MAX_CONCURRENT_RUNS_PER_USER` | `2` | ✅ gated | `_first_time_dispatch` 调用 `probe_quota.acquire_coordinator_concurrency`（`parallel_execution_subgraph.py:307`），基于 Redis 原子 `INCR` + `> cap` rollback。调高时确认 Redis 容量 + 用户事件配额；调低后已被 acquire 的 slot 由 `release_coordinator_quotas` 自然回落。**Pod crash recovery**：concurrency key 在每次成功 acquire 时刷 6h TTL（`CONCURRENCY_TTL_SECONDS = 21600`，codex round 3 P1-5），兜底 "dispatch INCR 后 pod crash、reducer DECR 永远不运行" 导致永久 slot 泄漏的场景；正常生命周期下 `reducer_node.finally` 的 DECR 在 TTL 触发前就已经释放槽位，TTL 只是 ceiling，不是常规清理路径。 |
| `ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_USER_PER_DAY` | `50.00` | 🚧 deferred | **PR-6 仅落地 service method**（`ProbeQuotaService.acquire_coordinator_daily_cost`，基于 Redis `INCRBYFLOAT` + 负向回滚 + 25h TTL，key 为 `actus:coord:daily_cost:{user_id}:{utc-date}`）；**dispatch 调用点延后到 PR-7+ wiring**（见 `parallel_execution_subgraph.py` `_first_time_dispatch` 的 TODO），flag flip 前该 cap 并未在生产路径生效，调整本 env 值在 PR-6 阶段无运行时效果。过低会让正常用户在跨日临近时被 reject；rollback 路径会自动负向 INCRBYFLOAT 抹掉超额。 |
| `ACTUS_COORDINATOR_MAX_TOOL_CALLS_PER_CHILD` | `25` | 🚧 deferred | 当前 dispatch 通过 `CoordinatorEnvelopeFactory.make_spawn_request` 的默认 `CoordinatorBudgetSnapshot(max_tool_calls=25, ...)` 写入 SPAWN_REQUEST envelope，child 收到但**累计 tool-count 上限尚未 enforce**：`ChildScopeGate._tool_call_budget_exhausted`（`child_scope_gate.py:174-176`）目前只判 static `max_tool_calls <= 0`，累计计数器尚未接入（cumulative counter 仍 out-of-scope of the rollout-readiness pipeline）。env 当前**不会**影响 envelope 默认值（factory 默认是 hardcoded `25`）；累计 enforcement 接入前调整本 env 无运行时效果。 |

任何覆盖都建议在 staging 环境跑一次 dispatch preflight + budget watchdog 烟测，确认日志中无 `coordinator_limits: ... non-positive, using default` 或 `>= supervisor backstop` 警告。✅ rows 立即生效；🚧 rows 在对应 callsite 接入前是惰性的（只影响 `CoordinatorLimits` 实例内字段，未被消费者读取）——这些接入 out-of-scope of the rollout-readiness pipeline。

### C2 Coordinator OTel monitoring

`CoordinatorMetrics`（`infrastructure/observability/coordinator_telemetry.py`）发出 4 个 OTel instrument。打开 `ACTUS_C2_COORDINATOR_ENABLED=true` 之前请先把它们接入 Prometheus / Grafana / Phoenix dashboards：

| Metric | Type | Unit | 建议告警 |
|---|---|---|---|
| `actus_coordinator_run_cost_usd` | Counter | usd | `rate(... [5m]) > 0.5` 触发支出速率告警 |
| `actus_coordinator_tool_calls` | Counter | 1 | `rate(... [1m]) > 30` 警示 tool-call 循环 |
| `actus_coordinator_duration_seconds` | Histogram | s | `p99 > 540s` 接近 supervisor 600s backstop |
| `actus_coordinator_budget_exhaustion_total` | Counter | 1 | `rate(... [10m]) > 0` 子任务触顶 budget |

Instrument 在 PR-6 已落地；**4 个 instrument 现已全部接入 call-site**（C2b rollout-readiness WS1b）：

- `actus_coordinator_budget_exhaustion_total` — 子任务 budget 终结器直接 `add`（`coordinator_child_runner.py:653`）。
- `actus_coordinator_tool_calls` — invoke-adapter `_drain` 每观察到一个 `CALLING` ToolEvent 经 `CoordinatorMetricsRecorder.record_tool_call` 计数（`agent_task_runner_invoke_adapter.py` `_drain`），覆盖**每个** child（含失败/取消，不止成功写盘的子任务）。
- `actus_coordinator_run_cost_usd` + `actus_coordinator_duration_seconds` — `reducer_node` 经 `CoordinatorMetricsRecorder.record_run_terminal` 记录（run 级）。

Run 级 metrics **每个 run 只在首次派发路径记录一次**：派发起点由 `_first_time_dispatch` 打 `dispatch_started_monotonic` 时间戳，崩溃后 rehydrate 走 `_rehydrate_dispatch`（不打戳）→ reducer 不重复记录，避免 monotonic `run_cost_usd` counter 双计（§3.4）。`run_cost_usd` 仅在 cost 权威时 `add`（无 `cost_unavailable` 诊断），绝不 `add(0)`（避免把 "未知" 伪装成 "零"）。

Cardinality 注意：metric label 携带高基数维度 `coordinator_run_id`（+ `user_id_hash` + `function_name`），在真实 Prometheus 上每个 run 会生成一组 series。当前无生产部署（scrape 仅本地），可接受；若将来部署，应把 `coordinator_run_id` 下放到 span/log 属性、保留可聚合维度、对 `function_name` 分桶（§3.7）。

### C2 Coordinator concurrency leak recovery

`ProbeQuotaService` 的 per-user concurrency 计数器（`actus:coord:concurrent:{user_id}`）内置 6h auto-expire（`CONCURRENCY_TTL_SECONDS=21600`）：dispatch INCR 成功之后立即刷 TTL，pod 即使在 `acquire_coordinator_concurrency()` 与 `release_coordinator_quotas()` 之间崩溃，6h 内 Redis 也会自动收回 slot。

若用户反馈 "concurrency cap reached" 但你确认无活跃 coordinator run：

```bash
# 1) 排查 — 当前计数 + 剩余 TTL：
docker compose exec redis redis-cli GET "actus:coord:concurrent:<user_id>"
docker compose exec redis redis-cli TTL "actus:coord:concurrent:<user_id>"

# 2) 手动恢复（确认无 live coordinator 时）：
# 完全清掉：
docker compose exec redis redis-cli DEL "actus:coord:concurrent:<user_id>"
# 或单步 DECR 回到真实并发数：
docker compose exec redis redis-cli DECR "actus:coord:concurrent:<user_id>"
```

操作前用 `GET` 复核当前计数；操作后再 `GET` 一次确认结果。`DEL` 比 `DECR` 安全（避免误判后变负值），但会丢掉 TTL —— 下次正常 acquire 时会重新设置。

### C2 Cost rollup PEL retry behavior

`MailboxSupervisor.ResultReadyHandler` 在 destroy sandbox 之前先跑 cost rollup PROLOGUE。如果 destroy 抛 `SandboxLifecycleError`，supervisor 会在 Redis PEL（Pending Entry List）保留 envelope，`XAUTOCLAIM` 后续重投递 → rollup 会重复触发。

`CostRollupService.rollup_to_parent(*, parent_session_id, cost, source, idempotency_key)` 契约要求幂等：`idempotency_key` 是 mailbox envelope 的 `envelope_id`（UUID，跨重试稳定）。具体 adapter（PR-7+ 落地）必须基于此 key 去重 —— UNIQUE 约束 或 "first-write wins" CostRecord 行 都可，但**禁止**累加。

若发现 parent session 成本被重复计入：
1. 查 supervisor 审计日志中是否有 `XAUTOCLAIM` 触发的 envelope redelivery；
2. 验证 adapter 是否对相同 `idempotency_key` 去重（grep `idempotency_key` 在 adapter 实现中的使用）；
3. PR-7+ 的 adapter 必须配套 "PEL retry 同 envelope 不重复 rollup" 集成测试 —— 这是回归守门，落地前不能 ship 整套。

### C2 PR-7 Crash recovery + rehydrate rollout

PR-7 落地了 coordinator 的 crash-recovery 主路径（spec §12）：pod 重启后通过 DB + envelope store 三源恢复 ↦ `ALREADY_APPLIED:{status}:{audit_id}` 短路防重复 apply ↦ 缺失子任务硬失败 (v1：raise RuntimeError，PR-7+ 引入幂等 spawn 替换)。

| Surface | PR-7 落地 | 当前生效状态 | 备注 |
|---------|-----------|----------------|------|
| `coordinator_result_envelope_store` table | ✅ migration `c2pr7_envelope_store` | upgrade head 后表已存在 | 7 columns + `(coordinator_run_id, work_unit_id)` UNIQUE；migration 在所有 `alembic upgrade head` 跑 |
| `CoordinatorRehydrateService` 7-step | ✅ application/services | ✅ live（C2 finish-core epic `bd400ac` 激活） | `detect_existing_run` 返回 `RehydrateResult(child_session_ids, pending, terminal, already_applied)` |
| `SessionRepository.find_children_by_coordinator_run` | ✅ ABC + DB impl | 已 ship | 按 `(coordinator_run_id, parent_session_id)` 排序 by `created_at` ASC |
| MailboxSupervisor `persist_terminal` PROLOGUE | ✅ ResultReady + CancelAck | ✅ wired | gate 双 None：`coordinator_envelope_store` AND `session_repo`。C2 finish-core epic（`bd400ac`）注入了 `coordinator_envelope_store`（`service_dependencies.py:785` AND `:964`）→ PROLOGUE live path 已激活 |
| `_rehydrate_dispatch` 4 branches | ✅ subgraph | ✅ live（C2 finish-core epic `bd400ac`；仅崩溃恢复路径触达） | already_applied 短路 + 终端 envelope pre-populate + 意外子任务 CANCEL_REQUEST + missing-child raise（v1 hard fail）|
| `main_graph._run_parallel_backend` `ALREADY_APPLIED:` 短路 | ✅ 4 status 分支 | 已 ship | success / rollback_partial / crash_mid_apply / in_progress_recent 各自有 operator-facing summary |
| `HealthEvent` rollback_partial + crash_mid_apply 告警 | ✅ rehydrate service | 已 ship | `HealthStatus.TERMINATING` + `metrics["code"]` = `coordinator_apply_rollback_partial` / `coordinator_apply_crash_mid_apply` |
| Envelope store payload safety（whitelist + 64KB + PII + 非序列化）| ✅ 4 path | 已 ship | minimum rehydrate fields = `{outcome, patch_manifest, cost_summary, needs_authorization_details, final_state}`；写入前 strip + truncate + PII redact + JSON fallback |

**Flip checklist (cleanup)**：
1. ✅ DONE（C2 finish-core epic `bd400ac`）：`service_dependencies.py` 已注入 `coordinator_envelope_store=DbCoordinatorResultEnvelopeStoreRepository(...)`（`:785` AND `:964`）和 `cost_rollup_service=...`；
2. 把 `parallel_execution_subgraph.py` `_first_time_dispatch` 的 PR-7+ TODO（daily cost cap + 缺失 child 幂等 spawn）逐项落地（仍 out-of-scope of the rollout-readiness pipeline）；
3. ✅ DONE（commit `7f2f853`）：`coordinator_child_runner.py` 的 wallclock watchdog（`start_wallclock_watchdog`，现位于 `:334`）+ budget callback 已跑通，`planner_react.py:_build_config` 的 coordinator DI 链也已接好；
4. 翻 `ACTUS_C2_COORDINATOR_ENABLED=true` 前必须确认 §15.2 7 AST gate + 6 pytest marker + 3 E2E integration test 全过。剩余收尾 = 本 PR（C2b rollout-readiness WS1b）落地的 3 个 OTel instrument call-site wiring（见上方 OTel monitoring 段）。

**PR-7 missing-child 硬失败（v1 契约）**：

`_rehydrate_dispatch` 检测到 `work_units` 中存在 wu_id 在 `existing.child_session_ids` 中找不到对应行时，会 raise `RuntimeError("rehydrate: cannot resume run ...")` 而非 silently fall through to reducer with 不完整 worker_results。原因：silently fall-through 会让 applier 提交"少了几个子任务结果"的不完整 patch plan。两种正常诱因：(a) planner 在 retry 时生成了不同的 work_unit 集合（plan 变化）；(b) 原始 dispatch 在 create_session_with_parent 循环中途 crash。PR-7+ 会引入 `_spawn_one(wu)` 幂等 spawn（依赖 `(coordinator_run_id, work_unit_id)` 上的 partial-unique INDEX）取代硬失败。

## C2 PR-9 Acceptance — Flip `ACTUS_C2_COORDINATOR_ENABLED=true`

> **Note on PR-9a / PR-9b nomenclature:** The plan at
> `docs/superpowers/plans/2026-05-25-c2-coordinator-task-runner-plan.md`
> defines a single PR-9 (lines 9390-10220). At implementation time, the
> tests + scaffolds + flip-SOP slice shipped as **PR-9a**; the remaining
> coordinator wirings (fixture harness, 4 deferred composition-root
> `_emit_event`/repo callables, `reducer_node` cost_total source,
> `CoordinatorApplyEvent` lineage threading) are tracked separately as
> **PR-9b**. PR-9b does not yet have its own plan document — when work
> begins, write a `docs/superpowers/plans/<date>-c2-coordinator-pr9b-*.md`
> follow-up plan and link it here.

PR-9 ships in two phases. Read both before flipping the flag.

### PR-9a (this PR) — scaffolds + invariants

Shipped:
- 7 AST + schema gates enforcing static C2 invariants (`tests/domain/**/test_executor_*.py`, `tests/domain/services/graphs/test_two_clock_extended_parallel.py`, `tests/application/services/test_reducer_purity.py`, `tests/application/services/test_coordinator_no_sandbox_destroy.py`, `tests/domain/services/permission/test_child_scope_gate_prologue.py`, `tests/domain/models/test_coordinator_lineage_mixin_enforced.py`, `tests/integration/test_coordinator_apply_audit_partial_unique.py`).
- 3 E2E integration test scaffolds (`tests/integration/test_coordinator_e2e_*.py`) — shipped `@pytest.mark.skip` in PR-9a; **unskipped + rewritten by the C2 finish-core epic (PR-F5)** to run flag-on in the `coordinator-e2e` CI job.
- 6 pytest markers (`coordinator_pure`, `coordinator_graph`, `coordinator_worker`, `coordinator_mailbox`, `coordinator_apply`, `coordinator_recovery`).
- CI yml marker-split execution.

NOT yet shipped *as of PR-9a* (ALL closed since — by PR-9b + the C2 finish-core epic PR-F1..F5; see "C2 finish-core (2026-05-30)" below. Kept for historical context):
- 7 fixtures the E2E tests need: `async_client`, `async_session` (or `async_session_factory` re-spec), `redis_real`, `minio_real`, `sandbox_real`, `fixture_mock_llm_3_workers`, `env_with_coordinator_flag_on`.
- 4 deferred composition-root wirings at `api/app/interfaces/service_dependencies.py:715-755`:
  - `SupervisorContext.cost_rollup_service` (PR-6 §14.4)
  - `SupervisorContext.coordinator_envelope_store` (PR-7 §12.4)
  - `PatchApplier._emit_event` (PR-8 §13.5)
  - `CoordinatorRunOrchestrator._emit_event` (PR-8 §13.6)
- `reducer_node` cost_total source wiring (`api/app/domain/services/graphs/parallel_execution_subgraph.py` inline comment, PR-8 R2 P2).
- `CoordinatorApplyEvent` lineage threading (`api/app/application/services/patch_applier.py:790` inline comment, PR-8 R1 P2).

While the flag stays `false`, the coordinator emit sites silently no-op and the supervisor behaves identically to pre-coordinator code.

### Acceptance gate (before flipping `ACTUS_C2_COORDINATOR_ENABLED=true`)

> **Superseded (2026-06-01) by "C2 finish-core (2026-05-30)" + the "Flag flip
> checklist" below.** PR-9b + the finish-core epic shipped the fixtures, the 4
> emit/repo wirings, cost_total, and lineage; the canonical pre-flip command set
> is now the **Flag flip checklist** (7 passed, 0 skipped). The PR-9a-era steps
> below are retained for history — the "3 passed" in step 4 predates the
> dark-launch + parametrized unskip-guard and is no longer the live count.

1. PR-1..8 + PR-9a merged to `develop` (already done by this PR's prerequisites).
2. PR-9b merged: fixtures land, 4 emit_event/repo wirings flip to non-None at the composition root, cost_total + lineage TODOs closed.
3. All 7 AST gates pass: `cd api && uv run pytest -m "not coordinator_recovery and not slow and not sandbox and not browser_eval" --tb=short` exits 0.
4. All 3 E2E tests pass (no longer skipped): `cd api && uv run pytest -m "coordinator_recovery and not slow and not sandbox and not browser_eval" --tb=short` exits 0 with 3 passed.
5. All `MailboxSupervisor` pods upgraded to the new envelope schema (operator check).
6. PR-9b codex xhigh review verdict READY (P0 = P1 = P2 = 0).

### Flip procedure

1. Update `.env` (or k8s ConfigMap): `ACTUS_C2_COORDINATOR_ENABLED=true`.
2. Roll restart `api` pods.
3. Monitor:
   - `actus_coordinator_run_cost_usd` — emitted from `reducer_node` via `CoordinatorMetricsRecorder.record_run_terminal` (call-site wired by the C2b rollout-readiness WS1b PR; run-level, recorded once per run on the first-time-dispatch path — rehydrate skips, §3.4).
   - `actus_coordinator_budget_exhaustion_total` — emitted from the child runner's budget finalizer (`CoordinatorChildRunner._finalize_needs_authorization_budget`, single aggregation point for both token/wallclock reasons; wired by the C2b in-flight budget PR).

   Precondition: drive a synthetic coordinator run (a planner step with 2-3 `parallel_work_units`) and verify `actus_coordinator_run_cost_usd > 0` via the metrics endpoint.
4. 24h observation window after the synthetic-run smoke check passes.
5. Rollback if anomaly: revert `ACTUS_C2_COORDINATOR_ENABLED=false`, roll restart `api` pods.

### Rollback safety

- Planner teaching is now injected (flag-gated) by the C2b rollout-readiness WS0 PR: the `PARALLEL_WORK_UNITS_TEACHING_{EN,ZH}` constants live in `api/app/domain/services/prompts/sections/parallel_work_units_teaching.py` and are registered as `parallel_work_units_teaching_section` at index 1 of BOTH the planner and updater registries in `bundles/en.py` + `bundles/zh.py`. The section's `_render` calls `is_coordinator_enabled()` per render, so flag-OFF emits nothing (planner never learns the `parallel_work_units` schema) and flag-ON teaches it — flipping `ACTUS_C2_COORDINATOR_ENABLED=true` now actually enables coordinator dispatch (the flag is no longer inert). Note the prompt teaching-gate is only a **probability reducer** — it lowers the chance the planner emits `parallel_work_units` while off, but it is NOT the hard safety control (an LLM could still emit the schema unprompted).
- Defense in depth — the HARD flag-off safety comes from two non-prompt layers, not the teaching-gate: (1) `_build_plan_from_response` + updater **sanitation** strip any `parallel_work_units` from a plan while the flag is `false`; (2) if a cold-code path or stale prompt ever produced `parallel_work_units` and it reached the executor, the executor's `assert_coordinator_enabled()` at `api/app/domain/services/graphs/main_graph.py:767` raises `RuntimeError` — fail-loud rather than silent-dispatch.
- Mid-run rollback caveat: flipping the flag back to `false` mid-run is NOT graceful. The supervisor continues consuming envelopes already dispatched, but any rehydrate-on-restart will hit `assert_coordinator_enabled()` and fail-fast. Drain in-flight coordinator runs (or wait for them to reach terminal) before flipping `false`.
- No data loss across flips; `coordinator_apply_audit`, `coordinator_result_envelope_store`, `coordinator_run_state` tables persist independently of the flag.

### C2 finish-core (2026-05-30) — flag-ready in CI, NOT flipped in production

The "C2 coordinator finish" epic (PR-F1..F5) closed the flag-on blocking gaps:
the child-runner is fully wired (invoke-adapter + per-child sandbox + cost +
cancel seam), apply/recovery is live (adapter factory + G2b path contract +
rehydrate emit + orchestrator group-create hoist), and a dedicated
`coordinator-e2e` CI job runs the 3 E2E + a flag-OFF dark-launch + the reverse
unskip guard GREEN flag-on against real pg+redis+minio+sandbox with a fake LLM.

**`ACTUS_C2_COORDINATOR_ENABLED=true` runs ONLY in test/CI. Production stays
default-off.** The minimal read-only SSE coordinator timeline is now **DONE**
(C2b rollout-readiness PR-3 / WS2: `ui/src/components/session/coordinator-timeline-item.tsx`
+ the 5 `coordinator_*` render branches in `ui/src/app/sessions/[id]/page.tsx`
+ the TS event types); only a fancy coordinator dashboard (NG4) remains out of
scope. Remaining deferrals (NOT done): production flip, live-provider
acceptance, canary/rollout automation, fancy dashboard UI (NG4), N≥10 perf,
full ChildScopeGate live-wiring (child confirmation is disabled; lease safety via
tool_filter + patch-extraction lease-check + reducer), `atomic_write_file` true
atomicity, per-RUN wallclock/token budget wiring (per-CHILD budgets ARE wired),
multi-level spawn.

## C2 shell-mode (S2)

Coordinator children may opt into **shell-mode** (raw shell inside an isolated
ephemeral sandbox) while still producing only a lease-validated `PatchManifest`.
See [`docs/shell-mode/c2-shell-mode.md`](./docs/shell-mode/c2-shell-mode.md) for the full design.

Hard invariants when touching this path:
- **Fail-safe default**: shell-mode requires `flag_on AND wu.shell_mode` (both
  affirmative). Flag OFF (`ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED` default
  OFF) ⇒ byte-for-byte pre-S2 behavior; `dispatch_node` actively coerces a
  stale `shell_mode`/`write_tree_lease` payload to typed-only (mixed units that
  keep a typed lease) or hard-rejects it (tree-only / shell-only units). Pinned by
  `tests/integration/test_coordinator_shell_mode_dark_launch.py` (which also
  asserts ZERO shell tools are bound while the flag is OFF).
- **Tree leases are ADD-only**: modify/delete need a seeded file lease;
  tree-only modify/delete zero-applies the whole group.
- **Group zero-apply is the integrity boundary**: any out-of-lease / special /
  symlink / mode-only / scan-truncated diff discards the WHOLE manifest.
- The live shell-mode E2E (`tests/integration/test_coordinator_e2e_shell_mode.py`)
  runs only in the `coordinator-e2e` CI job and must never be silently skipped —
  `tests/structure/test_shell_mode_e2e_unskip_guard.py` enforces this.

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

## C2 Coordinator Rollout SOP

### Concept anchors (do NOT drift from these)

1. **`cost_rollup_service` is a metric / observability hook, NOT a cost source.**
   The authoritative cost source is the `cost_records` ledger via
   `CostCallbackHandler` + `flush_pending()`. `CoordinatorReduceEvent.cost_total`
   is computed by `CostRollupService.aggregate(...)`, which queries the ledger.
   The legacy `rollup_to_parent(...)` push call is preserved for telemetry but
   does NOT participate in coordinator event payload computation.

2. **Business lineage is the system contract; OTel is observability.**
   `root_session_id / parent_session_id / child_session_id / coordinator_run_id /
   work_unit_id` are authoritative for SSE, DB joins, rehydrate, idempotency,
   UI grouping. OTel `traceparent` / span context is for logs / metrics / traces
   only and MUST NOT participate in business correlation.

3. **`CoordinatorApplyEvent` is group-level.** It carries 3 lineage fields
   (`root_session_id`, `parent_session_id`, `coordinator_run_id`). Per-child
   fields (`child_session_id`, `work_unit_id`) stay `None` — they belong on
   per-worker events.

### Flag flip checklist

> **✅ Now runnable (C2 finish-core epic, PR-F5).** The C2 finish-core epic landed
> `tests/integration/test_coordinator_dark_launch.py` and
> `tests/structure/test_coordinator_e2e_unskipped.py`, unskipped + rewrote the 3
> E2E tests, and retired `tests/structure/test_coordinator_e2e_skip_honesty.py`
> (replaced by the inverse unskip guard). These run flag-on in the dedicated
> `coordinator-e2e` CI job. `ACTUS_C2_COORDINATOR_ENABLED` still stays `false` in
> production — the flag is exercised only in test/CI (see "C2 finish-core
> (2026-05-30) — flag-ready in CI, NOT flipped in production" above).

Run these two commands on the candidate commit; both must pass:

1. **Coordinator acceptance** — exactly 7 passed, 0 skipped (1 dark-launch + 3 E2E + 3 structural guard items — `test_coordinator_e2e_unskipped.py` is parametrized over the 3 enumerated E2E files):

   ```bash
   cd api && uv run pytest \
     tests/integration/test_coordinator_dark_launch.py \
     tests/integration/test_coordinator_e2e_apply_rollback.py \
     tests/integration/test_coordinator_e2e_3_work_units.py \
     tests/integration/test_coordinator_e2e_sibling_cancel.py \
     tests/structure/test_coordinator_e2e_unskipped.py \
     --tb=short -rs --strict-markers --strict-config
   ```

2. **Non-coordinator-recovery regression** — no new failures vs. `develop`
   baseline:

   ```bash
   cd api && uv run pytest \
     -m "not slow and not sandbox and not browser_eval and not coordinator_recovery" \
     --tb=short --strict-config
   ```

Then flip `ACTUS_C2_COORDINATOR_ENABLED=true` via deploy config.

### Rollback caveat

Flipping `ACTUS_C2_COORDINATOR_ENABLED=false` while a coordinator run is
in-flight is **NOT a graceful abort**. Wiring stays live; in-flight runs continue
to emit events and drain to terminal state. The flag only gates the entry to
**new** coordinator dispatches. To fully drain before rollback, wait for all
sessions with a `parallel_work_units` step to reach terminal status, then flip.
