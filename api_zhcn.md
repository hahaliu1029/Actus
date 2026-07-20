# API 文档（简体中文）

本文档按当前代码中的实际路由整理，基础路径为 `/api`。

## 基础约定

### 响应包裹

大多数 JSON 接口返回统一结构：

```json
{
  "code": 200,
  "msg": "success",
  "data": {}
}
```

注意：

- 部分错误虽然 `HTTP status` 仍可能为 `200`，但 `code` 会是业务错误码
- 显式迁移接口会返回 `410`
- 文件下载、微信回调、SSE、WebSocket 不使用统一 JSON 包裹

### 认证

- 需要登录的接口使用 `Authorization: Bearer <access_token>`
- 管理员接口要求 `role=super_admin`
- WebSocket 鉴权通过 query 参数传递 `token`

### 限流

- 超限返回 `429`
- 响应体示例：`{"code":429,"msg":"请求过多，请稍后重试","data":{"retry_after":N}}`
- 限流依赖 Redis；Redis 不可用时相关接口会返回 `503`
- 认证类接口（`/auth/register`、`/auth/login`、`/auth/refresh`、`/auth/wechat/*`）应用独立的 `rate_limit_auth` 限流策略

### 流式与实时通道

- SSE：
  - `POST /sessions/stream`
  - `POST /sessions/{session_id}/chat`
  - `POST /sessions/{parent_session_id}/subagents/research`
  - `POST /v2/skills/create`
- HTTP 增量恢复：
  - `GET /sessions/{session_id}/events?since_seq=...&since=...`（按已知游标获取增量事件 + 当前会话/supervisor 状态；优先使用 `since_seq`，`since` 保留为旧事件兜底）
- WebSocket：
  - `/sessions/{session_id}/takeover/shell/ws?takeover_id=...&token=...`
  - `/sessions/{session_id}/vnc?token=...`

### 会话状态

`pending | running | takeover_pending | takeover | waiting | finishing | completed | timed_out`

- `finishing` 是 `running → completed` 之间的中间态，承载最终摘要、附件归档、未读计数刷新等异步收尾
- `timed_out` 是 watchdog 总超时或恢复失败时写入的终态

### 会话 SSE 事件类型

对话流可能发出：

`message | title | step | plan | tool | tool_confirmation | wait | control | context_status | compaction | finishing | health | sandbox_state_changed | session_mode_changed | coordinator_dispatch | coordinator_worker_spawned | coordinator_reduce | coordinator_apply | coordinator_sibling_cancel | done | error`

补充：
- `tool_confirmation` 用于工具审批与确认系统，前端展示工具确认卡片并把决策回传到 `/chat` 接口的 `tool_confirmation` 字段
- `finishing` 在会话进入 `finishing` 状态前发出，前端应解锁输入框并展示"收尾中"指示
- `health` 事件用于展示后端探活与降级状态
- `context_status` / `compaction` 上报上下文窗口压力和渐进压缩动作
- 开启 lifecycle 投影后，SSE 事件名采用点分格式：`lifecycle.{task|plan|step|tool|subagent}.{started|progress|completed|failed|cancelled|retried}`。每种 `lifecycle_type` 只允许其中一个受控子集；payload 还包含 `type=lifecycle`、`state`、`unit_id`、`epoch`、源事件引用，以及可选的父级/关联信息
- 研究子智能体流会发出 `child_started`、`child_done`、`joined_summary`

## 运行时配置说明

- Docker Compose 模式下，业务配置文件位于 `/app/data/config.yaml`
- 本地后端开发默认使用 `api/config.yaml`
- Skill v2 默认存储目录为 `/app/data/skills`
- `SANDBOX_PROVISION_MODE` 仅由环境变量配置：`always` 预先创建父沙箱，`on_demand` 延迟到首次沙箱操作再创建，`off` 关闭整个沙箱能力面。`GET /sessions/{session_id}` 通过 `data.sandbox_mode` 返回当前部署档位
- `EXTENSION_GOVERNANCE_MODE` 仅由环境变量配置，可取 `off | shadow | enforce`
- lifecycle 投影位于 `config.yaml` 的 `lifecycle_runtime` 下；`lifecycle_events_enabled` 是总开关，`lifecycle_subagent_events_enabled` 只有与总开关同时开启才生效，两者默认均为 `false`
- MinIO 区分内部和公开配置：`MINIO_ENDPOINT` 用于 API 侧对象读写，`MINIO_PUBLIC_ENDPOINT` / `MINIO_PUBLIC_SECURE` 用于生成浏览器或远程消费者可访问的 URL。设置公开 endpoint 时必须同时配置 `MINIO_REGION`

## 认证模块 `/auth`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `POST` | `/auth/register` | 否 | 注册用户，返回用户信息和 token |
| `POST` | `/auth/login` | 否 | 用户名或邮箱登录 |
| `POST` | `/auth/refresh` | 否 | 刷新 access token |
| `GET` | `/auth/me` | 是 | 获取当前用户 |
| `PUT` | `/auth/me` | 是 | 更新当前用户昵称、头像 |
| `GET` | `/auth/wechat/authorize` | 否 | 获取微信网页授权 URL |
| `GET` | `/auth/wechat/callback` | 否 | 微信回调，处理后重定向前端 |

## 状态模块 `/status`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/status/` | 是 | 检查 FastAPI、PostgreSQL、Redis、MinIO 健康状态 |
| `GET` | `/status/minio` | 是 | 检查 MinIO 连通性；`smoke=true` 时执行读写自检 |
| `POST` | `/status/minio/upload` | 管理员 | 通过 multipart/form-data 上传测试文件到 MinIO |

## 内部指标 `/v1/metrics`

`GET /v1/metrics` 是内部 Prometheus 抓取接口，刻意不出现在 OpenAPI 中。`ACTUS_METRICS_ENDPOINT_TOKEN`/`METRICS_ENDPOINT_TOKEN` 为空时接口禁用并返回 `404`；设置后要求 `Authorization: Bearer <token>`，返回 Prometheus exposition 文本。

## 设置模块 `/app-config`

### LLM 与 Agent

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/app-config/llm` | 是 | 获取 LLM 配置（不回传 `api_key`） |
| `POST` | `/app-config/llm` | 管理员 | 更新 LLM 配置 |
| `GET` | `/app-config/agent` | 是 | 获取 Agent 配置 |
| `POST` | `/app-config/agent` | 管理员 | 更新 Agent 配置 |

`AgentConfig` 除 `max_iterations`、`max_retries`、`max_search_results` 外，还包含 `skill_selection`、`skill_embedding`、`memory`、`tool_confirmation`、`execution`、`slash_commands` 子配置。

### MCP

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/app-config/mcp-servers` | 是 | 获取 MCP 服务列表与探测到的工具名 |
| `POST` | `/app-config/mcp-servers` | 管理员 | 新增或更新 MCP 服务配置；治理开启时支持 `dry_run`、`acknowledge`、`force` query 参数 |
| `POST` | `/app-config/mcp-servers/{server_name}/delete` | 管理员 | 删除 MCP 服务 |
| `POST` | `/app-config/mcp-servers/{server_name}/enabled` | 管理员 | 更新 MCP 服务全局启用状态 |

### A2A

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/app-config/a2a-servers` | 是 | 获取 A2A 服务列表 |
| `POST` | `/app-config/a2a-servers` | 管理员 | 新增 A2A 服务（传 `base_url`）；治理开启时支持 `dry_run`、`acknowledge`、`force` query 参数 |
| `POST` | `/app-config/a2a-servers/{a2a_id}/delete` | 管理员 | 删除 A2A 服务 |
| `POST` | `/app-config/a2a-servers/{a2a_id}/enabled` | 管理员 | 更新 A2A 服务全局启用状态 |

### 文件理解

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/app-config/file-understanding` | 是 | 获取文件理解配置（视觉降级、音频、视频） |
| `POST` | `/app-config/file-understanding` | 管理员 | 更新文件理解配置 |

## Skill 模块

### 旧接口（迁移提示）

以下接口保留为迁移提示，调用时返回 `410 SKILL_API_MOVED`：

- `GET /app-config/skills`
- `POST /app-config/skills/install`
- `POST /app-config/skills/{skill_id}/enabled`
- `POST /app-config/skills/{skill_id}/delete`

### Skill v2 `/v2/skills`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/v2/skills` | 管理员 | 获取 Skill 列表 |
| `POST` | `/v2/skills/install?force={bool}` | 管理员 | 从 GitHub 或本地目录安装 Skill；`force=true` 可放行 dangerous 扫描结果 |
| `POST` | `/v2/skills/create` | 管理员 | AI 创建 Skill，SSE 返回进度和结果；沙箱 `off` 档返回 `409 SANDBOX_DISABLED` |
| `POST` | `/v2/skills/{skill_key}/enabled` | 管理员 | 更新 Skill 全局启用状态 |
| `DELETE` | `/v2/skills/{skill_key}` | 管理员 | 删除 Skill |
| `GET` | `/v2/skills/policy` | 是 | 获取 Skill 风险策略 |
| `POST` | `/v2/skills/policy` | 管理员 | 更新 Skill 风险策略 |
| `GET` | `/v2/skills/{skill_key}/export?format={format}` | 管理员 | 以 ZIP 导出 Skill；`format=agent-skills|actus` |
| `GET` | `/v2/skills/{skill_key}` | 管理员 | 获取 Skill 详情、工具定义、bundle 文件索引和原始 `SKILL.md` |

关键字段：

- `source_type`: `local | github`
- `runtime_type`: `native | mcp | a2a`
- `mode`: `off | enforce_confirmation`

## 运行时扩展 `/v1/runtime`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/v1/runtime/extensions` | 是 | 返回 MCP、A2A、Skill、Plugin 的聚合运行时快照；非管理员只获得裁剪后的投影 |
| `GET` | `/v1/runtime/extensions/catalog` | 是 | 返回静态 MCP 推荐目录 |
| `POST` | `/v1/runtime/extensions/{kind}/{ext_id}/enabled` | 管理员 | 通过统一 façade 启停 `mcp`、`a2a` 或 `skill` 扩展 |
| `POST` | `/v1/runtime/extensions/{kind}/{ext_id}/probe` | 管理员 | 探测已启用的 `mcp`、`a2a` 或 `skill`；同一条目重复探测有 5 秒冷却窗 |

聚合条目包含 `config`、`health`、`liveness`、`stats`、按 kind 区分的 `details`，以及治理开启时仅管理员可见的 `governance` 块。顶层 `probe_enabled` / `stats_enabled` 表示实际生效的运行能力，不只是原始配置值。已隔离条目不能通过 runtime façade 重新启用，必须调用治理 reapprove 接口。

## 扩展治理 `/v2/extensions`

本节全部接口都要求管理员；`kind` 可取 `mcp | a2a | skill | plugin`。

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v2/extensions/governance` | 获取治理计数；治理模式为 `off` 时返回字面量零摘要 |
| `POST` | `/v2/extensions/refresh-observations` | 刷新全部或指定扩展的观测数据 |
| `POST` | `/v2/extensions/approve-pins` | 批量转正全部或指定 pin |
| `GET` | `/v2/extensions/audit` | 翻页查询审计记录；支持 `kind`、`ext_id`、`event`、`cursor`、`limit` |
| `POST` | `/v2/extensions/{kind}/{ext_id}/refresh-observation` | 刷新单个扩展观测 |
| `POST` | `/v2/extensions/{kind}/{ext_id}/quarantine` | 隔离单个扩展 |
| `POST` | `/v2/extensions/{kind}/{ext_id}/reapprove` | 解除隔离并重新 pin |
| `POST` | `/v2/extensions/{kind}/{ext_id}/governance-disable` | 通过治理状态机停用扩展 |
| `POST` | `/v2/extensions/{kind}/{ext_id}/governance-enable` | 通过治理状态机启用扩展 |

治理写操作使用乐观并发：请求体携带 `expected_row_revision`，隔离接口还可传 `note`；状态修改成功后返回新的 `row_revision`。除返回零摘要的 GET 外，`EXTENSION_GOVERNANCE_MODE=off` 时治理接口返回 `409 governance_disabled`。

准入原因存在两条不同的阻断边界：

- 检测类原因（`unknown`、`unpinned`、`pin_stale`、`config_drift`、`pin_mismatch`、`registry_unavailable`）在 `shadow` 下只观测并放行，在 `enforce` 下阻断。
- 行政/结构类原因（`quarantined`、`disabled`、`deleted`、`parent_blocked`）在 `shadow` 和 `enforce` 下都会阻断。因此，`shadow` 只对检测结果 fail-open；显式停用/隔离或父 Plugin 被阻断不会被放行。
- `off` 不构造 registry/admission 服务，两类治理判定都不会执行。

## Plugin `/v2/plugins`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/v2/plugins` | 管理员 | 获取已安装 Plugin、最新 operation 和成员扩展 |
| `POST` | `/v2/plugins/install` | 管理员 | 通过 JSON body 的 `source_type` + `source_ref` 安装或预检 Plugin bundle；支持 `dry_run`、`force`、`acknowledge` |
| `POST` | `/v2/plugins/{plugin_ext_id}/enabled` | 管理员 | 通过 `enabled` + `expected_row_revision` 启停 Plugin |
| `DELETE` | `/v2/plugins/{plugin_ext_id}` | 管理员 | 通过必填的 `expected_row_revision` body 卸载 Plugin |

`POST /v2/plugins/install` 接受以下请求体（未知字段会被拒绝）：

```json
{
  "source_type": "local",
  "source_ref": "/absolute/path/to/plugin",
  "dry_run": true,
  "acknowledge": false,
  "force": false
}
```

可实际安装的 `source_type` 为 `local` 和 `github`；历史枚举值 `mcp_registry` 会被 source loader 以 `422` 拒绝。本地来源必须是绝对目录，GitHub 来源使用受支持的仓库/目录 URL，压缩的 `.zip`/`.tar` 输入会被拒绝。

三个安装标志的含义彼此独立：

| 模式/请求 | 结果 |
|-----------|------|
| `dry_run=true` | 返回零写入 preview，字段包括 `plugin_id`、`name`、`version`、`aggregate_verdict`、`install_policy_decision`、`members`、`warnings`。它不执行碰撞查询和最终安装门，因此真实安装结果才是最终依据。 |
| `shadow`，真实安装 | `safe` 直接放行；`caution`、`dangerous` 带警告放行。检测策略不要求 `acknowledge`/`force`。 |
| `enforce` + `caution` | 必须传 `acknowledge=true`，否则返回 `409 acknowledge_required`。 |
| `enforce` + `dangerous` | 必须传 `force=true`，否则返回 `422 force_required`。 |
| MCP/A2A 成员 probe 失败 | 两种启用的治理模式下都必须传 `force=true`；`acknowledge` 不能绕过。 |
| 声明 hash 与实测不符，或 Plugin/成员身份碰撞 | 真实安装始终拒绝，两个标志都不能绕过。 |

混合 bundle 可以同时传 `acknowledge=true` 和 `force=true`，各自只作用于匹配的告警/阻断等级。安装成功只返回 `plugin_ext_id`、`operation_id` 和 `status=completed`，**不返回 revision**；随后应调用 `GET /v2/plugins` 取得条目的 `row_revision`，再用于启停或卸载请求。列表还返回 `status`、`artifact_hash`、`last_operation` 以及成员状态/扫描字段。安装失败但补偿成功时返回 `422 plugin_install_failed_compensated`，包含 `operation_id`、`error`、`collided_targets`；补偿也失败时返回 `500 plugin_install_failed_requires_admin` 和 `operation_id`。

`POST /v2/plugins/{id}/enabled` 与 `POST /v2/extensions/plugin/{id}/governance-enable|governance-disable` 是同一个 Plugin 父治理行、同一个 CAS 状态迁移的两种 API 形式，不是两个独立开关。停用父项执行 `active -> disabled`，其成员变为 `parent_blocked`，但不会改写各成员的 MCP/A2A/Skill 配置；重新启用执行 `disabled -> active`，也不会覆盖成员自身的 disabled/quarantined 状态或运行时配置。runtime façade 只负责子类 `mcp`、`a2a`、`skill`，不接受 `plugin`。

Plugin 路由只在扩展治理不为 `off` 时可用，否则返回 `409 governance_disabled`。切换为 `off` 不等于卸载：持久化的 Plugin registry 行、bundle 和已物化成员不会被删除。Plugin 列表/管理面和父项投影会消失；已物化的 MCP/A2A/Skill 成员仍通过原有 config/Skill store 工作，不再执行准入或 `parent_blocked` 阻断。启动时的 Plugin saga 收尾和治理 reconcile 也会跳过，直到重新启用治理。

推荐按 `off -> shadow -> enforce` 灰度：先以 `shadow` 重建 API，检查 `/v2/extensions/governance` 和 `/v2/extensions/audit`，刷新观测并转正计划启用的 pin；理解检测结果后再以 `enforce` 重建。行政/结构类原因在 `shadow` 已经阻断，最终切换前应明确保留或处理。模式在启动时读入进程级 settings 并决定 lifespan 服务装配，仅修改 `.env` 不会生效。Compose 每次切换使用：

```bash
docker compose --env-file .env up -d --force-recreate api
```

## 用户工具偏好

### 旧版 `/user/tools`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/user/tools/mcp` | 是 | 获取带用户偏好的 MCP 工具列表 |
| `POST` | `/user/tools/mcp/{server_name}/enabled` | 是 | 设置 MCP 工具个人启用状态 |
| `GET` | `/user/tools/a2a` | 是 | 获取带用户偏好的 A2A 工具列表 |
| `POST` | `/user/tools/a2a/{a2a_id}/enabled` | 是 | 设置 A2A 工具个人启用状态 |
| `GET` | `/user/tools/skills` | 是 | 已迁移，返回 `410` |
| `POST` | `/user/tools/skills/{skill_id}/enabled` | 是 | 已迁移，返回 `410` |

### Skill 偏好 v2 `/v2/user/tools`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/v2/user/tools/skills` | 是 | 获取 Skill 工具列表与个人偏好 |
| `POST` | `/v2/user/tools/skills/{skill_key}/enabled` | 是 | 设置 Skill 工具个人启用状态 |

### 工具审批策略 `/v2/user/tool-policies`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/v2/user/tool-policies` | 是 | 获取当前用户显式设置的逐工具审批策略 |
| `GET` | `/v2/user/tool-policies/{tool_name}` | 是 | 获取一条显式策略；缺行返回 `404` |
| `PUT` | `/v2/user/tool-policies/{tool_name}` | 是 | 设置 `policy=auto|ask|deny` |
| `DELETE` | `/v2/user/tool-policies/{tool_name}` | 是 | 清除显式策略，操作幂等 |

## 文件模块 `/files`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `POST` | `/files` | 是 | 上传附件到对象存储并关联用户 |
| `GET` | `/files/{file_id}` | 是 | 获取文件元数据 |
| `GET` | `/files/{file_id}/download` | 是 | 下载文件内容 |
| `DELETE` | `/files/{file_id}` | 是 | 删除文件 |

## 记忆管理 `/v2/memories`

全部记忆接口都限定为当前用户的数据。

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v2/memories` | 分页、搜索并按 source/category/时间过滤记忆；支持 `auto_promoted_after` |
| `POST` | `/v2/memories` | 手工创建记忆（`201`） |
| `GET` | `/v2/memories/cleanup-config` | 获取当前部署的 legacy 清理上线边界 |
| `DELETE` | `/v2/memories/legacy` | 在配置边界内删除未分类的 legacy `session_flush` 记忆 |
| `GET` | `/v2/memories/{chunk_id}` | 获取单条记忆 |
| `PATCH` | `/v2/memories/{chunk_id}` | `content` 或 `pinned` 严格二选一更新 |
| `DELETE` | `/v2/memories/{chunk_id}` | 删除单条记忆 |
| `POST` | `/v2/memories/{chunk_id}/reindex` | 从磁盘 memory 文件重建数据库/搜索索引 |
| `POST` | `/v2/memories/bulk-delete` | 删除指定 id 列表 |
| `POST` | `/v2/memories/delete-all` | 清空当前用户全部记忆 |

记忆分类为 `user | rule | fact`；`auto_promoted_after` 必须是带时区的 ISO 8601 时间。

## 系统通知 `/v2/notifications`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/v2/notifications/unread?limit=N` | 是 | 获取最多 50 条未过期未读通知，并返回完整未读总数 |
| `POST` | `/v2/notifications/{notification_id}/mark-read` | 是 | 标记单条通知已读；重复调用或非本人通知返回 `marked_read=false` |

## 管理员模块 `/admin`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/admin/users` | 管理员 | 分页获取用户列表 |
| `GET` | `/admin/users/{user_id}` | 管理员 | 获取用户详情 |
| `PUT` | `/admin/users/{user_id}/status` | 管理员 | 更新用户状态 |
| `DELETE` | `/admin/users/{user_id}` | 管理员 | 删除用户 |

## 会话模块 `/sessions`

### 普通 HTTP / SSE 接口

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `POST` | `/sessions` | 是 | 创建新会话 |
| `POST` | `/sessions/stream` | 是，SSE | 流式推送会话列表 |
| `GET` | `/sessions` | 是 | 获取会话列表 |
| `GET` | `/sessions/background-quota` | 是 | 获取系统/用户后台执行额度使用情况 |
| `GET` | `/sessions/{session_id}/children?depth=N` | 是 | 返回深度受限的扁平后代列表及截断信息 |
| `POST` | `/sessions/{session_id}/clear-unread-message-count` | 是 | 清空未读消息数 |
| `POST` | `/sessions/{session_id}/delete` | 是 | 删除会话 |
| `POST` | `/sessions/{session_id}/chat` | 是，SSE | 向会话发送消息并流式接收事件；body 可携带 `tool_confirmation` 提交工具确认决策 |
| `POST` | `/sessions/{session_id}/cancel` | 是 | 请求取消任务并返回 `cancel_requested` 状态 |
| `GET` | `/sessions/{session_id}` | 是 | 获取会话详情、历史事件、supervisor 快照和部署级 `sandbox_mode` |
| `GET` | `/sessions/{session_id}/events?since={event_id}&since_seq={seq}` | 是 | 获取增量事件和当前会话/supervisor 状态；`since_seq` 是首选单调游标 |
| `GET` | `/sessions/{session_id}/takeover` | 是 | 获取当前接管状态 |
| `POST` | `/sessions/{session_id}/takeover/start` | 是 | 发起接管；`request_status=starting` 时 HTTP 为 `202` |
| `POST` | `/sessions/{session_id}/takeover/renew` | 是 | 续期接管租约 |
| `POST` | `/sessions/{session_id}/takeover/reject` | 是 | 处理 AI 发起的接管请求 |
| `POST` | `/sessions/{session_id}/takeover/end` | 是 | 结束接管并选择继续或完成 |
| `POST` | `/sessions/{session_id}/takeover/reopen` | 是 | 在窗口期内补救重开接管 |
| `POST` | `/sessions/{session_id}/retry-from-suspend` | 是 | 恢复可重试的后台挂起任务 |
| `POST` | `/sessions/{session_id}/stop` | 是 | 停止当前任务 |
| `GET` | `/sessions/{session_id}/files` | 是 | 获取会话文件列表 |
| `POST` | `/sessions/{session_id}/file` | 是 | 读取沙箱中文件内容 |
| `GET` | `/sessions/{session_id}/file/download?filepath=...` | 是 | 从沙箱下载二进制或文本文件 |
| `POST` | `/sessions/{session_id}/shell` | 是 | 读取指定 shell 会话输出 |
| `POST` | `/sessions/{parent_session_id}/subagents/research` | 是，SSE | 扇出 1-3 个只读研究子任务，并流式返回确定性 join 进度 |

### 恢复游标约定

每个持久化 SSE payload 都有 `data.event_id`；新序列化事件还包含 `data.seq`，且 SSE 的 `id` 行与 `data.event_id` 相同。客户端重连时应保存最大的正数 `seq` 和最新事件 id，并把两个游标都传给增量事件接口：`since_seq` 负责已序列化事件的单调排序，`since` 用于补回没有序列号的旧事件。

响应 `data` 为 `{events, session_status, has_more, last_seq, supervisor_snapshot}`。客户端应把本地游标推进到 `last_seq`、按事件身份幂等合并，并在 `has_more=true` 时继续拉取。该接口只做时点回放，不是实时订阅：拿到完整快照或增量循环到 `has_more=false` 后，如果会话仍在运行，应通过 `POST /sessions/{session_id}/chat` 的 `event_id` 字段重新连接聊天 SSE。`GET /sessions/{session_id}` 是完整快照路径。

### WebSocket 接口

| 路径 | 认证 | 说明 |
|------|------|------|
| `/sessions/{session_id}/takeover/shell/ws?takeover_id=...&token=...` | Query token | 接管态终端双向交互 |
| `/sessions/{session_id}/vnc?token=...` | Query token | noVNC WebSocket 代理 |

补充说明：

- `takeover/shell/ws` 会发送 JSON 状态消息和终端字节流
- `/vnc` 会把浏览器的 WebSocket 数据转发到沙箱 VNC 服务
- `SANDBOX_PROVISION_MODE=off` 时，两个 WebSocket 接口都会发送 `SANDBOX_DISABLED` 状态并以 `4409` 关闭

### 研究子智能体与后代会话

`POST /sessions/{parent_session_id}/subagents/research` 请求体：

```json
{
  "prompts": ["研究问题"],
  "max_children": 3
}
```

`prompts` 和 `max_children` 上限均为 3。接口在开始流式响应前验证父会话归属，创建只读子会话，然后依次发出 `child_started`、`child_done`、`joined_summary`。子任务结果为 `completed | failed | timed_out | waiting | cancelled`；检测到 `waiting` 时上游按失败处理。

- `child_started`：`probe_run_id`、`child_session_id`、`prompt`
- `child_done`：`probe_run_id`、`child_session_id`、`outcome`、`final_answer?`、`transcript_tokens`、`error_summary?`
- `joined_summary`：`probe_run_id`、`summary`、`summary_tokens`、`completed_children`、`dropped_children`、`metrics`、`validation_warnings`

该 POST 不具备幂等语义，也没有重连游标或 join 状态查询接口。研究 SSE 断开会取消尚未完成的 child；重复 POST 会启动新的 probe run，并可能创建新的 child。客户端不得自动重试，应使用已收到的 child id 和后代接口检查已经创建的会话。

`GET /sessions/{session_id}/children` 返回 `data={parent_session_id, descendants, truncated, depth_applied}`。每个扁平后代项为 `{id, parent_session_id, worker_type, tool_filter_preset, status, title, created_at, updated_at}`，不内嵌事件；客户端按 `id` / `parent_session_id` 还原树。请求的 `depth` 会被部署级 `max_subagent_depth` 截断；命中深度或后代数量上限时 `truncated=true`。

### 生命周期事件投影

当 `lifecycle_runtime.lifecycle_events_enabled=true` 时，现有会话流和事件恢复接口会额外提供 `task`、`plan`、`step`、`tool` 的归一化 lifecycle 事件。再同时开启 `lifecycle_subagent_events_enabled=true`，才会增加 coordinator 和研究子任务的投影。

lifecycle payload 是加性投影，客户端仍应兼容原始事件。因为 lifecycle 事件也会写入会话历史，只能在全部 API pod 已升级到兼容版本后开启。

合法事件与派生状态为：

| 类型 | 事件及事件后状态 |
|------|------------------|
| `task` | `started→running`、`progress→running`、`completed→completed`、`failed→failed`、`cancelled→cancelled`、`retried→running` |
| `plan` | `started→pending`、`progress→running`、`completed→completed` |
| `step` | `started→running`、`completed→completed`、`failed→failed` |
| `tool` | `started→pending`、`progress→running`、`completed→completed`、`failed→failed`、`cancelled→cancelled` |
| `subagent` | `started→running`、`completed→completed`、`failed→failed`、`cancelled→cancelled` |

data 字段为 `type`、`lifecycle_type`、`event`、`state`、`unit_id`、`epoch`、`source_event_type`、`source_event_id`、`source_seq`、`reason`、`detail`、`parent_unit_id`、`correlation`。`detail` 只允许 `trigger`、`previous_state`、`retry_budget_remaining`、`original_outcome`、`note`；`correlation` 只允许 `work_unit_id`、`coordinator_run_id`、`coordinator_attempt_ix`。

当前没有 lifecycle 专用配置接口。需要修改文件型运行时配置并重建/重启 API，使启动快照重新加载：

```yaml
lifecycle_runtime:
  lifecycle_events_enabled: false
  lifecycle_subagent_events_enabled: false
```

### 成本 `/sessions/{session_id}/cost*`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/sessions/{session_id}/cost` | 是 | 按 node、model、provider 返回当前会话的 LLM 成本聚合 |
| `GET` | `/sessions/{session_id}/cost/tree?depth=N` | 是 | 聚合自身和后代成本，并返回深度/截断信息与归因来源 |

成本小数序列化为字符串。单个聚合包含 `total_usd`、`record_count`、`by_node`、`by_model`、`by_provider`、`pricing_version`、`cost_status`、`first_record_at`、`last_record_at`、`has_partial_records`；三个 breakdown 都是“字符串 key → 十进制字符串”的对象，`cost_status` 为 `actual | estimated | partial | unknown`。树级结果包含 `session_id`、`self_cost`、`descendants_cost`、`total_cost`、`descendant_ids`、`depth_reached`、`max_depth_applied`、`truncated`、`cost_source`；归因来源为 `none | direct | coordinator_subagent | research_subagent | mixed`。`truncated=true` 时返回的是受上限约束的聚合，不能当作完整后代树总成本。

### 对话压缩 `/sessions/{session_id}/compactions`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/sessions/{session_id}/compactions` | 是 | 获取持久化的压缩摘要和 token/message 变化 |
| `POST` | `/sessions/{session_id}/compactions` | 是 | 为符合条件的 completed/timed_out 会话排队执行手工压缩 |
| `GET` | `/sessions/{session_id}/compactions/{compaction_id}` | 是 | 获取压缩记录及操作详情 |
| `GET` | `/sessions/{session_id}/compactions/{compaction_id}/original-content` | 是 | 恢复脱敏后的压缩前消息；checkpoint 过期后返回 `410` |

列表通过 `items` 返回压缩 id、操作类型、摘要预览、token/message 变化、可见事件边界、原文可恢复性和创建时间。详情额外返回完整摘要、操作指标、父级/checkpoint id 与总计。手工 `POST` 没有请求体，成功时返回 `{"request_status":"queued"}`，但不会分配或返回 `compaction_id`；应观察后续 `compaction` SSE 事件或刷新列表。需要把请求与结果一一关联时，不要并发提交多个手工请求。当手工压缩或 overflow guard 未启用、会话仍在运行/等待/接管，或状态不符合条件时，可能以 `409` 返回机器可读的 `detail`。这些压缩接口直接返回上述响应体，不使用通用 JSON 包裹。

## 常用数据模型

### `LLMConfig`

包含：

- `base_url`
- `api_key`（读取接口不返回）
- `model_name`
- `temperature`
- `max_tokens`
- `api_type` — `chat_completions` / `responses` / `auto`（auto 走 fallback 适配器；默认 `chat_completions`）
- `timeout_seconds` — LLM 单次调用硬超时（默认 120s；范围 0-3600；0 = 关闭外层 `asyncio.wait_for`）
- `connect_timeout_seconds` — httpx 连接阶段独立超时（默认 60s，1.0 ≤ x ≤ 300.0），即使 `timeout_seconds=0` 也仍生效
- `context_window`
- `supports_response_format`
- `supports_vision`
- `supports_pdf_input`
- `context_overflow_guard_enabled`
- `overflow_retry_cap`
- `soft_trigger_ratio`
- `hard_trigger_ratio`
- `reserved_output_tokens`
- `reserved_output_tokens_cap_ratio`
- `token_estimator`
- `token_safety_factor`
- `unknown_model_context_window`
- `tool_result_max_chars`
- `tool_compress_trigger_ratio`
- `system_prompt_max_tokens`

### `FileInfo`

```json
{
  "id": "uuid",
  "filename": "string",
  "filepath": "string",
  "key": "string",
  "extension": "string",
  "mime_type": "string",
  "size": 0
}
```

### `ToolWithPreference`

```json
{
  "tool_id": "string",
  "tool_name": "string",
  "description": "string | null",
  "enabled_global": true,
  "enabled_user": true
}
```

### `SupervisorSnapshot`

后台会话会通过会话列表/详情/恢复接口返回该结构，关键字段：

- `execution_mode`: `foreground | background`
- `execution_phase`: `running | recovering | idle | suspended | terminating | terminated`
- `background_reason`: `explicit | auto_degrade | null`
- `retry_budget_remaining`
- `suspended_reason` / `terminal_reason`
- `last_progress_at` / `expires_at`
- `is_alive`
- `cancellation_state`: `none | cancelling | cancelled`
- `execution_revision`：单调递增的执行状态 revision；未显式提供值的旧/当前快照默认 `0`

### `StartTakeoverRequest`

```json
{
  "scope": "shell"
}
```

### `RenewTakeoverRequest`

```json
{
  "takeover_id": "string"
}
```

### `EndTakeoverRequest`

```json
{
  "handoff_mode": "continue"
}
```

## OpenAPI

- Swagger UI：`/docs`
- ReDoc：`/redoc`
- OpenAPI JSON：`/openapi.json`
