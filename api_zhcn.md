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
  - `POST /v2/skills/create`
- HTTP 增量恢复：
  - `GET /sessions/{session_id}/events?since=...`（断线重连后获取增量事件 + 当前状态）
- WebSocket：
  - `/sessions/{session_id}/takeover/shell/ws?takeover_id=...&token=...`
  - `/sessions/{session_id}/vnc?token=...`

### 会话状态

`pending | running | takeover_pending | takeover | waiting | finishing | completed | timed_out`

- `finishing` 是 `running → completed` 之间的中间态，承载最终摘要、附件归档、未读计数刷新等异步收尾
- `timed_out` 是 watchdog 总超时或恢复失败时写入的终态

### 对话 SSE 事件类型

`message | title | step | plan | tool | tool_confirmation | wait | control | context_status | compaction | finishing | health | done | error`

补充：
- `tool_confirmation` 用于工具审批与确认系统，前端展示工具确认卡片并把决策回传到 `/chat` 接口的 `tool_confirmation` 字段
- `finishing` 在会话进入 `finishing` 状态前发出，前端应解锁输入框并展示"收尾中"指示
- `health` 事件用于展示后端探活与降级状态
- `context_status` / `compaction` 上报上下文窗口压力和渐进压缩动作

## 运行时配置说明

- Docker Compose 模式下，业务配置文件位于 `/app/data/config.yaml`
- 本地后端开发默认使用 `api/config.yaml`
- Skill v2 默认存储目录为 `/app/data/skills`

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

## 设置模块 `/app-config`

### LLM 与 Agent

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/app-config/llm` | 是 | 获取 LLM 配置（不回传 `api_key`） |
| `POST` | `/app-config/llm` | 管理员 | 更新 LLM 配置 |
| `GET` | `/app-config/agent` | 是 | 获取 Agent 配置 |
| `POST` | `/app-config/agent` | 管理员 | 更新 Agent 配置 |

`AgentConfig` 除 `max_iterations`、`max_retries`、`max_search_results` 外，还包含 `skill_selection` 子配置。

### MCP

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/app-config/mcp-servers` | 是 | 获取 MCP 服务列表与探测到的工具名 |
| `POST` | `/app-config/mcp-servers` | 管理员 | 新增或更新 MCP 服务配置 |
| `POST` | `/app-config/mcp-servers/{server_name}/delete` | 管理员 | 删除 MCP 服务 |
| `POST` | `/app-config/mcp-servers/{server_name}/enabled` | 管理员 | 更新 MCP 服务全局启用状态 |

### A2A

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/app-config/a2a-servers` | 是 | 获取 A2A 服务列表 |
| `POST` | `/app-config/a2a-servers` | 管理员 | 新增 A2A 服务（传 `base_url`） |
| `POST` | `/app-config/a2a-servers/{a2a_id}/delete` | 管理员 | 删除 A2A 服务 |
| `POST` | `/app-config/a2a-servers/{a2a_id}/enabled` | 管理员 | 更新 A2A 服务全局启用状态 |

### 文件理解

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/app-config/file-understanding` | 是 | 获取文件理解配置（视觉降级、音频、视频�� |
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
| `POST` | `/v2/skills/install` | 管理员 | 从 GitHub 或本地目录安装 Skill |
| `POST` | `/v2/skills/create` | 管理员 | AI 创建 Skill，SSE 返回进度和结果 |
| `POST` | `/v2/skills/{skill_key}/enabled` | 管理员 | 更新 Skill 全局启用状态 |
| `DELETE` | `/v2/skills/{skill_key}` | 管理员 | 删除 Skill |
| `GET` | `/v2/skills/policy` | 是 | 获取 Skill 风险策略 |
| `POST` | `/v2/skills/policy` | 管理员 | 更新 Skill 风险策略 |
| `GET` | `/v2/skills/{skill_key}` | 管理员 | 获取 Skill 详情、工具定义、bundle 文件索引 |

关键字段：

- `source_type`: `local | github`
- `runtime_type`: `native | mcp | a2a`
- `mode`: `off | enforce_confirmation`

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

## 文件模块 `/files`

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `POST` | `/files` | 是 | 上传附件到对象存储并关联用户 |
| `GET` | `/files/{file_id}` | 是 | 获取文件元数据 |
| `GET` | `/files/{file_id}/download` | 是 | 下载文件内容 |
| `DELETE` | `/files/{file_id}` | 是 | 删除文件 |

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
| `POST` | `/sessions/{session_id}/clear-unread-message-count` | 是 | 清空未读消息数 |
| `POST` | `/sessions/{session_id}/delete` | 是 | 删除会话 |
| `POST` | `/sessions/{session_id}/chat` | 是，SSE | 向会话发送消息并流式接收事件；body 可携带 `tool_confirmation` 提交工具确认决策 |
| `GET` | `/sessions/{session_id}` | 是 | 获取会话详情和历史事件 |
| `GET` | `/sessions/{session_id}/events?since={event_id}` | 是 | 获取指定 event_id 之后的增量事件 + 当前会话状态，用于断线重连后的状态恢复 |
| `GET` | `/sessions/{session_id}/takeover` | 是 | 获取当前接管状态 |
| `POST` | `/sessions/{session_id}/takeover/start` | 是 | 发起接管；`request_status=starting` 时 HTTP 为 `202` |
| `POST` | `/sessions/{session_id}/takeover/renew` | 是 | 续期接管租约 |
| `POST` | `/sessions/{session_id}/takeover/reject` | 是 | 处理 AI 发起的接管请求 |
| `POST` | `/sessions/{session_id}/takeover/end` | 是 | 结束接管并选择继续或完成 |
| `POST` | `/sessions/{session_id}/takeover/reopen` | 是 | 在窗口期内补救重开接管 |
| `POST` | `/sessions/{session_id}/stop` | 是 | 停止当前任务 |
| `GET` | `/sessions/{session_id}/files` | 是 | 获取会话文件列表 |
| `POST` | `/sessions/{session_id}/file` | 是 | 读取沙箱中文件内容 |
| `POST` | `/sessions/{session_id}/shell` | 是 | 读取指定 shell 会话输出 |

### WebSocket 接口

| 路径 | 认证 | 说明 |
|------|------|------|
| `/sessions/{session_id}/takeover/shell/ws?takeover_id=...&token=...` | Query token | 接管态终端双向交互 |
| `/sessions/{session_id}/vnc?token=...` | Query token | noVNC WebSocket 代理 |

补充说明：

- `takeover/shell/ws` 会发送 JSON 状态消息和终端字节流
- `/vnc` 会把浏览器的 WebSocket 数据转发到沙箱 VNC 服务

## 常用数据模型

### `LLMConfig`

包含：

- `base_url`
- `api_key`（读取接口不返回）
- `model_name`
- `temperature`
- `max_tokens`
- `api_type` — `chat_completions` / `responses` / `auto`（auto 走 fallback 适配器；默认 `chat_completions`）
- `timeout_seconds` — LLM 单次调用硬超时（默认 120s，0 = 关闭外层 `asyncio.wait_for`）。建议复杂场景用 300s
- `connect_timeout_seconds` — httpx 连接阶段独立超时（默认 60s，1.0 ≤ x ≤ 300.0），即使 `timeout_seconds=0` 也仍生效
- `context_window`
- `context_overflow_guard_enabled`
- `overflow_retry_cap`
- `soft_trigger_ratio`
- `hard_trigger_ratio`
- `reserved_output_tokens`
- `reserved_output_tokens_cap_ratio`
- `token_estimator`
- `token_safety_factor`
- `unknown_model_context_window`

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
