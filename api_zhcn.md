# API 文档（简体中文）

基础路径：`/api`

所有 JSON 响应都使用统一包裹格式：

```json
{
  "code": 200,
  "msg": "success",
  "data": {}
}
```

认证说明：
- 需要登录的接口使用 `Authorization: Bearer <access_token>`。
- 管理员接口要求 `role=super_admin`。
- 未绑定认证依赖的接口可匿名访问。
- `sessions` / `files` 模块全部要求 Bearer token，普通用户仅可访问自己的资源。
- 管理员允许跨用户访问 `sessions` / `files` 资源。
- VNC WebSocket 需通过 query 参数传递 `token`：`/api/sessions/{session_id}/vnc?token=<access_token>`。

限流说明：
- 请求超限返回 HTTP `429`，并带 `Retry-After` 响应头。
- 统一响应体示例：`{"code":429,"msg":"请求过多，请稍后重试","data":{"retry_after":N}}`。
- 限流依赖 Redis，Redis 不可用时返回 HTTP `503`。

说明：大部分错误通过响应体 `code`/`msg` 返回，HTTP 状态码可能仍为 200。

## Skill 运行时同步（目录型 Skill）

- 该能力不变更 HTTP API，仅调整服务端运行时内部链路。
- 会话启动时先同步初始选中 Skill 子集，再后台补齐其余已启用 Skill。
- 单会话内采用静态版本：不做中途刷新，Skill 更新在下一会话生效。
- native Skill 在缺少 `entry.exec_dir` 时，默认执行目录为 `/home/ubuntu/workspace/.skills/<skill_id>`。

## OpenAPI

- Swagger UI：`/docs`
- ReDoc：`/redoc`
- 规范文件：`/openapi.json`
- 导出：

```bash
curl -o openapi.json http://localhost:8000/openapi.json
```

## 数据模型

### HealthStatus

```json
{
  "service": "string",
  "status": "ok | error",
  "details": "string"
}
```

### FileInfo

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

### LLMConfig

```json
{
  "base_url": "https://api.deepseek.com",
  "api_key": "string",
  "model_name": "deepseek-reasoner",
  "temperature": 0.7,
  "max_tokens": 8192,
  "context_window": 32768,
  "context_overflow_guard_enabled": false,
  "overflow_retry_cap": 2,
  "soft_trigger_ratio": 0.85,
  "hard_trigger_ratio": 0.95,
  "reserved_output_tokens": 4096,
  "reserved_output_tokens_cap_ratio": 0.25,
  "token_estimator": "hybrid",
  "token_safety_factor": 1.15,
  "unknown_model_context_window": 32768
}
```

说明：LLMConfig 响应不包含 `api_key`。
补充说明：
- `context_window` 为空时按模型映射自动推断。
- `context_overflow_guard_enabled` 用于开启上下文超限治理。
- `hard_trigger_ratio` 必须大于 `soft_trigger_ratio`。

### AgentConfig

```json
{
  "max_iterations": 100,
  "max_retries": 3,
  "max_search_results": 10
}
```

### SessionStatus

`pending | running | waiting | completed`

### CreateSessionResponse

```json
{
  "session_id": "string"
}
```

### ListSessionItem

```json
{
  "session_id": "string",
  "title": "string",
  "latest_message": "string",
  "latest_message_at": "string | null",
  "status": "pending | running | waiting | completed",
  "unread_message_count": 0
}
```

### ListSessionResponse

```json
{
  "sessions": [
    {
      "session_id": "string",
      "title": "string",
      "latest_message": "string",
      "latest_message_at": "string | null",
      "status": "pending | running | waiting | completed",
      "unread_message_count": 0
    }
  ]
}
```

### UserRole

`super_admin | user`

### UserStatus

`active | inactive | banned`

### RegisterRequest

```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "nickname": "string"
}
```

`username` 或 `email` 至少提供一个。

### LoginRequest

```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

`username` 或 `email` 至少提供一个。

### RefreshTokenRequest

```json
{
  "refresh_token": "string"
}
```

### UpdateUserRequest

```json
{
  "nickname": "string",
  "avatar": "string"
}
```

### TokenResponse

```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer"
}
```

### UserResponse

```json
{
  "id": "string",
  "username": "string",
  "email": "string",
  "nickname": "string",
  "avatar": "string",
  "role": "super_admin | user",
  "status": "active | inactive | banned",
  "created_at": "string"
}
```

### LoginResponse

```json
{
  "user": {
    "id": "string",
    "username": "string",
    "email": "string",
    "nickname": "string",
    "avatar": "string",
    "role": "super_admin | user",
    "status": "active | inactive | banned",
    "created_at": "string"
  },
  "tokens": {
    "access_token": "string",
    "refresh_token": "string",
    "token_type": "bearer"
  }
}
```

### UserStatusUpdateRequest

```json
{
  "status": "active | inactive | banned"
}
```

### UserListResponse

```json
{
  "users": [
    {
      "id": "string",
      "username": "string",
      "email": "string",
      "nickname": "string",
      "avatar": "string",
      "role": "super_admin | user",
      "status": "active | inactive | banned",
      "created_at": "string"
    }
  ],
  "total": 0
}
```

### ToolPreferenceRequest

```json
{
  "enabled": true
}
```

### ToolWithPreference

```json
{
  "tool_id": "string",
  "tool_name": "string",
  "description": "string",
  "enabled_global": true,
  "enabled_user": true
}
```

### ToolListResponse

```json
{
  "tools": [
    {
      "tool_id": "string",
      "tool_name": "string",
      "description": "string",
      "enabled_global": true,
      "enabled_user": true
    }
  ]
}
```

### MCPConfig

```json
{
  "mcpServers": {
    "server_name": {
      "transport": "stdio | sse | streamable_http",
      "enabled": true,
      "description": "string",
      "env": {
        "ANY": "value"
      },
      "command": "string",
      "args": [
        "string"
      ],
      "url": "string",
      "headers": {
        "ANY": "value"
      }
    }
  }
}
```

校验规则：
- `transport` 为 `sse` 或 `streamable_http` 时必须提供 `url`。
- `transport` 为 `stdio` 时必须提供 `command`。

### ListMCPServerResponse

```json
{
  "mcp_servers": [
    {
      "server_name": "string",
      "enabled": true,
      "transport": "stdio | sse | streamable_http",
      "tools": [
        "string"
      ]
    }
  ]
}
```

### ListA2AServerResponse

```json
{
  "a2a_servers": [
    {
      "id": "string",
      "name": "string",
      "description": "string",
      "input_modes": [
        "string"
      ],
      "output_modes": [
        "string"
      ],
      "streaming": false,
      "push_notifications": false,
      "enabled": true
    }
  ]
}
```

### MinIO Ping Response Data

```json
{
  "ok": true,
  "reachable": true,
  "endpoint": "string",
  "secure": true,
  "bucket": "string",
  "bucket_exists": true
}
```

### MinIO Smoke Test Response Data

```json
{
  "ok": true,
  "endpoint": "string",
  "secure": true,
  "bucket": "string",
  "bucket_exists": true,
  "object": "string",
  "uploaded_bytes": 0,
  "downloaded_bytes": 0,
  "match": true
}
```

当 bucket 不存在时，`bucket_exists` 为 false，且 `error` 为 `bucket_not_exists`。

## 认证模块

### POST `/api/auth/register`

用户注册。

请求体：`RegisterRequest`

响应：`Response[LoginResponse]`

### POST `/api/auth/login`

使用用户名或邮箱登录。

请求体：`LoginRequest`

响应：`Response[LoginResponse]`

### POST `/api/auth/refresh`

刷新访问令牌。

请求体：`RefreshTokenRequest`

响应：`Response[TokenResponse]`

### GET `/api/auth/me`

获取当前用户信息。

认证：需要 Bearer token。

响应：`Response[UserResponse]`

### PUT `/api/auth/me`

更新当前用户信息。

认证：需要 Bearer token。

请求体：`UpdateUserRequest`

响应：`Response[UserResponse]`

### GET `/api/auth/wechat/authorize`

获取微信 OAuth 授权 URL。

查询参数：
- `state` (string, 可选)
- `scope` (string, 默认 `snsapi_userinfo`)

响应数据：

```json
{
  "authorize_url": "string"
}
```

### GET `/api/auth/wechat/callback`

微信 OAuth 回调。

查询参数：
- `code` (string, 必填)
- `state` (string, 可选)

响应：HTTP 重定向到前端页面，URL 参数包含 token 或 error。

## 状态模块

### GET `/api/status`

系统健康检查。

响应：`Response[List[HealthStatus]]`
- 任一服务为 `error` 时返回 `code=503`。

### GET `/api/status/minio`

MinIO 健康检查。

查询参数：
- `smoke` (bool, 默认 false)：执行 put/get/remove 自检。
- `bucket` (string, 可选)：指定 bucket。

响应：`Response`，数据为 MinIO ping 或 smoke-test 信息。

### POST `/api/status/minio/upload`

MinIO 上传文件测试（multipart/form-data）。

表单：
- `file` (file, 必填)

查询参数：
- `bucket` (string, 可选)
- `object_name` (string, 可选)
- `prefix` (string, 默认 `uploads`)
- `presign` (bool, 默认 true)
- `expiry_seconds` (int, 默认 3600)

响应数据：

```json
{
  "bucket": "string",
  "object": "string",
  "etag": "string",
  "version_id": "string",
  "filename": "string",
  "content_type": "string",
  "size": 0,
  "presigned_get_url": "string"
}
```

说明：
- bucket 不存在时返回 `code=400` 且 `msg="bucket_not_exists"`。
- `presign=true` 时才包含 `presigned_get_url`。

## 应用配置模块

### GET `/api/app-config/llm`

获取 LLM 配置。

响应：`Response[LLMConfig]`（不包含 api_key）

### POST `/api/app-config/llm`

更新 LLM 配置。

请求体：`LLMConfig`
- `api_key` 为空时表示不更新该字段。

响应：`Response[LLMConfig]`（不包含 api_key）

### GET `/api/app-config/agent`

获取 Agent 配置。

响应：`Response[AgentConfig]`

### POST `/api/app-config/agent`

更新 Agent 配置。

请求体：`AgentConfig`

响应：`Response[AgentConfig]`

### GET `/api/app-config/mcp-servers`

获取 MCP 服务器列表。

响应：`Response[ListMCPServerResponse]`

### POST `/api/app-config/mcp-servers`

新增 MCP 服务器配置（支持一次传多个）。

请求体：`MCPConfig`

响应：`Response`（无 data）

### POST `/api/app-config/mcp-servers/{server_name}/delete`

删除 MCP 服务器配置。

响应：`Response`（无 data）

### POST `/api/app-config/mcp-servers/{server_name}/enabled`

设置 MCP 服务器启用状态。

请求体：

```json
{
  "enabled": true
}
```

响应：`Response`（无 data）

### GET `/api/app-config/a2a-servers`

获取 A2A 服务器列表。

响应：`Response[ListA2AServerResponse]`

### POST `/api/app-config/a2a-servers`

新增 A2A 服务器。

请求体：

```json
{
  "base_url": "string"
}
```

响应：`Response`（无 data）

### POST `/api/app-config/a2a-servers/{a2a_id}/delete`

删除 A2A 服务器。

响应：`Response`（无 data）

### POST `/api/app-config/a2a-servers/{a2a_id}/enabled`

设置 A2A 服务器启用状态。

请求体：

```json
{
  "enabled": true
}
```

响应：`Response`（无 data）

## 用户工具偏好模块

### GET `/api/user/tools/mcp`

获取 MCP 工具列表（含用户偏好）。

认证：需要 Bearer token。

响应：`Response[ToolListResponse]`

### POST `/api/user/tools/mcp/{server_name}/enabled`

设置 MCP 工具个人启用状态。

认证：需要 Bearer token。

请求体：`ToolPreferenceRequest`

响应：`Response`（无 data）

### GET `/api/user/tools/a2a`

获取 A2A 工具列表（含用户偏好）。

认证：需要 Bearer token。

响应：`Response[ToolListResponse]`

### POST `/api/user/tools/a2a/{a2a_id}/enabled`

设置 A2A 工具个人启用状态。

认证：需要 Bearer token。

请求体：`ToolPreferenceRequest`

响应：`Response`（无 data）

## 会话模块

### POST `/api/sessions`

为当前用户创建新会话。

认证：需要 Bearer token。

响应：`Response[CreateSessionResponse]`

### GET `/api/sessions`

获取当前用户会话列表。

认证：需要 Bearer token。

响应：`Response[ListSessionResponse]`

### POST `/api/sessions/stream`

流式获取会话列表（SSE）。

认证：需要 Bearer token。

说明：
- 受请求频率限制与 SSE 并发连接限制（按用户）。

### GET `/api/sessions/{session_id}`

获取指定会话详情。

认证：需要 Bearer token。

说明：
- 普通用户仅可访问自己的会话。
- 管理员可跨用户访问。

### POST `/api/sessions/{session_id}/chat`

向指定会话发起聊天（SSE）。

认证：需要 Bearer token。

说明：
- 普通用户仅可访问自己的会话。
- 普通用户仅可引用自己上传的附件文件。
- 管理员可跨用户访问会话并引用任意附件。
- 受 chat 请求频率限制与 SSE 并发连接限制（按用户）。

### POST `/api/sessions/{session_id}/stop`

停止指定会话。

认证：需要 Bearer token。

说明：
- 普通用户仅可停止自己的会话。
- 管理员可跨用户停止会话。

### POST `/api/sessions/{session_id}/clear-unread-message-count`

清除当前用户指定会话的未读消息数。

认证：需要 Bearer token。

响应：`Response`（无 data）

错误：
- 403：会话不属于当前用户。
- 404：会话不存在。

### POST `/api/sessions/{session_id}/delete`

删除当前用户指定会话。

认证：需要 Bearer token。

响应：`Response`（无 data）

错误：
- 403：会话不属于当前用户。
- 404：会话不存在。

### GET `/api/sessions/{session_id}/files`

获取指定会话文件列表。

认证：需要 Bearer token。

说明：
- 普通用户仅可访问自己的会话。
- 管理员可跨用户访问。

### POST `/api/sessions/{session_id}/file`

读取指定会话沙箱文件内容。

认证：需要 Bearer token。

说明：
- 普通用户仅可访问自己的会话。
- 管理员可跨用户访问。

### POST `/api/sessions/{session_id}/shell`

读取指定会话的 Shell 输出。

认证：需要 Bearer token。

说明：
- 普通用户仅可访问自己的会话。
- 管理员可跨用户访问。
- 长时间命令（如 `npm start` / `pip install`）通常会在执行端先返回 `running`，随后通过本接口持续读取输出。
- 安装类命令会自动补齐常见非交互参数以降低卡在确认提示符的概率。

### WS `/api/sessions/{session_id}/vnc?token=<access_token>`

建立会话 VNC WebSocket 连接。

认证：query token 必填。

说明：
- 普通用户仅可访问自己的会话。
- 管理员可跨用户访问。
- 受请求频率限制与 WS 并发连接限制（按用户）。

## 管理员模块

### GET `/api/admin/users`

获取用户列表。

认证：需要 Bearer token（仅 super_admin）。

查询参数：
- `skip` (int, 默认 0)
- `limit` (int, 默认 100)

响应：`Response[UserListResponse]`

### GET `/api/admin/users/{user_id}`

获取用户详情。

认证：需要 Bearer token（仅 super_admin）。

响应：`Response[UserResponse]`

### PUT `/api/admin/users/{user_id}/status`

更新用户状态。

认证：需要 Bearer token（仅 super_admin）。

请求体：`UserStatusUpdateRequest`

响应：`Response`（无 data）

说明：
- 不能修改自己的状态。
- 不能修改其他管理员的状态。

### DELETE `/api/admin/users/{user_id}`

删除用户。

认证：需要 Bearer token（仅 super_admin）。

响应：`Response`（无 data）

说明：
- 不能删除自己。
- 不能删除其他管理员。

## 文件模块

### POST `/api/files`

上传文件（multipart/form-data）。

表单：
- `file` (file, 必填)

认证：需要 Bearer token。

响应：`Response[FileInfo]`

### GET `/api/files/{file_id}`

获取文件信息。

认证：需要 Bearer token。

说明：
- 普通用户仅可访问自己的文件。
- 管理员可跨用户访问。

响应：`Response[FileInfo]`

### GET `/api/files/{file_id}/download`

下载文件内容。

认证：需要 Bearer token。

说明：
- 普通用户仅可访问自己的文件。
- 管理员可跨用户访问。

响应：文件流，响应头包含：
- `Content-Disposition: attachment; filename*=utf-8''<encoded>`
- `Content-Length: <bytes>`

### DELETE `/api/files/{file_id}`

删除文件。

认证：需要 Bearer token。

说明：
- 普通用户仅可删除自己的文件。
- 管理员可跨用户删除。

响应：`Response`（无 data）

## Skill 生态模块（v2）

### 数据模型

#### SkillInstallRequest

```json
{
  "source_type": "local | github",
  "source_ref": "string",
  "manifest": {},
  "skill_md": ""
}
```

说明：
- `source_type`：来源类型，`local` 为本地绝对路径，`github` 为 GitHub tree URL。
- `source_ref`：来源引用（本地路径如 `/path/to/skill`，GitHub URL 如 `https://github.com/owner/repo/tree/main/skills/my-skill`）。
- `manifest`：可选兼容字段，默认由 SKILL.md frontmatter 自动生成。
- `skill_md`：可选 SKILL.md 内容覆盖，优先级高于 bundle 中的 SKILL.md。

#### SkillItem

```json
{
  "id": "string",
  "slug": "string",
  "name": "string",
  "description": "",
  "version": "0.1.0",
  "source_type": "local | github",
  "source_ref": "string",
  "runtime_type": "native | mcp | a2a",
  "enabled": true,
  "installed_by": "string | null",
  "created_at": "datetime",
  "updated_at": "datetime",
  "bundle_file_count": 0,
  "context_ref_count": 0,
  "last_sync_at": "string | null"
}
```

#### SkillRiskPolicyItem

```json
{
  "mode": "off | enforce_confirmation"
}
```

### GET `/api/v2/skills`

获取 Skill 列表。

认证：需要 Bearer token（管理员）。

响应：`Response[SkillListResponse]`

### POST `/api/v2/skills/install`

安装 Skill（支持 GitHub / 本地目录来源）。

认证：需要 Bearer token（管理员）。

请求体：`SkillInstallRequest`

响应：`Response[SkillItem]`

说明：
- 安装流程：加载 bundle → 解析 SKILL.md frontmatter → 校验安全策略 → 构建 context_blob → 写入文件系统。
- 同 slug 的 Skill 会被更新（保留 enabled 状态和 ID）。
- Skill 存储路径：`SKILLS_ROOT_DIR`（默认 `/app/data/skills`）。

### POST `/api/v2/skills/{skill_key}/enabled`

更新 Skill 启用状态。

认证：需要 Bearer token（管理员）。

请求体：

```json
{
  "enabled": true
}
```

响应：`Response`（无 data）

### DELETE `/api/v2/skills/{skill_key}`

删除 Skill（同时清理用户工具偏好）。

认证：需要 Bearer token（管理员）。

响应：`Response`（无 data）

### GET `/api/v2/skills/policy`

获取 Skill 风险策略配置。

认证：需要 Bearer token（管理员）。

响应：`Response[SkillRiskPolicyItem]`

### POST `/api/v2/skills/policy`

更新 Skill 风险策略配置。

认证：需要 Bearer token（管理员）。

请求体：`SkillRiskPolicyItem`

响应：`Response`（无 data）

> **注意**：旧版 v1 Skill API（`/api/app-config/skills/*`）已返回 HTTP 410，请使用 v2 API。

## cURL 示例

基础地址：`http://localhost:8000`

### 注册

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"password123","nickname":"Demo"}'
```

### 登录

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"password123"}'
```

### 获取当前用户

```bash
curl -X GET http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer <access_token>"
```

### 获取用户列表（管理员）

```bash
curl -X GET "http://localhost:8000/api/admin/users?skip=0&limit=20" \
  -H "Authorization: Bearer <access_token>"
```

### 创建会话

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Authorization: Bearer <access_token>"
```

### 获取会话列表

```bash
curl -X GET http://localhost:8000/api/sessions \
  -H "Authorization: Bearer <access_token>"
```

### 清除会话未读消息数

```bash
curl -X POST http://localhost:8000/api/sessions/<session_id>/clear-unread-message-count \
  -H "Authorization: Bearer <access_token>"
```

### 删除会话

```bash
curl -X POST http://localhost:8000/api/sessions/<session_id>/delete \
  -H "Authorization: Bearer <access_token>"
```

### 获取 Skill 列表

```bash
curl -X GET http://localhost:8000/api/v2/skills \
  -H "Authorization: Bearer <access_token>"
```

### 安装 Skill（本地目录）

```bash
curl -X POST http://localhost:8000/api/v2/skills/install \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "source_ref": "/path/to/skill"}'
```

### 安装 Skill（GitHub）

```bash
curl -X POST http://localhost:8000/api/v2/skills/install \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "github", "source_ref": "https://github.com/owner/repo/tree/main/skills/my-skill"}'
```

### 启用/禁用 Skill

```bash
curl -X POST http://localhost:8000/api/v2/skills/<skill_key>/enabled \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

### 删除 Skill

```bash
curl -X DELETE http://localhost:8000/api/v2/skills/<skill_key> \
  -H "Authorization: Bearer <access_token>"
```

### 获取 Skill 风险策略

```bash
curl -X GET http://localhost:8000/api/v2/skills/policy \
  -H "Authorization: Bearer <access_token>"
```
