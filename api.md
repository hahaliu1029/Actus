# API Documentation

Base path: `/api`

All JSON responses use a common envelope:

```json
{
  "code": 200,
  "msg": "success",
  "data": {}
}
```

Authentication:
- Use `Authorization: Bearer <access_token>` for endpoints that require a logged-in user.
- Admin-only endpoints require `role=super_admin`.
- Endpoints without auth dependencies are publicly accessible.
- `sessions` and `files` modules require Bearer token for all endpoints.
- Normal users can only access their own `sessions` / `files`; admins can access across users.
- VNC WebSocket requires query token: `/api/sessions/{session_id}/vnc?token=<access_token>`.

Rate limit:
- Exceeded limits return HTTP `429` with `Retry-After` header.
- Unified body example: `{\"code\":429,\"msg\":\"请求过多，请稍后重试\",\"data\":{\"retry_after\":N}}`.
- Redis is required for rate limiting. If unavailable, endpoints return HTTP `503`.

Note: Most errors are returned via the response envelope (`code`/`msg`) while HTTP status may remain 200.

## Skill Runtime Sync (Directory Skills)

- HTTP APIs are unchanged for this behavior; sync is an internal runtime mechanism.
- At session startup, the runner syncs a selected initial skill subset to sandbox first, then backfills remaining enabled skills in background.
- During one session, skill bundle version is static (no mid-session refresh).
- For native skills, when `entry.exec_dir` is missing, runtime defaults to `/home/ubuntu/workspace/.skills/<skill_id>`.

## OpenAPI

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- Spec: `/openapi.json`
- Export:

```bash
curl -o openapi.json http://localhost:8000/openapi.json
```

## Data Models

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
  "max_tokens": 8192
}
```

Note: Responses for LLMConfig exclude `api_key`.

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

At least one of `username` or `email` is required.

### LoginRequest

```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

At least one of `username` or `email` is required.

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

Validation rules:
- `transport` in `sse` or `streamable_http` requires `url`.
- `transport` in `stdio` requires `command`.

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

When the bucket does not exist, `bucket_exists` is false and `error` is `bucket_not_exists`.

## Auth Module

### POST `/api/auth/register`

Register a new user.

Body: `RegisterRequest`

Response: `Response[LoginResponse]`

### POST `/api/auth/login`

Login by username or email.

Body: `LoginRequest`

Response: `Response[LoginResponse]`

### POST `/api/auth/refresh`

Refresh access token.

Body: `RefreshTokenRequest`

Response: `Response[TokenResponse]`

### GET `/api/auth/me`

Get current user profile.

Auth: Bearer token required.

Response: `Response[UserResponse]`

### PUT `/api/auth/me`

Update current user profile.

Auth: Bearer token required.

Body: `UpdateUserRequest`

Response: `Response[UserResponse]`

### GET `/api/auth/wechat/authorize`

Get WeChat OAuth authorization URL.

Query:
- `state` (string, optional)
- `scope` (string, default `snsapi_userinfo`)

Response data:

```json
{
  "authorize_url": "string"
}
```

### GET `/api/auth/wechat/callback`

WeChat OAuth callback.

Query:
- `code` (string, required)
- `state` (string, optional)

Response: HTTP redirect to frontend URL with token or error in query parameters.

## Status Module

### GET `/api/status`

System health check.

Response: `Response[List[HealthStatus]]`
- Returns `code=503` if any service is in `error` state.

### GET `/api/status/minio`

MinIO health check.

Query:
- `smoke` (bool, default false): run put/get/remove self-test.
- `bucket` (string, optional): override default bucket.

Response: `Response` with MinIO ping or smoke-test data.

### POST `/api/status/minio/upload`

Upload a file to MinIO for testing (multipart/form-data).

Form:
- `file` (file, required)

Query:
- `bucket` (string, optional)
- `object_name` (string, optional)
- `prefix` (string, default `uploads`)
- `presign` (bool, default true)
- `expiry_seconds` (int, default 3600)

Response data:

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

Notes:
- If the bucket does not exist, returns `code=400` with `msg="bucket_not_exists"`.
- `presigned_get_url` is included only when `presign=true`.

## App Config Module

### GET `/api/app-config/llm`

Get LLM configuration.

Response: `Response[LLMConfig]` (api_key excluded)

### POST `/api/app-config/llm`

Update LLM configuration.

Body: `LLMConfig`
- If `api_key` is empty, it is not updated.

Response: `Response[LLMConfig]` (api_key excluded)

### GET `/api/app-config/agent`

Get agent configuration.

Response: `Response[AgentConfig]`

### POST `/api/app-config/agent`

Update agent configuration.

Body: `AgentConfig`

Response: `Response[AgentConfig]`

### GET `/api/app-config/mcp-servers`

List MCP servers.

Response: `Response[ListMCPServerResponse]`

### POST `/api/app-config/mcp-servers`

Create MCP server configuration(s).

Body: `MCPConfig`

Response: `Response` (no data payload)

### POST `/api/app-config/mcp-servers/{server_name}/delete`

Delete an MCP server configuration by name.

Response: `Response` (no data payload)

### POST `/api/app-config/mcp-servers/{server_name}/enabled`

Set MCP server enabled state.

Body:

```json
{
  "enabled": true
}
```

Response: `Response` (no data payload)

### GET `/api/app-config/a2a-servers`

List A2A servers.

Response: `Response[ListA2AServerResponse]`

### POST `/api/app-config/a2a-servers`

Create an A2A server.

Body:

```json
{
  "base_url": "string"
}
```

Response: `Response` (no data payload)

### POST `/api/app-config/a2a-servers/{a2a_id}/delete`

Delete an A2A server by id.

Response: `Response` (no data payload)

### POST `/api/app-config/a2a-servers/{a2a_id}/enabled`

Set A2A server enabled state.

Body:

```json
{
  "enabled": true
}
```

Response: `Response` (no data payload)

## User Tools Module

### GET `/api/user/tools/mcp`

Get MCP tools with user preferences.

Auth: Bearer token required.

Response: `Response[ToolListResponse]`

### POST `/api/user/tools/mcp/{server_name}/enabled`

Set MCP tool enabled status for current user.

Auth: Bearer token required.

Body: `ToolPreferenceRequest`

Response: `Response` (no data payload)

### GET `/api/user/tools/a2a`

Get A2A tools with user preferences.

Auth: Bearer token required.

Response: `Response[ToolListResponse]`

### POST `/api/user/tools/a2a/{a2a_id}/enabled`

Set A2A tool enabled status for current user.

Auth: Bearer token required.

Body: `ToolPreferenceRequest`

Response: `Response` (no data payload)

## Session Module

### POST `/api/sessions`

Create a new session for the current user.

Auth: Bearer token required.

Response: `Response[CreateSessionResponse]`

### GET `/api/sessions`

List sessions for the current user.

Auth: Bearer token required.

Response: `Response[ListSessionResponse]`

### POST `/api/sessions/stream`

Stream sessions (SSE).

Auth: Bearer token required.

Notes:
- Protected by request-rate limit and SSE concurrent-connection limit per user.

### GET `/api/sessions/{session_id}`

Get session detail by id.

Auth: Bearer token required.

Notes:
- Normal users can only access their own sessions.
- Admin users can access across users.

### POST `/api/sessions/{session_id}/chat`

Start chat on a session (SSE).

Auth: Bearer token required.

Notes:
- Normal users can only access their own sessions.
- Normal users can only attach their own uploaded files.
- Admin users can access across users and attach any file.
- Protected by chat-rate limit and SSE concurrent-connection limit per user.

### POST `/api/sessions/{session_id}/stop`

Stop a session.

Auth: Bearer token required.

Notes:
- Normal users can only stop their own sessions.
- Admin users can stop sessions across users.

### POST `/api/sessions/{session_id}/clear-unread-message-count`

Clear unread count for a session (current user only).

Auth: Bearer token required.

Response: `Response` (no data payload)

Errors:
- 403 if the session does not belong to the current user.
- 404 if the session does not exist.

### POST `/api/sessions/{session_id}/delete`

Delete a session (current user only).

Auth: Bearer token required.

Response: `Response` (no data payload)

Errors:
- 403 if the session does not belong to the current user.
- 404 if the session does not exist.

### GET `/api/sessions/{session_id}/files`

List files for a session.

Auth: Bearer token required.

Notes:
- Normal users can only access their own sessions.
- Admin users can access across users.

### POST `/api/sessions/{session_id}/file`

Read file content inside session sandbox.

Auth: Bearer token required.

Notes:
- Normal users can only access their own sessions.
- Admin users can access across users.

### POST `/api/sessions/{session_id}/shell`

Read shell output for a session.

Auth: Bearer token required.

Notes:
- Normal users can only access their own sessions.
- Admin users can access across users.

### WS `/api/sessions/{session_id}/vnc?token=<access_token>`

Open VNC WebSocket for a session.

Auth: query token required.

Notes:
- Normal users can only access their own sessions.
- Admin users can access across users.
- Protected by request-rate limit and WS concurrent-connection limit per user.

## Admin Module

### GET `/api/admin/users`

List users.

Auth: Bearer token required (super_admin only).

Query:
- `skip` (int, default 0)
- `limit` (int, default 100)

Response: `Response[UserListResponse]`

### GET `/api/admin/users/{user_id}`

Get user detail.

Auth: Bearer token required (super_admin only).

Response: `Response[UserResponse]`

### PUT `/api/admin/users/{user_id}/status`

Update user status.

Auth: Bearer token required (super_admin only).

Body: `UserStatusUpdateRequest`

Response: `Response` (no data payload)

Notes:
- Cannot update your own status.
- Cannot update another admin user.

### DELETE `/api/admin/users/{user_id}`

Delete a user.

Auth: Bearer token required (super_admin only).

Response: `Response` (no data payload)

Notes:
- Cannot delete yourself.
- Cannot delete another admin user.

## File Module

### POST `/api/files`

Upload a file (multipart/form-data).

Form:
- `file` (file, required)

Auth: Bearer token required.

Response: `Response[FileInfo]`

### GET `/api/files/{file_id}`

Get file metadata by id.

Auth: Bearer token required.

Notes:
- Normal users can only access their own files.
- Admin users can access across users.

Response: `Response[FileInfo]`

### GET `/api/files/{file_id}/download`

Download file content.

Auth: Bearer token required.

Notes:
- Normal users can only access their own files.
- Admin users can access across users.

Response: streaming file content with headers:
- `Content-Disposition: attachment; filename*=utf-8''<encoded>`
- `Content-Length: <bytes>`

### DELETE `/api/files/{file_id}`

Delete a file by id.

Auth: Bearer token required.

Notes:
- Normal users can only delete their own files.
- Admin users can delete across users.

Response: `Response` (no data payload)

## Skill Ecosystem Module (v2)

### Data Models

#### SkillInstallRequest

```json
{
  "source_type": "local | github",
  "source_ref": "string",
  "manifest": {},
  "skill_md": ""
}
```

Notes:
- `source_type`: Source type. `local` for absolute local path, `github` for GitHub tree URL.
- `source_ref`: Source reference (local path e.g. `/path/to/skill`, GitHub URL e.g. `https://github.com/owner/repo/tree/main/skills/my-skill`).
- `manifest`: Optional compatibility field. Auto-generated from SKILL.md frontmatter by default.
- `skill_md`: Optional SKILL.md content override, takes priority over bundled SKILL.md.

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

List all installed Skills.

Auth: Bearer token (admin required).

Response: `Response[SkillListResponse]`

### POST `/api/v2/skills/install`

Install a Skill from GitHub or local directory.

Auth: Bearer token (admin required).

Body: `SkillInstallRequest`

Response: `Response[SkillItem]`

Notes:
- Installation flow: load bundle → parse SKILL.md frontmatter → validate security policy → build context_blob → write to filesystem.
- Skills with the same slug are updated (preserving enabled state and ID).
- Storage path: `SKILLS_ROOT_DIR` (default `/app/data/skills`).

### POST `/api/v2/skills/{skill_key}/enabled`

Toggle Skill enabled state.

Auth: Bearer token (admin required).

Body:

```json
{
  "enabled": true
}
```

Response: `Response` (no data payload)

### DELETE `/api/v2/skills/{skill_key}`

Delete a Skill (also cleans up user tool preferences).

Auth: Bearer token (admin required).

Response: `Response` (no data payload)

### GET `/api/v2/skills/policy`

Get Skill risk policy configuration.

Auth: Bearer token (admin required).

Response: `Response[SkillRiskPolicyItem]`

### POST `/api/v2/skills/policy`

Update Skill risk policy configuration.

Auth: Bearer token (admin required).

Body: `SkillRiskPolicyItem`

Response: `Response` (no data payload)

> **Note**: Legacy v1 Skill API (`/api/app-config/skills/*`) returns HTTP 410. Use the v2 API above.

## cURL Examples

Base URL: `http://localhost:8000`

### Register

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"password123","nickname":"Demo"}'
```

### Login

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"password123"}'
```

### Get Current User

```bash
curl -X GET http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer <access_token>"
```

### List Users (Admin)

```bash
curl -X GET "http://localhost:8000/api/admin/users?skip=0&limit=20" \
  -H "Authorization: Bearer <access_token>"
```

### Create Session

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Authorization: Bearer <access_token>"
```

### List Sessions

```bash
curl -X GET http://localhost:8000/api/sessions \
  -H "Authorization: Bearer <access_token>"
```

### Clear Session Unread Count

```bash
curl -X POST http://localhost:8000/api/sessions/<session_id>/clear-unread-message-count \
  -H "Authorization: Bearer <access_token>"
```

### Delete Session

```bash
curl -X POST http://localhost:8000/api/sessions/<session_id>/delete \
  -H "Authorization: Bearer <access_token>"
```

### List Skills

```bash
curl -X GET http://localhost:8000/api/v2/skills \
  -H "Authorization: Bearer <access_token>"
```

### Install Skill (Local Directory)

```bash
curl -X POST http://localhost:8000/api/v2/skills/install \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "local", "source_ref": "/path/to/skill"}'
```

### Install Skill (GitHub)

```bash
curl -X POST http://localhost:8000/api/v2/skills/install \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "github", "source_ref": "https://github.com/owner/repo/tree/main/skills/my-skill"}'
```

### Enable/Disable Skill

```bash
curl -X POST http://localhost:8000/api/v2/skills/<skill_key>/enabled \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

### Delete Skill

```bash
curl -X DELETE http://localhost:8000/api/v2/skills/<skill_key> \
  -H "Authorization: Bearer <access_token>"
```

### Get Skill Risk Policy

```bash
curl -X GET http://localhost:8000/api/v2/skills/policy \
  -H "Authorization: Bearer <access_token>"
```
