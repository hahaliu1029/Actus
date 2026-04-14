# API Reference

This document reflects the current routes implemented in the codebase. The base path for business endpoints is `/api`.

## Conventions

### Response envelope

Most JSON endpoints return:

```json
{
  "code": 200,
  "msg": "success",
  "data": {}
}
```

Notes:

- Some business errors still return HTTP `200` with a non-200 `code`
- Explicit migration endpoints return HTTP `410`
- File downloads, WeChat callback redirects, SSE, and WebSocket endpoints do not use the JSON envelope

### Authentication

- Authenticated endpoints use `Authorization: Bearer <access_token>`
- Admin endpoints require `role=super_admin`
- WebSocket auth uses a `token` query parameter

### Rate limiting

- Exceeded limits return HTTP `429`
- Example body: `{"code":429,"msg":"请求过多，请稍后重试","data":{"retry_after":N}}`
- Redis is required for rate limiting; related endpoints return `503` if Redis is unavailable
- Auth endpoints (`/auth/register`, `/auth/login`, `/auth/refresh`, `/auth/wechat/*`) carry an independent `rate_limit_auth` policy

### Streaming transports

- SSE:
  - `POST /sessions/stream`
  - `POST /sessions/{session_id}/chat`
  - `POST /v2/skills/create`
- HTTP incremental recovery:
  - `GET /sessions/{session_id}/events?since=...` — fetch events after a known event id plus the current session status, used to resume after disconnect
- WebSocket:
  - `/sessions/{session_id}/takeover/shell/ws?takeover_id=...&token=...`
  - `/sessions/{session_id}/vnc?token=...`

### Session status values

`pending | running | takeover_pending | takeover | waiting | finishing | completed | timed_out`

- `finishing` sits between `running` and `completed` and carries asynchronous wrap-up (final summary, attachment archival, unread-count refresh, lifespan cleanup)
- `timed_out` is a terminal state written by the execution watchdog when total timeout or recovery failure forces termination

### Chat SSE event types

`message | title | step | plan | tool | tool_confirmation | wait | control | context_status | compaction | finishing | health | done | error`

Notes:
- `tool_confirmation` drives the tool approval & confirmation system; the frontend renders a confirmation card and returns the user decision via the `tool_confirmation` field on the next `/chat` request
- `finishing` is emitted right before the session enters the `finishing` state — clients should unlock input / show a "wrapping up" indicator
- `health` carries backend liveness / degradation state for UI display
- `context_status` and `compaction` report context window pressure and gradual-compaction events

## Runtime configuration

- In Docker Compose mode, the runtime config file is `/app/data/config.yaml`
- In local backend development, the default file is `api/config.yaml`
- Skill v2 data is stored under `/app/data/skills`

## Auth `/auth`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/auth/register` | No | Register a user and return profile + tokens |
| `POST` | `/auth/login` | No | Login with username or email |
| `POST` | `/auth/refresh` | No | Refresh access token |
| `GET` | `/auth/me` | Yes | Get current user |
| `PUT` | `/auth/me` | Yes | Update nickname / avatar |
| `GET` | `/auth/wechat/authorize` | No | Generate WeChat OAuth URL |
| `GET` | `/auth/wechat/callback` | No | WeChat callback, then redirect to frontend |

## Status `/status`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/status/` | Yes | Check FastAPI, PostgreSQL, Redis, and MinIO health |
| `GET` | `/status/minio` | Yes | Check MinIO connectivity; `smoke=true` runs put/get/remove |
| `POST` | `/status/minio/upload` | Admin | Upload a test file to MinIO with multipart/form-data |

## App config `/app-config`

### LLM and agent

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/app-config/llm` | Yes | Get LLM config (`api_key` is excluded) |
| `POST` | `/app-config/llm` | Admin | Update LLM config |
| `GET` | `/app-config/agent` | Yes | Get agent config |
| `POST` | `/app-config/agent` | Admin | Update agent config |

`AgentConfig` includes `max_iterations`, `max_retries`, `max_search_results`, and a nested `skill_selection` policy object.

### MCP

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/app-config/mcp-servers` | Yes | List MCP servers and discovered tool names |
| `POST` | `/app-config/mcp-servers` | Admin | Create or update MCP server config |
| `POST` | `/app-config/mcp-servers/{server_name}/delete` | Admin | Delete an MCP server |
| `POST` | `/app-config/mcp-servers/{server_name}/enabled` | Admin | Toggle global enable state |

### A2A

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/app-config/a2a-servers` | Yes | List configured A2A servers |
| `POST` | `/app-config/a2a-servers` | Admin | Create an A2A server from `base_url` |
| `POST` | `/app-config/a2a-servers/{a2a_id}/delete` | Admin | Delete an A2A server |
| `POST` | `/app-config/a2a-servers/{a2a_id}/enabled` | Admin | Toggle global enable state |

### File understanding

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/app-config/file-understanding` | Yes | Get file understanding config (vision fallback, audio, video) |
| `POST` | `/app-config/file-understanding` | Admin | Update file understanding config |

## Skills

### Legacy skill routes

These legacy endpoints are preserved only to return `410 SKILL_API_MOVED`:

- `GET /app-config/skills`
- `POST /app-config/skills/install`
- `POST /app-config/skills/{skill_id}/enabled`
- `POST /app-config/skills/{skill_id}/delete`

### Skill v2 `/v2/skills`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v2/skills` | Admin | List installed skills |
| `POST` | `/v2/skills/install` | Admin | Install a skill from GitHub or a local directory |
| `POST` | `/v2/skills/create` | Admin | Create a skill with AI; progress is streamed over SSE |
| `POST` | `/v2/skills/{skill_key}/enabled` | Admin | Toggle global enable state |
| `DELETE` | `/v2/skills/{skill_key}` | Admin | Delete a skill |
| `GET` | `/v2/skills/policy` | Yes | Get the skill risk policy |
| `POST` | `/v2/skills/policy` | Admin | Update the skill risk policy |
| `GET` | `/v2/skills/{skill_key}` | Admin | Get skill metadata, tools, bundle file index, and raw `SKILL.md` |

Important fields:

- `source_type`: `local | github`
- `runtime_type`: `native | mcp | a2a`
- risk policy `mode`: `off | enforce_confirmation`

## User tool preferences

### Legacy `/user/tools`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/user/tools/mcp` | Yes | List MCP tools with user preference overrides |
| `POST` | `/user/tools/mcp/{server_name}/enabled` | Yes | Toggle MCP preference for the current user |
| `GET` | `/user/tools/a2a` | Yes | List A2A tools with user preference overrides |
| `POST` | `/user/tools/a2a/{a2a_id}/enabled` | Yes | Toggle A2A preference for the current user |
| `GET` | `/user/tools/skills` | Yes | Moved, returns `410` |
| `POST` | `/user/tools/skills/{skill_id}/enabled` | Yes | Moved, returns `410` |

### Skill preferences v2 `/v2/user/tools`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v2/user/tools/skills` | Yes | List skills with global and per-user enable states |
| `POST` | `/v2/user/tools/skills/{skill_key}/enabled` | Yes | Toggle a skill for the current user |

## Files `/files`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/files` | Yes | Upload an attachment and persist metadata |
| `GET` | `/files/{file_id}` | Yes | Get file metadata |
| `GET` | `/files/{file_id}/download` | Yes | Download file content |
| `DELETE` | `/files/{file_id}` | Yes | Delete a file |

## Admin `/admin`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/admin/users` | Admin | List users |
| `GET` | `/admin/users/{user_id}` | Admin | Get user details |
| `PUT` | `/admin/users/{user_id}/status` | Admin | Update user status |
| `DELETE` | `/admin/users/{user_id}` | Admin | Delete a user |

## Sessions `/sessions`

### HTTP and SSE

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/sessions` | Yes | Create a new session |
| `POST` | `/sessions/stream` | Yes, SSE | Stream the session list |
| `GET` | `/sessions` | Yes | List sessions |
| `POST` | `/sessions/{session_id}/clear-unread-message-count` | Yes | Clear unread count |
| `POST` | `/sessions/{session_id}/delete` | Yes | Delete a session |
| `POST` | `/sessions/{session_id}/chat` | Yes, SSE | Send a message and receive streamed events; body may carry `tool_confirmation` to submit an approval decision for a paused tool call |
| `GET` | `/sessions/{session_id}` | Yes | Get session details and event history |
| `GET` | `/sessions/{session_id}/events?since={event_id}` | Yes | Fetch incremental events after the given `event_id` plus the current session status — used by clients to resume after disconnect |
| `GET` | `/sessions/{session_id}/takeover` | Yes | Get takeover state |
| `POST` | `/sessions/{session_id}/takeover/start` | Yes | Start a takeover; returns HTTP `202` when `request_status=starting` |
| `POST` | `/sessions/{session_id}/takeover/renew` | Yes | Renew the takeover lease |
| `POST` | `/sessions/{session_id}/takeover/reject` | Yes | Respond to an AI-initiated takeover request |
| `POST` | `/sessions/{session_id}/takeover/end` | Yes | End takeover and either continue or complete |
| `POST` | `/sessions/{session_id}/takeover/reopen` | Yes | Reopen takeover during the recovery window |
| `POST` | `/sessions/{session_id}/stop` | Yes | Stop the current task |
| `GET` | `/sessions/{session_id}/files` | Yes | List session files |
| `POST` | `/sessions/{session_id}/file` | Yes | Read a file from the sandbox |
| `POST` | `/sessions/{session_id}/shell` | Yes | Read output from a shell session |

### WebSocket

| Path | Auth | Description |
|------|------|-------------|
| `/sessions/{session_id}/takeover/shell/ws?takeover_id=...&token=...` | Query token | Bidirectional shell takeover |
| `/sessions/{session_id}/vnc?token=...` | Query token | noVNC WebSocket proxy |

Additional notes:

- `takeover/shell/ws` sends JSON status frames and terminal bytes
- `/vnc` forwards browser WebSocket traffic to the sandbox VNC service

## Common models

### `LLMConfig`

Includes:

- `base_url`
- `api_key` (excluded from read responses)
- `model_name`
- `temperature`
- `max_tokens`
- `api_type` — `chat_completions` / `responses` / `auto` (auto uses the fallback adapter; default is `chat_completions`)
- `timeout_seconds` — Hard per-call timeout (default 120s, 0 disables the outer `asyncio.wait_for`). 300s recommended for complex flows
- `connect_timeout_seconds` — Independent httpx connect-phase timeout (default 60s, 1.0 ≤ x ≤ 300.0); still enforced even when `timeout_seconds=0`
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

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
