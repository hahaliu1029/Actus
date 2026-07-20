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
  - `POST /sessions/{parent_session_id}/subagents/research`
  - `POST /v2/skills/create`
- HTTP incremental recovery:
  - `GET /sessions/{session_id}/events?since_seq=...&since=...` — fetch events after a known cursor plus the current session/supervisor state; `since_seq` is preferred and `since` remains the legacy-event fallback
- WebSocket:
  - `/sessions/{session_id}/takeover/shell/ws?takeover_id=...&token=...`
  - `/sessions/{session_id}/vnc?token=...`

### Session status values

`pending | running | takeover_pending | takeover | waiting | finishing | completed | timed_out`

- `finishing` sits between `running` and `completed` and carries asynchronous wrap-up (final summary, attachment archival, unread-count refresh, lifespan cleanup)
- `timed_out` is a terminal state written by the execution watchdog when total timeout or recovery failure forces termination

### Session SSE event types

The chat stream can emit:

`message | title | step | plan | tool | tool_confirmation | wait | control | context_status | compaction | finishing | health | sandbox_state_changed | session_mode_changed | coordinator_dispatch | coordinator_worker_spawned | coordinator_reduce | coordinator_apply | coordinator_sibling_cancel | done | error`

Notes:
- `tool_confirmation` drives the tool approval & confirmation system; the frontend renders a confirmation card and returns the user decision via the `tool_confirmation` field on the next `/chat` request
- `finishing` is emitted right before the session enters the `finishing` state — clients should unlock input / show a "wrapping up" indicator
- `health` carries backend liveness / degradation state for UI display
- `context_status` and `compaction` report context window pressure and gradual-compaction events
- When lifecycle projection is enabled, lifecycle events use a dotted SSE name: `lifecycle.{task|plan|step|tool|subagent}.{started|progress|completed|failed|cancelled|retried}`. The exact allowed event subset depends on `lifecycle_type`; the payload also carries `type=lifecycle`, `state`, `unit_id`, `epoch`, source-event references, and optional parent/correlation data
- The research-subagent stream emits `child_started`, `child_done`, and `joined_summary`

## Runtime configuration

- In Docker Compose mode, the runtime config file is `/app/data/config.yaml`
- In local backend development, the default file is `api/config.yaml`
- Skill v2 data is stored under `/app/data/skills`
- `SANDBOX_PROVISION_MODE` is environment-only: `always` provisions the parent sandbox eagerly, `on_demand` waits for the first sandbox operation, and `off` removes the sandbox capability surface. `GET /sessions/{session_id}` exposes the effective deployment value as `data.sandbox_mode`
- `EXTENSION_GOVERNANCE_MODE` is environment-only: `off | shadow | enforce`
- Lifecycle projection lives in `config.yaml` under `lifecycle_runtime`. `lifecycle_events_enabled` is the master switch; `lifecycle_subagent_events_enabled` only takes effect together with the master switch. Both default to `false`
- MinIO uses separate internal and public settings. `MINIO_ENDPOINT` is used for API-side object I/O; `MINIO_PUBLIC_ENDPOINT` and `MINIO_PUBLIC_SECURE` are used to generate browser/remote-consumer URLs. If the public endpoint is set, `MINIO_REGION` is required

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

## Internal metrics `/v1/metrics`

`GET /v1/metrics` is an internal Prometheus scrape endpoint and is intentionally hidden from OpenAPI. It is disabled and returns `404` when `ACTUS_METRICS_ENDPOINT_TOKEN`/`METRICS_ENDPOINT_TOKEN` is empty; otherwise it requires `Authorization: Bearer <token>` and returns Prometheus exposition text.

## App config `/app-config`

### LLM and agent

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/app-config/llm` | Yes | Get LLM config (`api_key` is excluded) |
| `POST` | `/app-config/llm` | Admin | Update LLM config |
| `GET` | `/app-config/agent` | Yes | Get agent config |
| `POST` | `/app-config/agent` | Admin | Update agent config |

`AgentConfig` includes `max_iterations`, `max_retries`, `max_search_results`, plus nested `skill_selection`, `skill_embedding`, `memory`, `tool_confirmation`, `execution`, and `slash_commands` objects.

### MCP

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/app-config/mcp-servers` | Yes | List MCP servers and discovered tool names |
| `POST` | `/app-config/mcp-servers` | Admin | Create or update MCP server config; governance mode supports `dry_run`, `acknowledge`, and `force` query parameters |
| `POST` | `/app-config/mcp-servers/{server_name}/delete` | Admin | Delete an MCP server |
| `POST` | `/app-config/mcp-servers/{server_name}/enabled` | Admin | Toggle global enable state |

### A2A

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/app-config/a2a-servers` | Yes | List configured A2A servers |
| `POST` | `/app-config/a2a-servers` | Admin | Create an A2A server from `base_url`; governance mode supports `dry_run`, `acknowledge`, and `force` query parameters |
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
| `POST` | `/v2/skills/install?force={bool}` | Admin | Install a skill from GitHub or a local directory; `force=true` admits a dangerous scan result |
| `POST` | `/v2/skills/create` | Admin | Create a skill with AI; progress is streamed over SSE; returns `409 SANDBOX_DISABLED` in sandbox `off` mode |
| `POST` | `/v2/skills/{skill_key}/enabled` | Admin | Toggle global enable state |
| `DELETE` | `/v2/skills/{skill_key}` | Admin | Delete a skill |
| `GET` | `/v2/skills/policy` | Yes | Get the skill risk policy |
| `POST` | `/v2/skills/policy` | Admin | Update the skill risk policy |
| `GET` | `/v2/skills/{skill_key}/export?format={format}` | Admin | Export a Skill as ZIP; `format=agent-skills|actus` |
| `GET` | `/v2/skills/{skill_key}` | Admin | Get skill metadata, tools, bundle file index, and raw `SKILL.md` |

Important fields:

- `source_type`: `local | github`
- `runtime_type`: `native | mcp | a2a`
- risk policy `mode`: `off | enforce_confirmation`

## Runtime extensions `/v1/runtime`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v1/runtime/extensions` | Yes | Return the aggregated MCP, A2A, Skill, and Plugin runtime snapshot; non-admin callers receive a reduced projection |
| `GET` | `/v1/runtime/extensions/catalog` | Yes | Return the static recommended MCP catalog |
| `POST` | `/v1/runtime/extensions/{kind}/{ext_id}/enabled` | Admin | Toggle an `mcp`, `a2a`, or `skill` extension through the unified façade |
| `POST` | `/v1/runtime/extensions/{kind}/{ext_id}/probe` | Admin | Probe an enabled `mcp`, `a2a`, or `skill`; repeated probes for the same item have a five-second cooldown |

The aggregate item includes `config`, `health`, `liveness`, `stats`, kind-specific `details`, and, for admin callers while governance is active, a `governance` block. `probe_enabled` and `stats_enabled` describe effective runtime capability, not only raw configuration. A quarantined item cannot be re-enabled through the runtime façade; use the governance reapproval endpoint.

## Extension governance `/v2/extensions`

All routes in this section require an admin. `kind` is `mcp | a2a | skill | plugin`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v2/extensions/governance` | Get governance counts; when governance mode is `off`, returns a literal zero summary |
| `POST` | `/v2/extensions/refresh-observations` | Refresh all or a supplied list of observations |
| `POST` | `/v2/extensions/approve-pins` | Approve all or a supplied list of pins |
| `GET` | `/v2/extensions/audit` | Page audit records; supports `kind`, `ext_id`, `event`, `cursor`, and `limit` filters |
| `POST` | `/v2/extensions/{kind}/{ext_id}/refresh-observation` | Refresh one extension observation |
| `POST` | `/v2/extensions/{kind}/{ext_id}/quarantine` | Quarantine one extension |
| `POST` | `/v2/extensions/{kind}/{ext_id}/reapprove` | Remove quarantine and pin again |
| `POST` | `/v2/extensions/{kind}/{ext_id}/governance-disable` | Disable through the governance state machine |
| `POST` | `/v2/extensions/{kind}/{ext_id}/governance-enable` | Enable through the governance state machine |

Mutating administrative actions use optimistic concurrency: request bodies carry `expected_row_revision`; quarantine additionally accepts `note`. Successful state changes return the new `row_revision`. Except for the zero-summary GET, governance routes return `409 governance_disabled` when `EXTENSION_GOVERNANCE_MODE=off`.

Admission reasons have two different enforcement boundaries:

- Detection reasons (`unknown`, `unpinned`, `pin_stale`, `config_drift`, `pin_mismatch`, `registry_unavailable`) are observed but admitted in `shadow`; `enforce` blocks them.
- Administrative or structural reasons (`quarantined`, `disabled`, `deleted`, `parent_blocked`) block admission in both `shadow` and `enforce`. Therefore, `shadow` is fail-open only for detection findings, not for an explicit administrative disable/quarantine or a blocked Plugin parent.
- `off` does not construct the registry/admission services, so neither class of governance decision is evaluated.

## Plugins `/v2/plugins`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v2/plugins` | Admin | List installed Plugins, their latest operation, and member extensions |
| `POST` | `/v2/plugins/install` | Admin | Install or preview a Plugin bundle from `source_type` + `source_ref`; supports `dry_run`, `force`, and `acknowledge` in the JSON body |
| `POST` | `/v2/plugins/{plugin_ext_id}/enabled` | Admin | Toggle a Plugin using `enabled` + `expected_row_revision` |
| `DELETE` | `/v2/plugins/{plugin_ext_id}` | Admin | Uninstall a Plugin using a required `expected_row_revision` body |

`POST /v2/plugins/install` accepts this body (unknown fields are rejected):

```json
{
  "source_type": "local",
  "source_ref": "/absolute/path/to/plugin",
  "dry_run": true,
  "acknowledge": false,
  "force": false
}
```

Operational `source_type` values are `local` and `github`; the legacy enum value `mcp_registry` is rejected by the source loader with `422`. Local sources must be absolute directories, GitHub sources use a supported repository/directory URL, and compressed `.zip`/`.tar` inputs are rejected.

The install flags have deliberately separate meanings:

| Mode/request | Result |
|--------------|--------|
| `dry_run=true` | Returns a zero-write preview with `plugin_id`, `name`, `version`, `aggregate_verdict`, `install_policy_decision`, `members`, and `warnings`. It does not perform collision reads or the final install gates, so a real install must still be treated as authoritative. |
| `shadow`, real install | `safe` is allowed; `caution` and `dangerous` are allowed with warnings. Detection-policy findings do not require `acknowledge`/`force`. |
| `enforce`, `caution` | Requires `acknowledge=true`; otherwise returns `409 acknowledge_required`. |
| `enforce`, `dangerous` | Requires `force=true`; otherwise returns `422 force_required`. |
| MCP/A2A member probe failure | Requires `force=true` in either active governance mode. `acknowledge` does not bypass it. |
| Declared-hash mismatch or Plugin/member identity collision | Always rejected on a real install; neither flag bypasses these gates. |

`acknowledge=true` and `force=true` may be supplied together for a mixed bundle; each is consumed only by the matching warning/block tier. A completed install returns only `plugin_ext_id`, `operation_id`, and `status=completed`. It does **not** return a revision: call `GET /v2/plugins`, then use that item's `row_revision` for the enabled or uninstall request. The list response also exposes `status`, `artifact_hash`, `last_operation`, and member status/scan fields. A compensated install returns `422 plugin_install_failed_compensated` with `operation_id`, `error`, and `collided_targets`; compensation failure returns `500 plugin_install_failed_requires_admin` with `operation_id`.

`POST /v2/plugins/{id}/enabled` and `POST /v2/extensions/plugin/{id}/governance-enable|governance-disable` are two API shapes over the same Plugin-parent governance row and the same CAS state transition; they are not independent switches. Disabling the parent changes `active -> disabled` and makes its members `parent_blocked` without rewriting each member's MCP/A2A/Skill configuration. Re-enabling changes `disabled -> active`, but does not override a member's own disabled/quarantined state or runtime configuration. The runtime façade only toggles child kinds (`mcp`, `a2a`, `skill`), not `plugin`.

Plugin routes are available only when extension governance is not `off`; otherwise they return `409 governance_disabled`. Switching to `off` is not an uninstall: persisted Plugin registry rows, bundles, and materialized members are not deleted. The Plugin list/management surface and Plugin parent projection disappear, while materialized MCP/A2A/Skill members continue through their normal config/Skill stores without admission or `parent_blocked` enforcement. Startup Plugin-saga closure and governance reconcile are also skipped until governance is enabled again.

For rollout, use `off -> shadow -> enforce`: first recreate the API in `shadow`, inspect `/v2/extensions/governance` and `/v2/extensions/audit`, refresh observations and approve the intended pins, then recreate in `enforce` once detection findings are understood. Administrative/structural blockers already apply in `shadow` and should be resolved or retained intentionally before the final step. The mode is read into process-wide settings and lifespan services at startup; editing `.env` alone is insufficient. For Compose, apply each transition with:

```bash
docker compose --env-file .env up -d --force-recreate api
```

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

### Tool approval policies `/v2/user/tool-policies`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v2/user/tool-policies` | Yes | List explicit per-tool approval policies for the current user |
| `GET` | `/v2/user/tool-policies/{tool_name}` | Yes | Get one explicit policy; missing rows return `404` |
| `PUT` | `/v2/user/tool-policies/{tool_name}` | Yes | Set `policy=auto|ask|deny` |
| `DELETE` | `/v2/user/tool-policies/{tool_name}` | Yes | Clear an explicit policy; idempotent |

## Files `/files`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/files` | Yes | Upload an attachment and persist metadata |
| `GET` | `/files/{file_id}` | Yes | Get file metadata |
| `GET` | `/files/{file_id}/download` | Yes | Download file content |
| `DELETE` | `/files/{file_id}` | Yes | Delete a file |

## Memories `/v2/memories`

All memory routes are scoped to the current user.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v2/memories` | Page, search, and filter memories by source/category/time; supports `auto_promoted_after` |
| `POST` | `/v2/memories` | Create a manual memory (`201`) |
| `GET` | `/v2/memories/cleanup-config` | Get the deployment's legacy-cleanup rollout boundary |
| `DELETE` | `/v2/memories/legacy` | Delete legacy unclassified `session_flush` memories within the configured boundary |
| `GET` | `/v2/memories/{chunk_id}` | Get one memory |
| `PATCH` | `/v2/memories/{chunk_id}` | Update exactly one of `content` or `pinned` |
| `DELETE` | `/v2/memories/{chunk_id}` | Delete one memory |
| `POST` | `/v2/memories/{chunk_id}/reindex` | Rebuild the database/search index from the memory file on disk |
| `POST` | `/v2/memories/bulk-delete` | Delete a supplied list of ids |
| `POST` | `/v2/memories/delete-all` | Delete all memories for the current user |

Memory categories are `user | rule | fact`. `auto_promoted_after` must be an ISO 8601 timezone-aware timestamp.

## Notifications `/v2/notifications`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/v2/notifications/unread?limit=N` | Yes | List up to 50 non-expired unread notifications and return the full unread count |
| `POST` | `/v2/notifications/{notification_id}/mark-read` | Yes | Mark one notification read; repeat and cross-owner requests return `marked_read=false` |

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
| `GET` | `/sessions/background-quota` | Yes | Get system/user background execution quota usage |
| `GET` | `/sessions/{session_id}/children?depth=N` | Yes | Return a flat, depth-capped descendant list with truncation metadata |
| `POST` | `/sessions/{session_id}/clear-unread-message-count` | Yes | Clear unread count |
| `POST` | `/sessions/{session_id}/delete` | Yes | Delete a session |
| `POST` | `/sessions/{session_id}/chat` | Yes, SSE | Send a message and receive streamed events; body may carry `tool_confirmation` to submit an approval decision for a paused tool call |
| `POST` | `/sessions/{session_id}/cancel` | Yes | Request cancellation and return `cancel_requested` state |
| `GET` | `/sessions/{session_id}` | Yes | Get session details, event history, supervisor snapshot, and deployment `sandbox_mode` |
| `GET` | `/sessions/{session_id}/events?since={event_id}&since_seq={seq}` | Yes | Fetch incremental events and the current session/supervisor state; `since_seq` is the preferred monotonic cursor |
| `GET` | `/sessions/{session_id}/takeover` | Yes | Get takeover state |
| `POST` | `/sessions/{session_id}/takeover/start` | Yes | Start a takeover; returns HTTP `202` when `request_status=starting` |
| `POST` | `/sessions/{session_id}/takeover/renew` | Yes | Renew the takeover lease |
| `POST` | `/sessions/{session_id}/takeover/reject` | Yes | Respond to an AI-initiated takeover request |
| `POST` | `/sessions/{session_id}/takeover/end` | Yes | End takeover and either continue or complete |
| `POST` | `/sessions/{session_id}/takeover/reopen` | Yes | Reopen takeover during the recovery window |
| `POST` | `/sessions/{session_id}/retry-from-suspend` | Yes | Resume a recoverable suspended background run |
| `POST` | `/sessions/{session_id}/stop` | Yes | Stop the current task |
| `GET` | `/sessions/{session_id}/files` | Yes | List session files |
| `POST` | `/sessions/{session_id}/file` | Yes | Read a file from the sandbox |
| `GET` | `/sessions/{session_id}/file/download?filepath=...` | Yes | Download a binary or text file from the sandbox |
| `POST` | `/sessions/{session_id}/shell` | Yes | Read output from a shell session |
| `POST` | `/sessions/{parent_session_id}/subagents/research` | Yes, SSE | Fan out one to three read-only research children and stream deterministic join progress |

### Recovery cursor contract

Every persisted SSE payload carries `data.event_id` and, for newly sequenced events, `data.seq`; the SSE `id` line equals `data.event_id`. For reconnects, persist the highest positive `seq` and the latest event id, then call the events endpoint with both cursors. When both are present, `since_seq` orders sequenced events and `since` recovers legacy events without a sequence.

The response `data` is `{events, session_status, has_more, last_seq, supervisor_snapshot}`. Advance the local cursor to `last_seq`, merge events idempotently by event identity, and fetch again while `has_more=true`. This endpoint is a point-in-time replay, not a live subscription: after the page has a full snapshot or the incremental loop reaches `has_more=false`, reconnect an active chat SSE with `POST /sessions/{session_id}/chat` and the latest `event_id` in the body. `GET /sessions/{session_id}` is the full-snapshot path.

### WebSocket

| Path | Auth | Description |
|------|------|-------------|
| `/sessions/{session_id}/takeover/shell/ws?takeover_id=...&token=...` | Query token | Bidirectional shell takeover |
| `/sessions/{session_id}/vnc?token=...` | Query token | noVNC WebSocket proxy |

Additional notes:

- `takeover/shell/ws` sends JSON status frames and terminal bytes
- `/vnc` forwards browser WebSocket traffic to the sandbox VNC service
- In `SANDBOX_PROVISION_MODE=off`, both WebSocket endpoints send a `SANDBOX_DISABLED` status and close with code `4409`

### Research subagents and descendants

`POST /sessions/{parent_session_id}/subagents/research` accepts:

```json
{
  "prompts": ["research question"],
  "max_children": 3
}
```

`prompts` and `max_children` are capped at three. The route verifies parent ownership before streaming, creates read-only child sessions, and emits `child_started`, `child_done`, then `joined_summary`. Child outcomes are `completed | failed | timed_out | waiting | cancelled`; a detected `waiting` result is treated as failure upstream.

- `child_started`: `probe_run_id`, `child_session_id`, `prompt`
- `child_done`: `probe_run_id`, `child_session_id`, `outcome`, `final_answer?`, `transcript_tokens`, `error_summary?`
- `joined_summary`: `probe_run_id`, `summary`, `summary_tokens`, `completed_children`, `dropped_children`, `metrics`, `validation_warnings`

This POST is not idempotent and has no reconnect cursor or join-status endpoint. Disconnecting the research SSE cancels pending children; repeating the POST starts a new probe run and can create new children. Clients must not retry it automatically. Use the emitted child ids and the descendants endpoint to inspect sessions that were already created.

`GET /sessions/{session_id}/children` returns `data={parent_session_id, descendants, truncated, depth_applied}`. Each flat descendant is `{id, parent_session_id, worker_type, tool_filter_preset, status, title, created_at, updated_at}` and does not embed events. Rebuild a tree by joining `id` to `parent_session_id`. Requested `depth` is clamped to the deployment's `max_subagent_depth`, and `truncated` is true when either the depth or descendant cap is hit.

### Lifecycle event projection

When `lifecycle_runtime.lifecycle_events_enabled=true`, the existing session streams and event-recovery endpoints additionally expose normalized lifecycle events for `task`, `plan`, `step`, and `tool`. Setting `lifecycle_subagent_events_enabled=true` together with the master switch adds coordinator and research-child projection.

The lifecycle payload is additive; clients should continue to handle the source events. Enable it only after all API pods run a compatible version because lifecycle events are also persisted in session history.

Supported event/state pairs are:

| Type | Events and resulting states |
|------|-----------------------------|
| `task` | `started→running`, `progress→running`, `completed→completed`, `failed→failed`, `cancelled→cancelled`, `retried→running` |
| `plan` | `started→pending`, `progress→running`, `completed→completed` |
| `step` | `started→running`, `completed→completed`, `failed→failed` |
| `tool` | `started→pending`, `progress→running`, `completed→completed`, `failed→failed`, `cancelled→cancelled` |
| `subagent` | `started→running`, `completed→completed`, `failed→failed`, `cancelled→cancelled` |

The data fields are `type`, `lifecycle_type`, `event`, `state`, `unit_id`, `epoch`, `source_event_type`, `source_event_id`, `source_seq`, `reason`, `detail`, `parent_unit_id`, and `correlation`. `detail` is limited to `trigger`, `previous_state`, `retry_budget_remaining`, `original_outcome`, and `note`; `correlation` is limited to `work_unit_id`, `coordinator_run_id`, and `coordinator_attempt_ix`.

There is no lifecycle-specific configuration endpoint. Edit the file-backed runtime config and recreate/restart the API so its startup snapshot is rebuilt:

```yaml
lifecycle_runtime:
  lifecycle_events_enabled: false
  lifecycle_subagent_events_enabled: false
```

### Cost `/sessions/{session_id}/cost*`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/sessions/{session_id}/cost` | Yes | Return the session's LLM cost rollup by node, model, and provider |
| `GET` | `/sessions/{session_id}/cost/tree?depth=N` | Yes | Roll up self + descendant costs, truncation/depth metadata, and attribution source |

Cost decimals serialize as strings. An aggregate contains `total_usd`, `record_count`, `by_node`, `by_model`, `by_provider`, `pricing_version`, `cost_status`, `first_record_at`, `last_record_at`, and `has_partial_records`; each breakdown is an object from its string key to a decimal string. `cost_status` is `actual | estimated | partial | unknown`. A tree contains `session_id`, `self_cost`, `descendants_cost`, `total_cost`, `descendant_ids`, `depth_reached`, `max_depth_applied`, `truncated`, and `cost_source`; the source is `none | direct | coordinator_subagent | research_subagent | mixed`. When `truncated=true`, the response is a capped rollup and must not be treated as the complete descendant-tree total.

### Conversation compactions `/sessions/{session_id}/compactions`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/sessions/{session_id}/compactions` | Yes | List persisted compaction summaries and token/message deltas |
| `POST` | `/sessions/{session_id}/compactions` | Yes | Queue manual compaction for an eligible completed/timed-out session |
| `GET` | `/sessions/{session_id}/compactions/{compaction_id}` | Yes | Get a compaction record and its operations |
| `GET` | `/sessions/{session_id}/compactions/{compaction_id}/original-content` | Yes | Recover sanitized pre-compaction messages; returns `410` after checkpoint expiry |

The list returns `items` with the compaction id, operation kinds, summary preview, token/message deltas, visible-event bounds, recoverability, and creation time. Detail adds the full summary, operation metrics, parent/checkpoint ids, and totals. Manual `POST` has no request body, returns `{"request_status":"queued"}`, and does not allocate or return a `compaction_id`; observe the later `compaction` SSE event or refresh the list. Do not issue concurrent manual requests when request-to-result correlation is required. The POST may return `409` with a machine-readable `detail` when manual compaction or the overflow guard is disabled, the run is active/waiting/taken over, or the status is otherwise ineligible. These compaction endpoints return their documented body directly rather than the common JSON envelope.

## Common models

### `LLMConfig`

Includes:

- `base_url`
- `api_key` (excluded from read responses)
- `model_name`
- `temperature`
- `max_tokens`
- `api_type` — `chat_completions` / `responses` / `auto` (auto uses the fallback adapter; default is `chat_completions`)
- `timeout_seconds` — Hard per-call timeout (default 120s; range 0-3600; 0 disables the outer `asyncio.wait_for`)
- `connect_timeout_seconds` — Independent httpx connect-phase timeout (default 60s, 1.0 ≤ x ≤ 300.0); still enforced even when `timeout_seconds=0`
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

Returned for background sessions by session list/detail/recovery APIs. Important fields:

- `execution_mode`: `foreground | background`
- `execution_phase`: `running | recovering | idle | suspended | terminating | terminated`
- `background_reason`: `explicit | auto_degrade | null`
- `retry_budget_remaining`
- `suspended_reason` / `terminal_reason`
- `last_progress_at` / `expires_at`
- `is_alive`
- `cancellation_state`: `none | cancelling | cancelled`
- `execution_revision`: monotonic execution-state revision, default `0` for legacy/current snapshots without an explicit value

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
