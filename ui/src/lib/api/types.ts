/**
 * API 统一响应格式
 */
export type ApiResponse<T = unknown> = {
  code: number;
  msg: string;
  data: T | null;
};

export type UserRole = "super_admin" | "user";
export type UserStatus = "active" | "inactive" | "banned";

export type UserProfile = {
  id: string;
  username: string | null;
  email: string | null;
  nickname: string | null;
  avatar: string | null;
  role: UserRole;
  status: UserStatus;
  created_at: string;
};

export type TokenResponse = {
  access_token: string;
  refresh_token: string;
  token_type: string;
};

export type LoginResponse = {
  user: UserProfile;
  tokens: TokenResponse;
};

export type RegisterParams = {
  username?: string;
  email?: string;
  password: string;
  nickname?: string;
};

export type LoginParams = {
  username?: string;
  email?: string;
  password: string;
};

export type RefreshParams = {
  refresh_token: string;
};

export type UpdateMeParams = {
  nickname?: string;
  avatar?: string;
};

/**
 * 会话状态
 */
export type SessionStatus =
  | "pending"
  | "running"
  | "takeover_pending"
  | "takeover"
  | "waiting"
  | "finishing"
  | "completed"
  | "timed_out";

/**
 * 执行状态
 */
export type ExecutionStatus = "pending" | "running" | "completed" | "failed";

/**
 * 工具事件状态
 */
export type ToolEventStatus = "calling" | "called";

/**
 * MCP 传输类型
 */
export type MCPTransport = "stdio" | "sse" | "streamable_http";

// ==================== 配置模块类型 ====================

export type LLMConfig = {
  base_url: string;
  api_key?: string;
  model_name: string;
  supports_vision: boolean;
  supports_pdf_input: boolean;
  api_type: "chat_completions" | "responses" | "auto";
  temperature: number;
  max_tokens: number;
  context_window: number | null;
  context_overflow_guard_enabled: boolean;
  overflow_retry_cap: number;
  soft_trigger_ratio: number;
  hard_trigger_ratio: number;
  reserved_output_tokens: number;
  reserved_output_tokens_cap_ratio: number;
  token_estimator: "hybrid" | "char" | "provider_api";
  token_safety_factor: number;
  unknown_model_context_window: number;
};

export type ToolConfirmationConfig = {
  enabled: boolean;
  timeout_seconds: number;
  smart_approve_enabled: boolean;
  smart_approve_medium_only: boolean;
};

export type AgentConfig = {
  max_iterations: number;
  max_retries: number;
  max_search_results: number;
  tool_confirmation?: ToolConfirmationConfig;
};

export type VisionFallbackConfig = {
  enabled: boolean;
  base_url: string;
  api_key?: string;
  model_name: string;
  api_type: "chat_completions" | "responses" | "auto";
};

export type AudioProcessorConfig = {
  provider: "disabled" | "sandbox_whisper" | "openai_api";
  openai_api_key?: string;
  openai_base_url: string;
  openai_model: string;
};

export type VideoProcessorConfig = {
  max_keyframes: number;
  extract_audio: boolean;
  frame_strategy: "scene" | "uniform";
  scene_threshold: number;
};

export type FileUnderstandingConfig = {
  vision_fallback: VisionFallbackConfig;
  audio: AudioProcessorConfig;
  video: VideoProcessorConfig;
};

export type ListMCPServerItem = {
  server_name: string;
  enabled: boolean;
  transport: MCPTransport;
  tools: string[];
};

export type MCPServersData = {
  mcp_servers: ListMCPServerItem[];
};

export type MCPServerConfig = {
  transport?: MCPTransport;
  enabled?: boolean;
  description?: string | null;
  env?: Record<string, unknown> | null;
  command?: string | null;
  args?: string[] | null;
  url?: string | null;
  headers?: Record<string, unknown> | null;
  [key: string]: unknown;
};

export type MCPConfig = {
  mcpServers: Record<string, MCPServerConfig>;
};

export type ListA2AServerItem = {
  id: string;
  name: string;
  description: string;
  input_modes: string[];
  output_modes: string[];
  streaming: boolean;
  push_notifications: boolean;
  enabled: boolean;
};

export type A2AServersData = {
  a2a_servers: ListA2AServerItem[];
};

export type CreateA2AServerParams = {
  base_url: string;
};

export type SkillSourceType = "local" | "github";
export type SkillRuntimeType = "native" | "mcp" | "a2a";

export type SkillItem = {
  id: string;
  slug: string;
  name: string;
  description: string;
  version: string;
  source_type: SkillSourceType;
  source_ref: string;
  runtime_type: SkillRuntimeType;
  enabled: boolean;
  installed_by?: string | null;
  created_at: string;
  updated_at: string;
  bundle_file_count?: number;
  context_ref_count?: number;
  last_sync_at?: string | null;
};

export type SkillListData = {
  skills: SkillItem[];
};

export type SkillToolItem = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  required: string[];
  entry?: Record<string, unknown> | null;
};

export type BundleFileItem = {
  path: string;
  size: number;
  sha256: string;
  is_text: boolean;
};

export type SkillDetailData = SkillItem & {
  tools: SkillToolItem[];
  skill_md: string;
  bundle_files: BundleFileItem[];
  activation: Record<string, unknown>;
  policy: Record<string, unknown>;
  security: Record<string, unknown>;
};

export type InstallSkillParams = {
  source_type: SkillSourceType;
  source_ref: string;
  skill_md?: string;
  manifest?: Record<string, unknown>;
};

export type SkillRiskPolicy = {
  mode: "off" | "enforce_confirmation";
};


export type ToolWithPreference = {
  tool_id: string;
  tool_name: string;
  description: string | null;
  enabled_global: boolean;
  enabled_user: boolean;
};

export type ToolPreferenceListResponse = {
  tools: ToolWithPreference[];
};

// ==================== 文件模块类型 ====================

export type FileInfo = {
  id: string;
  filename: string;
  filepath: string;
  key: string;
  extension: string;
  mime_type: string;
  size: number;
  user_id?: string | null;
};

export type FileUploadParams = {
  file: File;
  session_id?: string;
};

// ==================== 会话模块类型 ====================

export type ListSessionItem = {
  session_id: string;
  title: string;
  latest_message: string;
  latest_message_at: string | null;
  status: SessionStatus;
  unread_message_count: number;
};

export type ListSessionResponse = {
  sessions: ListSessionItem[];
};

// B3-core PR-1 §3.3 — supervisor snapshot mirror of api/app/interfaces/schemas/session.py:SupervisorSnapshot
export type SupervisorExecutionMode = "foreground" | "background";
export type SupervisorExecutionPhase =
  | "running"
  | "recovering"
  | "idle"
  | "suspended"
  | "terminating"
  | "terminated";

export type SupervisorSnapshot = {
  execution_mode: SupervisorExecutionMode;
  execution_phase: SupervisorExecutionPhase;
  background_reason?: "explicit" | "auto_degrade" | null;
  expires_at?: string | null;
  retry_budget_remaining: number;
  suspended_reason?: string | null;
  terminal_reason?: string | null;
  last_progress_at?: string | null;
  is_alive: boolean;
  cancellation_state: "none" | "cancelling" | "cancelled";
};

export type Session = {
  session_id: string;
  title: string | null;
  status: SessionStatus;
  events: AgentSSEEvent[];
  // B3-core PR-1 — null/0 when backend hasn't populated yet
  last_seq?: number;
  supervisor_snapshot?: SupervisorSnapshot | null;
};

export type EventsSinceResponse = {
  events: AgentSSEEvent[];
  session_status: SessionStatus;
  has_more: boolean;
  // B3-core PR-1 §3.3 additions
  last_seq: number;
  supervisor_snapshot: SupervisorSnapshot | null;
};

export type CreateSessionParams = {
  title?: string;
};

export type CreateSessionResponse = {
  session_id: string;
};

export type TakeoverScope = "shell" | "browser";

export type GetTakeoverResponse = {
  status: SessionStatus;
  takeover_id?: string;
  request_status?: string;
  reason?: string;
  scope?: string;
  handoff_mode?: string;
  expires_at?: number;
};

export type StartTakeoverParams = {
  scope?: TakeoverScope;
};

export type StartTakeoverResponse = {
  status: SessionStatus;
  request_status: string;
  scope: string;
  takeover_id?: string;
  reason?: string;
  expires_at?: number;
};

export type RejectTakeoverParams = {
  decision: "continue" | "terminate";
};

export type RejectTakeoverResponse = {
  status: SessionStatus;
  reason: string;
};

export type EndTakeoverParams = {
  handoff_mode?: "continue" | "complete";
};

export type EndTakeoverResponse = {
  status: SessionStatus;
  handoff_mode: string;
};

export type RenewTakeoverParams = {
  takeover_id: string;
};

export type RenewTakeoverResponse = {
  status: SessionStatus;
  request_status: string;
  takeover_id: string;
  expires_at?: number;
};

export type ChatMessageData = {
  event_id?: string;
  created_at?: number;
  role: "user" | "assistant" | "system";
  message: string;
  stream_id?: string;
  partial?: boolean;
  attachments: FileInfo[];
};

export type ChatParams = {
  message?: string;
  attachments?: string[];
  skill_confirmation_action?: "generate" | "revise" | "install" | "cancel";
  tool_confirmation?: {
    action: "approve" | "deny";
    scope: "once" | "session" | "always";
    tool_call_id: string;
  };
  event_id?: string;
  timestamp?: number;
};

export type PlanStep = {
  id: string;
  description: string;
  status: ExecutionStatus;
};

export type PlanEvent = {
  event_id?: string;
  created_at?: number;
  steps: PlanStep[];
};

export type StepEvent = {
  event_id?: string;
  created_at?: number;
  id: string;
  status: ExecutionStatus;
  description: string;
};

// ==================== R4 CS3 ToolEventEnvelopeV1 ====================

export type ToolStatusV1 =
  | "ok"
  | "error"
  | "denied"
  | "timeout"
  | "passthrough";
  // 注意: asked 不在 envelope, 走独立 tool_confirmation 事件

export type DecisionReasonWire = {
  type: string;        // domain 白名单 6 值 + wire-only fallback "unknown_variant"
                       //   domain: approval_policy | smart_approve | ast_validator |
                       //           risk_enforce | exception | timeout
                       //   wire-only: unknown_variant (projector fallback)
  code: string;
  message: string;
};

export type FunctionResultV1 = {
  status: ToolStatusV1;
  message: string;
  data: unknown;
  retryable: boolean;
  user_action_required: boolean;  // 恒为 false in v1 (asked 走 tool_confirmation)
  reason?: DecisionReasonWire | null;
  result_blocks?: Array<Record<string, unknown>> | null;
};

export type RenderStyle = "text" | "code" | "table" | "image" | "document";

export type ToolSource = {
  source: "native" | "mcp" | "a2a" | "skill";
  category: string;
  canonical_name: string;
};

export type ToolEventEnvelopeV1 = {
  envelope_version: 1;
  event_id?: string;
  created_at?: number;

  tool_call_id: string;
  name: string;                          // wire 短名 (backend alias from tool_name)
  tool_source?: ToolSource | null;
  function: string;                      // wire 短名 (backend alias from function_name)
  args: Record<string, unknown>;         // wire 短名 (backend alias from function_args)
  status: "calling" | "called";
  activity_description: string;
  display_icon?: string | null;
  render_style?: RenderStyle | null;
  media_type?: string | null;

  function_result?: FunctionResultV1 | null;
  content?: Record<string, unknown> | null;  // 保留 tool_content enrichment channel
};

// R4: backward compat alias. 旧 consumer 逐步迁到 ToolEventEnvelopeV1.
/** @deprecated Use ToolEventEnvelopeV1 */
export type ToolEvent = ToolEventEnvelopeV1;

export type TitleEvent = {
  event_id?: string;
  created_at?: number;
  title: string;
};

export type ErrorEvent = {
  event_id?: string;
  created_at?: number;
  error: string;
};

export type ReopenTakeoverResponse = {
  status: SessionStatus;
  request_status: string;
  reason: string | null;
  remaining_seconds: number | null;
};

export type ControlAction =
  | "requested"
  | "started"
  | "rejected"
  | "renewed"
  | "expired"
  | "ended"
  | "reopened";

export type ControlSource = "agent" | "user" | "system";

export type ControlEvent = {
  event_id?: string;
  created_at?: number;
  action: ControlAction;
  scope?: "shell" | "browser";
  source: ControlSource;
  reason?: string;
  handoff_mode?: "continue" | "complete";
  request_status?: string;
  takeover_id?: string;
  expires_at?: number;
};

export type WaitEvent = {
  event_id?: string;
  created_at?: number;
  pending_action?: "generate" | "install" | null;
  [key: string]: unknown;
};

export type ExecutionMetrics = {
  tool_success_rate?: number;
  avg_tool_latency_ms?: number;
  avg_llm_latency_ms?: number;
  tool_calls_total?: number;
  tool_calls_failed?: number;
  llm_calls_total?: number;
  steps_completed?: number;
  steps_failed?: number;
  context_usage_ratio?: number;
  compaction_count?: number;
};

export type DoneEvent = {
  event_id?: string;
  created_at?: number;
  metrics?: ExecutionMetrics | null;
  [key: string]: unknown;
};

export type HealthEventStatus =
  | "healthy"
  | "degraded"
  | "terminating"
  | "terminated";

export type HealthEvent = {
  event_id?: string;
  created_at?: number;
  status: HealthEventStatus;
  reason: string;
  last_node?: string | null;
  idle_seconds?: number | null;
  tool_failures?: number;
  action?: string;
  metrics?: ExecutionMetrics | null;
};

export type FinishingEvent = {
  event_id?: string;
  created_at?: number;
  [key: string]: unknown;
};

export type SandboxStateChangedEvent = {
  event_id?: string;
  created_at?: number;
  session_id?: string;
  old_state?: string;
  new_state: string;
  generation?: number | null;
  reason?: string | null;
  [key: string]: unknown;
};

export type OwnerConflictEventData = {
  event_id?: string;
  created_at?: number;
  seq?: number | null;
  payload: {
    current_owner_connection_id: string;
    conflicting_connection_id: string;
    session_id: string;
    suggested_action: "wait_lease_expire" | "request_takeover";
  };
  [key: string]: unknown;
};

export type CompactionEventData = {
  event_id?: string;
  created_at?: number | string;
  compaction_id?: string | null;
  level: number;             // 2 = llm_summary, 3 = hard_truncate
  tokens_before: number;
  tokens_after: number;
  messages_removed: number;
  usage_ratio_after?: number;
};

export type SSEEventType =
  | "message"
  | "title"
  | "plan"
  | "step"
  | "tool"
  | "control"
  | "wait"
  | "tool_confirmation"
  | "finishing"
  | "health"
  | "sandbox_state_changed"
  | "owner_conflict"
  | "compaction"
  | "done"
  | "error"
  | "sessions";

export type ToolConfirmationEventData = {
  event_id?: string;
  created_at?: number;
  tool_call_id: string;
  tool_name: string;
  tool_args: Record<string, unknown>;
  risk_level: "high" | "medium";
  risk_reason: string;
  matched_patterns: string[];
  suggested_alternative: string | null;
  approval_options: string[];
  timeout_seconds: number;
};

export type SSEEventData =
  | { type: "message"; data: ChatMessageData }
  | { type: "title"; data: TitleEvent }
  | { type: "plan"; data: PlanEvent }
  | { type: "step"; data: StepEvent }
  | { type: "tool"; data: ToolEvent }
  | { type: "control"; data: ControlEvent }
  | { type: "wait"; data: WaitEvent }
  | { type: "tool_confirmation"; event_id?: string; created_at?: number; data: ToolConfirmationEventData }
  | { type: "finishing"; data: FinishingEvent }
  | { type: "health"; data: HealthEvent }
  | { type: "sandbox_state_changed"; data: SandboxStateChangedEvent }
  | { type: "owner_conflict"; data: OwnerConflictEventData }
  | { type: "compaction"; data: CompactionEventData }
  | { type: "done"; data: DoneEvent }
  | { type: "error"; data: ErrorEvent }
  | { type: "sessions"; data: ListSessionResponse };

export type SSEEventHandler = (event: SSEEventData) => void;

export type SessionFile = FileInfo;

export type GetSessionFilesResponse = {
  files: SessionFile[];
};

export type ViewFileParams = {
  filepath: string;
};

export type FileReadResponse = {
  filepath: string;
  content: string;
};

export type ViewShellParams = {
  session_id: string;
};

export type ShellConsoleRecord = {
  ps1: string;
  command: string;
  output: string;
};

export type ShellReadResponse = {
  session_id: string;
  output: string;
  console_records: ShellConsoleRecord[];
};

export type AgentSSEEvent = {
  event: SSEEventType | string;
  data: Record<string, unknown>;
};

// ==================== 管理员用户管理 ====================

export type UserListResponse = {
  users: UserProfile[];
  total: number;
};

export type UserStatusUpdateRequest = {
  status: UserStatus;
};

// ==================== Memory Management ====================

/**
 * PR-1 起后端把 memory 分三类。旧数据 category=NULL（不强制回填），
 * 所以 UI 侧在列表里要能显式表达 "legacy（未分类）" 的场景。
 */
export type MemoryCategory = "user" | "rule" | "fact";

export interface MemoryItem {
  id: string;
  content: string;
  source: string;
  created_at: string;
  updated_at: string;
  session_id: string | null;
  // PR-1 新增字段——列表视图按 category 过滤 + 显示 pinned badge。
  // legacy 行 category=null 表示 PR-1 前的数据，UI 显示为 "未分类"。
  category: MemoryCategory | null;
  pinned: boolean;
  auto_promoted_at: string | null;
}

export interface MemoryDetail extends MemoryItem {
  content_hash: string;
  metadata: Record<string, unknown>;
  // fs_synced 只在详情暴露（设计 L35）：list 视图不关心同步状态。
  fs_synced: boolean;
}

export interface MemoryListResponse {
  items: MemoryItem[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
}

export interface MemoryListParams {
  query?: string;
  source?: string;
  /** 传 undefined 返回全部（含 legacy null）；传具体 category 只返回该类。 */
  category?: MemoryCategory;
  created_from?: string;
  created_to?: string;
  updated_from?: string;
  updated_to?: string;
  page?: number;
  page_size?: number;
}

/**
 * POST /v2/memories 请求体。source 由服务端固定为 "manual"，客户端不传。
 * pinned=true 仅在 category="user" 时合法——后端 400，UI 侧也提前 disable。
 *
 * ``tags``（可选）：落到 frontmatter ``tags:`` + ``metadata.tags``。后端侧已做
 * strip/dedupe/单条 64 字符/最多 20 条的校验，客户端只需把用户输入切开传上去；
 * 空数组可以直接不传（或传 undefined）。
 */
export interface CreateMemoryRequest {
  content: string;
  category: MemoryCategory;
  pinned?: boolean;
  tags?: string[];
}

export interface DeleteCountResponse {
  deleted_count: number;
}

/**
 * M3-A codex fix P1: legacy cleanup 的时间边界配置。
 *
 * - `rollout_at` 非空（ISO 8601）：后端 SQL 会加 `AND created_at < rollout_at`，
 *   dialog 显示具体 cutoff 时间
 * - `rollout_at` 为 null：未配置 → 后端沿用旧谓词（清所有未分类 session_flush），
 *   UI 显示显式警告
 */
export interface LegacyCleanupConfigResponse {
  rollout_at: string | null;
}

/**
 * Post-M3 reindex endpoint 响应（Option A 权威契约；codex round-4 收口）。
 *
 * 契约表（与后端 service / route / schema docstring 四层对齐）：
 * - `body` → apply 到 DB
 * - `id` mismatch → 409（不在 warnings 里）
 * - `source` / `created_at` / `auto_promoted_at` → warnings（系统字段）
 * - `title` / `category` / `pinned` / `tags` → warnings + **file-only**
 *
 * - `reindexed_fields`: 实际 apply 到 DB 的字段；`["content"]` = body 改动已同步；
 *   `[]` = no-op 幂等（盘上与 DB 一致）。未来扩展新字段 append 到 Literal。
 * - `warnings`: hand-edit 改了 file-only / 系统字段被忽略的说明；**这些字段
 *   只停留在文件侧（sandbox `file_read` 能看到），不进入 DB /
 *   `memory_search` / prompt；当前没有受支持的自动同步路径**。前端需原样
 *   展示，不能承诺"走 PATCH"或"等 reconciler"——两个路径都不真实可用。
 * - `fs_synced`: reindex 后 DB 的 fs_synced 值（正常路径恒 True）。
 */
export interface ReindexResponse {
  reindexed_fields: ("content")[];
  warnings: string[];
  fs_synced: boolean;
}

// ---------------------------------------------------------------------------
// B4 M0: cost ledger
// ---------------------------------------------------------------------------

/**
 * Aggregate-level confidence in the cost number.
 *
 * - `actual`: every row's usage came from the provider.
 * - `unknown`: provider didn't report usage, or no rows yet.
 * - `partial`: session mixes actual + unknown rows, or a degraded persist
 *   marker landed → ledger is known-incomplete.
 * - `estimated`: reserved (M0 does not produce; see backend
 *   `CostStatus` docstring).
 */
export type CostStatus = "actual" | "unknown" | "partial" | "estimated";

/**
 * Response shape for `GET /api/sessions/{session_id}/cost`.
 *
 * `total_usd` and the breakdown maps are serialized as decimal strings
 * (backend `Decimal → format(v, "f")`) so we don't lose precision on small
 * cache-hit deltas. The UI is responsible for parsing if it needs math; for
 * display, render the string verbatim.
 */
export interface CostAggregateResponse {
  total_usd: string;
  record_count: number;
  by_node: Record<string, string>;
  by_model: Record<string, string>;
  by_provider: Record<string, string>;
  pricing_version: string;
  cost_status: CostStatus;
  first_record_at: string | null;
  last_record_at: string | null;
  has_partial_records: boolean;
}
