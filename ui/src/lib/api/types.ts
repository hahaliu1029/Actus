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
export type SessionStatus = "pending" | "running" | "waiting" | "completed";

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
  temperature: number;
  max_tokens: number;
};

export type AgentConfig = {
  max_iterations: number;
  max_retries: number;
  max_search_results: number;
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

export type Session = {
  session_id: string;
  title: string | null;
  status: SessionStatus;
  events: AgentSSEEvent[];
};

export type CreateSessionParams = {
  title?: string;
};

export type CreateSessionResponse = {
  session_id: string;
};

export type ChatMessageData = {
  event_id?: string;
  created_at?: number;
  role: "user" | "assistant";
  message: string;
  stream_id?: string;
  partial?: boolean;
  attachments: FileInfo[];
};

export type ChatParams = {
  message?: string;
  attachments?: string[];
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

export type ToolEvent = {
  event_id?: string;
  created_at?: number;
  tool_call_id: string;
  name: string;
  function: string;
  args: Record<string, unknown>;
  content?: unknown;
  status?: ToolEventStatus;
};

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

export type WaitEvent = {
  event_id?: string;
  created_at?: number;
  [key: string]: unknown;
};

export type DoneEvent = {
  event_id?: string;
  created_at?: number;
  [key: string]: unknown;
};

export type SSEEventType =
  | "message"
  | "title"
  | "plan"
  | "step"
  | "tool"
  | "wait"
  | "done"
  | "error"
  | "sessions";

export type SSEEventData =
  | { type: "message"; data: ChatMessageData }
  | { type: "title"; data: TitleEvent }
  | { type: "plan"; data: PlanEvent }
  | { type: "step"; data: StepEvent }
  | { type: "tool"; data: ToolEvent }
  | { type: "wait"; data: WaitEvent }
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
