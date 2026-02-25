import type { FileInfo } from "@/lib/api/types";

type PreviewSource = {
  filename?: string;
  extension?: string;
  mime_type?: string;
};

type SessionEventLike = {
  event: string;
  data: Record<string, unknown>;
};

export type FilePreviewKind = "text" | "image" | "pdf" | "unsupported";
export type ToolDisplayKind = "tool" | "progress";
export type WorkbenchMode = "shell" | "browser";
export type TimelineCursorState =
  | "live_following"
  | "live_locked"
  | "history_scrubbing"
  | "history_paused";

export type ToolDisplayCopy = {
  kind: ToolDisplayKind;
  title: string;
  detail: string;
};

export type WorkbenchConsoleRecord = {
  ps1?: string;
  command?: string;
  output?: string;
};

export type WorkbenchSnapshot = {
  id: string;
  timestamp: number; // 秒级时间戳
  mode: WorkbenchMode;
  shellSessionId: string | null;
  command: string | null;
  url: string | null;
  screenshot: string | null;
  consoleRecords: WorkbenchConsoleRecord[] | null;
};

export type SessionProgressStepStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "started";

export type SessionProgressStep = {
  id: string;
  description: string;
  status: SessionProgressStepStatus;
};

export type SessionProgressSummary = {
  completed: number;
  total: number;
  currentStep: string;
  hasPlan: boolean;
  steps: SessionProgressStep[];
};

const TEXT_EXTENSIONS = new Set([
  "txt",
  "md",
  "json",
  "csv",
  "log",
  "ts",
  "tsx",
  "js",
  "jsx",
  "py",
  "java",
  "go",
  "c",
  "cc",
  "cpp",
  "h",
  "hpp",
  "css",
  "html",
  "xml",
  "yml",
  "yaml",
  "toml",
  "sh",
  "sql",
]);

const IMAGE_EXTENSIONS = new Set([
  "png",
  "jpg",
  "jpeg",
  "gif",
  "webp",
  "bmp",
  "svg",
]);

function normalizeExtension(source: PreviewSource): string {
  if (source.extension) {
    return source.extension.replace(/^\./, "").toLowerCase();
  }

  const parts = source.filename?.split(".");
  if (!parts || parts.length < 2) {
    return "";
  }
  return parts.at(-1)?.toLowerCase() ?? "";
}

export function getFilePreviewKind(source: PreviewSource): FilePreviewKind {
  const extension = normalizeExtension(source);
  const mimeType = source.mime_type?.toLowerCase() ?? "";

  if (mimeType === "application/pdf" || extension === "pdf") {
    return "pdf";
  }

  if (mimeType.startsWith("image/") || IMAGE_EXTENSIONS.has(extension)) {
    return "image";
  }

  if (mimeType.startsWith("text/") || TEXT_EXTENSIONS.has(extension)) {
    return "text";
  }

  return "unsupported";
}

function parseTimestamp(value: unknown): number | null {
  if (value == null) {
    return null;
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    return value > 10 ** 12 ? value : value * 1000;
  }

  if (typeof value === "string") {
    const parsed = Date.parse(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }

  return null;
}

function parseTimestampSeconds(value: unknown): number | null {
  const ms = parseTimestamp(value);
  if (!ms) {
    return null;
  }
  return Math.floor(ms / 1000);
}

export function formatRelativeTime(value: unknown, now = Date.now()): string {
  const timestamp = parseTimestamp(value);
  if (!timestamp) {
    return "刚刚";
  }

  const diffMs = Math.max(0, now - timestamp);
  const minutes = Math.floor(diffMs / 60000);

  if (minutes < 1) {
    return "刚刚";
  }
  if (minutes < 60) {
    return `${minutes}分钟前`;
  }

  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}小时前`;
  }

  const days = Math.floor(hours / 24);
  if (days <= 7) {
    return `${days}天前`;
  }

  return new Date(timestamp).toISOString().slice(0, 10);
}

export function formatFileSize(size: number): string {
  if (!Number.isFinite(size) || size <= 0) {
    return "0 B";
  }

  if (size < 1024) {
    return `${Math.floor(size)} B`;
  }

  const kb = size / 1024;
  if (kb < 1024) {
    return `${kb.toFixed(1)} KB`;
  }

  const mb = kb / 1024;
  if (mb < 1024) {
    return `${mb.toFixed(1)} MB`;
  }

  const gb = mb / 1024;
  return `${gb.toFixed(1)} GB`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function getSessionEventStableKey(
  event: SessionEventLike,
  index: number
): string {
  if (event.event === "message") {
    const streamId = asString(event.data.stream_id).trim();
    if (streamId) {
      return `message:${streamId}`;
    }
  }

  if (event.event === "tool") {
    const toolCallId = asString(event.data.tool_call_id).trim();
    if (toolCallId) {
      return `tool:${toolCallId}`;
    }
  }

  if (event.event === "step") {
    const stepId = asString(event.data.id).trim();
    if (stepId) {
      return `step:${stepId}`;
    }
  }

  const eventId = asString(event.data.event_id).trim();
  if (eventId) {
    return eventId;
  }

  return `${event.event}-${index}`;
}

function formatPathTail(path: string): string {
  const clean = path.split("?")[0]?.split("#")[0] || path;
  const parts = clean.split("/");
  return parts[parts.length - 1] || path;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

const SESSION_PROGRESS_STEP_STATUS_SET = new Set<SessionProgressStepStatus>([
  "pending",
  "running",
  "completed",
  "failed",
  "started",
]);

function normalizeSessionProgressStepStatus(value: unknown): SessionProgressStepStatus {
  const status = asString(value).trim().toLowerCase();
  if (SESSION_PROGRESS_STEP_STATUS_SET.has(status as SessionProgressStepStatus)) {
    return status as SessionProgressStepStatus;
  }
  return "pending";
}

function normalizeFileInfoFromRecord(value: Record<string, unknown>): FileInfo | null {
  const id = asString(value.id).trim();
  if (!id) {
    return null;
  }
  return {
    id,
    filename: asString(value.filename),
    filepath: asString(value.filepath),
    key: asString(value.key),
    extension: asString(value.extension),
    mime_type: asString(value.mime_type) || "application/octet-stream",
    size: typeof value.size === "number" && Number.isFinite(value.size) ? value.size : 0,
  };
}

function fallbackAttachmentById(id: string): FileInfo {
  return {
    id,
    filename: `文件 ${id.slice(0, 8)}`,
    filepath: "",
    key: "",
    extension: "",
    mime_type: "application/octet-stream",
    size: 0,
  };
}

export function normalizeMessageAttachments(
  rawAttachments: unknown,
  sessionFiles: FileInfo[]
): FileInfo[] {
  if (!Array.isArray(rawAttachments)) {
    return [];
  }
  const fileMap = new Map(sessionFiles.map((file) => [file.id, file]));

  return rawAttachments
    .map((item) => {
      if (typeof item === "string") {
        return fileMap.get(item) ?? fallbackAttachmentById(item);
      }
      if (!isRecord(item)) {
        return null;
      }
      const normalized = normalizeFileInfoFromRecord(item);
      if (!normalized) {
        return null;
      }
      const matched = fileMap.get(normalized.id);
      if (!matched) {
        return normalized.filename ? normalized : fallbackAttachmentById(normalized.id);
      }
      return {
        ...matched,
        ...normalized,
        filename: normalized.filename || matched.filename,
        filepath: normalized.filepath || matched.filepath,
        key: normalized.key || matched.key,
        extension: normalized.extension || matched.extension,
        mime_type: normalized.mime_type || matched.mime_type,
        size: normalized.size > 0 ? normalized.size : matched.size,
      };
    })
    .filter((item): item is FileInfo => item !== null);
}

function getStepListFromPlanEvent(
  events: SessionEventLike[]
): SessionProgressStep[] | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!event || event.event !== "plan") {
      continue;
    }
    const rawSteps = event.data.steps;
    if (!Array.isArray(rawSteps)) {
      return [];
    }
    return rawSteps.map((item, itemIndex) => {
      const step = isRecord(item) ? item : {};
      const description = asString(step.description).trim() || "未命名步骤";
      return {
        id: asString(step.id).trim() || `step-${itemIndex + 1}`,
        status: normalizeSessionProgressStepStatus(step.status),
        description,
      };
    });
  }
  return null;
}

function pickCurrentStepDescription(
  steps: SessionProgressStep[]
): string {
  const running =
    steps.find((step) => step.status === "running" || step.status === "started") ||
    steps.find((step) => step.status === "failed") ||
    steps.find((step) => step.status === "pending");
  if (running) {
    return running.description || "未命名步骤";
  }
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    if (step?.status === "completed") {
      return step.description || "未命名步骤";
    }
  }
  return "暂无步骤";
}

export function deriveSessionProgressSummary(
  events: SessionEventLike[]
): SessionProgressSummary | null {
  const steps = getStepListFromPlanEvent(events);
  if (steps == null) {
    return null;
  }
  const total = steps.length;
  const completed = steps.filter((step) => step.status === "completed").length;
  return {
    completed,
    total,
    currentStep: pickCurrentStepDescription(steps),
    hasPlan: true,
    steps,
  };
}

export function formatTimeOfDay(value: unknown): string {
  const ms = parseTimestamp(value);
  if (!ms) {
    return "--:--:--";
  }
  const date = new Date(ms);
  const hh = String(date.getHours()).padStart(2, "0");
  const mm = String(date.getMinutes()).padStart(2, "0");
  const ss = String(date.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

export function formatWorkbenchClock(value: unknown): string {
  return formatTimeOfDay(value);
}

/**
 * 从 MCP 工具的函数名中提取可读的短名称。
 * 函数名格式: mcp_servername_toolname → 提取 toolname 部分。
 */
function extractMCPToolShortName(functionName: string): string {
  if (!functionName) return "";
  // 去掉 mcp_ 前缀后，按第一个 _ 分割出 servername 和 toolname
  const withoutPrefix = functionName.replace(/^mcp_/, "");
  const firstUnderscore = withoutPrefix.indexOf("_");
  if (firstUnderscore > 0 && firstUnderscore < withoutPrefix.length - 1) {
    return withoutPrefix.substring(firstUnderscore + 1);
  }
  return functionName;
}

function pickToolDetail(
  toolName: string,
  functionName: string,
  args: Record<string, unknown>
): string {
  if (functionName === "search_web") {
    const query = asString(args.query);
    return query ? `关键词：${query}` : "正在检索相关信息";
  }

  if (functionName === "browser_navigate" || functionName === "browser_restart") {
    const url = asString(args.url);
    return url ? `目标网页：${url}` : "正在访问目标网页";
  }

  if (
    functionName === "read_file" ||
    functionName === "write_file" ||
    functionName === "replace_in_file" ||
    functionName === "search_in_file"
  ) {
    const filepath = asString(args.filepath);
    return filepath ? `文件：${formatPathTail(filepath)}` : "正在处理文件";
  }

  if (functionName === "find_files") {
    const dirPath = asString(args.dir_path);
    const pattern = asString(args.glob_pattern);
    if (dirPath && pattern) {
      return `目录：${dirPath}，模式：${pattern}`;
    }
    return "正在查找文件";
  }

  if (functionName.startsWith("shell_")) {
    const command = asString(args.command);
    if (command) {
      return `命令：${command}`;
    }
    const sessionId = asString(args.session_id);
    return sessionId ? `终端会话：${sessionId.slice(0, 8)}` : "正在操作终端";
  }

  const genericText =
    asString(args.text) ||
    asString(args.query) ||
    asString(args.url) ||
    asString(args.filepath);

  // MCP工具：显示完整的工具函数名和参数摘要
  if (toolName === "mcp" || toolName === "a2a" || toolName === "skill") {
    return functionName || genericText || "正在处理中";
  }

  return genericText || "正在处理中";
}

function getToolActionTitle(
  toolName: string,
  functionName: string,
  called: boolean
): string {
  const phase = called ? "已完成" : "正在";

  const titleByFunction: Record<string, string> = {
    search_web: "搜索资料",
    browser_view: "查看网页内容",
    browser_navigate: "访问网页",
    browser_restart: "重启浏览器",
    browser_click: "点击页面元素",
    browser_input: "填写页面输入",
    browser_move_mouse: "移动鼠标",
    browser_press_key: "触发按键",
    browser_select_option: "选择页面选项",
    browser_scroll_up: "向上滚动页面",
    browser_scroll_down: "向下滚动页面",
    browser_console_exec: "执行网页脚本",
    browser_console_view: "查看控制台输出",
    read_file: "读取文件",
    write_file: "写入文件",
    replace_in_file: "替换文件内容",
    search_in_file: "搜索文件内容",
    find_files: "查找文件",
    shell_execute: "执行终端命令",
    shell_read_output: "读取终端输出",
    shell_wait_process: "等待终端进程",
    shell_write_input: "写入终端输入",
    shell_kill_process: "终止终端进程",
    get_remote_agent_cards: "获取远程 Agent 列表",
    call_remote_agent: "调用远程 Agent",
  };

  const mapped = titleByFunction[functionName];
  if (mapped) {
    return `${phase}${mapped}`;
  }

  const toolLabelByName: Record<string, string> = {
    search: "搜索工具",
    browser: "浏览器工具",
    shell: "终端工具",
    file: "文件工具",
    mcp: "MCP 工具",
    a2a: "远程 Agent",
    skill: "Skill 工具",
    message: "消息工具",
  };

  // MCP/A2A工具：从函数名中提取可读的工具名
  if (toolName === "mcp" || toolName === "a2a" || toolName === "skill") {
    const label = toolLabelByName[toolName] || "工具";
    // 函数名格式: mcp_servername_toolname，提取最后的工具名部分
    const readableName = extractMCPToolShortName(functionName);
    if (readableName) {
      return called ? `${label} ${readableName} 已完成` : `正在调用${label} ${readableName}`;
    }
    return called ? `${label}调用已完成` : `正在调用${label}`;
  }

  const label = toolLabelByName[toolName] || "工具";
  return called ? `${label}调用已完成` : `正在调用${label}`;
}

export function getToolDisplayCopy(eventData: Record<string, unknown>): ToolDisplayCopy {
  const toolName = asString(eventData.name);
  const functionName = asString(eventData.function);
  const args = isRecord(eventData.args) ? eventData.args : {};
  const status = asString(eventData.status);
  const called = status === "called";

  if (toolName === "message" && functionName === "message_notify_user") {
    const text = asString(args.text);
    return {
      kind: "progress",
      title: "进度更新",
      detail: text || "任务正在推进中",
    };
  }

  if (toolName === "message" && functionName === "message_ask_user") {
    const text = asString(args.text);
    return {
      kind: "progress",
      title: "需要你确认",
      detail: text || "请补充下一步操作信息",
    };
  }

  return {
    kind: "tool",
    title: getToolActionTitle(toolName, functionName, called),
    detail: pickToolDetail(toolName, functionName, args),
  };
}

function toSnapshotId(event: SessionEventLike, index: number): string {
  const eventId = asString(event.data.event_id);
  if (eventId) {
    return eventId;
  }
  return `snapshot-${index}`;
}

function toSnapshotFromToolEvent(
  event: SessionEventLike,
  index: number
): WorkbenchSnapshot | null {
  if (event.event !== "tool") {
    return null;
  }
  const status = asString(event.data.status);
  if (status !== "called") {
    return null;
  }

  const timestamp = parseTimestampSeconds(event.data.created_at);
  if (!timestamp) {
    return null;
  }

  const toolName = asString(event.data.name);
  const args = isRecord(event.data.args) ? event.data.args : {};
  const content = isRecord(event.data.content) ? event.data.content : {};

  if (toolName === "browser") {
    const screenshot = asString(content.screenshot);
    if (!screenshot) {
      return null;
    }

    return {
      id: toSnapshotId(event, index),
      timestamp,
      mode: "browser",
      shellSessionId: null,
      command: null,
      url: asString(args.url) || asString(content.url) || null,
      screenshot,
      consoleRecords: null,
    };
  }

  if (toolName === "shell") {
    const shellSessionId = asString(args.session_id) || null;
    const command = asString(args.command) || null;
    const rawConsole = content.console ?? content.console_records;
    const consoleRecords = Array.isArray(rawConsole)
      ? rawConsole.map((record) => {
          const item = isRecord(record) ? record : {};
          return {
            ps1: asString(item.ps1),
            command: asString(item.command),
            output: asString(item.output),
          };
        })
      : [];

    if (!shellSessionId && consoleRecords.length === 0 && !command) {
      return null;
    }

    return {
      id: toSnapshotId(event, index),
      timestamp,
      mode: "shell",
      shellSessionId,
      command,
      url: null,
      screenshot: null,
      consoleRecords: consoleRecords.length > 0 ? consoleRecords : null,
    };
  }

  return null;
}

export function deriveWorkbenchSnapshots(events: SessionEventLike[]): WorkbenchSnapshot[] {
  const snapshots: WorkbenchSnapshot[] = [];
  const dedupe = new Set<string>();

  events.forEach((event, index) => {
    const snapshot = toSnapshotFromToolEvent(event, index);
    if (!snapshot) {
      return;
    }
    if (dedupe.has(snapshot.id)) {
      return;
    }
    dedupe.add(snapshot.id);
    snapshots.push(snapshot);
  });

  snapshots.sort((a, b) => {
    if (a.timestamp === b.timestamp) {
      return a.id.localeCompare(b.id);
    }
    return a.timestamp - b.timestamp;
  });
  return snapshots;
}

export function getLatestSnapshotByMode(
  snapshots: WorkbenchSnapshot[],
  mode: WorkbenchMode
): WorkbenchSnapshot | null {
  for (let index = snapshots.length - 1; index >= 0; index -= 1) {
    const snapshot = snapshots[index];
    if (snapshot?.mode === mode) {
      return snapshot;
    }
  }
  return null;
}

export function findSnapshotAtOrBefore(
  snapshots: WorkbenchSnapshot[],
  timestamp: number | null
): WorkbenchSnapshot | null {
  if (snapshots.length === 0) {
    return null;
  }
  if (!timestamp) {
    return snapshots[snapshots.length - 1] || null;
  }

  let candidate: WorkbenchSnapshot | null = null;
  for (let index = 0; index < snapshots.length; index += 1) {
    const snapshot = snapshots[index];
    if (!snapshot) {
      continue;
    }
    if (snapshot.timestamp <= timestamp) {
      candidate = snapshot;
      continue;
    }
    break;
  }

  if (candidate) {
    return candidate;
  }
  return snapshots[0] || null;
}

export function pickSnapshotAtTime(
  snapshots: WorkbenchSnapshot[],
  timestamp: number | null
): WorkbenchSnapshot | null {
  return findSnapshotAtOrBefore(snapshots, timestamp);
}

export function getShellSessionIdsFromSnapshots(
  snapshots: WorkbenchSnapshot[]
): string[] {
  const ids: string[] = [];
  const dedupe = new Set<string>();
  for (let index = snapshots.length - 1; index >= 0; index -= 1) {
    const snapshot = snapshots[index];
    if (!snapshot || snapshot.mode !== "shell" || !snapshot.shellSessionId) {
      continue;
    }
    if (dedupe.has(snapshot.shellSessionId)) {
      continue;
    }
    dedupe.add(snapshot.shellSessionId);
    ids.push(snapshot.shellSessionId);
  }
  return ids;
}

export function getShellSessionIds(events: SessionEventLike[]): string[] {
  const ids: string[] = [];
  const dedupe = new Set<string>();

  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!event || event.event !== "tool") {
      continue;
    }

    const toolName = String(event.data.name || "");
    const functionName = String(event.data.function || "");
    const isShellTool = toolName === "shell" || functionName.startsWith("shell_");
    if (!isShellTool) {
      continue;
    }

    const args = event.data.args;
    if (!isRecord(args)) {
      continue;
    }

    const sessionId = args.session_id;
    if (typeof sessionId !== "string" || !sessionId.trim()) {
      continue;
    }

    if (dedupe.has(sessionId)) {
      continue;
    }

    dedupe.add(sessionId);
    ids.push(sessionId);
  }

  return ids;
}
