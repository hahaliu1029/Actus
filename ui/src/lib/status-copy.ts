export type StatusTone = "muted" | "warning" | "success" | "danger" | "info";

export type StatusIconKey =
  | "circle-dashed"
  | "loader"
  | "clock"
  | "check-circle"
  | "x-circle"
  | "alert-circle"
  | "hand"
  | "circle-help";

export type StatusMeta = {
  text: string;
  tone: StatusTone;
  icon: StatusIconKey;
  spinning?: boolean;
  rawStatus?: string;
};

const SESSION_STATUS_MAP: Record<string, StatusMeta> = {
  pending: { text: "待执行", tone: "muted", icon: "circle-dashed" },
  running: { text: "执行中", tone: "warning", icon: "loader", spinning: true },
  waiting: { text: "等待中", tone: "warning", icon: "clock" },
  finishing: { text: "整理中", tone: "info", icon: "loader" },
  completed: { text: "已完成", tone: "success", icon: "check-circle" },
  failed: { text: "失败", tone: "danger", icon: "x-circle" },
  takeover_pending: { text: "待接管", tone: "warning", icon: "alert-circle" },
  takeover: { text: "接管中", tone: "info", icon: "hand" },
};

const STEP_STATUS_MAP: Record<string, StatusMeta> = {
  pending: { text: "待执行", tone: "muted", icon: "circle-dashed" },
  started: { text: "执行中", tone: "warning", icon: "loader", spinning: true },
  running: { text: "执行中", tone: "warning", icon: "loader", spinning: true },
  completed: { text: "已完成", tone: "success", icon: "check-circle" },
  failed: { text: "失败", tone: "danger", icon: "x-circle" },
};

const UNKNOWN_META: StatusMeta = {
  text: "未知状态",
  tone: "muted",
  icon: "circle-help",
};

function normalizeStatus(status: unknown): string {
  return typeof status === "string" ? status.trim().toLowerCase() : "";
}

function toFallbackMeta(status: unknown): StatusMeta {
  return {
    ...UNKNOWN_META,
    rawStatus: String(status),
  };
}

export function getSessionStatusMeta(status: unknown): StatusMeta {
  const key = normalizeStatus(status);
  if (!key) {
    return SESSION_STATUS_MAP.pending;
  }
  return SESSION_STATUS_MAP[key] ?? toFallbackMeta(status);
}

export function getStepStatusMeta(status: unknown): StatusMeta {
  const key = normalizeStatus(status);
  if (!key) {
    return STEP_STATUS_MAP.pending;
  }
  return STEP_STATUS_MAP[key] ?? toFallbackMeta(status);
}
