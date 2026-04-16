// 记忆管理相关的小工具与常量，供 memory-management.tsx / memory-detail-drawer.tsx 共用，
// 避免两处维护同一份 source label 字典。

export const MEMORY_SOURCE_LABEL: Record<string, string> = {
  session_flush: "对话自动记录",
  file: "文件",
  manual: "手动",
};

export function memorySourceLabel(source: string): string {
  return MEMORY_SOURCE_LABEL[source] ?? source;
}

export const MEMORY_SOURCE_OPTIONS: ReadonlyArray<{
  value: string;
  label: string;
}> = [
  { value: "", label: "全部来源" },
  { value: "session_flush", label: "对话自动记录 (session_flush)" },
  { value: "file", label: "文件 (file)" },
  { value: "manual", label: "手动 (manual)" },
];

export function truncateText(text: string, max = 200): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max)}…`;
}

export function formatRelativeTime(
  value: string | null | undefined,
): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const diffMs = Date.now() - date.getTime();
  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} 天前`;
  return date.toLocaleString();
}
