/**
 * B10 FE display registry (spec §5.1).
 *
 * 分工 (D1): envelope 只承载 machine 位 (display_icon 词表 / read_only /
 * destructive / tool_source); 执行中/完成文案与 icon 组件映射由本模块按
 * function 名生成 (i18n 留位: 文案集中在此单模块, N8).
 *
 * INV-B10-4: 折叠/高亮/badge 策略只信 envelope 显式位; 本 registry 仅供
 * icon/文案, 禁止推断策略位.
 *
 * 文案吸收 session-ui.ts titleByFunction/pickToolDetail — 原函数不删
 * (legacy 分支与既有测试继续用), 新卡走本模块.
 */
import type { LucideIcon } from "lucide-react";
import {
  Brain,
  FilePen,
  FileText,
  Globe,
  MessageSquare,
  Plug,
  Puzzle,
  Search,
  Share2,
  Terminal,
  Wrench,
} from "lucide-react";

import type { ToolEventEnvelopeV1 } from "@/lib/api/types";

/** spec §3.3 受控词表 → lucide 组件. 未知词表值走 fallback 链 (fail-closed). */
const ICON_BY_VOCAB: Record<string, LucideIcon> = {
  file: FileText,
  "file-edit": FilePen,
  terminal: Terminal,
  browser: Globe,
  search: Search,
  message: MessageSquare,
  memory: Brain,
  mcp: Plug,
  skill: Puzzle,
  a2a: Share2,
  generic: Wrench,
};

export interface ToolDisplayEntry {
  icon: LucideIcon;
  verbPending: string; // "正在写入文件"
  verbDone: string;    // "已写入文件"
  detail: (args: Record<string, unknown>) => string | null;
}

// R4#1: resolveToolDisplay 的完整返回合同 (spec §5.1 照抄)
export interface ResolvedToolDisplay {
  icon: LucideIcon;
  title: string;
  detail: string | null;
  collapsedByDefault: boolean; // === (envelope.read_only === true)
  destructive: boolean;        // === (envelope.destructive === true)
  sourceBadge: "MCP" | "Skill" | "A2A" | null;
  sourceTooltip: string | null; // `${category} / ${canonical_name}`
}

function asString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function formatPathTail(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.length > 2 ? `…/${parts.slice(-2).join("/")}` : path;
}

const fileDetail = (args: Record<string, unknown>): string | null => {
  const filepath = asString(args.filepath);
  return filepath ? `文件：${formatPathTail(filepath)}` : null;
};

const shellDetail = (args: Record<string, unknown>): string | null => {
  const command = asString(args.command);
  if (command) return `命令：${command}`;
  const sessionId = asString(args.session_id);
  return sessionId ? `终端会话：${sessionId.slice(0, 8)}` : null;
};

const urlDetail = (args: Record<string, unknown>): string | null => {
  const url = asString(args.url);
  return url ? `目标网页：${url}` : null;
};

const queryDetail = (args: Record<string, unknown>): string | null => {
  const query = asString(args.query);
  return query ? `关键词：${query}` : null;
};

const textDetail = (args: Record<string, unknown>): string | null =>
  asString(args.text) || null;

const genericDetail = (args: Record<string, unknown>): string | null => {
  const primary =
    asString(args.text) || asString(args.query) ||
    asString(args.url) || asString(args.filepath);
  if (primary) return primary;
  const keys = Object.keys(args);
  if (keys.length === 0) return null;
  return keys
    .slice(0, 3)
    .map((k) => {
      const v = args[k];
      const s = typeof v === "string" ? v : JSON.stringify(v);
      return `${k}: ${s && s.length > 40 ? `${s.slice(0, 40)}…` : s}`;
    })
    .join("，");
};

function entry(
  icon: LucideIcon,
  verb: string,
  detail: ToolDisplayEntry["detail"] = genericDetail
): ToolDisplayEntry {
  return { icon, verbPending: `正在${verb}`, verbDone: `已${verb}`, detail };
}

/** 38 canonical 工具 (与 BE tool_display_registry 名单一一对应). */
export const TOOL_DISPLAY_REGISTRY: Record<string, ToolDisplayEntry> = {
  browser_view: entry(Globe, "查看网页内容"),
  browser_navigate: entry(Globe, "访问网页", urlDetail),
  browser_click: entry(Globe, "点击页面元素"),
  browser_input: entry(Globe, "填写页面输入"),
  browser_move_mouse: entry(Globe, "移动鼠标"),
  browser_press_key: entry(Globe, "触发按键"),
  browser_select_option: entry(Globe, "选择页面选项"),
  browser_scroll_up: entry(Globe, "向上滚动页面"),
  browser_scroll_down: entry(Globe, "向下滚动页面"),
  browser_console_exec: entry(Globe, "执行网页脚本"),
  browser_console_view: entry(Globe, "查看控制台输出"),
  browser_restart: entry(Globe, "重启浏览器", urlDetail),
  shell_execute: entry(Terminal, "执行终端命令", shellDetail),
  shell_read_output: entry(Terminal, "读取终端输出", shellDetail),
  shell_wait_process: entry(Terminal, "等待终端进程", shellDetail),
  shell_write_input: entry(Terminal, "写入终端输入", shellDetail),
  shell_kill_process: entry(Terminal, "终止终端进程", shellDetail),
  file_read: entry(FileText, "读取文件", fileDetail),
  file_write: entry(FilePen, "写入文件", fileDetail),
  file_str_replace: entry(FilePen, "替换文件内容", fileDetail),
  file_find_in_content: entry(FileText, "搜索文件内容", fileDetail),
  file_find_by_name: entry(FileText, "查找文件"),
  file_list: entry(FileText, "列出目录"),
  file_view: entry(FileText, "查看多媒体文件", fileDetail),
  message_notify_user: entry(MessageSquare, "发送进度通知", textDetail),
  message_ask_user: entry(MessageSquare, "请求用户回复", textDetail),
  search_web: entry(Search, "搜索资料", queryDetail),
  memory_search: entry(Brain, "检索长期记忆", queryDetail),
  memory_get: entry(Brain, "读取记忆条目"),
  memory_save: entry(Brain, "保存记忆"),
  get_remote_agent_cards: entry(Share2, "获取远程 Agent 列表"),
  call_remote_agent: entry(Share2, "调用远程 Agent"),
  list_mcp_tools: entry(Plug, "列出 MCP 工具"),
  get_mcp_tool: entry(Plug, "加载 MCP 工具"),
  brainstorm_skill: entry(Puzzle, "构思 Skill 蓝图"),
  generate_skill: entry(Puzzle, "生成 Skill"),
  install_skill: entry(Puzzle, "安装 Skill"),
  get_skill_guide: entry(Puzzle, "读取 Skill 指南"),
};

function fallbackTitle(functionName: string, called: boolean): string {
  const readable = functionName ? functionName.replace(/_/g, " ") : "工具调用";
  return called ? `已完成 ${readable}` : `正在调用 ${readable}`;
}

export function resolveToolDisplay(
  envelope: ToolEventEnvelopeV1
): ResolvedToolDisplay {
  const registryEntry = TOOL_DISPLAY_REGISTRY[envelope.function];

  // Fallback 链 (spec §5.1): envelope.display_icon 词表命中 → registry → generic
  const vocabIcon = envelope.display_icon
    ? ICON_BY_VOCAB[envelope.display_icon]
    : undefined;
  const icon = vocabIcon ?? registryEntry?.icon ?? Wrench;

  const called = envelope.status === "called";
  const title = registryEntry
    ? called
      ? registryEntry.verbDone
      : registryEntry.verbPending
    : fallbackTitle(envelope.function, called);

  const args = envelope.args ?? {};
  const detail = registryEntry
    ? registryEntry.detail(args)
    : genericDetail(args);

  // INV-B10-4: 策略位只读 envelope 显式位; null/undefined → 现状行为
  const collapsedByDefault = envelope.read_only === true;
  const destructive = envelope.destructive === true;

  const source = envelope.tool_source?.source;
  const sourceBadge =
    source === "mcp" ? "MCP"
    : source === "skill" ? "Skill"
    : source === "a2a" ? "A2A"
    : null;
  const sourceTooltip =
    envelope.tool_source && sourceBadge !== null
      ? `${envelope.tool_source.category} / ${envelope.tool_source.canonical_name}`
      : null;

  return {
    icon,
    title,
    detail,
    collapsedByDefault,
    destructive,
    sourceBadge,
    sourceTooltip,
  };
}

/** R3#3: override Map key — page 组件跨 session 存活, 裸 tool_call_id 会跨 session 碰撞. */
export function toolCardOverrideKey(sessionId: string, toolCallId: string): string {
  return `${sessionId}:${toolCallId}`;
}
