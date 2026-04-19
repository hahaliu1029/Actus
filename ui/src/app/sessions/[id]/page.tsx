"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  CircleDashed,
  Loader2,
  MessageCircleQuestion,
  PanelRightClose,
  PanelRightOpen,
  XCircle,
} from "lucide-react";

import { ChatInput } from "@/components/chat-input";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { SessionHeader } from "@/components/session-header";
import { ToolConfirmationCard } from "@/components/tool-confirmation-card";
import { StatusIndicator } from "@/components/status-indicator";
import { SessionTaskDock } from "@/components/session-task-dock";
import { WorkbenchPanel } from "@/components/workbench-panel";
import { useIsMobile } from "@/hooks/use-mobile";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { sessionApi } from "@/lib/api/session";
import type { FileInfo } from "@/lib/api/types";
import {
  deriveSessionProgressSummary,
  deriveWorkbenchSnapshots,
  formatFileSize,
  formatRelativeTime,
  getFilePreviewKind,
  getSessionEventStableKey,
  getToolDisplayCopy,
  normalizeMessageAttachments,
} from "@/lib/session-ui";
import { getSessionStatusMeta } from "@/lib/status-copy";
import { cn } from "@/lib/utils";
import { normalizeUnixSeconds } from "@/lib/takeover/normalize";
import { useSessionStore } from "@/lib/store/session-store";
import { useTransferStore } from "@/lib/store/transfer-store";
import { useUIStore } from "@/lib/store/ui-store";

type SessionEvent = {
  event: string;
  data: Record<string, unknown>;
};

type SearchResultCard = {
  url: string;
  title: string;
  snippet: string;
};

type ToolVisualContent = {
  screenshots: Array<{ src: string; title: string; filepath: string }>;
  searchResults: SearchResultCard[];
  filepath: string | null;
  mcpResult: string | null;
  mcpAttachments: string[] | null;
};

type TakeoverMeta = {
  takeoverId: string | null;
  takeoverScope: "shell" | "browser" | null;
  takeoverExpiresAt: number | null;
};

function renderMessageAttachments(
  attachments: FileInfo[] | undefined,
  onPreviewFile: (file: FileInfo) => void
) {
  if (!attachments || attachments.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 grid gap-2 sm:grid-cols-2">
      {attachments.map((file) => (
        <button
          key={file.id}
          className="flex min-w-0 items-center justify-between rounded-xl border border-border bg-muted px-3 py-2 text-left text-xs hover:border-border-strong hover:bg-card"
          onClick={() => onPreviewFile(file)}
        >
          <span className="truncate font-medium text-foreground/85">{file.filename}</span>
          <span className="ml-2 shrink-0 text-muted-foreground">{formatFileSize(file.size)}</span>
        </button>
      ))}
    </div>
  );
}

function renderStepStatusIcon(status: string) {
  if (status === "completed") {
    return <CheckCircle2 size={16} className="text-emerald-500" />;
  }
  if (status === "failed") {
    return <XCircle size={16} className="text-red-500" />;
  }
  if (status === "running" || status === "started") {
    return <Loader2 size={16} className="animate-spin text-amber-500" />;
  }
  return <CircleDashed size={16} className="animate-pulse text-muted-foreground" />;
}

function getEventTime(eventData: Record<string, unknown>) {
  return formatRelativeTime(eventData.created_at);
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

function getPathTail(path: string): string {
  const normalized = path.split("?")[0]?.split("#")[0] || path;
  const parts = normalized.split("/");
  return parts[parts.length - 1] || path;
}

function shouldUseSandboxFile(file: FileInfo): boolean {
  return !file.key && Boolean(file.filepath);
}

function toSearchThumbnail(url: string, width: number): string {
  return `https://s.wordpress.com/mshots/v1/${encodeURIComponent(url)}?w=${width}`;
}

function toImageProxyUrl(url: string): string {
  return `/api/image-proxy?url=${encodeURIComponent(url)}`;
}

function toDisplayImageUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) {
    return toImageProxyUrl(url);
  }
  return url;
}

function isSandboxDestroyed(events: SessionEvent[]): boolean {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!event || event.event !== "sandbox_state_changed") {
      continue;
    }
    const newState = String(event.data.new_state || "");
    if (newState === "destroyed") {
      return true;
    }
  }
  return false;
}

function deriveTakeoverMeta(events: SessionEvent[]): TakeoverMeta {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!event || event.event !== "control") {
      continue;
    }
    const takeoverId = String(event.data.takeover_id || "").trim() || null;
    const rawScope = String(event.data.scope || "").trim();
    const takeoverScope =
      rawScope === "shell" || rawScope === "browser" ? rawScope : null;
    const takeoverExpiresAt = normalizeUnixSeconds(event.data.expires_at);
    if (takeoverId || takeoverScope || takeoverExpiresAt != null) {
      return {
        takeoverId,
        takeoverScope,
        takeoverExpiresAt,
      };
    }
  }

  return {
    takeoverId: null,
    takeoverScope: null,
    takeoverExpiresAt: null,
  };
}

function deriveLatestSkillConfirmationPendingAction(
  events: SessionEvent[],
  sessionStatus: string | undefined
): "generate" | "install" | null {
  if (sessionStatus !== "waiting") {
    return null;
  }

  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!event || event.event !== "wait") {
      continue;
    }
    const pendingAction = String(event.data.pending_action || "").trim();
    if (pendingAction === "generate" || pendingAction === "install") {
      return pendingAction;
    }
  }

  return null;
}

function parseToolVisual(eventData: Record<string, unknown>): ToolVisualContent {
  const content = asRecord(eventData.content);
  const args = asRecord(eventData.args);
  const toolName = String(eventData.name || "");
  const functionName = String(eventData.function || "");

  const screenshots: Array<{ src: string; title: string; filepath: string }> = [];
  const screenshot = content.screenshot;
  if (typeof screenshot === "string" && screenshot.trim()) {
    screenshots.push({
      src: toDisplayImageUrl(screenshot),
      title: "网页截图",
      filepath: screenshot,
    });
  }

  const searchResults: SearchResultCard[] = [];
  const rawResults = content.results;
  if (Array.isArray(rawResults)) {
    rawResults.forEach((item) => {
      const entry = asRecord(item);
      const url = typeof entry.url === "string" ? entry.url : "";
      if (!url) {
        return;
      }
      const title = typeof entry.title === "string" ? entry.title : url;
      const snippet = typeof entry.snippet === "string" ? entry.snippet : "";
      searchResults.push({ url, title, snippet });
    });
  }

  const filepath =
    toolName === "file" &&
    /write|append|edit|create|replace|move|copy/i.test(functionName) &&
    typeof args.filepath === "string"
      ? args.filepath
      : null;

  // MCP/A2A 工具调用结果
  let mcpResult: string | null = null;
  let mcpAttachments: string[] | null = null;
  if ((toolName === "mcp" || toolName === "a2a" || toolName === "skill") && content) {
    const rawResult = (toolName === "a2a")
      ? (content as Record<string, unknown>).a2a_result
      : (toolName === "skill")
        ? (content as Record<string, unknown>).skill_result
        : (content as Record<string, unknown>).result;
    if (rawResult !== undefined && rawResult !== null) {
      if (typeof rawResult === "object" && rawResult !== null && !Array.isArray(rawResult)) {
        const obj = rawResult as Record<string, unknown>;
        // 结构化结果：提取 result 文本和 attachments
        if (typeof obj.result === "string") {
          mcpResult = obj.result;
          if (Array.isArray(obj.attachments) && obj.attachments.length > 0) {
            mcpAttachments = obj.attachments.filter((a): a is string => typeof a === "string");
          }
        } else {
          mcpResult = JSON.stringify(rawResult, null, 2);
        }
      } else {
        mcpResult = typeof rawResult === "string" ? rawResult : JSON.stringify(rawResult, null, 2);
      }
    }
  }

  return { screenshots, searchResults, filepath, mcpResult, mcpAttachments };
}

/** 可展开/折叠的工具详情文本 */
function ExpandableToolDetail({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <p
      className={cn(
        "mt-1 text-xs text-muted-foreground cursor-pointer hover:text-foreground/70 transition-colors",
        expanded ? "whitespace-pre-wrap break-words" : "truncate"
      )}
      onClick={() => setExpanded(!expanded)}
      title={expanded ? undefined : text}
    >
      {text}
    </p>
  );
}

/** 流式纯文本渲染时，清理 XML 标签为可读文本 */
function stripXmlTags(text: string): string {
  let result = text;
  result = result.replace(/<\/?think(?:ing)?>/gi, "");
  result = result.replace(/<\/?tool_code>/g, "");
  result = result.replace(/<tool\b[^>]*>/g, "");
  result = result.replace(/<\/tool>/g, "");
  return result;
}

/**
 * 从消息文本中提取嵌入的结构化 JSON 结果。
 * LLM 有时会在回复中直接包含 {"success":..., "result":"...", "attachments":[...]} 格式的原始 JSON，
 * 此函数将其拆解为纯文本 + 附件路径，以便前端正常渲染。
 */
function extractEmbeddedJsonResult(message: string): {
  text: string;
  embeddedAttachments: string[];
} {
  // 从消息末尾向前查找最后一个 `{` 开头的 JSON 块
  const lastBrace = message.lastIndexOf("}");
  if (lastBrace === -1) return { text: message, embeddedAttachments: [] };

  // 尝试从不同的 `{` 位置解析 JSON，从后往前查找有效的顶层 JSON 对象
  let searchFrom = lastBrace;
  while (searchFrom >= 0) {
    const openBrace = message.lastIndexOf("{", searchFrom);
    if (openBrace === -1) break;

    const candidate = message.slice(openBrace, lastBrace + 1).trim();
    try {
      const parsed = JSON.parse(candidate);
      if (typeof parsed === "object" && parsed !== null) {
        // Support both "result" (MCP/skill tools) and "message" (summarizer) keys
        const resultText =
          typeof parsed.result === "string" ? parsed.result
          : typeof parsed.message === "string" ? parsed.message
          : null;
        if (resultText !== null) {
          // Strip trailing code fence markers (e.g. ```json) from prefix —
          // LLMs sometimes wrap JSON output in markdown code blocks
          const prefix = message.slice(0, openBrace).trim().replace(/```\w*\s*$/, "").trim();
          const text = prefix ? `${prefix}\n\n${resultText}` : resultText;
          const attachments: string[] = Array.isArray(parsed.attachments)
            ? parsed.attachments.filter((a: unknown): a is string => typeof a === "string")
            : [];
          return { text, embeddedAttachments: attachments };
        }
      }
    } catch {
      // 该位置不是有效 JSON，继续向前查找
    }
    searchFrom = openBrace - 1;
  }

  return { text: message, embeddedAttachments: [] };
}

function renderEventItem(
  event: SessionEvent,
  index: number,
  sessionFiles: FileInfo[],
  onPreviewFile: (file: FileInfo) => void,
  onPreviewFilePath: (filepath: string) => void,
  onPreviewImage: (src: string, title?: string) => void,
  streamingAssistantEventId?: string | null
) {
  const eventKey = getSessionEventStableKey(event, index);

  if (event.event === "tool_confirmation") {
    return <ToolConfirmationCard key={eventKey} data={event.data as Parameters<typeof ToolConfirmationCard>[0]["data"]} />;
  }

  if (event.event === "message") {
    const role = String(event.data.role || "assistant");
    if (role === "system") {
      return null;
    }
    const message = String(event.data.message || "");
    const isPartial = Boolean(event.data.partial);
    const attachments = normalizeMessageAttachments(event.data.attachments, sessionFiles);
    const timeText = getEventTime(event.data);
    const isStreamingAssistant = role === "assistant" && streamingAssistantEventId === eventKey;

    if (role === "user") {
      return (
        <div key={eventKey} className="mt-4 flex flex-col items-end">
          <div className="mb-1 text-xs text-muted-foreground">{timeText}</div>
          <div className="max-w-[90%] rounded-2xl border border-border bg-card px-4 py-3 text-sm text-foreground/85 shadow-[var(--shadow-subtle)]">
            <p className="whitespace-pre-wrap leading-7">{message || "（空消息）"}</p>
            {renderMessageAttachments(attachments, onPreviewFile)}
          </div>
        </div>
      );
    }

    // 提取嵌入的结构化 JSON 结果（如 skill/a2a 工具返回的 {"success","result","attachments"}）
    // 对 partial 消息也尝试提取——不完整的 JSON 会安全地回退为原文
    const extracted = extractEmbeddedJsonResult(message);
    const displayMessage = extracted ? extracted.text : message;
    const embeddedAttachments = extracted?.embeddedAttachments || [];

    return (
      <div key={eventKey} className="mt-4">
        <div className="mb-1 flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-sm font-semibold text-foreground/85">
            <Bot size={16} />
            Actus
          </div>
          <span className="text-xs text-muted-foreground">{timeText}</span>
        </div>
        <div className="rounded-2xl border border-border bg-card px-4 py-3 text-sm text-foreground/85 shadow-[var(--shadow-subtle)]">
          <MarkdownRenderer content={isPartial ? (stripXmlTags(displayMessage) || "（空消息）") : (displayMessage || "（空消息）")} />
          {renderMessageAttachments(attachments, onPreviewFile)}
          {embeddedAttachments.length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {embeddedAttachments.map((filePath) => (
                <button
                  key={filePath}
                  className="inline-flex items-center rounded-lg border border-blue-200 bg-blue-50 px-2 py-1 text-xs text-blue-700 hover:bg-blue-100 dark:border-blue-500/30 dark:bg-blue-500/10 dark:text-blue-400 dark:hover:bg-blue-500/20"
                  onClick={() => onPreviewFilePath(filePath)}
                >
                  📎 {getPathTail(filePath)}
                </button>
              ))}
            </div>
          ) : null}
          {isStreamingAssistant || isPartial ? (
            <div className="mt-2 inline-flex items-center gap-1 text-xs text-amber-600">
              <Loader2 size={12} className="animate-spin" />
              流式输出中
            </div>
          ) : null}
        </div>
      </div>
    );
  }

  if (event.event === "plan") {
    const steps = (event.data.steps || []) as Array<Record<string, unknown>>;
    const done = steps.filter((step) => String(step.status || "") === "completed").length;

    return (
      <div key={eventKey} className="mt-3 rounded-xl border border-border bg-card px-3 py-2 text-sm text-foreground/85">
        <div className="flex items-center justify-between gap-2">
          <p className="font-medium">进度已更新</p>
          <span className="text-xs text-muted-foreground">
            {done}/{steps.length}
          </span>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">完整步骤请查看底部任务摘要。</p>
      </div>
    );
  }

  if (event.event === "step") {
    return (
      <div key={eventKey} className="mt-3 flex items-start gap-2 rounded-xl border border-border bg-card px-3 py-2">
        <div className="mt-[3px]">{renderStepStatusIcon(String(event.data.status || "pending"))}</div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm text-foreground/85">{String(event.data.description || "执行步骤")}</p>
        </div>
        <span className="shrink-0 text-xs text-muted-foreground">{getEventTime(event.data)}</span>
      </div>
    );
  }

  if (event.event === "tool") {
    const display = getToolDisplayCopy(event.data);
    if (display.kind === "ask") {
      return (
        <div key={eventKey} className="mt-4">
          <div className="mb-1 flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-sm font-semibold text-foreground/85">
              <Bot size={16} />
              Actus
            </div>
            <span className="text-xs text-muted-foreground">{getEventTime(event.data)}</span>
          </div>
          <div className="rounded-2xl border-l-[3px] border-l-primary border border-border bg-card px-4 py-3 shadow-[var(--shadow-subtle)]">
            <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-primary">
              <MessageCircleQuestion size={14} />
              {display.title}
            </div>
            <div className="text-sm text-foreground/85">
              <MarkdownRenderer content={display.detail || "请补充下一步操作信息"} />
            </div>
          </div>
        </div>
      );
    }
    if (display.kind === "progress") {
      return (
        <div key={eventKey} className="mt-4">
          <div className="mb-1 flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-sm font-semibold text-foreground/85">
              <Bot size={16} />
              Actus
            </div>
            <span className="text-xs text-muted-foreground">{getEventTime(event.data)}</span>
          </div>
          <div className="rounded-2xl border border-border bg-card px-4 py-3 text-sm text-foreground/85 shadow-[var(--shadow-subtle)]">
            <MarkdownRenderer content={display.detail || "请补充下一步操作信息"} />
          </div>
        </div>
      );
    }

    const isRunning = event.data.status !== "called";
    const statusText = isRunning ? "执行中" : "已完成";
    const visual = parseToolVisual(event.data);

    return (
      <div key={eventKey} className="mt-3 rounded-xl border border-border bg-card px-3 py-2">
        <div className="flex items-center justify-between gap-2">
          <p className="truncate text-sm font-medium text-foreground/85">{display.title}</p>
          <span
            className={cn(
              "inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-xs",
              isRunning
                ? "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400"
                : "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400"
            )}
          >
            {isRunning ? <Loader2 size={12} className="animate-spin" /> : <CheckCircle2 size={12} />}
            {statusText}
          </span>
        </div>
        {display.detail ? <ExpandableToolDetail text={display.detail} /> : null}

        {visual.screenshots.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-2">
            {visual.screenshots.map((shot) => (
              <button
                key={shot.src}
                className="overflow-hidden rounded-lg border border-border"
                onClick={() => onPreviewImage(shot.src, shot.title)}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={shot.src} alt={shot.title} className="h-24 w-36 object-cover" />
              </button>
            ))}
          </div>
        ) : null}

        {visual.searchResults.length > 0 ? (
          <div className="mt-2 grid gap-2 sm:grid-cols-2">
            {visual.searchResults.slice(0, 4).map((item) => (
              <button
                key={item.url}
                className="flex min-w-0 gap-2 rounded-lg border border-border bg-muted p-2 text-left hover:bg-card"
                onClick={() =>
                  onPreviewImage(
                    toDisplayImageUrl(toSearchThumbnail(item.url, 1200)),
                    item.title
                  )
                }
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={toDisplayImageUrl(toSearchThumbnail(item.url, 360))}
                  alt={item.title}
                  className="h-16 w-24 shrink-0 rounded-md border border-border object-cover"
                />
                <div className="min-w-0">
                  <p className="truncate text-xs font-medium text-foreground/85">{item.title}</p>
                  <p className="line-clamp-2 text-[11px] text-muted-foreground">{item.snippet || item.url}</p>
                </div>
              </button>
            ))}
          </div>
        ) : null}

        {!isRunning && visual.filepath ? (
          <button
            className="mt-2 inline-flex items-center rounded-lg border border-blue-200 bg-blue-50 px-2 py-1 text-xs text-blue-700 hover:bg-blue-100 dark:border-blue-500/30 dark:bg-blue-500/10 dark:text-blue-400 dark:hover:bg-blue-500/20"
            onClick={() => onPreviewFilePath(visual.filepath!)}
          >
            查看文件：{getPathTail(visual.filepath)}
          </button>
        ) : null}

        {!isRunning && visual.mcpResult ? (
          <details className="mt-2">
            <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
              查看调用结果
            </summary>
            <div className="mt-1 max-h-60 overflow-auto rounded-lg bg-muted p-2">
              <MarkdownRenderer content={visual.mcpResult} className="text-xs leading-6 text-muted-foreground" />
            </div>
          </details>
        ) : null}

        {!isRunning && visual.mcpAttachments && visual.mcpAttachments.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-2">
            {visual.mcpAttachments.map((filePath) => (
              <button
                key={filePath}
                className="inline-flex items-center rounded-lg border border-blue-200 bg-blue-50 px-2 py-1 text-xs text-blue-700 hover:bg-blue-100 dark:border-blue-500/30 dark:bg-blue-500/10 dark:text-blue-400 dark:hover:bg-blue-500/20"
                onClick={() => onPreviewFilePath(filePath)}
              >
                📎 {getPathTail(filePath)}
              </button>
            ))}
          </div>
        ) : null}
      </div>
    );
  }

  if (event.event === "error") {
    return (
      <div
        key={eventKey}
        className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400"
      >
        错误：{String(event.data.error || "未知错误")}
      </div>
    );
  }

  // D5: Health event — watchdog恢复/终止提示
  if (event.event === "health") {
    const data = event.data as Record<string, unknown>;
    const status = String(data.status || "");
    const reason = String(data.reason || "");
    const lastNode = typeof data.last_node === "string" ? data.last_node : null;
    const idleSeconds = typeof data.idle_seconds === "number" ? data.idle_seconds : null;
    const metrics = data.metrics && typeof data.metrics === "object"
      ? (data.metrics as Record<string, unknown>)
      : null;

    // DEGRADED: 黄色警告 (恢复中)
    // TERMINATING: 红色警告 (即将终止)
    // TERMINATED: 红色终态 (已终止 + 指标摘要)
    // HEALTHY: 不渲染 (正常态无需提示)
    if (status === "healthy") {
      return null;
    }

    const toneClass =
      status === "degraded"
        ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300"
        : "border-red-200 bg-red-50 text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400";

    const title =
      status === "degraded"
        ? "执行正在恢复"
        : status === "terminating"
          ? "执行即将终止"
          : status === "terminated"
            ? "执行已终止"
            : "执行状态";

    return (
      <div key={eventKey} className={`mt-3 rounded-xl border px-3 py-2 text-sm ${toneClass}`}>
        <div className="flex items-center gap-2 font-medium">
          <AlertCircle size={14} />
          {title}
        </div>
        {reason ? <p className="mt-1 text-xs">{reason}</p> : null}
        {(lastNode || idleSeconds !== null) && (
          <p className="mt-1 text-xs opacity-80">
            {lastNode ? `最后活跃节点: ${lastNode}` : null}
            {lastNode && idleSeconds !== null ? " · " : null}
            {idleSeconds !== null ? `空闲 ${idleSeconds.toFixed(1)}s` : null}
          </p>
        )}
        {metrics ? (
          <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs opacity-80 sm:grid-cols-4">
            {typeof metrics.tool_calls_total === "number" && (
              <span>工具调用 {String(metrics.tool_calls_total)}</span>
            )}
            {typeof metrics.tool_success_rate === "number" && (
              <span>成功率 {(Number(metrics.tool_success_rate) * 100).toFixed(0)}%</span>
            )}
            {typeof metrics.llm_calls_total === "number" && (
              <span>LLM {String(metrics.llm_calls_total)}</span>
            )}
            {typeof metrics.steps_completed === "number" && (
              <span>步骤 {String(metrics.steps_completed)}</span>
            )}
          </div>
        ) : null}
      </div>
    );
  }

  const fallbackText =
    (typeof event.data.text === "string" && event.data.text) ||
    (typeof event.data.message === "string" && event.data.message);
  if (fallbackText) {
    return (
      <div key={eventKey} className="mt-3 rounded-xl border border-border bg-card px-3 py-2 text-sm text-foreground/85">
        {fallbackText}
      </div>
    );
  }

  return null;
}

export default function SessionPage() {
  const params = useParams<{ id: string }>();
  const sessionId = params?.id;

  const router = useRouter();
  const createSession = useSessionStore((state) => state.createSession);
  const currentSession = useSessionStore((state) => state.currentSession);
  const currentSessionFiles = useSessionStore((state) => state.currentSessionFiles);
  const setActiveSession = useSessionStore((state) => state.setActiveSession);
  const fetchSessionById = useSessionStore((state) => state.fetchSessionById);
  const fetchSessionFiles = useSessionStore((state) => state.fetchSessionFiles);
  const recoverSession = useSessionStore((state) => state.recoverSession);
  const downloadFile = useSessionStore((state) => state.downloadFile);
  const downloadSandboxFile = useSessionStore((state) => state.downloadSandboxFile);
  const isLoadingCurrentSession = useSessionStore((state) => state.isLoadingCurrentSession);
  const isChatting = useSessionStore((state) => state.isChatting);
  const chatSessionId = useSessionStore((state) => state.chatSessionId);
  const setMessage = useUIStore((state) => state.setMessage);
  const addTransferTask = useTransferStore((s) => s.addTask);
  const updateTransferProgress = useTransferStore((s) => s.updateProgress);
  const completeTransferTask = useTransferStore((s) => s.completeTask);
  const failTransferTask = useTransferStore((s) => s.failTask);
  const isMobile = useIsMobile();

  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewTitle, setPreviewTitle] = useState("预览");
  const [previewKind, setPreviewKind] = useState<ReturnType<typeof getFilePreviewKind>>("unsupported");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewTextContent, setPreviewTextContent] = useState("");
  const [previewBlobUrl, setPreviewBlobUrl] = useState<string | null>(null);
  const [previewFile, setPreviewFile] = useState<FileInfo | null>(null);
  const [sandboxDownloadPath, setSandboxDownloadPath] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<{
    src: string;
    title: string;
  } | null>(null);
  const [desktopWorkbenchVisible, setDesktopWorkbenchVisible] = useState(true);
  const [mobileWorkbenchOpen, setMobileWorkbenchOpen] = useState(false);

  const eventScrollRef = useRef<HTMLDivElement | null>(null);

  const resetPreviewState = useCallback(() => {
    setPreviewError(null);
    setPreviewTextContent("");
    setPreviewKind("unsupported");
    setPreviewLoading(false);
    setSandboxDownloadPath(null);
  }, []);

  useEffect(() => {
    return () => {
      if (previewBlobUrl) {
        URL.revokeObjectURL(previewBlobUrl);
      }
    };
  }, [previewBlobUrl]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    setActiveSession(sessionId);
  }, [sessionId, setActiveSession]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    void fetchSessionById(sessionId);
    void fetchSessionFiles(sessionId);
  }, [fetchSessionById, fetchSessionFiles, sessionId]);

  const visibleSession = useMemo(() => {
    if (!currentSession || currentSession.session_id !== sessionId) {
      return null;
    }
    return currentSession;
  }, [currentSession, sessionId]);

  const eventList = useMemo(() => {
    return (visibleSession?.events || []) as SessionEvent[];
  }, [visibleSession]);
  const workbenchSnapshots = useMemo(
    () => deriveWorkbenchSnapshots(eventList),
    [eventList]
  );
  const takeoverMeta = useMemo(
    () => deriveTakeoverMeta(eventList),
    [eventList]
  );
  const progressSummary = useMemo(
    () => deriveSessionProgressSummary(eventList),
    [eventList]
  );
  const skillConfirmationPendingAction = useMemo(
    () =>
      deriveLatestSkillConfirmationPendingAction(
        eventList,
        visibleSession?.status
      ),
    [eventList, visibleSession?.status]
  );
  const currentStatusMeta = useMemo(
    () => getSessionStatusMeta(visibleSession?.status || "pending"),
    [visibleSession?.status]
  );
  const sandboxDestroyed = useMemo(
    () => isSandboxDestroyed(eventList),
    [eventList]
  );
  const workbenchVisible = (!isMobile && desktopWorkbenchVisible) || (isMobile && mobileWorkbenchOpen);
  const isCurrentSessionStreaming = Boolean(sessionId) && isChatting && chatSessionId === sessionId;
  const sessionRunning =
    isCurrentSessionStreaming ||
    visibleSession?.status === "running" ||
    visibleSession?.status === "waiting";

  useEffect(() => {
    if (!isMobile) {
      return;
    }
    if (visibleSession?.status !== "takeover") {
      return;
    }
    if (takeoverMeta.takeoverScope === "browser") {
      return;
    }
    setMobileWorkbenchOpen(true);
  }, [isMobile, takeoverMeta.takeoverScope, visibleSession?.status]);

  useEffect(() => {
    if (!sessionId || !sessionRunning) {
      return;
    }

    let stopped = false;
    const refresh = () => {
      if (stopped) {
        return;
      }
      // SSE 流已经实时推送事件，轮询 session 会导致状态交替抖动（闪烁），
      // 因此当前会话处于流式活跃期间只轮询文件列表
      if (!isCurrentSessionStreaming) {
        void fetchSessionById(sessionId, { silent: true });
      }
      void fetchSessionFiles(sessionId, { silent: true });
    };

    // 任务运行期间做轻量轮询，确保进度与文件列表持续更新
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [
    fetchSessionById,
    fetchSessionFiles,
    sessionId,
    sessionRunning,
    isCurrentSessionStreaming,
  ]);

  // E2: SSE 状态恢复 — visibilitychange 触发
  useEffect(() => {
    if (!sessionId) return;

    let debounceTimer: ReturnType<typeof setTimeout> | null = null;

    const handleVisibilityChange = () => {
      if (document.visibilityState !== "visible") return;

      const session = useSessionStore.getState().currentSession;
      if (!session || session.session_id !== sessionId) return;

      // 已 COMPLETED 不触发
      if (session.status === "completed") return;

      // 正在流式中（stream 会实时推送），不需要恢复
      const { isChatting, chatSessionId } = useSessionStore.getState();
      if (isChatting && chatSessionId === sessionId) return;

      // 1s debounce
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        void recoverSession(sessionId);
      }, 1000);
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (debounceTimer) clearTimeout(debounceTimer);
    };
  }, [sessionId, recoverSession]);

  const streamingAssistantEventId = useMemo(() => {
    if (!isCurrentSessionStreaming) {
      return null;
    }
    for (let index = eventList.length - 1; index >= 0; index -= 1) {
      const event = eventList[index];
      if (event?.event !== "message") {
        continue;
      }
      if (String(event.data.role || "assistant") !== "assistant") {
        continue;
      }
      return getSessionEventStableKey(event, index);
    }
    return null;
  }, [eventList, isCurrentSessionStreaming]);

  useEffect(() => {
    const node = eventScrollRef.current;
    if (!node) {
      return;
    }
    node.scrollTo({
      top: node.scrollHeight,
      behavior:
        eventList.at(-1)?.event === "message" &&
        Boolean(eventList.at(-1)?.data?.partial)
          ? "auto"
          : "smooth",
    });
  }, [eventList]);

  // Retry watcher: when global TransferPanel retries a download task,
  // detect the status flip back to "pending" and re-initiate the download.
  useEffect(() => {
    const retryDownload = async (taskId: string, sourceRef: string, filename: string) => {
      const signal = useTransferStore.getState().getSignal(taskId);
      if (!signal) return;

      try {
        const isSandbox = sourceRef.includes("/");
        let blob: Blob;

        if (isSandbox && sessionId) {
          blob = await downloadSandboxFile(sessionId, sourceRef, {
            signal,
            onProgress: (loaded: number, total: number) => updateTransferProgress(taskId, loaded, total),
          });
        } else {
          blob = await downloadFile(sourceRef, {
            signal,
            onProgress: (loaded: number, total: number) => updateTransferProgress(taskId, loaded, total),
          });
        }

        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        completeTransferTask(taskId);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        if (typeof error === "object" && error !== null && "name" in error && (error as { name: string }).name === "CanceledError") return;
        failTransferTask(taskId, error instanceof Error ? error.message : "下载失败");
      }
    };

    const unsubscribe = useTransferStore.subscribe((state, prevState) => {
      for (const [id, task] of Object.entries(state.tasks)) {
        const prev = prevState.tasks[id];
        if (
          task.type === "download" &&
          task.status === "pending" &&
          task.sourceRef &&
          prev &&
          (prev.status === "failed" || prev.status === "cancelled")
        ) {
          void retryDownload(id, task.sourceRef, task.filename);
        }
      }
    });

    return unsubscribe;
  }, [sessionId, downloadFile, downloadSandboxFile, updateTransferProgress, completeTransferTask, failTransferTask]);

  const closePreview = useCallback(() => {
    setPreviewOpen(false);
    setPreviewFile(null);
    if (previewBlobUrl) {
      URL.revokeObjectURL(previewBlobUrl);
      setPreviewBlobUrl(null);
    }
    resetPreviewState();
  }, [previewBlobUrl, resetPreviewState]);

  const openFilePreview = useCallback(
    async (file: FileInfo) => {
      if (!sessionId) {
        return;
      }

      resetPreviewState();
      setPreviewFile(file);
      setPreviewTitle(file.filename);
      setPreviewOpen(true);
      setPreviewLoading(true);

      if (previewBlobUrl) {
        URL.revokeObjectURL(previewBlobUrl);
        setPreviewBlobUrl(null);
      }

      const nextKind = getFilePreviewKind(file);
      setPreviewKind(nextKind);

      try {
        if (nextKind === "text") {
          const result = await sessionApi.viewFile(sessionId, { filepath: file.filepath });
          setPreviewTextContent(result.content || "文件为空。");
          return;
        }

        if (nextKind === "image" || nextKind === "pdf") {
          const blob = shouldUseSandboxFile(file)
            ? await downloadSandboxFile(sessionId, file.filepath)
            : await downloadFile(file.id);
          const url = URL.createObjectURL(blob);
          setPreviewBlobUrl(url);
          return;
        }
      } catch (error) {
        setPreviewError(error instanceof Error ? error.message : "文件预览失败");
      } finally {
        setPreviewLoading(false);
      }

      setPreviewLoading(false);
    },
    [downloadFile, downloadSandboxFile, previewBlobUrl, resetPreviewState, sessionId]
  );

  const openFilePathPreview = useCallback(
    async (filepath: string) => {
      const target = currentSessionFiles.find(
        (file) => file.filepath === filepath || file.filename === getPathTail(filepath)
      );
      if (target) {
        await openFilePreview(target);
        return;
      }

      if (!sessionId) {
        return;
      }

      // 文件不在当前列表中，刷新文件列表后重新查找
      await fetchSessionFiles(sessionId, { silent: true });
      const refreshed = useSessionStore.getState().currentSessionFiles;
      const retryTarget = refreshed.find(
        (file) => file.filepath === filepath || file.filename === getPathTail(filepath)
      );
      if (retryTarget) {
        await openFilePreview(retryTarget);
        return;
      }

      // 仍未找到：根据扩展名判断文件类型，直接从沙箱读取/下载
      resetPreviewState();
      const filename = getPathTail(filepath);
      setPreviewTitle(filename);
      setPreviewOpen(true);
      setPreviewLoading(true);

      const nextKind = getFilePreviewKind({ filename });
      setPreviewKind(nextKind);

      try {
        if (nextKind === "pdf" || nextKind === "image") {
          // 二进制可预览文件：从沙箱下载 blob 后用 iframe/img 展示
          const blob = await downloadSandboxFile(sessionId, filepath);
          const url = URL.createObjectURL(blob);
          setPreviewBlobUrl(url);
        } else if (nextKind === "text") {
          const result = await sessionApi.viewFile(sessionId, { filepath });
          setPreviewTextContent(result.content || "文件为空。");
        }
        // unsupported 类型：不读取内容，显示提示 + 下载按钮
      } catch (error) {
        setPreviewError(error instanceof Error ? error.message : "文件预览失败");
      } finally {
        setPreviewLoading(false);
      }

      // 为非文本文件设置 sandboxFilepath，供下载按钮使用
      if (nextKind !== "text") {
        setSandboxDownloadPath(filepath);
      }
    },
    [currentSessionFiles, downloadSandboxFile, fetchSessionFiles, openFilePreview, resetPreviewState, sessionId]
  );

  const handleFileDownload = useCallback(
    async (file: FileInfo) => {
      const useSandboxFile = shouldUseSandboxFile(file);
      const sourceRef = useSandboxFile ? file.filepath : file.id;
      // Dedup: skip if active transfer exists for this file
      const existingTasks = useTransferStore.getState().tasks;
      const hasActive = Object.values(existingTasks).some(
        (t) => t.sourceRef === sourceRef && (t.status === "pending" || t.status === "transferring")
      );
      if (hasActive) return;

      const { taskId, signal } = addTransferTask({
        type: "download",
        filename: file.filename,
        totalBytes: file.size,
        sourceRef,
      });

      try {
        const blob = useSandboxFile
          ? await downloadSandboxFile(sessionId!, file.filepath, {
              signal,
              onProgress: (loaded: number, total: number) =>
                updateTransferProgress(taskId, loaded, total),
            })
          : await downloadFile(file.id, {
              signal,
              onProgress: (loaded: number, total: number) =>
                updateTransferProgress(taskId, loaded, total),
            });

        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = file.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        completeTransferTask(taskId);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        if (typeof error === "object" && error !== null && "name" in error && (error as { name: string }).name === "CanceledError") return;
        failTransferTask(taskId, error instanceof Error ? error.message : "下载失败");
      }
    },
    [addTransferTask, completeTransferTask, downloadFile, downloadSandboxFile, failTransferTask, sessionId, updateTransferProgress]
  );

  const handleSandboxDownload = useCallback(
    async (filepath: string) => {
      if (!sessionId) return;

      const existingTasks = useTransferStore.getState().tasks;
      const hasActive = Object.values(existingTasks).some(
        (t) => t.sourceRef === filepath && (t.status === "pending" || t.status === "transferring")
      );
      if (hasActive) return;

      const { taskId, signal } = addTransferTask({
        type: "download",
        filename: getPathTail(filepath),
        totalBytes: 0,
        sourceRef: filepath,
      });

      try {
        const blob = await downloadSandboxFile(sessionId, filepath, {
          signal,
          onProgress: (loaded: number, total: number) => updateTransferProgress(taskId, loaded, total),
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = getPathTail(filepath);
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        completeTransferTask(taskId);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        if (typeof error === "object" && error !== null && "name" in error && (error as { name: string }).name === "CanceledError") return;
        failTransferTask(taskId, error instanceof Error ? error.message : "下载失败");
      }
    },
    [addTransferTask, completeTransferTask, downloadSandboxFile, failTransferTask, sessionId, updateTransferProgress]
  );

  const handlePreviewImage = useCallback(
    (src: string, title?: string) => {
      setImagePreview({ src, title: title || "图片预览" });
    },
    []
  );

  const handleTaskDockPreviewFile = useCallback(
    (file: FileInfo) => {
      void openFilePreview(file);
    },
    [openFilePreview]
  );

  const handleTaskDockDownloadFile = useCallback(
    (file: FileInfo) => {
      void handleFileDownload(file);
    },
    [handleFileDownload]
  );

  if (!sessionId) {
    return null;
  }

  return (
    <div className="flex min-h-screen flex-col">
      <SessionHeader sessionId={sessionId} />

      <div className="mx-auto flex w-full max-w-[1700px] flex-1 gap-4 px-4 py-4">
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                <span>当前状态：</span>
                <StatusIndicator meta={currentStatusMeta} />
              </span>
              {sessionRunning ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400">
                  <Loader2 size={12} className="animate-spin" />
                  正在执行中
                </span>
              ) : null}
            </div>
            <div className="flex items-center gap-2">
              {isMobile ? (
                <Button
                  variant="outline"
                  className="rounded-xl border-border"
                >
                  <PanelRightOpen size={16} />
                  打开工作区
                </Button>
              ) : (
                <Button
                  variant="outline"
                  className="rounded-xl border-border"
                  onClick={() => setDesktopWorkbenchVisible(!desktopWorkbenchVisible)}
                >
                  {desktopWorkbenchVisible ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
                  {desktopWorkbenchVisible ? "隐藏工作区" : "显示工作区"}
                </Button>
              )}
            </div>
          </div>

          {isLoadingCurrentSession ? (
            <div className="mb-4 rounded-2xl border border-border bg-card p-3 text-sm text-muted-foreground">
              正在加载会话内容...
            </div>
          ) : null}

          <div ref={eventScrollRef} className="flex-1 space-y-0 overflow-y-auto pb-4">
            {eventList.length === 0 ? (
              <div className="rounded-2xl border border-border bg-card p-3 text-sm text-muted-foreground">
                暂无会话事件，输入消息后开始。
              </div>
            ) : (
              eventList.map((event, index) =>
                renderEventItem(
                  event,
                  index,
                  currentSessionFiles,
                  (file) => {
                    void openFilePreview(file);
                  },
                  (filepath) => {
                    void openFilePathPreview(filepath);
                  },
                  (src, title) => {
                    setImagePreview({
                      src,
                      title: title || "图片预览",
                    });
                  },
                  streamingAssistantEventId
                )
              )
            )}
            {isCurrentSessionStreaming ? (
              <div className="mt-3 inline-flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-400">
                <Loader2 size={14} className="animate-spin" />
                正在持续生成执行结果...
              </div>
            ) : null}
          </div>

          <div className="mt-3 border-t border-border bg-surface-1 pt-3">
            <SessionTaskDock
              className="mb-3"
              summary={progressSummary}
              files={currentSessionFiles}
              running={sessionRunning}
              onPreviewFile={handleTaskDockPreviewFile}
              onDownloadFile={handleTaskDockDownloadFile}
            />
            {sandboxDestroyed ? (
              <div className="flex flex-col items-center gap-3 rounded-2xl border border-red-200 bg-red-50 px-4 py-5 text-center dark:border-red-500/30 dark:bg-red-500/10">
                <XCircle size={24} className="text-red-500" />
                <p className="text-sm font-medium text-red-700 dark:text-red-400">
                  会话已终结
                </p>
                <p className="text-xs text-red-600/80 dark:text-red-400/70">
                  该会话的运行环境已被销毁，无法继续交互。
                </p>
                <Button
                  variant="outline"
                  className="mt-1 rounded-xl border-red-200 text-red-700 hover:bg-red-100 dark:border-red-500/30 dark:text-red-400 dark:hover:bg-red-500/20"
                  onClick={async () => {
                    try {
                      const newId = await createSession();
                      router.push(`/sessions/${newId}`);
                    } catch {
                      setMessage({ type: "error", text: "创建新会话失败" });
                    }
                  }}
                >
                  开新会话
                </Button>
              </div>
            ) : (
              <ChatInput
                sessionId={sessionId}
                skillConfirmationPendingAction={skillConfirmationPendingAction}
              />
            )}
          </div>
        </main>

        {!isMobile && desktopWorkbenchVisible ? (
          <aside className="sticky top-[84px] hidden h-[calc(100vh-104px)] min-h-[620px] w-[620px] shrink-0 self-start lg:block xl:w-[660px]">
            <WorkbenchPanel
              sessionId={sessionId}
              status={visibleSession?.status || "pending"}
              takeoverId={takeoverMeta.takeoverId}
              takeoverScope={takeoverMeta.takeoverScope}
              takeoverExpiresAt={takeoverMeta.takeoverExpiresAt}
              snapshots={workbenchSnapshots}
              running={sessionRunning}
              visible={workbenchVisible}
              onPreviewImage={handlePreviewImage}
            />
          </aside>
        ) : null}
      </div>

      <Sheet open={mobileWorkbenchOpen} onOpenChange={setMobileWorkbenchOpen}>
        <SheetContent side="right" className="w-full max-w-none border-l-border p-3 sm:max-w-[620px]">
          <WorkbenchPanel
            sessionId={sessionId}
            status={visibleSession?.status || "pending"}
            takeoverId={takeoverMeta.takeoverId}
            takeoverScope={takeoverMeta.takeoverScope}
            takeoverExpiresAt={takeoverMeta.takeoverExpiresAt}
            snapshots={workbenchSnapshots}
            running={sessionRunning}
            visible={mobileWorkbenchOpen}
            onPreviewImage={handlePreviewImage}
          />
        </SheetContent>
      </Sheet>

      <Sheet
        open={previewOpen}
        onOpenChange={(open) => {
          if (!open) {
            closePreview();
            return;
          }
          setPreviewOpen(true);
        }}
      >
        <SheetContent side="right" className="w-full max-w-none p-0 sm:max-w-xl">
          <SheetHeader className="border-b border-border px-5 py-4">
            <SheetTitle className="truncate">{previewTitle}</SheetTitle>
            <SheetDescription>文件内容预览</SheetDescription>
          </SheetHeader>

          <div className="min-h-0 flex-1 overflow-auto p-5">
            {previewLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
              </div>
            ) : null}

            {!previewLoading && previewError ? (
              <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
                {previewError}
              </div>
            ) : null}

            {!previewLoading && !previewError && previewKind === "text" ? (
              <pre className="overflow-auto rounded-xl border border-border bg-muted p-4 text-xs leading-6 text-foreground/85">
                {previewTextContent}
              </pre>
            ) : null}

            {!previewLoading && !previewError && previewKind === "image" && previewBlobUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewBlobUrl} alt={previewTitle} className="h-auto max-w-full rounded-xl border border-border" />
            ) : null}

            {!previewLoading && !previewError && previewKind === "pdf" && previewBlobUrl ? (
              <iframe
                title={previewTitle}
                src={previewBlobUrl}
                className="h-[72vh] w-full rounded-xl border border-border"
              />
            ) : null}

            {!previewLoading && !previewError && previewKind === "unsupported" ? (
              <div className="rounded-xl border border-border bg-muted p-4 text-sm text-muted-foreground">
                此文件暂不支持在线预览，可下载后查看。
                {previewFile ? (
                  <Button
                    className="mt-3 rounded-xl"
                    variant="outline"
                    onClick={() => {
                      void handleFileDownload(previewFile);
                    }}
                  >
                    下载文件
                  </Button>
                ) : null}
                {!previewFile && sandboxDownloadPath && sessionId ? (
                  <Button
                    className="mt-3 rounded-xl"
                    variant="outline"
                    onClick={() => void handleSandboxDownload(sandboxDownloadPath)}
                  >
                    下载文件
                  </Button>
                ) : null}
              </div>
            ) : null}
          </div>
        </SheetContent>
      </Sheet>

      <Dialog
        open={Boolean(imagePreview)}
        onOpenChange={(open) => {
          if (!open) {
            setImagePreview(null);
          }
        }}
      >
        <DialogContent className="max-w-5xl border-border p-3">
          <DialogTitle className="px-2 text-sm text-foreground/85">{imagePreview?.title || "图片预览"}</DialogTitle>
          <div className="max-h-[80vh] overflow-auto">
            {imagePreview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={imagePreview.src} alt={imagePreview.title} className="h-auto w-full rounded-lg border border-border" />
            ) : null}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
