"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "next/navigation";
import {
  Bot,
  CheckCircle2,
  CircleDashed,
  Loader2,
  PanelRightClose,
  PanelRightOpen,
  XCircle,
} from "lucide-react";

import { ChatInput } from "@/components/chat-input";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { SessionHeader } from "@/components/session-header";
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
  getToolDisplayCopy,
  normalizeMessageAttachments,
} from "@/lib/session-ui";
import { cn } from "@/lib/utils";
import { useSessionStore } from "@/lib/store/session-store";
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

function getEventKey(event: SessionEvent, index: number): string {
  return String(event.data.event_id || `${event.event}-${index}`);
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
  if ((toolName === "mcp" || toolName === "a2a") && content) {
    const rawResult = (toolName === "a2a")
      ? (content as Record<string, unknown>).a2a_result
      : (content as Record<string, unknown>).result;
    if (rawResult !== undefined && rawResult !== null) {
      mcpResult = typeof rawResult === "string" ? rawResult : JSON.stringify(rawResult, null, 2);
    }
  }

  return { screenshots, searchResults, filepath, mcpResult };
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
  const eventKey = getEventKey(event, index);

  if (event.event === "message") {
    const role = String(event.data.role || "assistant");
    const message = String(event.data.message || "");
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
          <MarkdownRenderer content={message || "（空消息）"} />
          {renderMessageAttachments(attachments, onPreviewFile)}
          {isStreamingAssistant ? (
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
            <p className="whitespace-pre-wrap leading-7">{display.detail}</p>
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
        {display.detail ? <p className="mt-1 truncate text-xs text-muted-foreground">{display.detail}</p> : null}

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
            <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg bg-muted p-2 text-xs text-muted-foreground">
              {visual.mcpResult}
            </pre>
          </details>
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

  const currentSession = useSessionStore((state) => state.currentSession);
  const currentSessionFiles = useSessionStore((state) => state.currentSessionFiles);
  const setActiveSession = useSessionStore((state) => state.setActiveSession);
  const fetchSessionById = useSessionStore((state) => state.fetchSessionById);
  const fetchSessionFiles = useSessionStore((state) => state.fetchSessionFiles);
  const downloadFile = useSessionStore((state) => state.downloadFile);
  const isLoadingCurrentSession = useSessionStore((state) => state.isLoadingCurrentSession);
  const isChatting = useSessionStore((state) => state.isChatting);
  const setMessage = useUIStore((state) => state.setMessage);
  const isMobile = useIsMobile();

  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewTitle, setPreviewTitle] = useState("预览");
  const [previewKind, setPreviewKind] = useState<ReturnType<typeof getFilePreviewKind>>("unsupported");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewTextContent, setPreviewTextContent] = useState("");
  const [previewBlobUrl, setPreviewBlobUrl] = useState<string | null>(null);
  const [previewFile, setPreviewFile] = useState<FileInfo | null>(null);
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
  const progressSummary = useMemo(
    () => deriveSessionProgressSummary(eventList),
    [eventList]
  );
  const workbenchVisible = (!isMobile && desktopWorkbenchVisible) || (isMobile && mobileWorkbenchOpen);
  const sessionRunning =
    isChatting || visibleSession?.status === "running" || visibleSession?.status === "waiting";

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
      // 因此 isChatting（SSE 活跃）期间只轮询文件列表
      if (!isChatting) {
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
  }, [fetchSessionById, fetchSessionFiles, sessionId, sessionRunning, isChatting]);

  const streamingAssistantEventId = useMemo(() => {
    if (!isChatting) {
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
      return getEventKey(event, index);
    }
    return null;
  }, [eventList, isChatting]);

  useEffect(() => {
    const node = eventScrollRef.current;
    if (!node) {
      return;
    }
    node.scrollTo({
      top: node.scrollHeight,
      behavior: "smooth",
    });
  }, [eventList]);

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
          const blob = await downloadFile(file.id);
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
    [downloadFile, previewBlobUrl, resetPreviewState, sessionId]
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

      resetPreviewState();
      setPreviewTitle(getPathTail(filepath));
      setPreviewOpen(true);
      setPreviewLoading(true);
      setPreviewKind("text");

      try {
        const result = await sessionApi.viewFile(sessionId, { filepath });
        setPreviewTextContent(result.content || "文件为空。");
      } catch (error) {
        setPreviewError(error instanceof Error ? error.message : "文件预览失败");
      } finally {
        setPreviewLoading(false);
      }
    },
    [currentSessionFiles, openFilePreview, resetPreviewState, sessionId]
  );

  const handleFileDownload = useCallback(
    async (file: FileInfo) => {
      try {
        const blob = await downloadFile(file.id);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = file.filename;
        a.click();
        URL.revokeObjectURL(url);
      } catch (error) {
        setMessage({
          type: "error",
          text: error instanceof Error ? error.message : "下载文件失败",
        });
      }
    },
    [downloadFile, setMessage]
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
              <span>当前状态：{visibleSession?.status || "pending"}</span>
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
            {isChatting ? (
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
            <ChatInput sessionId={sessionId} />
          </div>
        </main>

        {!isMobile && desktopWorkbenchVisible ? (
          <aside className="sticky top-[84px] hidden h-[calc(100vh-104px)] min-h-[620px] w-[620px] shrink-0 self-start lg:block xl:w-[660px]">
            <WorkbenchPanel
              sessionId={sessionId}
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
