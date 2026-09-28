"use client";

import { useCallback, useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { MessageCircle, Moon, Plus, RotateCcw, Search, Sun, Trash, X } from "lucide-react";
import { useTheme } from "next-themes";

import { StatusIndicator } from "@/components/status-indicator";
import { ActusMark } from "@/components/actus-mark";
import { Sidebar, SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { sessionApi } from "@/lib/api/session";
import { formatRelativeTime } from "@/lib/session-ui";
import { getSessionStatusMeta } from "@/lib/status-copy";
import type { BackgroundQuotaResponse, SupervisorSnapshot } from "@/lib/api/types";
import { useFilteredSessionsForList, useSessionStore } from "@/lib/store/session-store";
import { useUIStore } from "@/lib/store/ui-store";

function getBackgroundPhaseLabel(
  phase: SupervisorSnapshot["execution_phase"]
): string {
  switch (phase) {
    case "running":
      return "运行中";
    case "recovering":
      return "恢复中";
    case "idle":
      return "空闲";
    case "suspended":
      return "已挂起";
    case "terminating":
      return "结束中";
    case "terminated":
      return "已结束";
  }
}

export function LeftPanel() {
  const router = useRouter();
  const pathname = usePathname();
  const { setOpenMobile } = useSidebar();
  const [search, setSearch] = useState("");
  const [creating, setCreating] = useState(false);

  // C1a: filter probe child sessions (parent_session_id non-null) out of
  // the LeftPanel list view.
  const sessions = useFilteredSessionsForList();
  const isLoadingSessions = useSessionStore((state) => state.isLoadingSessions);
  const fetchSessions = useSessionStore((state) => state.fetchSessions);
  const streamSessions = useSessionStore((state) => state.streamSessions);
  const stopStreamSessions = useSessionStore((state) => state.stopStreamSessions);
  const createSession = useSessionStore((state) => state.createSession);
  const deleteSession = useSessionStore((state) => state.deleteSession);
  const retryFromSuspend = useSessionStore((state) => state.retryFromSuspend);
  const setMessage = useUIStore((state) => state.setMessage);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [retryingSessionId, setRetryingSessionId] = useState<string | null>(null);
  const [backgroundQuota, setBackgroundQuota] =
    useState<BackgroundQuotaResponse | null>(null);
  const { resolvedTheme, setTheme } = useTheme();
  const backgroundQuotaRefreshKey = sessions
    .map((session) => {
      const snapshot = session.supervisor_snapshot;
      return [
        session.session_id,
        snapshot?.execution_mode || "foreground",
        snapshot?.execution_phase || "",
      ].join(":");
    })
    .join("|");

  const refreshBackgroundQuota = useCallback(async () => {
    try {
      const quota = await sessionApi.getBackgroundQuota();
      setBackgroundQuota(quota);
    } catch {
      setBackgroundQuota(null);
    }
  }, []);

  useEffect(() => {
    void fetchSessions();
    streamSessions();
    return () => {
      stopStreamSessions();
    };
  }, [fetchSessions, streamSessions, stopStreamSessions]);

  useEffect(() => {
    void refreshBackgroundQuota();
  }, [backgroundQuotaRefreshKey, refreshBackgroundQuota]);

  const currentSessionId =
    pathname?.startsWith("/sessions/") === true
      ? pathname.replace("/sessions/", "").split("/")[0]
      : null;
  const isDarkMode = resolvedTheme === "dark";
  const themeLabel =
    resolvedTheme == null ? "切换主题" : isDarkMode ? "浅色模式" : "深色模式";
  const searchTerm = search.trim().toLocaleLowerCase();
  const visibleSessions = sessions.filter((session) =>
    `${session.title || ""} ${session.latest_message || ""}`.toLocaleLowerCase().includes(searchTerm)
  );

  const navigate = (path: string) => {
    router.push(path);
    setOpenMobile(false);
  };

  const handleCreate = async () => {
    setCreating(true);
    try {
      const createdId = await createSession();
      navigate(`/sessions/${createdId}`);
    } catch (error) {
      setMessage({ type: "error", text: error instanceof Error ? error.message : "新建对话失败，请重试" });
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (sessionId: string) => {
    await deleteSession(sessionId);
    if (currentSessionId === sessionId) {
      router.push("/");
    }
    setDeletingSessionId(null);
  };

  const handleRetryFromSuspend = async (sessionId: string) => {
    setRetryingSessionId(sessionId);
    try {
      await retryFromSuspend(sessionId);
      await refreshBackgroundQuota();
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "重试后台任务失败",
      });
    } finally {
      setRetryingSessionId(null);
    }
  };

  return (
    <Sidebar className="border-border-subtle">
    <aside className="flex h-full min-h-0 flex-col px-3 pb-3 pt-4" aria-label="对话导航">
      <div className="mb-6 flex items-center justify-between px-2">
        <button onClick={() => navigate("/")} className="flex items-center gap-2.5 rounded-lg text-lg font-semibold tracking-tight focus-visible:outline-2 focus-visible:outline-ring" aria-label="Actus 主页">
          <ActusMark className="size-8" />Actus
        </button>
        <SidebarTrigger aria-label="收起对话列表" className="size-8 rounded-full text-muted-foreground" />
      </div>
      <button
        onClick={() => {
          void handleCreate();
        }}
        disabled={creating}
        className="mb-3 flex w-full items-center gap-2.5 rounded-xl border border-border bg-card px-3 py-2.5 text-sm font-medium text-foreground/90 transition-colors hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring disabled:opacity-50"
      >
        <Plus size={17} /> {creating ? "正在创建…" : "新建对话"}
      </button>

      <div className="mb-6 flex items-center gap-2 rounded-xl px-3 py-2 text-muted-foreground focus-within:bg-card focus-within:ring-1 focus-within:ring-border-strong">
        <Search size={15} className="shrink-0" />
        <input aria-label="搜索对话" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索对话" className="w-full min-w-0 bg-transparent text-sm outline-none placeholder:text-muted-foreground" />
        {search ? <button onClick={() => setSearch("")} aria-label="清除搜索" className="rounded hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"><X size={14} /></button> : null}
      </div>
      <h2 className="mb-2 px-3 text-[11px] font-medium text-muted-foreground">最近对话</h2>

      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto" aria-label="最近对话列表">
        {isLoadingSessions ? (
          <div className="rounded-xl border border-border bg-muted px-3 py-2 text-sm text-muted-foreground">
            正在加载会话...
          </div>
        ) : null}

        {!isLoadingSessions && visibleSessions.length === 0 ? (
          <p className="px-3 py-6 text-center text-xs leading-6 text-muted-foreground">{searchTerm ? "没有找到匹配的对话，试试其他关键词。" : "从一个新对话开始。你的想法和进展会留在这里。"}</p>
        ) : null}
        {visibleSessions.map((session) => {
          const backgroundSnapshot =
            session.supervisor_snapshot?.execution_mode === "background"
              ? session.supervisor_snapshot
              : null;
          const canRetryFromSuspend =
            backgroundSnapshot?.execution_phase === "suspended" &&
            backgroundSnapshot.retry_budget_remaining > 0;

          return (
            <div
              key={session.session_id}
              className={`group rounded-xl border px-3 py-3 transition-colors duration-150 ${
                currentSessionId === session.session_id
                  ? "border-border bg-accent"
                  : "border-transparent hover:bg-accent/70"
              }`}
            >
              <button
                onClick={() => navigate(`/sessions/${session.session_id}`)}
                aria-current={currentSessionId === session.session_id ? "page" : undefined}
                className="w-full rounded text-left focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring"
              >
                <p className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <MessageCircle size={14} className="shrink-0 text-muted-foreground" />
                  <span className="truncate">
                  {session.title || "未命名会话"}
                  </span>
                </p>
                <p className="mt-1.5 truncate text-xs text-muted-foreground">
                  {session.latest_message || "暂无消息"}
                </p>
              </button>
              <div className="mt-1 flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <StatusIndicator
                    meta={getSessionStatusMeta(session.status)}
                    className="text-[11px]"
                  />
                  {session.unread_message_count > 0 ? (
                    <span className="rounded-full bg-primary px-1.5 py-0.5 text-[10px] text-primary-foreground">
                      {session.unread_message_count}
                    </span>
                  ) : null}
                  <span className="text-[11px] text-muted-foreground/60">
                    {formatRelativeTime(session.latest_message_at)}
                  </span>
                </div>
                <button
                  onClick={() => {
                    setDeletingSessionId(session.session_id);
                  }}
                  className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive focus-visible:outline-2 focus-visible:outline-ring md:opacity-0 md:group-hover:opacity-100 md:group-focus-within:opacity-100"
                  aria-label="删除会话"
                >
                  <Trash size={14} />
                </button>
              </div>
              {backgroundSnapshot ? (
                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[10px] text-muted-foreground">
                  <span className="rounded-full border border-amber-300/60 bg-amber-50 px-1.5 py-0.5 text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
                    后台
                  </span>
                  <span>{getBackgroundPhaseLabel(backgroundSnapshot.execution_phase)}</span>
                  <span>剩余重试 {backgroundSnapshot.retry_budget_remaining}</span>
                  {canRetryFromSuspend ? (
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 rounded border border-border px-1.5 py-0.5 text-foreground transition-colors hover:bg-accent disabled:opacity-50"
                      disabled={retryingSessionId === session.session_id}
                      onClick={() => {
                        void handleRetryFromSuspend(session.session_id);
                      }}
                    >
                      <RotateCcw size={11} />
                      重试
                    </button>
                  ) : null}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <div className="mt-3 border-t border-border-subtle pt-3">
        {backgroundQuota ? (
          <div className="mb-2 flex items-center gap-2 px-3 text-[10px] text-muted-foreground">
            <span>后台额度</span>
            <span>{backgroundQuota.user_used}/{backgroundQuota.user_limit}</span>
            <span className="ml-auto">全局 {backgroundQuota.system_used}/{backgroundQuota.system_limit}</span>
          </div>
        ) : null}
        <button
          onClick={() => setTheme(isDarkMode ? "light" : "dark")}
          className="relative flex w-full items-center gap-2 rounded-xl px-3 py-2 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
        >
          <span className="relative size-4">
            <Sun
              size={16}
              className="absolute inset-0 rotate-0 scale-100 transition-transform dark:-rotate-90 dark:scale-0"
            />
            <Moon
              size={16}
              className="absolute inset-0 rotate-90 scale-0 transition-transform dark:rotate-0 dark:scale-100"
            />
          </span>
          <span>{themeLabel}</span>
        </button>
      </div>

      <Dialog open={Boolean(deletingSessionId)} onOpenChange={(open) => !open && setDeletingSessionId(null)}>
        <DialogContent className="max-w-md rounded-2xl border-border">
          <DialogHeader>
            <DialogTitle className="text-xl">要删除任务信息吗？</DialogTitle>
            <DialogDescription className="leading-6">
              删除任务后，该任务下的消息和文件将无法恢复，请确认是否继续。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" className="rounded-xl" onClick={() => setDeletingSessionId(null)}>
              取消
            </Button>
            <Button
              className="rounded-xl"
              onClick={() => {
                if (!deletingSessionId) {
                  return;
                }
                void handleDelete(deletingSessionId);
              }}
            >
              确认
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </aside>
    </Sidebar>
  );
}
