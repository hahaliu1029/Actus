"use client";

import { useCallback, useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Moon, Plus, RotateCcw, Sun, Trash } from "lucide-react";
import { useTheme } from "next-themes";

import { StatusIndicator } from "@/components/status-indicator";
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

  const handleCreate = async () => {
    const createdId = await createSession();
    router.push(`/sessions/${createdId}`);
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
    <aside className="hidden h-screen w-[280px] border-r border-border bg-card p-3 md:flex md:flex-col">
      <button
        onClick={() => {
          void handleCreate();
        }}
        className="mb-3 flex w-full items-center justify-center gap-2 rounded-xl border border-border px-3 py-2 text-sm text-foreground/80 transition-colors hover:bg-accent"
      >
        <Plus size={16} /> 新建任务
      </button>

      {backgroundQuota ? (
        <div className="mb-2 flex items-center justify-between border-b border-border pb-2 text-[11px] text-muted-foreground">
          <span>后台额度</span>
          <span className="font-medium text-foreground">
            {backgroundQuota.user_used}/{backgroundQuota.user_limit}
          </span>
          <span>
            全局 {backgroundQuota.system_used}/{backgroundQuota.system_limit}
          </span>
        </div>
      ) : null}

      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        {isLoadingSessions ? (
          <div className="rounded-xl border border-border bg-muted px-3 py-2 text-sm text-muted-foreground">
            正在加载会话...
          </div>
        ) : null}

        {sessions.map((session) => {
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
              className={`group rounded-xl border px-3 py-2 transition-colors duration-150 ${
                currentSessionId === session.session_id
                  ? "border-border-strong bg-accent"
                  : "border-transparent hover:border-border hover:bg-accent/60"
              }`}
            >
              <button
                onClick={() => router.push(`/sessions/${session.session_id}`)}
                className="w-full text-left"
              >
                <p className="truncate text-sm font-medium text-foreground">
                  {session.title || "未命名会话"}
                </p>
                <p className="truncate text-xs text-muted-foreground">
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
                  className="invisible rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive group-hover:visible"
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

      <div className="mt-2 border-t border-border pt-2">
        <button
          onClick={() => setTheme(isDarkMode ? "light" : "dark")}
          className="relative flex w-full items-center gap-2 rounded-xl px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
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
  );
}
