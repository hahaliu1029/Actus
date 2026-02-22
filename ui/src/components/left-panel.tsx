"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Loader2, Moon, Plus, Sun, Trash } from "lucide-react";
import { useTheme } from "next-themes";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { formatRelativeTime } from "@/lib/session-ui";
import { useSessionStore } from "@/lib/store/session-store";

export function LeftPanel() {
  const router = useRouter();
  const pathname = usePathname();

  const sessions = useSessionStore((state) => state.sessions);
  const isLoadingSessions = useSessionStore((state) => state.isLoadingSessions);
  const fetchSessions = useSessionStore((state) => state.fetchSessions);
  const streamSessions = useSessionStore((state) => state.streamSessions);
  const stopStreamSessions = useSessionStore((state) => state.stopStreamSessions);
  const createSession = useSessionStore((state) => state.createSession);
  const deleteSession = useSessionStore((state) => state.deleteSession);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const { resolvedTheme, setTheme } = useTheme();

  useEffect(() => {
    void fetchSessions();
    streamSessions();
    return () => {
      stopStreamSessions();
    };
  }, [fetchSessions, streamSessions, stopStreamSessions]);

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

      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        {isLoadingSessions ? (
          <div className="rounded-xl border border-border bg-muted px-3 py-2 text-sm text-muted-foreground">
            正在加载会话...
          </div>
        ) : null}

        {sessions.map((session) => (
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
                <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                  {session.status === "running" ? (
                    <Loader2 size={11} className="animate-spin text-amber-500" />
                  ) : null}
                  {session.status}
                </span>
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
          </div>
        ))}
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
