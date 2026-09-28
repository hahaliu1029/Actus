"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Ellipsis, House, LogOut, Square, Trash2 } from "lucide-react";

import { ManusSettings } from "@/components/manus-settings";
import { SessionCostSummary } from "@/components/session-cost-summary";
import { StatusIndicator } from "@/components/status-indicator";
import { Button } from "@/components/ui/button";
import { SidebarTrigger } from "@/components/ui/sidebar";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/hooks/use-auth";
import { sessionApi } from "@/lib/api/session";
import { getSessionStatusMeta } from "@/lib/status-copy";
import { useSessionStore } from "@/lib/store/session-store";
import { useUIStore } from "@/lib/store/ui-store";

export function SessionHeader({ sessionId }: Readonly<{ sessionId: string }>) {
  const router = useRouter();
  const { logout } = useAuth();
  const session = useSessionStore((state) => state.currentSession);
  const stopSession = useSessionStore((state) => state.stopSession);
  const deleteSession = useSessionStore((state) => state.deleteSession);
  const fetchSessionById = useSessionStore((state) => state.fetchSessionById);
  const setMessage = useUIStore((state) => state.setMessage);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [takeoverSubmitting, setTakeoverSubmitting] = useState(false);
  const [stopSubmitting, setStopSubmitting] = useState(false);

  // A4-0 follow-up (a): only trust currentSession when it is THIS route's
  // session. During an A→B route switch, fetchSessionById(B) is async, so
  // currentSession can still be A — deriving the takeover control from a stale
  // A would let "结束接管" fire endTakeover(B) on the wrong session. Mirror the
  // page's visibleSession guard (app/sessions/[id]/page.tsx).
  const isSessionLoaded = session?.session_id === sessionId;
  const status = isSessionLoaded ? session?.status : undefined;
  const canEndTakeover = status === "takeover";
  const canStop = status !== undefined && ["running", "waiting", "finishing", "takeover_pending", "takeover"].includes(status);

  const handleStop = async () => {
    if (!canStop || stopSubmitting) return;
    setStopSubmitting(true);
    try {
      await stopSession(sessionId);
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "停止任务失败",
      });
    } finally {
      setStopSubmitting(false);
    }
  };

  const handleDelete = async () => {
    await deleteSession(sessionId);
    router.replace("/");
  };

  const handleEndTakeover = async () => {
    setTakeoverSubmitting(true);
    try {
      await sessionApi.endTakeover(sessionId, { handoff_mode: "continue" });
      await fetchSessionById(sessionId, { silent: true });
      setMessage({
        type: "success",
        text: "已结束接管并交还给 AI 继续执行",
      });
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "结束接管失败",
      });
    } finally {
      setTakeoverSubmitting(false);
    }
  };

  return (
    <header className="z-10 flex min-h-16 shrink-0 items-center justify-between gap-3 bg-surface-1 px-4 py-3 sm:px-6">
      <div className="flex min-w-0 items-center gap-3">
        <SidebarTrigger aria-label="打开会话侧栏" className="size-9 shrink-0 rounded-full" />
        <div className="min-w-0">
          <h1 className="truncate text-sm font-medium text-foreground sm:text-base">
            {isSessionLoaded ? session?.title || "未命名任务" : "正在加载会话"}
          </h1>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            {isSessionLoaded ? (
              <>
                <StatusIndicator meta={getSessionStatusMeta(status)} />
                <SessionCostSummary key={sessionId} sessionId={sessionId} />
              </>
            ) : null}
          </div>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-1 sm:gap-2">
        {canEndTakeover ? (
          <Button
            variant="outline"
            size="sm"
            disabled={takeoverSubmitting}
            className="rounded-full border-border text-foreground/80"
            onClick={() => {
              void handleEndTakeover();
            }}
          >
            结束接管
          </Button>
        ) : null}
        <Button
          variant="ghost"
          size="icon"
          className="size-9 rounded-full text-muted-foreground hover:text-foreground"
          aria-label="停止"
          title="停止任务"
          disabled={!canStop || stopSubmitting}
          onClick={() => {
            void handleStop();
          }}
        >
          <Square size={15} />
        </Button>
        <ManusSettings />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="size-9 rounded-full text-muted-foreground hover:text-foreground"
              aria-label="更多操作"
            >
              <Ellipsis size={18} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-64 rounded-xl p-1.5">
            <DropdownMenuLabel className="break-all text-xs font-normal text-muted-foreground">
              会话 ID：{sessionId}
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <Link href="/">
                <House size={15} />
                返回主页
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => logout()}>
              <LogOut size={15} />
              退出登录
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem variant="destructive" onClick={() => setDeleteDialogOpen(true)}>
              <Trash2 size={15} />
              删除
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="max-w-md rounded-2xl border-border">
          <DialogHeader>
            <DialogTitle className="text-xl">要删除任务信息吗？</DialogTitle>
            <DialogDescription className="leading-6">
              删除任务信息后，历史消息及任务文件将不可恢复，请确认。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" className="rounded-xl" onClick={() => setDeleteDialogOpen(false)}>
              取消
            </Button>
            <Button
              className="rounded-xl"
              onClick={() => {
                void handleDelete();
              }}
            >
              确认
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </header>
  );
}
