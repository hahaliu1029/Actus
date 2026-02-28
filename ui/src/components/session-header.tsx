"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { EllipsisVertical, House } from "lucide-react";

import { ManusSettings } from "@/components/manus-settings";
import { Button } from "@/components/ui/button";
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
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/hooks/use-auth";
import { useIsMobile } from "@/hooks/use-mobile";
import { sessionApi } from "@/lib/api/session";
import { useSessionStore } from "@/lib/store/session-store";
import { useUIStore } from "@/lib/store/ui-store";

export function SessionHeader({ sessionId }: Readonly<{ sessionId: string }>) {
  const router = useRouter();
  const { logout } = useAuth();
  const isMobile = useIsMobile();
  const session = useSessionStore((state) => state.currentSession);
  const stopSession = useSessionStore((state) => state.stopSession);
  const deleteSession = useSessionStore((state) => state.deleteSession);
  const fetchSessionById = useSessionStore((state) => state.fetchSessionById);
  const setMessage = useUIStore((state) => state.setMessage);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [takeoverSubmitting, setTakeoverSubmitting] = useState(false);

  const status = session?.status;
  const canEndTakeover = status === "takeover";

  const handleStop = async () => {
    await stopSession(sessionId);
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
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-surface-1/95 px-4 py-3 backdrop-blur-sm">
      <div className="min-w-0">
        <Link
          href="/"
          className="mb-1 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <House size={13} />
          返回主页
        </Link>
        <h1 className="text-base font-semibold text-foreground">
          {session?.title || "未命名任务"}
        </h1>
        <p className="text-xs text-muted-foreground">会话 ID：{sessionId}</p>
      </div>
      {isMobile ? (
        <div className="flex items-center gap-2">
          <ManusSettings />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className="rounded-xl border-border text-foreground/80"
                aria-label="更多操作"
              >
                <EllipsisVertical size={16} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-40">
              {canEndTakeover ? (
                <DropdownMenuItem
                  disabled={takeoverSubmitting}
                  onClick={() => {
                    void handleEndTakeover();
                  }}
                >
                  结束接管
                </DropdownMenuItem>
              ) : null}
              <DropdownMenuItem
                onClick={() => {
                  logout();
                }}
              >
                退出登录
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => {
                  void handleStop();
                }}
              >
                停止
              </DropdownMenuItem>
              <DropdownMenuItem
                className="text-red-600 focus:text-red-600"
                onClick={() => {
                  setDeleteDialogOpen(true);
                }}
              >
                删除
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <ManusSettings />
          {canEndTakeover ? (
            <Button
              variant="outline"
              disabled={takeoverSubmitting}
              className="rounded-xl border-border text-foreground/80"
              onClick={() => {
                void handleEndTakeover();
              }}
            >
              结束接管
            </Button>
          ) : null}
          <Button
            variant="outline"
            className="rounded-xl border-border text-foreground/80"
            onClick={() => {
              logout();
            }}
          >
            退出登录
          </Button>
          <Button
            variant="outline"
            className="rounded-xl border-border text-foreground/80"
            onClick={() => {
              void handleStop();
            }}
          >
            停止
          </Button>
          <Button
            variant="outline"
            className="rounded-xl border-destructive/30 text-destructive hover:bg-destructive/10 hover:text-destructive"
            onClick={() => {
              setDeleteDialogOpen(true);
            }}
          >
            删除
          </Button>
        </div>
      )}

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
