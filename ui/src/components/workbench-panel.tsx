"use client";

import Link from "next/link";
import { ChevronDown, Expand, Globe, TerminalSquare } from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useIsMobile } from "@/hooks/use-mobile";
import { usePageVisibility } from "@/hooks/use-page-visibility";
import { useShellPreview } from "@/hooks/use-shell-preview";
import { sessionApi } from "@/lib/api/session";
import type { SessionStatus, TakeoverScope } from "@/lib/api/types";
import {
  findSnapshotAtOrBefore,
  getLatestSnapshotByMode,
  getShellSessionIdsFromSnapshots,
  type TimelineCursorState,
  type WorkbenchMode,
  type WorkbenchSnapshot,
} from "@/lib/session-ui";
import { useSessionStore } from "@/lib/store/session-store";
import { useUIStore } from "@/lib/store/ui-store";
import { normalizeUnixSeconds } from "@/lib/takeover/normalize";
import { cn } from "@/lib/utils";

import { WorkbenchBrowserPreview } from "./workbench-browser-preview";
import { WorkbenchInteractiveTerminal } from "./workbench-interactive-terminal";
import { WorkbenchTerminalPreview } from "./workbench-terminal-preview";
import { WorkbenchTimeline } from "./workbench-timeline";

type WorkbenchPanelProps = {
  sessionId: string;
  status: SessionStatus;
  takeoverId?: string | null;
  takeoverScope?: TakeoverScope | null;
  takeoverExpiresAt?: number | null;
  snapshots: WorkbenchSnapshot[];
  running: boolean;
  visible: boolean;
  onPreviewImage: (src: string, title?: string) => void;
};

function findModeSnapshotAtOrBefore(
  snapshots: WorkbenchSnapshot[],
  mode: WorkbenchMode,
  timestamp: number | null,
  shellSessionId?: string | null
): WorkbenchSnapshot | null {
  for (let index = snapshots.length - 1; index >= 0; index -= 1) {
    const snapshot = snapshots[index];
    if (!snapshot || snapshot.mode !== mode) {
      continue;
    }
    if (timestamp != null && snapshot.timestamp > timestamp) {
      continue;
    }
    if (mode === "shell" && shellSessionId && snapshot.shellSessionId !== shellSessionId) {
      continue;
    }
    return snapshot;
  }
  return null;
}

const DEFAULT_TAKEOVER_TTL_MS = 900_000;

function computeRenewIntervalMs(ttlMs: number): number {
  return Math.max(10_000, Math.min(300_000, Math.floor(ttlMs / 3)));
}

function normalizeErrorHttpStatus(error: unknown): number | null {
  if (typeof error !== "object" || error == null) {
    return null;
  }
  const status = (error as { httpStatus?: unknown }).httpStatus;
  if (typeof status === "number" && Number.isFinite(status)) {
    return status;
  }
  return null;
}

export const WorkbenchPanel = memo(function WorkbenchPanel({
  sessionId,
  status,
  takeoverId = null,
  takeoverScope = null,
  takeoverExpiresAt = null,
  snapshots,
  running,
  visible,
  onPreviewImage,
}: WorkbenchPanelProps) {
  const [cursorState, setCursorState] = useState<TimelineCursorState>("live_following");
  const [liveLocked, setLiveLocked] = useState(false);
  const [historyTimestamp, setHistoryTimestamp] = useState<number | null>(null);
  const [historyAnchorTimestamp, setHistoryAnchorTimestamp] = useState<number | null>(null);
  const [manualMode, setManualMode] = useState<WorkbenchMode>("shell");
  const [selectedShellSessionId, setSelectedShellSessionId] = useState<string | null>(null);
  const [contentHovered, setContentHovered] = useState(false);
  const [takeoverSubmitting, setTakeoverSubmitting] = useState(false);
  const isMobile = useIsMobile();
  const isPageVisible = usePageVisibility();
  const renewTimerRef = useRef<number | null>(null);
  const prevPageVisibleRef = useRef(isPageVisible);
  const fetchSessionById = useSessionStore((state) => state.fetchSessionById);
  const setMessage = useUIStore((state) => state.setMessage);

  const canStartTakeover =
    status === "running" || status === "waiting" || status === "takeover_pending";
  const canEndTakeover = status === "takeover";
  const canRejectTakeover = status === "takeover_pending";
  const effectiveTakeoverScope: TakeoverScope =
    takeoverScope === "browser" ? "browser" : "shell";
  const interactiveShellTakeover =
    status === "takeover" && effectiveTakeoverScope === "shell" && Boolean(takeoverId);

  const latestSnapshot = snapshots[snapshots.length - 1] || null;
  const latestTimestamp = latestSnapshot?.timestamp ?? null;
  const shellSessionIds = useMemo(
    () => getShellSessionIdsFromSnapshots(snapshots),
    [snapshots]
  );
  const effectiveShellSessionId = useMemo(() => {
    if (shellSessionIds.length === 0) {
      return null;
    }
    if (selectedShellSessionId && shellSessionIds.includes(selectedShellSessionId)) {
      return selectedShellSessionId;
    }
    return shellSessionIds[0] || null;
  }, [selectedShellSessionId, shellSessionIds]);

  const isHistoryMode =
    cursorState === "history_scrubbing" || cursorState === "history_paused";
  const activeTimestamp = isHistoryMode
    ? historyTimestamp ?? latestTimestamp
    : latestTimestamp;

  const selectedMode: WorkbenchMode =
    !isHistoryMode && !liveLocked && latestSnapshot?.mode
      ? latestSnapshot.mode
      : manualMode;

  const timelineSnapshot = findSnapshotAtOrBefore(snapshots, activeTimestamp);
  const browserSnapshot = findModeSnapshotAtOrBefore(
    snapshots,
    "browser",
    activeTimestamp
  );
  const shellSnapshot =
    findModeSnapshotAtOrBefore(
      snapshots,
      "shell",
      activeTimestamp,
      effectiveShellSessionId
    ) || findModeSnapshotAtOrBefore(snapshots, "shell", activeTimestamp);

  const currentModeSnapshot = selectedMode === "browser" ? browserSnapshot : shellSnapshot;
  const latestModeSnapshot = getLatestSnapshotByMode(snapshots, selectedMode);

  const hasNewRealtime =
    isHistoryMode &&
    latestTimestamp != null &&
    historyAnchorTimestamp != null &&
    latestTimestamp > historyAnchorTimestamp;
  const timelineAutoOverlayDisabled =
    status === "takeover" && selectedMode === "shell";
  const showTimelineOverlay =
    (isMobile || contentHovered || isHistoryMode || hasNewRealtime) &&
    !timelineAutoOverlayDisabled;

  const shouldPollShell =
    visible &&
    running &&
    !interactiveShellTakeover &&
    (cursorState === "live_following" || cursorState === "live_locked") &&
    selectedMode === "shell" &&
    Boolean(effectiveShellSessionId);

  const shellPreview = useShellPreview({
    sessionId,
    shellSessionId: effectiveShellSessionId,
    enabled: shouldPollShell,
  });

  const handleBackToLive = () => {
    setCursorState(liveLocked ? "live_locked" : "live_following");
    setHistoryTimestamp(null);
    setHistoryAnchorTimestamp(null);
  };

  const handleScrubStart = () => {
    setCursorState("history_scrubbing");
    setHistoryAnchorTimestamp(latestTimestamp);
    if (historyTimestamp == null) {
      setHistoryTimestamp(activeTimestamp);
    }
  };

  const handleScrub = (timestamp: number) => {
    setHistoryTimestamp(timestamp);
  };

  const handleScrubEnd = (timestamp: number) => {
    setHistoryTimestamp(timestamp);
    if (latestTimestamp != null && timestamp >= latestTimestamp) {
      setCursorState(liveLocked ? "live_locked" : "live_following");
      setHistoryTimestamp(null);
      setHistoryAnchorTimestamp(null);
      return;
    }
    setCursorState("history_paused");
    if (historyAnchorTimestamp == null) {
      setHistoryAnchorTimestamp(latestTimestamp);
    }
  };

  const toggleLock = () => {
    setLiveLocked((prev) => {
      const next = !prev;
      if (!isHistoryMode) {
        setCursorState(next ? "live_locked" : "live_following");
      }
      return next;
    });
  };

  const atEarliestRecord =
    isHistoryMode &&
    snapshots.length > 0 &&
    activeTimestamp != null &&
    activeTimestamp <= (snapshots[0]?.timestamp ?? 0);

  const lockAndSwitchMode = useCallback((mode: WorkbenchMode) => {
    setManualMode(mode);
    if (!isHistoryMode && !liveLocked) {
      setLiveLocked(true);
      setCursorState("live_locked");
    }
  }, [isHistoryMode, liveLocked]);

  const takeoverTtlMs = useMemo(() => {
    if (takeoverExpiresAt == null || !Number.isFinite(takeoverExpiresAt)) {
      return DEFAULT_TAKEOVER_TTL_MS;
    }
    const expiresAtMs =
      takeoverExpiresAt > 10 ** 12 ? takeoverExpiresAt : takeoverExpiresAt * 1000;
    const remainingMs = Math.floor(expiresAtMs - Date.now());
    if (remainingMs <= 0) {
      return DEFAULT_TAKEOVER_TTL_MS;
    }
    return remainingMs;
  }, [takeoverExpiresAt]);

  const renewIntervalMs = useMemo(
    () => computeRenewIntervalMs(takeoverTtlMs),
    [takeoverTtlMs]
  );

  const clearRenewTimer = useCallback(() => {
    if (renewTimerRef.current != null) {
      window.clearTimeout(renewTimerRef.current);
      renewTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (status !== "takeover") {
      return;
    }
    if (effectiveTakeoverScope === "browser") {
      lockAndSwitchMode("browser");
      return;
    }
    lockAndSwitchMode("shell");
  }, [effectiveTakeoverScope, lockAndSwitchMode, status]);

  useEffect(() => {
    const wasHidden = !prevPageVisibleRef.current;
    prevPageVisibleRef.current = isPageVisible;

    if (status !== "takeover" || !takeoverId || !isPageVisible) {
      clearRenewTimer();
      return;
    }

    let disposed = false;

    const scheduleNext = (intervalMsOverride?: number | null) => {
      clearRenewTimer();
      const delayMs =
        intervalMsOverride != null && Number.isFinite(intervalMsOverride)
          ? Math.max(1, Math.floor(intervalMsOverride))
          : renewIntervalMs;
      renewTimerRef.current = window.setTimeout(() => {
        void renewLoop();
      }, delayMs);
    };

    const renewOnce = async (): Promise<{
      shouldContinue: boolean;
      nextIntervalMs: number | null;
    }> => {
      try {
        const renewResult = await sessionApi.renewTakeover(sessionId, {
          takeover_id: takeoverId,
        });
        const renewedExpiresAt = normalizeUnixSeconds(renewResult.expires_at);
        if (renewedExpiresAt != null) {
          const renewedTtlMs = Math.max(renewedExpiresAt * 1000 - Date.now(), 1);
          return {
            shouldContinue: true,
            nextIntervalMs: computeRenewIntervalMs(renewedTtlMs),
          };
        }
        return { shouldContinue: true, nextIntervalMs: null };
      } catch (error) {
        const httpStatus = normalizeErrorHttpStatus(error);
        if (httpStatus === 409) {
          setMessage({
            type: "error",
            text: error instanceof Error ? error.message : "接管已失效",
          });
          await fetchSessionById(sessionId, { silent: true });
          return { shouldContinue: false, nextIntervalMs: null };
        }

        // 4xx 客户端错误不重试，直接停止续期（403=无权限，400=参数错误等均不可恢复）
        if (httpStatus != null && httpStatus >= 400 && httpStatus < 500) {
          setMessage({
            type: "error",
            text:
              error instanceof Error
                ? error.message
                : "接管续期失败（客户端错误），已停止续期",
          });
          await fetchSessionById(sessionId, { silent: true });
          return { shouldContinue: false, nextIntervalMs: null };
        }

        try {
          const renewResult = await sessionApi.renewTakeover(sessionId, {
            takeover_id: takeoverId,
          });
          const renewedExpiresAt = normalizeUnixSeconds(renewResult.expires_at);
          if (renewedExpiresAt != null) {
            const renewedTtlMs = Math.max(renewedExpiresAt * 1000 - Date.now(), 1);
            return {
              shouldContinue: true,
              nextIntervalMs: computeRenewIntervalMs(renewedTtlMs),
            };
          }
          return { shouldContinue: true, nextIntervalMs: null };
        } catch (retryError) {
          setMessage({
            type: "error",
            text:
              retryError instanceof Error
                ? retryError.message
                : "接管续期失败，将在下一周期重试",
          });
          return { shouldContinue: true, nextIntervalMs: null };
        }
      }
    };

    const renewLoop = async () => {
      if (disposed) {
        return;
      }
      const renewState = await renewOnce();
      if (disposed) {
        return;
      }
      if (!renewState.shouldContinue) {
        clearRenewTimer();
        return;
      }
      scheduleNext(renewState.nextIntervalMs);
    };

    if (wasHidden) {
      // 页面从隐藏恢复可见 → 立即续期一次
      void renewLoop();
    } else {
      // 首次进入接管 → 基于剩余 TTL 计算初始延迟，防止 TTL<5s 时过期
      const SAFETY_MARGIN_MS = 2_000;
      const initialDelayMs = Math.min(5_000, Math.max(0, takeoverTtlMs - SAFETY_MARGIN_MS));
      scheduleNext(initialDelayMs);
    }

    return () => {
      disposed = true;
      clearRenewTimer();
    };
  }, [
    clearRenewTimer,
    fetchSessionById,
    isPageVisible,
    renewIntervalMs,
    sessionId,
    setMessage,
    status,
    takeoverId,
    takeoverTtlMs,
  ]);

  const handleStartTakeover = async (scope: TakeoverScope) => {
    setTakeoverSubmitting(true);
    try {
      const result = await sessionApi.startTakeover(sessionId, { scope });
      lockAndSwitchMode(scope === "browser" ? "browser" : "shell");
      await fetchSessionById(sessionId, { silent: true });
      setMessage({
        type: "success",
        text:
          result.request_status === "starting"
            ? "已发起接管，等待任务暂停后进入接管模式"
            : "已进入接管模式",
      });
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "主动接管失败",
      });
    } finally {
      setTakeoverSubmitting(false);
    }
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

  const handleRejectTakeover = async (decision: "continue" | "terminate") => {
    setTakeoverSubmitting(true);
    try {
      await sessionApi.rejectTakeover(sessionId, { decision });
      await fetchSessionById(sessionId, { silent: true });
      setMessage({
        type: "success",
        text:
          decision === "terminate"
            ? "已拒绝接管并结束任务"
            : "已拒绝接管，继续由 AI 执行",
      });
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "拒绝接管失败",
      });
    } finally {
      setTakeoverSubmitting(false);
    }
  };

  const handleTerminalTakeoverInvalidated = useCallback(
    (reason: string) => {
      setMessage({
        type: "error",
        text: reason,
      });
      void fetchSessionById(sessionId, { silent: true });
    },
    [fetchSessionById, sessionId, setMessage]
  );

  return (
    <section className="flex h-full min-h-[560px] flex-col rounded-3xl border border-border bg-surface-2 p-3 shadow-[var(--shadow-card)]">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-[32px] font-semibold tracking-tight text-foreground">Actus 的电脑</h2>
        <div className="flex items-center gap-2">
          {canStartTakeover ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="rounded-xl border-border text-foreground/80"
                  disabled={takeoverSubmitting}
                >
                  主动接管
                  <ChevronDown size={14} />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-40">
                <DropdownMenuItem
                  disabled={takeoverSubmitting}
                  onClick={() => {
                    void handleStartTakeover("shell");
                  }}
                >
                  接管终端
                </DropdownMenuItem>
                <DropdownMenuItem
                  disabled={takeoverSubmitting}
                  onClick={() => {
                    void handleStartTakeover("browser");
                  }}
                >
                  接管浏览器
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}

          {canEndTakeover ? (
            <Button
              variant="outline"
              size="sm"
              className="rounded-xl border-border text-foreground/80"
              disabled={takeoverSubmitting}
              onClick={() => {
                void handleEndTakeover();
              }}
            >
              结束接管
            </Button>
          ) : null}

          {canRejectTakeover ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="rounded-xl border-border text-foreground/80"
                  disabled={takeoverSubmitting}
                >
                  拒绝接管（继续执行）
                  <ChevronDown size={14} />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem
                  disabled={takeoverSubmitting}
                  onClick={() => {
                    void handleRejectTakeover("continue");
                  }}
                >
                  拒绝接管（继续执行）
                </DropdownMenuItem>
                <DropdownMenuItem
                  disabled={takeoverSubmitting}
                  onClick={() => {
                    void handleRejectTakeover("terminate");
                  }}
                >
                  拒绝并结束任务
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}

          <Link
            href={`/sessions/${sessionId}/novnc`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center rounded-xl border border-border bg-card p-2 text-muted-foreground transition-colors hover:bg-accent"
            title="打开实时窗口（VNC）"
          >
            <Expand size={16} />
          </Link>
        </div>
      </div>

      <div className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
        {selectedMode === "shell" ? <TerminalSquare size={14} /> : <Globe size={14} />}
        <span>
          {selectedMode === "shell"
            ? "Actus 正在使用终端"
            : "Actus 正在使用浏览器"}
        </span>
      </div>

      <div className="mb-3 flex items-center gap-2">
        <button
          type="button"
          onClick={() => {
            lockAndSwitchMode("shell");
          }}
          className={cn(
            "rounded-full border px-2.5 py-1 text-xs transition-colors",
            selectedMode === "shell"
              ? "border-accent-brand bg-accent-brand text-accent-brand-foreground"
              : "border-border bg-card text-muted-foreground hover:bg-accent"
          )}
        >
          终端
        </button>
        <button
          type="button"
          onClick={() => {
            lockAndSwitchMode("browser");
          }}
          className={cn(
            "rounded-full border px-2.5 py-1 text-xs transition-colors",
            selectedMode === "browser"
              ? "border-accent-brand bg-accent-brand text-accent-brand-foreground"
              : "border-border bg-card text-muted-foreground hover:bg-accent"
          )}
        >
          浏览器
        </button>
        {latestModeSnapshot == null ? (
          <span className="text-xs text-muted-foreground">当前模式暂无可展示快照</span>
        ) : null}
      </div>

      <div
        className="relative min-h-0 flex-1"
        onMouseEnter={() => setContentHovered(true)}
        onMouseLeave={() => setContentHovered(false)}
      >
        {interactiveShellTakeover ? (
          <>
            <div className={cn("h-full", selectedMode === "shell" ? "block" : "hidden")}>
              <WorkbenchInteractiveTerminal
                sessionId={sessionId}
                takeoverId={takeoverId as string}
                active={selectedMode === "shell"}
                panelVisible={visible}
                onTakeoverInvalidated={handleTerminalTakeoverInvalidated}
              />
            </div>
            <div className={cn("h-full", selectedMode === "browser" ? "block" : "hidden")}>
              <WorkbenchBrowserPreview
                snapshot={browserSnapshot}
                onPreviewImage={onPreviewImage}
              />
            </div>
          </>
        ) : selectedMode === "browser" ? (
          <WorkbenchBrowserPreview
            snapshot={browserSnapshot}
            onPreviewImage={onPreviewImage}
          />
        ) : (
          <WorkbenchTerminalPreview
            snapshot={shellSnapshot}
            shellSessionIds={shellSessionIds}
            selectedShellSessionId={effectiveShellSessionId}
            onSelectShellSessionId={setSelectedShellSessionId}
            isHistoryMode={isHistoryMode}
            realtimeRecords={shellPreview.consoleRecords}
            realtimeOutput={shellPreview.output}
            realtimeLoading={shellPreview.loading}
            realtimeError={shellPreview.error}
          />
        )}

        <div
          className={cn(
            "absolute inset-x-2 bottom-2 z-20 transition-all duration-200",
            showTimelineOverlay
              ? "translate-y-0 opacity-100 pointer-events-auto"
              : "translate-y-2 opacity-0 pointer-events-none"
          )}
        >
          <WorkbenchTimeline
            snapshots={snapshots}
            cursorTimestamp={activeTimestamp}
            cursorState={cursorState}
            isLocked={liveLocked}
            hasNewRealtime={hasNewRealtime}
            onScrubStart={handleScrubStart}
            onScrub={handleScrub}
            onScrubEnd={handleScrubEnd}
            onBackToLive={handleBackToLive}
            onToggleLock={toggleLock}
          />
        </div>
      </div>

      {atEarliestRecord ? (
        <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-700 dark:border-amber-500/20 dark:bg-amber-500/10 dark:text-amber-400">
          已到最早记录
        </div>
      ) : null}

      {snapshots.length === 0 ? (
        <div className="mt-3 rounded-xl border border-dashed border-border-strong bg-card px-3 py-3 text-sm text-muted-foreground">
          暂无可回看状态
        </div>
      ) : null}

      {currentModeSnapshot == null && timelineSnapshot != null ? (
        <p className="mt-2 text-xs text-muted-foreground">
          该时间点没有 {selectedMode === "shell" ? "终端" : "浏览器"} 快照，已保留当前模式。
        </p>
      ) : null}
    </section>
  );
});
