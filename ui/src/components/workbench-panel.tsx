"use client";

import Link from "next/link";
import { Expand, Globe, TerminalSquare } from "lucide-react";
import { memo, useMemo, useState } from "react";

import { useIsMobile } from "@/hooks/use-mobile";
import { useShellPreview } from "@/hooks/use-shell-preview";
import {
  findSnapshotAtOrBefore,
  getLatestSnapshotByMode,
  getShellSessionIdsFromSnapshots,
  type TimelineCursorState,
  type WorkbenchMode,
  type WorkbenchSnapshot,
} from "@/lib/session-ui";
import { cn } from "@/lib/utils";

import { WorkbenchBrowserPreview } from "./workbench-browser-preview";
import { WorkbenchTerminalPreview } from "./workbench-terminal-preview";
import { WorkbenchTimeline } from "./workbench-timeline";

type WorkbenchPanelProps = {
  sessionId: string;
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

export const WorkbenchPanel = memo(function WorkbenchPanel({
  sessionId,
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
  const isMobile = useIsMobile();

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
  const showTimelineOverlay =
    isMobile || contentHovered || isHistoryMode || hasNewRealtime;

  const shouldPollShell =
    visible &&
    running &&
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

  return (
    <section className="flex h-full min-h-[560px] flex-col rounded-3xl border border-border bg-surface-2 p-3 shadow-[var(--shadow-card)]">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-[32px] font-semibold tracking-tight text-foreground">Actus 的电脑</h2>
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
            setManualMode("shell");
            if (!isHistoryMode && !liveLocked) {
              setLiveLocked(true);
              setCursorState("live_locked");
            }
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
            setManualMode("browser");
            if (!isHistoryMode && !liveLocked) {
              setLiveLocked(true);
              setCursorState("live_locked");
            }
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
        {selectedMode === "browser" ? (
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
