"use client";

import { Lock, Unlock } from "lucide-react";
import { useRef } from "react";

import type { TimelineCursorState, WorkbenchSnapshot } from "@/lib/session-ui";
import { formatWorkbenchClock } from "@/lib/session-ui";
import { cn } from "@/lib/utils";

type WorkbenchTimelineProps = {
  snapshots: WorkbenchSnapshot[];
  cursorTimestamp: number | null;
  cursorState: TimelineCursorState;
  isLocked: boolean;
  hasNewRealtime: boolean;
  onScrubStart: () => void;
  onScrub: (timestamp: number) => void;
  onScrubEnd: (timestamp: number) => void;
  onBackToLive: () => void;
  onToggleLock: () => void;
};

function findIndexByTimestamp(snapshots: WorkbenchSnapshot[], timestamp: number | null): number {
  if (snapshots.length === 0) {
    return 0;
  }
  if (timestamp == null) {
    return snapshots.length - 1;
  }
  let index = 0;
  for (let i = 0; i < snapshots.length; i += 1) {
    const snapshot = snapshots[i];
    if (!snapshot) {
      continue;
    }
    if (snapshot.timestamp <= timestamp) {
      index = i;
      continue;
    }
    break;
  }
  return index;
}

export function WorkbenchTimeline({
  snapshots,
  cursorTimestamp,
  cursorState,
  isLocked,
  hasNewRealtime,
  onScrubStart,
  onScrub,
  onScrubEnd,
  onBackToLive,
  onToggleLock,
}: WorkbenchTimelineProps) {
  const lastScrubTimestampRef = useRef<number | null>(null);
  const hasSnapshots = snapshots.length > 0;
  const canScrub = snapshots.length > 1;
  const currentIndex = findIndexByTimestamp(snapshots, cursorTimestamp);
  const selectedSnapshot = hasSnapshots ? snapshots[currentIndex] : null;
  const isHistory = cursorState === "history_paused" || cursorState === "history_scrubbing";
  const currentPercent =
    snapshots.length <= 1 ? 0 : (currentIndex / (snapshots.length - 1)) * 100;

  const handleRangeChange = (nextValue: string) => {
    const index = Number.parseInt(nextValue, 10);
    if (Number.isNaN(index)) {
      return;
    }
    const snapshot = snapshots[index];
    if (!snapshot) {
      return;
    }
    lastScrubTimestampRef.current = snapshot.timestamp;
    onScrub(snapshot.timestamp);
  };

  const handleRangeCommit = (nextValue: string) => {
    const index = Number.parseInt(nextValue, 10);
    if (Number.isNaN(index)) {
      return;
    }
    const snapshot = snapshots[index];
    const commitTimestamp = lastScrubTimestampRef.current ?? snapshot?.timestamp ?? null;
    if (commitTimestamp == null) {
      return;
    }
    onScrub(commitTimestamp);
    onScrubEnd(commitTimestamp);
    lastScrubTimestampRef.current = null;
  };

  return (
    <div className="space-y-2 rounded-xl border border-border bg-card/95 p-3 backdrop-blur-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onToggleLock}
            className="inline-flex items-center gap-1 rounded-lg border border-border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent"
            title={isLocked ? "取消锁定视图" : "锁定当前视图"}
          >
            {isLocked ? <Lock size={12} /> : <Unlock size={12} />}
            {isLocked ? "已锁定" : "自动跟随"}
          </button>
          {hasNewRealtime && isHistory ? (
            <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-500/10 dark:text-amber-400">
              有新实时状态
            </span>
          ) : null}
        </div>
        {isHistory ? (
          <button
            type="button"
            onClick={onBackToLive}
            className="rounded-lg bg-accent-brand px-2.5 py-1 text-xs font-medium text-accent-brand-foreground transition-colors hover:opacity-90"
          >
            回到实时
          </button>
        ) : null}
      </div>

      <div className="space-y-2">
        <div className="relative h-5">
          <div className="pointer-events-none absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-border" />
          <div
            className="pointer-events-none absolute left-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-accent-brand"
            style={{ width: `${currentPercent}%` }}
          />
          {snapshots.map((snapshot, index) => {
            const markerPercent =
              snapshots.length <= 1 ? 0 : (index / (snapshots.length - 1)) * 100;
            const isBrowser = snapshot.mode === "browser";
            return (
              <span
                key={`${snapshot.id}-${snapshot.mode}`}
                data-testid={`timeline-marker-${snapshot.mode}`}
                className={cn(
                  "pointer-events-none absolute top-1/2 block h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white shadow-sm",
                  isBrowser ? "bg-sky-500" : "bg-emerald-500"
                )}
                style={{ left: `${markerPercent}%` }}
                title={isBrowser ? "浏览器快照点" : "终端快照点"}
              />
            );
          })}
          <span
            className="pointer-events-none absolute top-1/2 block h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-accent-brand bg-card shadow"
            style={{ left: `${currentPercent}%` }}
          />
          <input
            type="range"
            min={0}
            max={Math.max(0, snapshots.length - 1)}
            value={currentIndex}
            disabled={!canScrub}
            onMouseDown={onScrubStart}
            onTouchStart={onScrubStart}
            onChange={(event) => handleRangeChange(event.target.value)}
            onMouseUp={(event) => handleRangeCommit(event.currentTarget.value)}
            onTouchEnd={(event) => handleRangeCommit(event.currentTarget.value)}
            onKeyUp={(event) => handleRangeCommit(event.currentTarget.value)}
            className={cn(
              "absolute inset-0 h-5 w-full cursor-pointer opacity-0",
              !canScrub && "cursor-not-allowed"
            )}
          />
        </div>
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            Shell
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-sky-500" />
            Browser
          </span>
        </div>
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{selectedSnapshot ? formatWorkbenchClock(selectedSnapshot.timestamp * 1000) : "--:--:--"}</span>
          <span
            className={cn(
              "rounded-full px-2 py-0.5",
              isHistory ? "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400" : "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400"
            )}
          >
            {isHistory ? "历史" : "实时"}
          </span>
        </div>
      </div>
    </div>
  );
}
