"use client";

import { Loader2, TerminalSquare } from "lucide-react";
import { memo, useState } from "react";

import type { ShellConsoleRecord } from "@/lib/api/types";
import type { WorkbenchSnapshot } from "@/lib/session-ui";
import { cn } from "@/lib/utils";

type WorkbenchTerminalPreviewProps = {
  snapshot: WorkbenchSnapshot | null;
  shellSessionIds: string[];
  selectedShellSessionId: string | null;
  onSelectShellSessionId: (shellSessionId: string) => void;
  isHistoryMode: boolean;
  realtimeRecords: ShellConsoleRecord[];
  realtimeOutput: string;
  realtimeLoading: boolean;
  realtimeError: string | null;
};

const PREVIEW_RECORD_LIMIT = 80;

function formatRecord(record: Pick<ShellConsoleRecord, "ps1" | "command" | "output">): string {
  return `${record.ps1}${record.command}\n${record.output}`.trim();
}

export const WorkbenchTerminalPreview = memo(function WorkbenchTerminalPreview({
  snapshot,
  shellSessionIds,
  selectedShellSessionId,
  onSelectShellSessionId,
  isHistoryMode,
  realtimeRecords,
  realtimeOutput,
  realtimeLoading,
  realtimeError,
}: WorkbenchTerminalPreviewProps) {
  const [expanded, setExpanded] = useState(false);
  const historyRecords = !snapshot?.consoleRecords
    ? []
    : snapshot.consoleRecords.map((record) => ({
        ps1: record.ps1 || "",
        command: record.command || "",
        output: record.output || "",
      }));

  const rawRecords = isHistoryMode ? historyRecords : realtimeRecords;
  const records = expanded ? rawRecords : rawRecords.slice(-PREVIEW_RECORD_LIMIT);
  const joinedRecords = records.map(formatRecord).filter(Boolean).join("\n\n");
  const fallbackText = isHistoryMode ? "" : realtimeOutput;
  const terminalText = joinedRecords || fallbackText || "暂无终端输出";

  return (
    <div className="flex h-full flex-col rounded-xl border border-border bg-card">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <TerminalSquare size={14} className="text-muted-foreground" />
        <p className="text-sm text-foreground/85">终端会话</p>
        {snapshot?.command ? (
          <span className="max-w-[50%] truncate rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
            {snapshot.command}
          </span>
        ) : null}
      </div>

      {shellSessionIds.length > 0 ? (
        <div className="flex flex-wrap gap-2 border-b border-border px-3 py-2">
          {shellSessionIds.map((shellSessionId) => (
            <button
              key={shellSessionId}
              type="button"
              onClick={() => onSelectShellSessionId(shellSessionId)}
              className={cn(
                "rounded-full border px-2 py-0.5 text-xs transition-colors",
                selectedShellSessionId === shellSessionId
                  ? "border-accent-brand bg-accent-brand text-accent-brand-foreground"
                  : "border-border bg-card text-muted-foreground hover:bg-accent"
              )}
            >
              {shellSessionId.slice(0, 8)}
            </button>
          ))}
        </div>
      ) : null}

      <div className="min-h-0 flex-1 p-2">
        <div className="h-full overflow-auto rounded-lg border border-border bg-surface-1 p-3">
          {realtimeLoading && !isHistoryMode ? (
            <div className="mb-2 inline-flex items-center gap-1 text-xs text-muted-foreground">
              <Loader2 size={12} className="animate-spin" />
              刷新终端输出中...
            </div>
          ) : null}

          {realtimeError && !isHistoryMode ? (
            <div className="mb-2 rounded-md border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
              {realtimeError}
            </div>
          ) : null}

          <pre className="whitespace-pre-wrap break-words text-xs leading-6 text-foreground/85">{terminalText}</pre>
        </div>
      </div>

      {rawRecords.length > PREVIEW_RECORD_LIMIT ? (
        <div className="border-t border-border px-3 py-2">
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="text-xs text-accent-brand hover:underline"
          >
            {expanded ? "收起输出" : `展开完整输出（${rawRecords.length} 条）`}
          </button>
        </div>
      ) : null}
    </div>
  );
});
