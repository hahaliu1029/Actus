"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { memo, useMemo, useState } from "react";

import type { FileInfo } from "@/lib/api/types";
import { formatFileSize, type SessionProgressSummary } from "@/lib/session-ui";
import { cn } from "@/lib/utils";

type SessionTaskDockProps = {
  summary: SessionProgressSummary | null;
  files: FileInfo[];
  running?: boolean;
  onPreviewFile: (file: FileInfo) => void;
  onDownloadFile: (file: FileInfo) => void;
  className?: string;
};

export const SessionTaskDock = memo(function SessionTaskDock({
  summary,
  files,
  running = false,
  onPreviewFile,
  onDownloadFile,
  className,
}: Readonly<SessionTaskDockProps>) {
  const [expanded, setExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<"progress" | "files">("progress");

  const sortedFiles = useMemo(() => [...files].reverse(), [files]);

  if (!summary) {
    return null;
  }

  const progressText = `${summary.completed}/${summary.total}`;

  return (
    <div className={cn("w-full", className)}>
      <div className="mx-auto w-full max-w-4xl">
        <div className="rounded-2xl border border-border bg-card shadow-[var(--shadow-elevated)]">
          <button
            type="button"
            aria-label={expanded ? "收起任务摘要" : "展开任务摘要"}
            className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
            onClick={() => setExpanded((prev) => !prev)}
          >
            <div className="min-w-0">
              <p className="text-xs text-muted-foreground">任务摘要</p>
              <p className="truncate text-sm font-medium text-foreground">{summary.currentStep}</p>
            </div>
            <div className="flex shrink-0 items-center gap-2 text-xs text-muted-foreground">
              {running ? (
                <span className="rounded-full bg-amber-50 px-2 py-0.5 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400">运行中</span>
              ) : null}
              <span>{progressText}</span>
              <span>{`文件 ${files.length}`}</span>
              {expanded ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            </div>
          </button>

          {expanded ? (
            <div className="border-t border-border px-4 pb-4 pt-3">
              <div role="tablist" className="mb-3 flex items-center gap-2">
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === "progress"}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs transition-colors",
                    activeTab === "progress"
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border bg-card text-muted-foreground hover:bg-accent"
                  )}
                  onClick={() => setActiveTab("progress")}
                >
                  进度
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === "files"}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs transition-colors",
                    activeTab === "files"
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border bg-card text-muted-foreground hover:bg-accent"
                  )}
                  onClick={() => setActiveTab("files")}
                >
                  文件
                </button>
              </div>

              {activeTab === "progress" ? (
                <div className="space-y-2 rounded-xl border border-border bg-muted p-3 text-sm text-foreground/85">
                  <p>{`完成度：${summary.completed}/${summary.total}`}</p>
                  <p>{`当前步骤：${summary.currentStep}`}</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {sortedFiles.length === 0 ? (
                    <p className="rounded-xl border border-border bg-muted p-3 text-sm text-muted-foreground">
                      暂无文件
                    </p>
                  ) : (
                    sortedFiles.map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center justify-between gap-2 rounded-xl border border-border bg-muted p-2"
                      >
                        <button
                          type="button"
                          aria-label={`预览文件 ${file.filename}`}
                          className="min-w-0 flex-1 text-left"
                          onClick={() => onPreviewFile(file)}
                        >
                          <p className="truncate text-sm font-medium text-foreground/85">{file.filename}</p>
                          <p className="text-xs text-muted-foreground">{formatFileSize(file.size)}</p>
                        </button>
                        <button
                          type="button"
                          aria-label={`下载文件 ${file.filename}`}
                          className="shrink-0 rounded-lg border border-border bg-card px-2 py-1 text-xs text-foreground/80 transition-colors hover:bg-accent"
                          onClick={(event) => {
                            event.stopPropagation();
                            onDownloadFile(file);
                          }}
                        >
                          下载
                        </button>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
});
